"""Download DCT-Net (Alibaba DAMO Academy, Apache-2.0) and convert it to ptoart ONNX styles.

Usage:
    pip install "tensorflow==2.15.*" "tf2onnx==1.16.1" "onnx==1.16.*" "numpy<2" onnxruntime
    python tools/export_dctnet.py -o weights/styles          # anime, 3d, handdrawn
    python tools/export_dctnet.py anime -o weights/styles

Source: ModelScope damo/cv_unet_person-image-cartoon*_compound-models
(https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models),
code https://github.com/menyifang/DCT-Net. Both Apache License 2.0.

Each upstream frozen graph takes one BGR HWC float image in [0,255] (sides multiples of 16)
and returns BGR uint8. tf2onnx converts it, then a small wrapper gives it the same contract
as every other ptoart style (NCHW RGB float in [-1,1], input ``input``, output ``output``,
any batch of 1), so desktop StyleBackend and Android StyleModel run it unchanged.
Metadata: ``ptoart_multiple=16`` (side multiple) and, on head models, ``ptoart_face_side=288``
plus ``ptoart_face=dctnet`` (DCT-Net's own 0.75-ratio 5-point head alignment).

Files written per variant: ``dct_<v>.onnx`` (head model, used for faces) and, when the
upstream background model differs from the head model, ``dct_<v>_bg.onnx``.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "https://modelscope.cn/api/v1/models/damo/{}/repo?Revision=master&FilePath={}"
VARIANTS = {
    "anime": "cv_unet_person-image-cartoon_compound-models",
    "3d": "cv_unet_person-image-cartoon-3d_compound-models",
    "handdrawn": "cv_unet_person-image-cartoon-handdrawn_compound-models",
    "sketch": "cv_unet_person-image-cartoon-sketch_compound-models",
    "artstyle": "cv_unet_person-image-cartoon-artstyle_compound-models",
}
DEFAULT = ("anime", "3d", "handdrawn")
# SHA-256 of the upstream frozen graphs as listed by the ModelScope file API (head, background).
EXPECTED = {
    "anime": ("338e7bf7ab4014ba9da864c2dda104f26d8af6c7f468038782d92ccc8e2dedda",
              "71efec51b4566cce3a3123822fb4e17794ad062c3e324b33f2a65df850c037ad"),
    "3d": ("bca966dded317de32b6870b758fd01c6ab8f6ab76a285149fbf450a71624fe80",) * 2,
    "handdrawn": ("b5c8c54cfc96420feacdde140c7c81c3b8e6612162d58bc70a6fd4d35b340ce2",
                  "fc1553a316698b1e7fa85a286bd2994e29ae8ece233004cb42192b5b17accc64"),
    "sketch": ("a6acab53c080318a6c8f0571e8db0f7922ad6bd84005ae30115a4dc6c89ed90f",) * 2,
    "artstyle": ("34ceb2100f1fcb665afa9af44d2bd49f280a4a031fc74681370303e61993bec9",) * 2,
}


def download(url, target, sha256):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=300) as response, target.open("wb") as handle:
            handle.write(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != sha256:
        target.unlink()
        raise SystemExit(f"{target}: SHA-256 {digest} does not match upstream {sha256}")
    return target


def tf_to_onnx(pb_path):
    import tensorflow as tf
    import tf2onnx

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(Path(pb_path).read_bytes())
    model, _ = tf2onnx.convert.from_graph_def(
        graph_def, input_names=["input_image:0"], output_names=["output_image:0"], opset=13)
    return model


def wrap(model, *, head):
    """NCHW RGB [-1,1] -> upstream BGR HWC [0,255] -> NCHW RGB [-1,1]."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    graph = model.graph
    old_in, old_out = graph.input[0].name, graph.output[0].name
    raw_in, raw_out = "dct_in_hwc", "dct_out_hwc"
    for node in graph.node:
        node.input[:] = [raw_in if x == old_in else x for x in node.input]
        node.output[:] = [raw_out if x == old_out else x for x in node.output]
    consts = {
        "dct_half": np.array(127.5, np.float32),
        "dct_bgr": np.array([2, 1, 0], np.int64),
        "dct_axis0": np.array([0], np.int64),
    }
    graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in consts.items())
    pre = [
        helper.make_node("Mul", ["input", "dct_half"], ["dct_a"]),
        helper.make_node("Add", ["dct_a", "dct_half"], ["dct_b"]),
        helper.make_node("Gather", ["dct_b", "dct_bgr"], ["dct_c"], axis=1),
        helper.make_node("Squeeze", ["dct_c", "dct_axis0"], ["dct_d"]),
        helper.make_node("Transpose", ["dct_d"], [raw_in], perm=[1, 2, 0]),
    ]
    post = [
        helper.make_node("Cast", [raw_out], ["dct_e"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Transpose", ["dct_e"], ["dct_f"], perm=[2, 0, 1]),
        helper.make_node("Gather", ["dct_f", "dct_bgr"], ["dct_g"], axis=0),
        helper.make_node("Unsqueeze", ["dct_g", "dct_axis0"], ["dct_h"]),
        helper.make_node("Div", ["dct_h", "dct_half"], ["dct_i"]),
        helper.make_node("Sub", ["dct_i", "dct_one"], ["output"]),
    ]
    graph.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), "dct_one"))
    nodes = pre + list(graph.node) + post
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.input[:]
    del graph.output[:]
    dims = [1, 3, "H", "W"]
    graph.input.append(helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, dims))
    graph.output.append(helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, dims))
    meta = {"ptoart_kind": "style", "ptoart_multiple": "16", "ptoart_source": "DCT-Net (Apache-2.0)"}
    if head:
        meta.update(ptoart_face="dctnet", ptoart_face_side="288")
    for key, value in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def verify(pb_path, onnx_path):
    """Compare the wrapped ONNX with the upstream TF graph on a random 256x320 image."""
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (256, 320, 3)).astype(np.float32)
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(Path(pb_path).read_bytes())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="m")
        with tf.compat.v1.Session(graph=graph) as session:
            ref = session.run("m/output_image:0", {"m/input_image:0": rgb[:, :, ::-1]})[:, :, ::-1]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tensor = (rgb / 127.5 - 1).transpose(2, 0, 1)[None].astype(np.float32)
    out = (session.run(None, {"input": tensor})[0][0].transpose(1, 2, 0) + 1) * 127.5
    diff = np.abs(out - ref.astype(np.float32))
    print(f"  max |onnx - tf| = {diff.max():.2f}, mean = {diff.mean():.4f} (uint8 levels)")
    if diff.mean() > 1.0:
        raise SystemExit(f"{onnx_path}: ONNX output does not match TensorFlow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", help=f"any of {', '.join(VARIANTS)} (default: {' '.join(DEFAULT)})")
    parser.add_argument("-o", "--out", default="styles")
    parser.add_argument("--cache", default=".dctnet-src")
    args = parser.parse_args()
    import onnx

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {unknown}")
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.variants or DEFAULT:
        repo = VARIANTS[name]
        head_sha, bg_sha = EXPECTED[name]
        head = download(REPO.format(repo, "cartoon_h.pb"), cache / name / "cartoon_h.pb", head_sha)
        bg = download(REPO.format(repo, "cartoon_bg.pb"), cache / name / "cartoon_bg.pb", bg_sha)
        pairs = [(head, out / f"dct_{name}.onnx", True)]
        if head_sha != bg_sha:
            pairs.append((bg, out / f"dct_{name}_bg.onnx", False))
        for pb, target, is_head in pairs:
            onnx.save(wrap(tf_to_onnx(pb), head=is_head), target)
            print("ok", target)
            verify(pb, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
