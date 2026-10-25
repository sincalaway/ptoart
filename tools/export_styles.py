"""Download AnimeGANv2 style weights and export them to ONNX for ptoart.

Usage:
    pip install "torch>=2.5" onnx onnxscript onnxruntime   # plus tensorflow for hayao/shinkai
    python tools/export_styles.py                 # every style into ./styles
    python tools/export_styles.py hayao paprika -o styles

Each ``<style>.onnx`` takes float32 NCHW RGB in [-1,1] with dynamic height and
width (multiples of 4 are safest) and returns the same shape, output name ``output``.

Licensing: AnimeGANv2 weights are for non-commercial use only
(https://github.com/TachibanaYoshino/AnimeGANv2#license). The PyTorch port and
its converted weights come from https://github.com/bryandlee/animegan2-pytorch (MIT).
"""
import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PT_BASE = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/"
TF_BASE = "https://github.com/TachibanaYoshino/AnimeGANv2/raw/master/checkpoint/"

# name -> (source, location, Chinese label)
STYLES = {
    "hayao": ("tf", "generator_Hayao_weight/Hayao-99.ckpt", "宫崎骏"),
    "shinkai": ("tf", "generator_Shinkai_weight/Shinkai-53.ckpt", "新海诚"),
    "paprika": ("pt", "paprika.pt", "今敏 Paprika"),
    "celeba_distill": ("pt", "celeba_distill.pt", "CelebA 蒸馏"),
    "face_paint_512_v1": ("pt", "face_paint_512_v1.pt", "Face Paint v1"),
    "face_paint_512_v2": ("pt", "face_paint_512_v2.pt", "Face Paint v2"),
}


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
        handle.write(response.read())
    return target


def _convert_key(name):
    """Map a TF AnimeGANv2 generator variable to the PyTorch state_dict key.

    Logic follows bryandlee/animegan2-pytorch convert_weights.py (MIT).
    """
    k = name.replace("Conv/", "Conv_0/").replace("LayerNorm/", "LayerNorm_0/")
    keys = k.split("/")[2:]
    is_dconv = False
    if keys[0] == "C":
        if keys[1] in ["Conv_1", "LayerNorm_1"]:
            keys[1] = keys[1].replace("1", "5")
        if len(keys) == 4:
            if keys[1] == keys[2]:
                is_dconv = True
                keys[2] = "1.1"
            keys[2] = {"1": "1.2", "Conv_1": "2", "2": "3"}.get(keys[2], keys[2])
            keys[1] = keys[1].replace("r", "") + ".layers." + keys[2]
            keys[2] = keys[3]
            keys.pop(-1)
    if "out" in keys[0]:
        keys[1] = "0"
    if keys[0] in ["A", "B", "C", "D", "E"]:
        keys[0] = "block_" + keys[0].lower()
    if "LayerNorm_" in keys[1]:
        keys[1] = keys[1].replace("LayerNorm_", "") + ".2"
    if "Conv_" in keys[1]:
        keys[1] = keys[1].replace("Conv_", "") + ".1"
    keys[2] = {"weights": "weight", "w": "weight", "bias": "bias",
               "gamma": "weight", "beta": "bias"}[keys[2]]
    return ".".join(keys), is_dconv


def load_tf_checkpoint(prefix):
    """Read generator variables straight from the checkpoint; no TF graph code needed."""
    import tensorflow as tf
    import torch
    from models.animegan2 import Generator

    reader = tf.train.load_checkpoint(str(prefix))
    expected = Generator().state_dict()
    state = {}
    for name in reader.get_variable_to_shape_map():
        if not name.startswith("generator/") or "Adam" in name:
            continue
        key, is_dconv = _convert_key(name)
        value = torch.from_numpy(reader.get_tensor(name))
        if value.ndim == 4:
            value = value.permute(2, 3, 0, 1) if is_dconv else value.permute(3, 2, 0, 1)
        if key not in expected or expected[key].shape != value.shape:
            raise ValueError(f"Unexpected variable {name} -> {key} {tuple(value.shape)}")
        state[key] = value.contiguous()
    missing = set(expected) - set(state)
    if missing:
        raise ValueError(f"Checkpoint is missing {sorted(missing)[:5]}")
    return state


def build(style, workdir):
    import torch
    from models.animegan2 import Generator

    source, location, _ = STYLES[style]
    if source == "pt":
        path = download(PT_BASE + location, workdir / location)
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        for suffix in (".index", ".data-00000-of-00001"):
            download(TF_BASE + location + suffix, workdir / (location + suffix))
        state = load_tf_checkpoint(workdir / location)
    net = Generator()
    net.load_state_dict(state, strict=True)
    return net.eval()


def export(net, target, opset):
    import numpy as np
    import onnxruntime as ort
    import torch

    dummy = torch.rand(1, 3, 256, 256) * 2 - 1
    torch.onnx.export(net, dummy, str(target), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "height", 3: "width"},
                                    "output": {2: "height", 3: "width"}},
                      opset_version=opset, dynamo=False)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    for h, w in ((256, 256), (320, 480)):
        x = torch.rand(1, 3, h, w) * 2 - 1
        with torch.no_grad():
            expected = net(x).numpy()
        actual = session.run(["output"], {"input": x.numpy()})[0]
        error = float(np.abs(expected - actual).max())
        if actual.shape != expected.shape or error > 1e-2:
            raise RuntimeError(f"{target.name}: ONNX mismatch at {h}x{w} (max error {error:.2e})")
    print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB), verified")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("styles", nargs="*", default=[], metavar="STYLE", help=f"Any of: {', '.join(STYLES)} (default: all)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("styles"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)
    unknown = [s for s in args.styles if s not in STYLES]
    if unknown:
        parser.error(f"unknown style(s): {', '.join(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for style in args.styles or list(STYLES):
            print(f"{style} ({STYLES[style][2]})")
            export(build(style, Path(tmp)), args.output_dir / f"{style}.onnx", args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
