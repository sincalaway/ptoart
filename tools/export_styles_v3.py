"""Download AnimeGANv3 ONNX models and rewrap them to ptoart's NCHW [-1,1] style contract.

AnimeGANv3 ships NHWC graphs; we add a Transpose on the input and output so the
same StyleBackend (desktop) and StyleModel (Android) run them unchanged.
AnimeGANv3 weights are for non-commercial use (https://github.com/TachibanaYoshino/AnimeGANv3).
"""
import argparse
import urllib.request
from pathlib import Path

import onnx
from onnx import helper

SOURCES = {
    "hayao_v3": "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Hayao_36.onnx",
    "shinkai_v3": "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Shinkai_37.onnx",
    "ghibli_face": "https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/releases/download/1.0/"
                   "AnimeGANv3_large_Ghibli_c1_e299.onnx",
    "portrait_sketch": "https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/releases/download/1.0/"
                       "AnimeGANv3_PortraitSketch_25.onnx",
    "jp_face": "https://huggingface.co/vumichien/AnimeGANv3_JP_face/resolve/main/AnimeGANv3_JP_face.onnx",
}


def to_nchw(model):
    graph = model.graph
    old_in, old_out = graph.input[0], graph.output[0]
    nhwc_in, nhwc_out = old_in.name + "_nhwc", old_out.name + "_nhwc"
    for node in graph.node:
        node.input[:] = [nhwc_in if x == old_in.name else x for x in node.input]
        node.output[:] = [nhwc_out if x == old_out.name else x for x in node.output]
    graph.node.insert(0, helper.make_node("Transpose", ["input"], [nhwc_in], perm=[0, 2, 3, 1]))
    graph.node.append(helper.make_node("Transpose", [nhwc_out], ["output"], perm=[0, 3, 1, 2]))
    dims = ["N", 3, "H", "W"]
    del graph.input[:]
    del graph.output[:]
    graph.input.append(helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, dims))
    graph.output.append(helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, dims))
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", default="styles")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        raw = out / f"{name}.raw.onnx"
        urllib.request.urlretrieve(url, raw)
        onnx.save(to_nchw(onnx.load(raw)), out / f"{name}.onnx")
        raw.unlink()
        print("ok", name)


if __name__ == "__main__":
    main()
