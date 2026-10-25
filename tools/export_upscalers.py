"""Download Real-ESRGAN compact x4 models and export them to ONNX for ptoart.

Usage:
    pip install "torch>=2.5" onnx onnxscript onnxruntime
    python tools/export_upscalers.py                      # both into ./upscale
    python tools/export_upscalers.py anime -o weights/upscale

Each ``<name>.onnx`` takes float32 NCHW RGB in [0,1] with dynamic height and
width and returns the 4x image. The file carries ``ptoart_kind=upscaler`` metadata
so the Android app recognises it. Models: https://github.com/xinntao/Real-ESRGAN (BSD-3).
"""
import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
# name -> (file, num_conv, description)
UPSCALERS = {
    "anime": ("realesr-animevideov3.pth", 16, "动漫/卡通，速度快"),
    "general": ("realesr-general-x4v3.pth", 32, "通用照片"),
}


def build(name, workdir):
    import torch
    from models.srvgg import SRVGGNetCompact

    filename, num_conv, _ = UPSCALERS[name]
    path = workdir / filename
    print(f"  downloading {BASE + filename}")
    with urllib.request.urlopen(BASE + filename, timeout=120) as response:
        path.write_bytes(response.read())
    params = torch.load(path, map_location="cpu", weights_only=True)
    net = SRVGGNetCompact(num_conv=num_conv)
    net.load_state_dict(params.get("params_ema", params.get("params", params)), strict=True)
    return net.eval()


def export(net, target, opset):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    torch.onnx.export(net, torch.rand(1, 3, 64, 64), str(target), input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {2: "height", 3: "width"},
                                    "output": {2: "height4", 3: "width4"}},
                      opset_version=opset, dynamo=False)
    model = onnx.load(str(target))
    entry = model.metadata_props.add()
    entry.key, entry.value = "ptoart_kind", "upscaler"
    entry = model.metadata_props.add()
    entry.key, entry.value = "ptoart_scale", "4"
    onnx.save(model, str(target))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    for h, w in ((64, 64), (48, 80)):
        x = torch.rand(1, 3, h, w)
        with torch.no_grad():
            expected = net(x).numpy()
        actual = session.run(["output"], {"input": x.numpy()})[0]
        error = float(np.abs(expected - actual).max())
        if actual.shape != (1, 3, h * 4, w * 4) or error > 1e-3:
            raise RuntimeError(f"{target.name}: ONNX mismatch at {h}x{w} (max error {error:.2e})")
    print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB), verified")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=[], metavar="NAME",
                        help=f"Any of: {', '.join(UPSCALERS)} (default: all)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("upscale"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)
    unknown = [n for n in args.names if n not in UPSCALERS]
    if unknown:
        parser.error(f"unknown upscaler(s): {', '.join(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.names or list(UPSCALERS):
            print(f"{name} ({UPSCALERS[name][2]})")
            export(build(name, Path(tmp)), args.output_dir / f"{name}.onnx", args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
