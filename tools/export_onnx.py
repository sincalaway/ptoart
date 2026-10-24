"""Export photo2cartoon_weights.pt (Torch, genA2B) to photo2cartoon_weights.onnx.

Usage:
    pip install "torch>=2.5" onnx onnxscript onnxruntime
    python tools/export_onnx.py photo2cartoon_weights.pt photo2cartoon_weights.onnx

The ONNX file is what the Android app and ``--backend onnx`` expect:
input float32 [1,3,256,256] in [-1,1], output named ``output`` of the same shape.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Generator:
    @staticmethod
    def build(weights):
        import torch
        from models.networks import ResnetGenerator

        net = ResnetGenerator(ngf=32, img_size=256, light=True)
        params = torch.load(weights, map_location="cpu", weights_only=True)
        net.load_state_dict(params["genA2B"], strict=True)
        net.eval()

        class OnlyImage(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x)[0]

        return OnlyImage(net)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", type=Path, help="photo2cartoon_weights.pt")
    parser.add_argument("output", type=Path, nargs="?", default=Path("photo2cartoon_weights.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    import torch

    model = _Generator.build(args.weights)
    dummy = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        # The legacy TorchScript exporter cannot export this network's
        # InstanceNorm with an unknown channel size, so use torch.export (torch >= 2.5).
        torch.onnx.export(model, (dummy,), str(args.output), input_names=["input"],
                          output_names=["output"], opset_version=args.opset,
                          dynamo=True, external_data=False)
        probe = torch.rand(1, 3, 256, 256) * 2 - 1
        reference = model(probe).numpy()
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        result = session.run(["output"], {"input": probe.numpy()})[0]
        print(f"ONNX Runtime check: max abs diff {float(np.abs(result - reference).max()):.2e}")
    except ImportError:
        print("onnxruntime not installed; skipped numeric check")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
