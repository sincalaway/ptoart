"""Legacy compatibility entry point. Public inference() still returns BGR."""
from pathlib import Path

from ptoart import ArtConfig, ArtPipeline, NoFaceError
from ptoart.cli import main


class Ptoart:
    def __init__(self):
        root = Path(__file__).resolve().parent
        self.pipeline = ArtPipeline(ArtConfig(
            weights=root / "models/ptoart_weights.pt",
            segmentation=root / "utils/seg_model_384.pb",
        ))

    def inference(self, img):
        try:
            return self.pipeline.convert(img)[:, :, ::-1].copy()
        except NoFaceError:
            return None

    def close(self):
        self.pipeline.close()


def legacy_main(backend="torch"):
    import argparse
    parser = argparse.ArgumentParser(description="ptoart legacy inference entry")
    parser.add_argument("--photo_path", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    filename = "ptoart_weights.pt" if backend == "torch" else "ptoart_weights.onnx"
    return main(["convert", args.photo_path, args.save_path, "--backend", backend,
                 "--weights", str(root / "models" / filename),
                 "--segmentation", str(root / "utils/seg_model_384.pb")])


if __name__ == "__main__":
    raise SystemExit(legacy_main())
