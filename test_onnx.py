"""Legacy ONNX entry point; no CLI parsing or model loading on import."""
from pathlib import Path

from ptoart import ArtConfig, ArtPipeline, NoFaceError
from test import legacy_main


class Ptoart:
    def __init__(self):
        root = Path(__file__).resolve().parent
        self.pipeline = ArtPipeline(ArtConfig(
            backend="onnx", weights=root / "models/ptoart_weights.onnx",
            segmentation=root / "utils/seg_model_384.pb",
        ))

    def inference(self, img):
        try:
            return self.pipeline.convert(img)[:, :, ::-1].copy()
        except NoFaceError:
            return None

    def close(self):
        self.pipeline.close()


if __name__ == "__main__":
    raise SystemExit(legacy_main("onnx"))
