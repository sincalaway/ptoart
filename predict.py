"""Optional Cog adapter; install Cog separately to use this deployment entry."""
import tempfile
from pathlib import Path as FilePath

from cog import BasePredictor, Input, Path

from ptoart import ArtConfig, ArtPipeline
from ptoart.io import write_image


def legacy_config():
    root = FilePath(__file__).resolve().parent
    return ArtConfig(weights=root / "models/photo2cartoon_weights.pt",
                     segmentation=root / "utils/seg_model_384.pb")


class Predictor(BasePredictor):
    def setup(self):
        self.pipeline = ArtPipeline(legacy_config())

    def predict(self, photo: Path = Input(description="A clear portrait photo")) -> Path:
        output = FilePath(tempfile.mkdtemp(prefix="ptoart-")) / "out.png"
        self.pipeline.convert_file(photo, output)
        return Path(output)


def gen_cartoon(img):
    """Legacy RGB helper. Prefer a reused ArtPipeline for repeated calls."""
    output = FilePath(tempfile.mkdtemp(prefix="ptoart-")) / "out.png"
    with ArtPipeline(legacy_config()) as pipeline:
        write_image(output, pipeline.convert(img))
    return output
