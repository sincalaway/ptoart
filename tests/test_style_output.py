"""Whole-image style output (AnimeGANv2-type ONNX) with a fake backend."""

import numpy as np
import pytest

from ptoart.cli import build_parser, config_from
from ptoart.config import ArtConfig
from ptoart.errors import ModelError
from ptoart.pipeline import ArtPipeline


class Invert:
    def __init__(self):
        self.shape = None

    def run(self, x):
        self.shape = x.shape
        return -x


def test_style_keeps_size_and_uses_multiple_of_four_inference(tmp_path):
    backend = Invert()
    cfg = ArtConfig(output="style", style="hayao", model_dir=tmp_path, style_max_side=256)
    with ArtPipeline(cfg, backend=backend) as p:
        out = p.convert(np.full((301, 499, 3), 10, dtype=np.uint8))
    assert out.shape == (301, 499, 3)
    assert backend.shape[2] % 4 == 0 and backend.shape[3] % 4 == 0
    assert max(backend.shape[2:]) <= 256
    assert abs(int(out[150, 250, 0]) - 245) <= 2


def test_style_strength_zero_returns_original(tmp_path):
    img = np.random.default_rng(1).integers(0, 255, (64, 80, 3), dtype=np.uint8)
    cfg = ArtConfig(output="style", style="x", model_dir=tmp_path, strength=0)
    with ArtPipeline(cfg, backend=Invert()) as p:
        np.testing.assert_array_equal(p.convert(img), img)


def test_style_name_and_path_resolution(tmp_path):
    assert ArtConfig(output="style", style="shinkai", model_dir=tmp_path).style == \
        (tmp_path / "styles" / "shinkai.onnx").resolve()
    custom = tmp_path / "mine.onnx"
    assert ArtConfig(output="style", style=str(custom)).style == custom.resolve()


def test_style_requires_model_file(tmp_path):
    with pytest.raises(ModelError):
        ArtPipeline(ArtConfig(output="style", style="missing", model_dir=tmp_path))


def test_style_output_needs_style_and_cli_infers_output(tmp_path):
    with pytest.raises(ValueError):
        ArtConfig(output="style")
    args = build_parser().parse_args(["convert", "a.jpg", "b.png", "--style", "paprika",
                                      "--model-dir", str(tmp_path)])
    assert config_from(args).output == "style"
    args = build_parser().parse_args(["convert", "a.jpg", "b.png"])
    assert config_from(args).output == "face"


def test_bad_style_output_shape_raises(tmp_path):
    class Bad:
        def run(self, x):
            return x[:, :, :8]
    with ArtPipeline(ArtConfig(output="style", style="s", model_dir=tmp_path), backend=Bad()) as p:
        with pytest.raises(ModelError):
            p.convert(np.zeros((40, 40, 3), dtype=np.uint8))


class FakeUpscaler:
    scale = 4

    def __init__(self):
        self.calls = []

    def run(self, image):
        self.calls.append(image.shape)
        return np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)


def test_face_size_uses_upscaler_when_enlarging():
    from tests.test_photo_output import ConstantBackend, FakeMultiFace
    from utils.preprocess import FaceRegion
    up = FakeUpscaler()
    rgba = np.full((600, 600, 4), 255, dtype=np.uint8)
    with ArtPipeline(ArtConfig(size=768), backend=ConstantBackend(0.0), upscaler=up,
                     preprocessor=FakeMultiFace([FaceRegion(rgba, np.eye(2, 3))])) as p:
        out = p.convert(np.zeros((700, 700, 3), dtype=np.uint8))
    assert out.shape == (768, 768, 3) and up.calls == [(256, 256, 3)]


def test_style_uses_upscaler_only_when_downscaled(tmp_path):
    up = FakeUpscaler()
    cfg = ArtConfig(output="style", style="s", model_dir=tmp_path, style_max_side=256)
    with ArtPipeline(cfg, backend=Invert(), upscaler=up) as p:
        assert p.convert(np.zeros((512, 512, 3), dtype=np.uint8)).shape == (512, 512, 3)
        p.convert(np.zeros((128, 128, 3), dtype=np.uint8))
    assert up.calls == [(256, 256, 3)]


def test_upscale_name_resolution(tmp_path):
    assert ArtConfig(upscale="anime", model_dir=tmp_path).upscale == \
        (tmp_path / "upscale" / "anime.onnx").resolve()
