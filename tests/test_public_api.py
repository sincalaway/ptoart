from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ptoart.backends import validate_output
from ptoart.config import ArtConfig
from ptoart.errors import ImageError, ModelError, NoFaceError
from ptoart.io import read_image, write_image
from ptoart.pipeline import ArtPipeline


class RecordingBackend:
    def __init__(self, output_transform=None):
        self.input = None
        self.closed = 0
        self.output_transform = output_transform

    def run(self, nchw):
        self.input = nchw.copy()
        if self.output_transform:
            return self.output_transform(nchw)
        return nchw.copy()

    def close(self):
        self.closed += 1


class RecordingPreprocessor:
    def __init__(self, result):
        self.result = result
        self.input = None
        self.closed = 0

    def process(self, rgb):
        self.input = rgb.copy()
        return self.result

    def close(self):
        self.closed += 1


def rgba_fixture(size=256, alpha=255):
    y, x = np.mgrid[:size, :size]
    rgb = np.stack(((x * 3) % 256, (y * 5) % 256, (x + y) % 256), axis=2).astype(np.uint8)
    a = np.full((size, size, 1), alpha, dtype=np.uint8)
    return np.concatenate((rgb, a), axis=2)


def pipeline_for(rgba, *, config=None, backend=None):
    pre = RecordingPreprocessor(rgba)
    backend = backend or RecordingBackend()
    pipeline = ArtPipeline(config or ArtConfig(aligned=False), backend=backend, preprocessor=pre)
    return pipeline, backend, pre


def test_default_white_composite_and_original_normalization_math_are_preserved():
    rgba = rgba_fixture(alpha=91)
    pipeline, backend, pre = pipeline_for(rgba)
    try:
        actual = pipeline.convert(np.full((256, 256, 3), 17, dtype=np.uint8))
        source = rgba[:, :, :3].astype(np.float32)
        mask = rgba[:, :, 3:4].astype(np.float32) / 255.0
        expected_model_input = (source * mask + (1 - mask) * 255) / 127.5 - 1
        np.testing.assert_allclose(backend.input[0], expected_model_input.transpose(2, 0, 1), atol=1e-6)
        # Identity model output follows the original denormalize -> mask -> white composite math.
        reconstructed = (expected_model_input + 1) * 127.5
        expected = np.clip(reconstructed * mask + 255 * (1 - mask), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)
        assert pipeline.config.background == (255, 255, 255)
        np.testing.assert_array_equal(pre.input, np.full((256, 256, 3), 17, dtype=np.uint8))
    finally:
        pipeline.close()


def test_strength_zero_returns_original_foreground_with_default_white_background():
    rgba = rgba_fixture(alpha=255)
    backend = RecordingBackend(lambda x: np.full_like(x, 8.0))
    pipeline, _, _ = pipeline_for(rgba, config=ArtConfig(strength=0), backend=backend)
    try:
        result = pipeline.convert(np.zeros((256, 256, 3), dtype=np.uint8))
        np.testing.assert_array_equal(result, rgba[:, :, :3])
    finally:
        pipeline.close()


def test_transparent_output_retains_alpha_and_clips_out_of_range_values():
    rgba = rgba_fixture(alpha=128)
    def extremes(_):
        out = np.empty((1, 3, 256, 256), dtype=np.float32)
        out[:, 0] = 5.0
        out[:, 1] = -5.0
        out[:, 2] = 0.0
        return out
    pipeline, _, _ = pipeline_for(rgba, config=ArtConfig(background=None), backend=RecordingBackend(extremes))
    try:
        result = pipeline.convert(np.zeros((256, 256, 3), dtype=np.uint8))
        assert result.shape == (256, 256, 4)
        np.testing.assert_array_equal(result[:, :, 3], rgba[:, :, 3])
        assert result.dtype == np.uint8
        assert int(result[:, :, :3].max()) <= 255 and int(result[:, :, :3].min()) >= 0
        assert np.all(result[:, :, 0] == 255)
        assert np.all(result[:, :, 1] == 0)
    finally:
        pipeline.close()


@pytest.mark.parametrize("bad", [np.zeros((1, 3, 32, 32), dtype=np.float32),
                                  np.full((1, 3, 256, 256), np.nan, dtype=np.float32),
                                  np.full((1, 3, 256, 256), np.inf, dtype=np.float32)])
def test_invalid_model_shape_or_nonfinite_output_raises_model_error(bad):
    pipeline, _, _ = pipeline_for(rgba_fixture(), backend=RecordingBackend(lambda _: bad))
    with pytest.raises(ModelError, match="Invalid model output"):
        pipeline.convert(np.zeros((256, 256, 3), dtype=np.uint8))


def test_no_face_and_aligned_requires_square():
    pre = RecordingPreprocessor(None)
    pipeline = ArtPipeline(ArtConfig(aligned=False), backend=RecordingBackend(), preprocessor=pre)
    with pytest.raises(NoFaceError, match="No face"):
        pipeline.convert(np.zeros((32, 32, 3), dtype=np.uint8))
    pipeline.close()

    aligned = ArtPipeline(ArtConfig(aligned=True), backend=RecordingBackend())
    try:
        with pytest.raises(ImageError, match="square"):
            aligned.convert(np.zeros((20, 21, 3), dtype=np.uint8))
    finally:
        aligned.close()


def test_rgb_preprocessor_contract_and_close_release_each_injected_component_once():
    rgba = rgba_fixture()
    backend, pre = RecordingBackend(), RecordingPreprocessor(rgba)
    pipeline = ArtPipeline(ArtConfig(), backend=backend, preprocessor=pre)
    rgb = np.full((256, 256, 3), (13, 29, 47), dtype=np.uint8)
    pipeline.convert(rgb)
    np.testing.assert_array_equal(pre.input, rgb)
    assert pre.input.shape[2] == 3
    pipeline.close()
    pipeline.close()
    assert pre.closed == 1
    assert backend.closed == 1
    with pytest.raises(Exception, match="closed"):
        pipeline.convert(rgb)


def test_pipeline_rejects_non_rgb_and_oversized_images():
    pipeline = ArtPipeline(ArtConfig(aligned=True, max_pixels=16), backend=RecordingBackend())
    try:
        with pytest.raises(ImageError, match="uint8 RGB"):
            pipeline.convert(np.zeros((4, 4), dtype=np.uint8))
        with pytest.raises(ImageError, match="exceeds"):
            pipeline.convert(np.zeros((5, 5, 3), dtype=np.uint8))
    finally:
        pipeline.close()


def test_validate_output_accepts_only_finite_expected_nchw():
    valid = np.zeros((1, 3, 256, 256), dtype=np.float32)
    assert validate_output(valid) is valid
    with pytest.raises(ModelError):
        validate_output(np.zeros((1, 3, 256, 256), dtype=np.float64) + np.nan)


def test_io_unicode_exif_grayscale_and_transparent_inputs(tmp_path):
    unicode_dir = tmp_path / "中文路径"
    unicode_dir.mkdir()
    exif_path = unicode_dir / "旋转.jpg"
    image = Image.new("RGB", (3, 2), (200, 10, 20))
    exif = Image.Exif()
    exif[274] = 6
    image.save(exif_path, exif=exif)
    oriented = read_image(exif_path)
    assert oriented.shape == (3, 2, 3)
    np.testing.assert_allclose(oriented[0, 0], [200, 10, 20], atol=3)

    gray_path = unicode_dir / "灰度.png"
    Image.new("L", (2, 2), 73).save(gray_path)
    gray = read_image(gray_path)
    assert gray.shape == (2, 2, 3)
    assert np.all(gray == 73)

    alpha_path = unicode_dir / "透明.png"
    Image.new("RGBA", (1, 1), (250, 20, 0, 0)).save(alpha_path)
    transparent = read_image(alpha_path)
    np.testing.assert_array_equal(transparent[0, 0], [255, 255, 255])


def test_io_rejects_pixel_limit_corruption_existing_output_and_transparent_jpeg(tmp_path):
    over = tmp_path / "too-large.png"
    Image.new("RGB", (3, 2), "black").save(over)
    with pytest.raises(ImageError, match="exceeds"):
        read_image(over, max_pixels=5)

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ImageError, match="Cannot read"):
        read_image(corrupt)

    output = tmp_path / "result.png"
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    write_image(output, original)
    with pytest.raises(ImageError):
        write_image(output, np.full_like(original, 255))
    np.testing.assert_array_equal(np.asarray(Image.open(output)), original)
    write_image(output, np.full_like(original, 255), overwrite=True)
    np.testing.assert_array_equal(np.asarray(Image.open(output)), np.full_like(original, 255))

    with pytest.raises(ImageError, match="Transparent output"):
        write_image(tmp_path / "transparent.jpg", np.zeros((2, 2, 4), dtype=np.uint8))


def test_batch_preserves_same_stem_different_extensions_and_recurses_and_continues(tmp_path):
    from ptoart.cli import batch

    src, dst = tmp_path / "src", tmp_path / "out"
    (src / "sub").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(src / "same.jpg")
    Image.new("RGB", (8, 8), "blue").save(src / "same.png")
    Image.new("RGB", (8, 8), "green").save(src / "sub" / "nested.webp")
    (src / "broken.png").write_bytes(b"broken image data")
    pipeline = ArtPipeline(ArtConfig(aligned=True), backend=RecordingBackend())
    try:
        report = batch(pipeline, src, dst, recursive=True)
    finally:
        pipeline.close()
    assert report["total"] == 4
    assert report["succeeded"] == 3
    assert report["failed"] == 1
    assert (dst / "same.jpg.png").is_file()
    assert (dst / "same.png.png").is_file()
    assert (dst / "sub" / "nested.webp.png").is_file()
    assert next(item for item in report["results"] if item["input"].endswith("broken.png"))["status"] == "error"


def test_batch_skips_nested_output_tree_and_empty_directory_raises(tmp_path):
    from ptoart.cli import batch

    src, dst = tmp_path / "photos", tmp_path / "photos" / "generated"
    dst.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(src / "real.png")
    Image.new("RGB", (4, 4), "blue").save(dst / "prior.png")
    pipeline = ArtPipeline(ArtConfig(aligned=True), backend=RecordingBackend())
    try:
        report = batch(pipeline, src, dst)
        assert report["total"] == 1
        assert report["results"][0]["input"].endswith("real.png")
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No supported images"):
            batch(pipeline, empty, tmp_path / "empty-out")
    finally:
        pipeline.close()


def run_cli_without_ml(*args):
    # Block ML imports at Python's import boundary; CLI help/doctor must not request them.
    code = r'''
import builtins, sys
blocked = {"torch", "tensorflow", "onnxruntime", "face_alignment", "dlib"}
real_import = builtins.__import__
def guarded(name, *a, **kw):
    if name.split(".")[0] in blocked:
        raise AssertionError("unexpected ML import: " + name)
    return real_import(name, *a, **kw)
builtins.__import__ = guarded
from ptoart.cli import main
try:
    result = main(sys.argv[1:])
except SystemExit as exc:
    result = exc.code
sys.exit(result or 0)
'''
    return subprocess.run([sys.executable, "-c", code, *args], cwd=Path(__file__).parents[1],
                          text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])})


def test_cli_help_does_not_import_ml_frameworks():
    result = run_cli_without_ml("--help")
    assert result.returncode == 0
    assert "usage: ptoart" in result.stdout
    assert "unexpected ML import" not in result.stderr


def test_cli_doctor_checks_presence_without_loading_ml_frameworks(tmp_path):
    result = run_cli_without_ml("doctor", "--aligned", "--model-dir", str(tmp_path), "--device", "cpu")
    assert result.returncode in (0, 2)
    report = json.loads(result.stdout)
    assert report["backend"] == "torch"
    assert report["models"] == {str((tmp_path / "photo2cartoon_weights.pt").resolve()): False}
    assert "Presence checks only" in report["note"]
    assert "unexpected ML import" not in result.stderr
