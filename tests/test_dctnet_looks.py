"""DCT-Net face looks: config, CLI and GUI wiring, metadata-driven sizes, whole-photo scene."""
from pathlib import Path

import numpy as np
import pytest

from ptoart.cli import build_parser, config_from
from ptoart.config import ArtConfig
from ptoart.gui import (FACE_LOOKS, build_config, describe_options, output_name, scan_models,
                        selection_summary, stale_message)
from ptoart.looks import LOOK_HINTS, scene_for
from ptoart.pipeline import ArtPipeline

DCT_FILES = ("dct_anime.onnx", "dct_anime_bg.onnx", "dct_3d.onnx", "dct_handdrawn.onnx",
             "dct_handdrawn_bg.onnx")


def make(root, *names):
    for name in names:
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def dct_dir(tmp_path):
    make(tmp_path, "seg_model_384.pb", "ptoart_weights.pt", "styles/face_paint_512_v2.onnx",
         "styles/hayao.onnx", *(f"styles/{n}" for n in DCT_FILES))
    return scan_models(tmp_path)


def test_dct_looks_are_listed_after_default_and_existing_looks_unchanged(tmp_path):
    models = dct_dir(tmp_path)
    assert models.looks == ["paint", "dct_anime", "dct_3d", "dct_handdrawn", "classic"]
    assert FACE_LOOKS["paint"] == ("动漫脸（推荐）", "face_paint_512_v2")
    assert FACE_LOOKS["classic"] == ("经典卡通", None)
    assert all(LOOK_HINTS[k] for k in FACE_LOOKS)
    # Face-only DCT head models are not offered as whole-image styles; their *_bg twins are.
    assert list(models.whole_styles) == ["hayao", "face_paint_512_v2", "dct_anime_bg", "dct_3d",
                                         "dct_handdrawn_bg"]


def test_dct_build_config_adds_scene_only_for_whole_photo(tmp_path):
    models = dct_dir(tmp_path)
    styles = tmp_path / "styles"
    face = build_config(models, mode="face", look="dct_anime")
    assert face.style == (styles / "dct_anime.onnx").resolve() and face.scene_style is None
    photo = build_config(models, mode="photo", look="dct_anime")
    assert photo.scene_style == (styles / "dct_anime_bg.onnx").resolve()
    assert build_config(models, mode="photo", look="dct_3d").scene_style == (styles / "dct_3d.onnx").resolve()
    assert build_config(models, mode="photo", look="paint").scene_style is None


def test_scene_falls_back_to_head_model():
    assert scene_for("dct_handdrawn", {"dct_handdrawn"}) == "dct_handdrawn"
    assert scene_for("dct_anime", {"dct_anime", "dct_anime_bg"}) == "dct_anime_bg"
    assert scene_for("paint", {"face_paint_512_v2"}) is None


def test_scene_style_needs_photo_output(tmp_path):
    with pytest.raises(ValueError):
        ArtConfig(model_dir=tmp_path, output="face", scene_style="dct_anime_bg")


def test_cli_look_option(tmp_path):
    make(tmp_path, "styles/dct_anime.onnx", "styles/dct_anime_bg.onnx")
    args = build_parser().parse_args(["convert", "a.jpg", "b.jpg", "--look", "dct_anime",
                                      "--output", "photo", "--model-dir", str(tmp_path)])
    config = config_from(args)
    assert config.output == "photo"
    assert config.style == (tmp_path / "styles" / "dct_anime.onnx").resolve()
    assert config.scene_style == (tmp_path / "styles" / "dct_anime_bg.onnx").resolve()
    args = build_parser().parse_args(["convert", "a.jpg", "b.png", "--look", "dct_3d",
                                      "--model-dir", str(tmp_path)])
    config = config_from(args)
    assert config.output == "face" and config.scene_style is None
    args = build_parser().parse_args(["convert", "a.jpg", "b.png", "--look", "classic",
                                      "--model-dir", str(tmp_path)])
    assert config_from(args).style is None
    with pytest.raises(ValueError):
        config_from(build_parser().parse_args(["convert", "a.jpg", "b.png", "--look", "dct_3d",
                                               "--output", "style", "--model-dir", str(tmp_path)]))


def test_gui_selection_summary_and_stale_message():
    assert selection_summary("face", look="dct_anime") == "✓ 头像画风：DCT 日漫  ·  模式：卡通头像"
    assert selection_summary("style", style="ghibli_face") == "✓ 当前风格：吉卜力人像  ·  模式：整图风格"
    a = dict(mode="face", look="paint", size=1024, strength=1.0, upscale=None)
    assert stale_message(a, dict(a)) is None
    assert stale_message(None, a) is None
    message = stale_message(a, dict(a, look="dct_3d"))
    assert "动漫脸" in message and "DCT 3D" in message
    assert describe_options(dict(a, look="dct_handdrawn")) == "卡通头像 · DCT 手绘 · 1024px · 强度 100%"
    assert output_name("kid.jpg", "photo", look="dct_anime") == "kid-整张-DCT 日漫.jpg"


# --- pipeline behaviour with fake models -------------------------------------------------


class Painter:
    """Fake style model with DCT-Net metadata attributes."""

    def __init__(self, value, face_side=288, multiple=16):
        self.value, self.face_side, self.multiple = value, face_side, multiple
        self.shapes = []

    def run(self, x):
        self.shapes.append(x.shape)
        return np.full_like(x, self.value)


class FaceRegions:
    def __init__(self, regions):
        self.regions = regions

    def process(self, rgb):
        return self.regions[0].rgba

    def process_all(self, rgb, max_faces=10):
        return self.regions[:max_faces]


def test_dct_face_painter_runs_at_its_own_face_side(monkeypatch, tmp_path):
    import ptoart.pipeline as pl
    from utils.preprocess import FaceRegion

    painter = Painter(0.0)
    monkeypatch.setattr(pl, "StyleBackend", lambda path: painter)
    rgba = np.full((300, 300, 4), 255, dtype=np.uint8)
    config = ArtConfig(model_dir=tmp_path, style="dct_anime", size=512)
    with ArtPipeline(config, preprocessor=FaceRegions([FaceRegion(rgba, np.eye(2, 3))])) as p:
        out = p.convert(np.zeros((400, 400, 3), dtype=np.uint8))
    assert painter.shapes == [(1, 3, 288, 288)]
    assert out.shape == (512, 512, 3)


def test_dct_whole_photo_stylizes_scene_then_pastes_faces(monkeypatch, tmp_path):
    import ptoart.pipeline as pl
    from utils.preprocess import FaceRegion

    head, scene = Painter(1.0), Painter(-1.0)   # head -> white face, scene -> black background
    models = {str((tmp_path / "styles" / "dct_anime.onnx").resolve()): head,
              str((tmp_path / "styles" / "dct_anime_bg.onnx").resolve()): scene}
    monkeypatch.setattr(pl, "StyleBackend", lambda path: models[str(path)])
    rgba = np.full((100, 100, 4), 255, dtype=np.uint8)
    region = FaceRegion(rgba, np.array([[1, 0, -50], [0, 1, -40]], dtype=np.float64))
    config = ArtConfig(model_dir=tmp_path, output="photo", style="dct_anime", scene_style="dct_anime_bg",
                       style_max_side=256)
    image = np.full((301, 403, 3), 128, dtype=np.uint8)
    with ArtPipeline(config, preprocessor=FaceRegions([region])) as p:
        out = p.convert(image)
    assert out.shape == image.shape
    # DCT-Net needs sides that are multiples of 16.
    h, w = scene.shapes[0][2:]
    assert h % 16 == 0 and w % 16 == 0 and max(h, w) <= 256
    assert head.shapes == [(1, 3, 288, 288)]
    assert out[5, 5].max() <= 2                      # background came from the scene model
    assert out[90, 100].min() >= 250                 # face centre came from the head model


def test_style_backend_reads_ptoart_metadata(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    from ptoart.backends import StyleBackend

    dims = [1, 3, "H", "W"]
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "g",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, dims)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, dims)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for key, value in (("ptoart_multiple", "16"), ("ptoart_face_side", "288")):
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    path = tmp_path / "dct.onnx"
    onnx.save(model, path)
    backend = StyleBackend(path)
    assert backend.multiple == 16 and backend.face_side == 288
    plain = tmp_path / "plain.onnx"
    del model.metadata_props[:]
    onnx.save(model, plain)
    backend = StyleBackend(plain)
    assert backend.multiple == 8 and backend.face_side == 512
