from pathlib import Path

from ptoart.errors import NoFaceError
from ptoart.gui import build_config, friendly_error, output_name, scan_models, style_label


def make(root, *names):
    for name in names:
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_scan_models_detects_everything(tmp_path):
    make(tmp_path, "ptoart_weights.pt", "seg_model_384.pb", "styles/shinkai.onnx",
         "styles/hayao.onnx", "styles/custom.onnx", "upscale/anime.onnx")
    models = scan_models(tmp_path)
    assert models.face and models.any
    assert list(models.styles) == ["hayao", "shinkai", "custom"]
    assert list(models.upscalers) == ["anime"]


def test_scan_models_needs_both_face_files(tmp_path):
    make(tmp_path, "ptoart_weights.pt")
    assert not scan_models(tmp_path).face


def test_build_config_maps_ui_choices(tmp_path):
    make(tmp_path, "styles/hayao.onnx", "upscale/anime.onnx")
    models = scan_models(tmp_path)
    face = build_config(models, mode="face", size=1024, transparent=True, upscale="anime")
    assert face.size == 1024 and face.background is None and face.detector == "sfd"
    assert face.upscale == (tmp_path / "upscale" / "anime.onnx").resolve()
    style = build_config(models, mode="style", style="hayao", transparent=True)
    assert style.output == "style" and style.background == (255, 255, 255)


def test_labels_and_names():
    assert style_label("hayao") == "宫崎骏"
    assert output_name("a/me.jpg", "face") == "me-头像.png"
    assert output_name("me.jpg", "style", "shinkai") == "me-新海诚.jpg"
    assert "人脸" in friendly_error(NoFaceError("x"))


def test_friendly_error_out_of_memory():
    from ptoart.gui import friendly_error

    msg = friendly_error(RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 5110235136 bytes."))
    assert "内存不足" in msg
