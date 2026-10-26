"""The release step that bakes every model into the Windows ZIP (tools/bundle_windows_models.py)."""
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

STYLES = ["celeba_distill", "face_paint_512_v1", "face_paint_512_v2", "hayao", "paprika", "shinkai",
          "hayao_v3", "shinkai_v3", "ghibli_face", "jp_face", "portrait_sketch"]
DCT = ["dct_3d", "dct_anime", "dct_anime_bg", "dct_handdrawn", "dct_handdrawn_bg"]


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "bundle_windows_models.py"
    spec = importlib.util.spec_from_file_location("bundle_windows_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name in members:
            z.writestr(name, f"fake {name}")


def fake_artifacts(root):
    models, dct = root / "models", root / "dct"
    models.mkdir()
    dct.mkdir()
    for name in ("ptoart_weights.pt", "ptoart_weights.onnx", "seg_model_384.pb", "SHA256SUMS-models.txt"):
        (models / name).write_bytes(b"fake")
    fake_zip(models / "ptoart-styles.zip",
             [f"{s}.onnx" for s in STYLES] + ["LICENSE-AnimeGANv2.txt", "LICENSE-AnimeGANv3.txt"])
    fake_zip(models / "ptoart-upscalers-realesrgan.zip", ["anime.onnx", "general.onnx", "LICENSE-Real-ESRGAN.txt"])
    fake_zip(dct / "ptoart-dctnet.zip",
             [f"{s}.onnx" for s in DCT] + ["README-DCT-Net.txt", "LICENSE-DCT-Net-Apache-2.0.txt"])
    return models, dct


# The release step runs on Linux. On Windows, zipfile itself rewrites "\\" to "/" when writing,
# so the backslash case (what Compress-Archive may produce) can only be reproduced off Windows.
SEPARATORS = ["/"] if sys.platform == "win32" else ["\\", "/"]


@pytest.mark.parametrize("sep", SEPARATORS)
def test_every_model_lands_in_ptoart_weights(tmp_path, sep):
    app = tmp_path / "ptoart-windows-x64.zip"
    exe_build = ["ptoart/ptoart.exe", "ptoart/_internal/python311.dll", "ptoart/weights/README.txt",
                 "ptoart/weights/torch/hub/checkpoints/s3fd-619a316812.pth"]
    fake_zip(app, [n.replace("/", sep) for n in exe_build])
    models, dct = fake_artifacts(tmp_path)

    added = load_tool().bundle(app, models, dct)

    names = zipfile.ZipFile(app).namelist()
    assert all(("\\" in n) == (sep == "\\") for n in names), "path separator must match the EXE build"
    flat = {n.replace("\\", "/") for n in names}
    expected = {"ptoart/weights/ptoart_weights.pt", "ptoart/weights/seg_model_384.pb",
                "ptoart/weights/ptoart_weights.onnx",
                "ptoart/weights/upscale/anime.onnx", "ptoart/weights/upscale/general.onnx",
                "ptoart/weights/upscale/LICENSE-Real-ESRGAN.txt",
                "ptoart/weights/styles/LICENSE-DCT-Net-Apache-2.0.txt",
                "ptoart/weights/styles/LICENSE-AnimeGANv2.txt"}
    expected |= {f"ptoart/weights/styles/{s}.onnx" for s in STYLES + DCT}
    assert expected <= flat
    assert len(added) == len(expected) + 2  # + AnimeGANv3 licence and DCT readme, nothing else
    assert not any("SHA256SUMS" in n for n in flat)
    assert len(names) == len(set(names)), "no duplicate entries"
    # Running it again adds nothing (existing entries are kept).
    assert load_tool().bundle(app, models, dct) == []


def test_bundled_zip_is_found_by_the_frozen_gui_without_import(tmp_path, monkeypatch):
    app = tmp_path / "ptoart-windows-x64.zip"
    fake_zip(app, ["ptoart/ptoart.exe", "ptoart/weights/README.txt"])
    models, dct = fake_artifacts(tmp_path)
    load_tool().bundle(app, models, dct)
    zipfile.ZipFile(app).extractall(tmp_path / "unzipped")

    from ptoart.gui import find_model_dir, scan_models

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "unzipped" / "ptoart" / "ptoart.exe"))
    monkeypatch.delenv("PTOART_MODEL_DIR", raising=False)
    root = find_model_dir()
    assert root == (tmp_path / "unzipped" / "ptoart" / "weights").resolve()
    found = scan_models(root)
    assert found.face and found.seg
    assert set(found.styles) == set(STYLES + DCT)
    assert list(found.upscalers) == ["anime", "general"]


def test_missing_artifact_fails_loudly(tmp_path):
    app = tmp_path / "ptoart-windows-x64.zip"
    fake_zip(app, ["ptoart/ptoart.exe"])
    models, dct = fake_artifacts(tmp_path)
    (models / "ptoart-upscalers-realesrgan.zip").unlink()
    with pytest.raises(SystemExit, match="upscalers"):
        load_tool().bundle(app, models, dct)
