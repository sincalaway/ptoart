"""Invalid requests fail before model setup; framework errors stay actionable."""
import pytest
from PIL import Image

from ptoart import ArtConfig, ModelError
from ptoart.cli import main
from ptoart.pipeline import ArtPipeline


@pytest.mark.parametrize("case,expected", [
    ("missing", "Not an input image"),
    ("bad-image", "Cannot read"),
    ("bad-output", "Unsupported output"),
    ("transparent-jpeg", "Transparent output"),
    ("oversized", "exceeds"),
    ("empty-batch", "No supported images"),
    ("missing-batch", "Not an input directory"),
    ("non-square", "square"),
])
def test_cli_preflight_precedes_model_setup(tmp_path, monkeypatch, capsys, case, expected):
    import ptoart.pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Model setup must not run for invalid input")
    monkeypatch.setattr(ptoart.pipeline, "ArtPipeline", forbidden)
    source = tmp_path / "photo.png"
    target = tmp_path / "output.png"
    Image.new("RGB", (8, 8), "red").save(source)
    args = ["convert", str(source), str(target), "--aligned"]
    if case == "missing":
        source.unlink()
    elif case == "bad-image":
        source.write_bytes(b"broken")
    elif case == "bad-output":
        args[2] = str(tmp_path / "output.xyz")
    elif case == "transparent-jpeg":
        args[2] = str(tmp_path / "output.jpg")
        args += ["--background", "transparent"]
    elif case == "oversized":
        args += ["--max-pixels", "16"]
    elif case in {"empty-batch", "missing-batch"}:
        directory = tmp_path / "photos"
        if case == "empty-batch":
            directory.mkdir()
        args = ["batch", str(directory), str(tmp_path / "output")]
    elif case == "non-square":
        Image.new("RGB", (8, 10), "blue").save(source)
    assert main(args) == 2
    assert expected in capsys.readouterr().err


def test_framework_initialization_failure_is_wrapped(tmp_path, monkeypatch):
    import ptoart.pipeline
    weights = tmp_path / "exists.pt"
    weights.touch()
    def broken(_):
        raise RuntimeError("simulated device initialization failure")
    monkeypatch.setattr(ptoart.pipeline, "create_backend", broken)
    with pytest.raises(ModelError, match="device initialization failure") as failure:
        ArtPipeline(ArtConfig(aligned=True, weights=weights))
    assert isinstance(failure.value.__cause__, RuntimeError)
