"""Test the frozen launcher policy without importing optional ML frameworks."""
import importlib.util
from pathlib import Path

import pytest

from ptoart import cli


def load_entry():
    path = Path(__file__).resolve().parents[1] / "packaging" / "windows_entry.py"
    spec = importlib.util.spec_from_file_location("ptoart_windows_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", ["doctor", "convert", "batch"])
def test_windows_defaults_to_sfd(monkeypatch, command):
    monkeypatch.setattr(cli, "main", lambda args: args)
    assert load_entry().run([command]) == [command, "--detector", "sfd"]


@pytest.mark.parametrize("args", [
    ["doctor", "--detector", "dlib"],
    ["doctor", "--detector=sfd"],
    ["--help"],
    ["--version"],
    [],
])
def test_windows_preserves_explicit_options(monkeypatch, args):
    monkeypatch.setattr(cli, "main", lambda args: args)
    assert load_entry().run(args) == args


def test_frozen_exe_uses_adjacent_weights(monkeypatch, tmp_path):
    (tmp_path / "weights").mkdir()
    monkeypatch.setattr(cli, "main", lambda args: args)
    monkeypatch.delenv("PTOART_MODEL_DIR", raising=False)
    entry = load_entry()
    monkeypatch.setattr(entry.sys, "frozen", True, raising=False)
    monkeypatch.setattr(entry.sys, "executable", str(tmp_path / "ptoart.exe"))
    assert entry.run(["doctor"]) == ["doctor", "--detector", "sfd", "--model-dir",
                                     str((tmp_path / "weights").resolve())]
    assert entry.run(["doctor", "--model-dir", "x"]) == ["doctor", "--model-dir", "x",
                                                         "--detector", "sfd"]
