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
