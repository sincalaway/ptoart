

def test_describe_options_names_mode_and_look():
    from ptoart.gui import describe_options

    text = describe_options(dict(mode="face", look="jp", size=1024, strength=0.8, upscale="anime"))
    assert text == "卡通头像 · 日漫二次元 · 1024px · 强度 80% · 超分"
    assert describe_options(dict(mode="style", style="hayao_v3")) == "整图风格 · 宫崎骏 v3"


def test_unique_path_and_output_name(tmp_path):
    from ptoart.gui import output_name, unique_path

    name = output_name("baby.jpg", "face", look="paint")
    assert name == "baby-头像-动漫脸.png"
    (tmp_path / name).write_bytes(b"x")
    assert unique_path(tmp_path, name).name == "baby-头像-动漫脸-2.png"


def test_bundled_torch_hub(tmp_path, monkeypatch):
    from ptoart.gui import use_bundled_torch_hub

    monkeypatch.delenv("TORCH_HOME", raising=False)
    assert not use_bundled_torch_hub(tmp_path)
    (tmp_path / "torch" / "hub" / "checkpoints").mkdir(parents=True)
    assert use_bundled_torch_hub(tmp_path)
    import os

    assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")
