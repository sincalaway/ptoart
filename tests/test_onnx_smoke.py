"""Synthetic ONNX graph tests, not official-model quality verification."""
import numpy as np
import pytest

from ptoart import ArtConfig, ArtPipeline, ImageError, ModelError

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def identity_model(path, shape=(1, 3, 256, 256)):
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["portrait"], ["result"])], "identity-fixture",
        [helper.make_tensor_value_info("portrait", TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, list(shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_real_onnx_runtime_uses_model_tensor_names(tmp_path):
    model = tmp_path / "identity.onnx"
    identity_model(model)
    config = ArtConfig(backend="onnx", device="cpu", aligned=True, weights=model)
    image = np.full((256, 256, 3), (25, 75, 175), dtype=np.uint8)
    with ArtPipeline(config) as pipeline:
        assert pipeline.backend.input_name == "portrait"
        assert pipeline.backend.output_name == "result"
        result = pipeline.convert(image)
    assert result.dtype == np.uint8
    np.testing.assert_allclose(result, image, atol=1)


def test_onnx_wrong_input_shape_is_rejected(tmp_path):
    model = tmp_path / "wrong.onnx"
    identity_model(model, (1, 256, 256, 3))
    with pytest.raises(ModelError, match="Expected NCHW"):
        ArtPipeline(ArtConfig(backend="onnx", aligned=True, weights=model))


def test_onnx_corrupt_file_is_actionable(tmp_path):
    model = tmp_path / "bad.onnx"
    model.write_bytes(b"invalid model")
    with pytest.raises(ModelError, match="Unable to load ONNX"):
        ArtPipeline(ArtConfig(backend="onnx", aligned=True, weights=model))


def test_original_file_is_protected_even_with_overwrite(tmp_path):
    from PIL import Image
    model = tmp_path / "identity.onnx"
    identity_model(model)
    source = tmp_path / "original.png"
    Image.new("RGB", (16, 16), "red").save(source)
    content = source.read_bytes()
    with ArtPipeline(ArtConfig(backend="onnx", device="cpu", aligned=True, weights=model)) as pipeline:
        with pytest.raises(ImageError, match="original photos"):
            pipeline.convert_file(source, source, overwrite=True)
    assert source.read_bytes() == content


def test_style_sessions_do_not_keep_a_memory_arena():
    """A 1536 px style pass peaks near 2 GB; with the default arena every loaded style kept it."""
    import onnxruntime as ort

    from ptoart.backends import _session_options

    assert _session_options(ort).enable_cpu_mem_arena is False
