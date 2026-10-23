"""Random-checkpoint structure smoke tests; these do not measure pretrained quality."""

import os

import numpy as np
import pytest

from ptoart.config import ArtConfig
from ptoart.errors import ModelError
from ptoart.pipeline import ArtPipeline

torch = pytest.importorskip("torch")


torch.set_num_threads(2)


class PickleSideEffect:
    """Pickle payload that would create a file if unrestricted loading executed it."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def save_random_checkpoint(path):
    from models.networks import ResnetGenerator

    generator = ResnetGenerator(ngf=32, img_size=256, light=True)
    torch.save({"genA2B": generator.state_dict()}, path)
    return generator


def test_torch_backend_random_checkpoint_load_eval_and_cpu_inference(tmp_path):
    weights = tmp_path / "random.pt"
    save_random_checkpoint(weights)
    config = ArtConfig(aligned=True, device="cpu", weights=weights)
    image = np.random.default_rng(2026).integers(0, 256, (256, 256, 3), dtype=np.uint8)

    with ArtPipeline(config) as pipeline:
        assert pipeline.backend.device.type == "cpu"
        assert pipeline.backend.net.training is False
        output = pipeline.convert(image)

    assert output.shape == (256, 256, 3)
    assert output.dtype == np.uint8
    assert np.isfinite(output).all()
    # Random initialized weights validate architecture and loading only, not the quality of
    # actual photo2cartoon pretrained weights or their visual results.


def test_torch_backend_rejects_bad_weights_without_unrestricted_pickle_execution(tmp_path):
    marker = tmp_path / "pickle-executed"
    malicious = tmp_path / "untrusted.pt"
    torch.save({"genA2B": PickleSideEffect(marker)}, malicious)

    with pytest.raises(ModelError, match="Unable to load trusted genA2B checkpoint"):
        ArtPipeline(ArtConfig(aligned=True, device="cpu", weights=malicious))
    assert not marker.exists(), "weights_only loading must not execute arbitrary pickle globals"
