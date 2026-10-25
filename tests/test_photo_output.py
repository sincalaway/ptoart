"""Paste-back (--output photo), multi-face and --size behaviour with fake components."""
import numpy as np
import pytest

from ptoart.config import ArtConfig
from ptoart.errors import NoFaceError
from ptoart.pipeline import ArtPipeline
from utils.preprocess import FaceRegion


class ConstantBackend:
    """Returns a flat colour so pasted pixels are easy to find."""
    def __init__(self, value=1.0):
        self.value = value
        self.calls = 0

    def run(self, nchw):
        self.calls += 1
        return np.full((1, 3, 256, 256), self.value, dtype=np.float32)


class FakeMultiFace:
    def __init__(self, regions):
        self.regions = regions
        self.max_faces = None

    def process(self, rgb):
        return self.regions[0].rgba if self.regions else None

    def process_all(self, rgb, max_faces=10):
        self.max_faces = max_faces
        return self.regions[:max_faces]


def region(left, top, side):
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    # Original pixel (x, y) maps to crop pixel (x - left, y - top).
    return FaceRegion(rgba, np.array([[1, 0, -left], [0, 1, -top]], dtype=np.float64))


def test_photo_output_keeps_size_and_pastes_each_face_center():
    photo = np.zeros((300, 400, 3), dtype=np.uint8)
    pre = FakeMultiFace([region(20, 20, 120), region(250, 120, 100)])
    backend = ConstantBackend(1.0)  # white cartoon
    with ArtPipeline(ArtConfig(output="photo"), backend=backend, preprocessor=pre) as p:
        out = p.convert(photo)
    assert out.shape == photo.shape and out.dtype == np.uint8
    assert backend.calls == 2
    assert out[80, 80].min() > 240      # first face centre is cartoon
    assert out[170, 300].min() > 240    # second face centre is cartoon
    assert out[290, 10].max() == 0      # background untouched
    assert out[21, 21].max() < 40       # crop border is feathered, no hard seam


def test_photo_output_respects_max_faces_and_strength():
    photo = np.zeros((200, 200, 3), dtype=np.uint8)
    pre = FakeMultiFace([region(0, 0, 100), region(100, 100, 100)])
    backend = ConstantBackend(1.0)
    cfg = ArtConfig(output="photo", max_faces=1, strength=0.5)
    with ArtPipeline(cfg, backend=backend, preprocessor=pre) as p:
        out = p.convert(photo)
    assert pre.max_faces == 1 and backend.calls == 1
    assert 110 < out[50, 50, 0] < 145
    assert out[150, 150].max() == 0


def test_photo_output_without_faces_raises():
    with ArtPipeline(ArtConfig(output="photo"), backend=ConstantBackend(),
                     preprocessor=FakeMultiFace([])) as p:
        with pytest.raises(NoFaceError):
            p.convert(np.zeros((64, 64, 3), dtype=np.uint8))


@pytest.mark.parametrize("size", [128, 512, 1024])
def test_face_output_size(size):
    rgba = np.full((600, 600, 4), 255, dtype=np.uint8)
    with ArtPipeline(ArtConfig(size=size), backend=ConstantBackend(0.0),
                     preprocessor=FakeMultiFace([FaceRegion(rgba, np.eye(2, 3))])) as p:
        out = p.convert(np.zeros((700, 700, 3), dtype=np.uint8))
    assert out.shape == (size, size, 3)
    assert abs(int(out[size // 2, size // 2, 0]) - 128) <= 2


@pytest.mark.parametrize("kwargs", [dict(output="x"), dict(size=32), dict(size=4096),
                                    dict(max_faces=0), dict(output="photo", aligned=True)])
def test_invalid_new_options(kwargs):
    with pytest.raises(ValueError):
        ArtConfig(**kwargs)


def test_crop_matrix_maps_original_pixels_onto_crop():
    """The affine stored in FaceRegion must reproduce the real crop (rotation + offset)."""
    import cv2
    import math
    from utils.preprocess import Preprocess
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    angle = math.radians(12)
    h, w = image.shape[:2]
    cos, sin = math.cos(angle), math.sin(angle)
    new_w, new_h = int(w * abs(cos) + h * abs(sin)), int(w * abs(sin) + h * abs(cos))
    m = np.array([[cos, sin, (1 - cos) * w / 2 - sin * h / 2 + new_w // 2 - w // 2],
                  [-sin, cos, sin * w / 2 + (1 - cos) * h / 2 + new_h // 2 - h // 2]])
    rotated = cv2.warpAffine(image, m, (new_w, new_h), borderValue=(255, 255, 255))
    landmarks = np.array([[150.0, 110.0], [210.0, 170.0]])
    crop, left, top = Preprocess._Preprocess__crop(rotated, landmarks)
    to_crop = m.copy()
    to_crop[0, 2] -= left
    to_crop[1, 2] -= top
    direct = cv2.warpAffine(image, to_crop, crop.shape[1::-1], borderValue=(255, 255, 255))
    inner = (slice(5, -5), slice(5, -5))
    diff = np.abs(direct[inner].astype(int) - crop[inner].astype(int))
    assert np.median(diff) <= 2
