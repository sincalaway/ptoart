import sys
import types

import numpy as np


def _load(monkeypatch):
    fake = types.SimpleNamespace(LandmarksType=types.SimpleNamespace(TWO_D=1))
    monkeypatch.setitem(sys.modules, "face_alignment", fake)
    sys.modules.pop("utils.face_detect", None)
    from utils import face_detect
    return face_detect


class _Detector:
    def __init__(self):
        self.shapes = []

    def detect_from_image(self, image):
        self.shapes.append(image.shape)
        return [np.array([100, 50, 200, 150, 0.99], dtype=np.float32)]


class _FA:
    def __init__(self):
        self.face_detector = _Detector()
        self.calls = []

    def get_landmarks(self, image, detected_faces=None):
        self.calls.append((image.shape, detected_faces))
        return [np.zeros((68, 2), dtype=np.float32)]


def test_large_photo_detects_on_downscaled_copy(monkeypatch):
    module = _load(monkeypatch)
    detect = module.FaceDetect.__new__(module.FaceDetect)
    detect.fa = _FA()
    image = np.zeros((4000, 5000, 3), dtype=np.uint8)
    detect._landmarks(image)
    assert max(detect.fa.face_detector.shapes[0][:2]) == module.DETECT_MAX_SIDE
    shape, boxes = detect.fa.calls[0]
    assert shape == image.shape  # landmarks still on full resolution
    np.testing.assert_allclose(boxes[0][:4], np.array([100, 50, 200, 150]) * 5000 / 1280, rtol=1e-4)


def test_small_photo_untouched(monkeypatch):
    module = _load(monkeypatch)
    detect = module.FaceDetect.__new__(module.FaceDetect)
    detect.fa = _FA()
    detect._landmarks(np.zeros((800, 600, 3), dtype=np.uint8))
    assert detect.fa.face_detector.shapes == []
    assert detect.fa.calls[0][1] is None
