"""Reusable sequential inference pipeline with injectable components for tests."""
from pathlib import Path

import cv2
import numpy as np

from .backends import create_backend, require_file, validate_output
from .config import ArtConfig
from .errors import ImageError, ModelError, NoFaceError, PtoartError
from .io import read_image, write_image


class ArtPipeline:
    """RGB input/output. Reuse one instance per sequential worker, not across threads.

    ``aligned=True`` explicitly bypasses face detection and segmentation. It is
    only for already aligned, tightly cropped square portraits, never full photos.
    """

    def __init__(self, config=None, *, backend=None, preprocessor=None):
        self.config = config or ArtConfig()
        self.backend = None
        self.preprocessor = preprocessor
        self.closed = False
        try:
            if backend is None:
                require_file(self.config.weights)
            if not self.config.aligned and self.preprocessor is None:
                require_file(self.config.segmentation)
                try:
                    from utils.preprocess import Preprocess
                    # Keep upstream detector on CPU unless GPU was explicitly requested.
                    device = "cuda" if self.config.device.startswith("cuda") else "cpu"
                    self.preprocessor = Preprocess(
                        device=device, detector=self.config.detector,
                        seg_model_path=str(self.config.segmentation),
                    )
                except ImportError as exc:
                    raise ModelError("Face preprocessing requires: pip install 'ptoart[face]'") from exc
            self.backend = backend if backend is not None else create_backend(self.config)
        except Exception as exc:
            try:
                self.close()
            except Exception:
                pass  # Preserve the initialization cause if best-effort cleanup also fails.
            if isinstance(exc, PtoartError):
                raise
            raise ModelError(f"Unable to initialize pipeline: {exc}") from exc

    def convert(self, image):
        if self.closed:
            raise PtoartError("Pipeline is closed")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) < 2):
            raise ImageError("Expected a nonempty uint8 RGB image")
        if image.shape[0] * image.shape[1] > self.config.max_pixels:
            raise ImageError(f"Image exceeds {self.config.max_pixels} pixels")
        if self.config.aligned:
            if image.shape[0] != image.shape[1]:
                raise ImageError("--aligned requires a square, pre-aligned portrait")
            rgba = np.dstack((image, np.full(image.shape[:2], 255, dtype=np.uint8)))
        else:
            rgba = self.preprocessor.process(image)
            if rgba is None:
                raise NoFaceError("No face detected; use a clear, front-facing portrait")
            if (not isinstance(rgba, np.ndarray) or rgba.ndim != 3 or rgba.shape[2] != 4
                    or min(rgba.shape[:2]) < 1 or rgba.dtype != np.uint8):
                raise ImageError("Preprocessor must return uint8 RGBA")
        rgba = cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA)
        source = rgba[:, :, :3].astype(np.float32)
        mask = rgba[:, :, 3:4].astype(np.float32) / 255.0
        # White compositing and normalization intentionally match the upstream model.
        normalized = (source * mask + (1 - mask) * 255) / 127.5 - 1
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        try:
            output = validate_output(self.backend.run(tensor))
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Inference failed: {exc}") from exc
        cartoon = (output[0].transpose(1, 2, 0) + 1) * 127.5
        foreground = cartoon * self.config.strength + source * (1 - self.config.strength)
        if self.config.background is None:
            rgb = np.clip(foreground, 0, 255).astype(np.uint8)
            return np.concatenate((rgb, rgba[:, :, 3:4]), axis=2)
        background = np.asarray(self.config.background, dtype=np.float32)
        return np.clip(foreground * mask + background * (1 - mask), 0, 255).astype(np.uint8)

    def convert_file(self, source, destination, *, overwrite=False):
        if Path(source).resolve() == Path(destination).resolve():
            raise ImageError("Input and output must differ; original photos are never overwritten")
        image = read_image(source, self.config.max_pixels)
        result = self.convert(image)
        return write_image(destination, result, overwrite=overwrite)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.preprocessor is not None and hasattr(self.preprocessor, "close"):
                self.preprocessor.close()
        finally:
            if self.backend is not None and hasattr(self.backend, "close"):
                self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
