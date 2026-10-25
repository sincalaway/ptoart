"""Reusable sequential inference pipeline with injectable components for tests."""
from pathlib import Path

import cv2
import numpy as np

from .backends import StyleBackend, UpscaleBackend, create_backend, require_file, validate_output
from .config import ArtConfig
from .errors import ImageError, ModelError, NoFaceError, PtoartError
from .io import read_image, write_image


class ArtPipeline:
    """RGB input/output. Reuse one instance per sequential worker, not across threads.

    ``aligned=True`` explicitly bypasses face detection and segmentation. It is
    only for already aligned, tightly cropped square portraits, never full photos.
    """

    def __init__(self, config=None, *, backend=None, preprocessor=None, upscaler=None):
        self.config = config or ArtConfig()
        self.backend = None
        self.preprocessor = preprocessor
        self.upscaler = upscaler
        self.closed = False
        self.face_painter = None
        try:
            if self.upscaler is None and self.config.upscale is not None:
                self.upscaler = UpscaleBackend(self.config.upscale)
            if self.config.output == "style":
                self.backend = backend if backend is not None else StyleBackend(self.config.style)
                return
            if self.config.style is not None:
                # Face/photo modes with a style: AnimeGANv2 face painter replaces the generator.
                self.face_painter = StyleBackend(self.config.style)
            elif backend is None:
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
            if self.face_painter is None:
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
        """Return the cartoon result for one RGB photo.

        ``output="face"`` returns the largest face as a ``size``-square image
        (RGB, or RGBA with a transparent background). ``output="photo"`` returns
        the whole photo, same size as the input, with up to ``max_faces``
        cartoon faces blended back in place.
        """
        if self.closed:
            raise PtoartError("Pipeline is closed")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) < 2):
            raise ImageError("Expected a nonempty uint8 RGB image")
        if image.shape[0] * image.shape[1] > self.config.max_pixels:
            raise ImageError(f"Image exceeds {self.config.max_pixels} pixels")
        if self.config.output == "style":
            return self._convert_style(image)
        if self.config.output == "photo":
            return self._convert_photo(image)
        if self.config.aligned:
            if image.shape[0] != image.shape[1]:
                raise ImageError("--aligned requires a square, pre-aligned portrait")
            rgba = np.dstack((image, np.full(image.shape[:2], 255, dtype=np.uint8)))
        else:
            rgba = self._check_rgba(self.preprocessor.process(image))
        return self._compose_face(rgba, self._infer(rgba))

    @staticmethod
    def _check_rgba(rgba):
        if rgba is None:
            raise NoFaceError("No face detected; use a clear, front-facing portrait")
        if (not isinstance(rgba, np.ndarray) or rgba.ndim != 3 or rgba.shape[2] != 4
                or min(rgba.shape[:2]) < 1 or rgba.dtype != np.uint8):
            raise ImageError("Preprocessor must return uint8 RGBA")
        return rgba

    def _infer(self, rgba):
        """Run the generator on a face crop; returns float32 HWC cartoon (256px, or 512px painter)."""
        if self.face_painter is not None:
            return self._paint_face(rgba)
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
        return (output[0].transpose(1, 2, 0) + 1) * 127.5

    def _paint_face(self, rgba, side=512):
        """AnimeGANv2 face painter on the raw crop (keeps proportions; no white compositing)."""
        interpolation = cv2.INTER_AREA if rgba.shape[0] >= side else cv2.INTER_CUBIC
        rgb = cv2.resize(rgba[:, :, :3], (side, side), interpolation=interpolation)
        tensor = np.ascontiguousarray((rgb.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None])
        try:
            output = np.asarray(self.face_painter.run(tensor))
        except Exception as exc:
            raise ModelError(f"Inference failed: {exc}") from exc
        if output.shape != tensor.shape or not np.isfinite(output).all():
            raise ModelError(f"Invalid face painter output; expected {tensor.shape}, got {output.shape}")
        return (output[0].transpose(1, 2, 0) + 1) * 127.5

    def _upscale(self, cartoon, size):
        """Resize the 256px cartoon to ``size``.

        With an upscaler model (``upscale``), enlarge with super-resolution first;
        otherwise use Lanczos plus light sharpening.
        """
        if size == cartoon.shape[0]:
            return cartoon
        if size < cartoon.shape[0]:
            return cv2.resize(cartoon, (size, size), interpolation=cv2.INTER_AREA)
        if self.upscaler is not None:
            big = self._super_resolve(cartoon)
            interpolation = cv2.INTER_AREA if big.shape[0] >= size else cv2.INTER_LANCZOS4
            return cv2.resize(big, (size, size), interpolation=interpolation)
        big = cv2.resize(cartoon, (size, size), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(big, (0, 0), sigmaX=max(1.0, size / 256))
        return cv2.addWeighted(big, 1.4, blur, -0.4, 0)

    def _super_resolve(self, image):
        try:
            return np.asarray(self.upscaler.run(np.clip(image, 0, 255).astype(np.float32)),
                              dtype=np.float32)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Upscaling failed: {exc}") from exc

    def _compose_face(self, rgba, cartoon):
        size = self.config.size
        interpolation = cv2.INTER_AREA if rgba.shape[0] >= size else cv2.INTER_CUBIC
        rgba = cv2.resize(rgba, (size, size), interpolation=interpolation)
        source = rgba[:, :, :3].astype(np.float32)
        mask = rgba[:, :, 3:4].astype(np.float32) / 255.0
        cartoon = self._upscale(cartoon, size)
        foreground = cartoon * self.config.strength + source * (1 - self.config.strength)
        if self.config.background is None:
            rgb = np.clip(foreground, 0, 255).astype(np.uint8)
            return np.concatenate((rgb, rgba[:, :, 3:4]), axis=2)
        background = np.asarray(self.config.background, dtype=np.float32)
        return np.clip(foreground * mask + background * (1 - mask), 0, 255).astype(np.uint8)

    def _convert_photo(self, image):
        if not hasattr(self.preprocessor, "process_all"):
            raise ModelError("photo output requires a preprocessor with process_all()")
        regions = self.preprocessor.process_all(image, max_faces=self.config.max_faces)
        if not regions:
            raise NoFaceError("No face detected; use a clear, front-facing portrait")
        height, width = image.shape[:2]
        result = image.astype(np.float32)
        # Paint smaller faces first so the largest face wins any overlap.
        for region in reversed(list(regions)):
            rgba = self._check_rgba(region.rgba)
            matrix = np.asarray(region.matrix, dtype=np.float64)
            if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
                raise ImageError("Face region matrix must be a finite 2x3 affine transform")
            side = rgba.shape[0]
            cartoon = self._upscale(self._infer(rgba), side)
            source = rgba[:, :, :3].astype(np.float32)
            foreground = cartoon * self.config.strength + source * (1 - self.config.strength)
            alpha = self._feather(rgba[:, :, 3].astype(np.float32) / 255.0)
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            warped = cv2.warpAffine(foreground, matrix, (width, height), flags=flags,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            weight = cv2.warpAffine(alpha, matrix, (width, height), flags=flags,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)[:, :, None]
            result = warped * weight + result * (1 - weight)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _convert_style(self, image):
        """Stylize the whole photo; the result has the input's size."""
        height, width = image.shape[:2]
        scale = min(1.0, self.config.style_max_side / max(height, width))
        # The generator downsamples twice; keep sides multiples of 4.
        h = max(4, int(round(height * scale / 4)) * 4)
        w = max(4, int(round(width * scale / 4)) * 4)
        small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        tensor = np.ascontiguousarray(
            (small.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None])
        try:
            output = np.asarray(self.backend.run(tensor))
        except Exception as exc:
            raise ModelError(f"Inference failed: {exc}") from exc
        if output.shape != tensor.shape or not np.isfinite(output).all():
            raise ModelError(f"Invalid style output; expected {tensor.shape}, got {output.shape}")
        styled = (output[0].transpose(1, 2, 0) + 1) * 127.5
        if (h, w) != (height, width):
            if self.upscaler is not None and scale < 1:
                styled = self._super_resolve(styled)
            interpolation = cv2.INTER_AREA if styled.shape[0] > height else cv2.INTER_CUBIC
            styled = cv2.resize(styled, (width, height), interpolation=interpolation)
        source = image.astype(np.float32)
        blended = styled * self.config.strength + source * (1 - self.config.strength)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _feather(alpha):
        """Soften the mask edge and fade the crop border so pasted faces show no seams."""
        side = alpha.shape[0]
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, side / 128))
        ramp = np.ones(side, dtype=np.float32)
        fade = max(1, int(side * 0.12))
        ramp[:fade] = np.linspace(0, 1, fade, endpoint=False)
        ramp[-fade:] = np.linspace(1, 0, fade)
        return alpha * np.minimum.outer(ramp, ramp)

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
            try:
                if self.backend is not None and hasattr(self.backend, "close"):
                    self.backend.close()
            finally:
                if self.upscaler is not None and hasattr(self.upscaler, "close"):
                    self.upscaler.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
