"""RGB/RGBA image I/O, Unicode paths and no-clobber atomic output."""
from pathlib import Path
import os
import tempfile
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ImageError

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_image(path, max_pixels=25_000_000):
    path = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.width * image.height > max_pixels:
                    raise ImageError(f"Image exceeds {max_pixels} pixels: {path}")
                image = ImageOps.exif_transpose(image)
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    canvas = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(canvas, rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
                return np.array(image)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ImageError(f"Cannot read image {path}: {exc}") from exc


def write_image(path, image, overwrite=False):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ImageError(f"Unsupported output extension: {suffix}")
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise ImageError("Output must be uint8 RGB or RGBA")
    if image.shape[2] == 4 and suffix not in {".png", ".webp", ".tif", ".tiff"}:
        raise ImageError("Transparent output requires PNG, WebP or TIFF")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        fd, temp = tempfile.mkstemp(prefix=".ptoart-", suffix=suffix, dir=path.parent)
        os.close(fd)
        Image.fromarray(image).save(temp)
        if overwrite:
            os.replace(temp, path)
        else:
            # Atomic no-clobber: racing writers cannot replace an existing result.
            os.link(temp, path)
    except OSError as exc:
        raise ImageError(f"Cannot write {path}: {exc}") from exc
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    return path
