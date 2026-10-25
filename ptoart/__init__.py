"""ptoart: stable, extensible portrait-to-cartoon inference."""
from .config import ArtConfig
from .errors import ImageError, ModelError, NoFaceError, PtoartError

__version__ = "0.3.2"
__all__ = ["ArtConfig", "ArtPipeline", "PtoartError", "ModelError", "ImageError", "NoFaceError"]


def __getattr__(name):
    if name == "ArtPipeline":
        from .pipeline import ArtPipeline
        return ArtPipeline
    raise AttributeError(name)
