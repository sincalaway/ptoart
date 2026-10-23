class PtoartError(RuntimeError):
    """An actionable conversion failure."""


class ModelError(PtoartError):
    pass


class ImageError(PtoartError):
    pass


class NoFaceError(PtoartError):
    pass
