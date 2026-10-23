"""Utility exports, loaded on demand to keep inference imports lightweight."""

from importlib import import_module

_UTILITY_EXPORTS = {
    'RGB2BGR', 'cam', 'check_folder', 'cv2', 'denorm', 'imagenet_norm',
    'imsave', 'inverse_transform', 'load_test_data', 'merge', 'misc', 'np',
    'os', 'preprocessing', 'save_images', 'str2bool', 'tensor2numpy', 'torch',
}
__all__ = ['Preprocess'] + sorted(_UTILITY_EXPORTS)


def __getattr__(name):
    if name == 'Preprocess':
        from .preprocess import Preprocess
        return Preprocess
    if name in _UTILITY_EXPORTS:
        utility_module = import_module('.utils', __name__)
        return getattr(utility_module, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
