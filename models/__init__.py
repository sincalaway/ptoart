"""Model exports, imported lazily to keep package initialization lightweight."""

__all__ = ['ResnetGenerator', 'UgatitSadalinHourglass']


def __getattr__(name):
    if name == 'ResnetGenerator':
        from .networks import ResnetGenerator
        return ResnetGenerator
    if name == 'UgatitSadalinHourglass':
        from .UGATIT_sadalin_hourglass import UgatitSadalinHourglass
        return UgatitSadalinHourglass
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
