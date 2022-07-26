import importlib

__all__ = ["algorithm", "analysis", "color", "messaging", "plot", "problems", "utils"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
