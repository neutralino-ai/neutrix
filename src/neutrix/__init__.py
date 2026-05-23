try:
    from neutrix._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
