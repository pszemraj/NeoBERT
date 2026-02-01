"""NeoBERT package entry points and shared utilities."""

from .config import Config, ConfigLoader, load_config_from_args
from .utils import configure_tf32, model_summary

# Import version information
try:
    from ._version import __version__
except ImportError:
    # Fallback if setuptools-scm hasn't generated the version file yet
    __version__ = "0.0.0+unknown"

__all__ = [
    "Config",
    "ConfigLoader",
    "load_config_from_args",
    "configure_tf32",
    "model_summary",
    "__version__",
]
