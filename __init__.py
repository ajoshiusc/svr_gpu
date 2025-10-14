"""Minimal standalone SVR CLI package."""

from .svr_cli import main, run_svr, preprocess_inputs

# Package version
__version__ = "0.1.0"

__all__ = ["main", "run_svr", "preprocess_inputs", "__version__"]
