"""ChromBPNet, with a choice of deep-learning engine.

``import chrombpnet`` deliberately imports no deep-learning framework: the
TensorFlow and JAX environments have no PyTorch, and the PyTorch model classes
re-exported below are resolved on first attribute access instead (PEP 562).
"""

# Some cluster login environments inject an older system libstdc++ ahead of
# conda without including the active prefix itself in LD_LIBRARY_PATH. Preload
# this environment's runtime before importing PyTorch/SciPy extensions. This is
# process-local and avoids conda environment-variable overrides and warnings.
import ctypes
from importlib import metadata as _metadata
import os
from pathlib import Path
import sys
import tempfile

_numba_cache = Path(tempfile.gettempdir()) / f"chrombpnet-numba-{os.getuid()}"
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))

_cxx_runtime = Path(sys.prefix) / "lib" / "libstdc++.so.6"
if _cxx_runtime.exists():
    try:
        ctypes.CDLL(str(_cxx_runtime), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


try:
    __version__ = _metadata.version("chrombpnet-refactor")
except _metadata.PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BPNet",
    "BiasModel",
    "ChromBPNet",
    "ModelConfig",
    "load_model",
    "save_model",
]


# The PyTorch model API, resolved lazily so that importing the package does not
# import torch. ``from chrombpnet import BPNet`` still works exactly as before.
_LAZY_PYTORCH = frozenset(__all__)


def __getattr__(name):
    if name in _LAZY_PYTORCH:
        from chrombpnet import model as _model
        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_PYTORCH)
