"""Make the package's libstdc++ preload run before anything else imports.

`chrombpnet/__init__.py` preloads this environment's `libstdc++.so.6` with
`RTLD_GLOBAL`, because some cluster login shells put an older system copy
ahead of conda on the loader path. That preload only helps if it happens
before the first compiled extension is loaded: `torch` binds the system
library on import, after which SciPy's `spatial` extensions fail to resolve
`CXXABI_1.3.15`.

Whether that happened was previously an accident of collection order.
`pytest` alone was fine -- alphabetically `test_data_io.py` imports
`chrombpnet` early -- but `pytest tests/test_models.py tests/test_metrics.py`
loaded torch first and died during collection. Importing the package here
makes the ordering explicit, since conftest is imported before any test
module.

For the same reason ``TF_USE_LEGACY_KERAS`` is set here: TensorFlow decides
whether ``tf.keras`` is Keras 3 or tf-keras when it is first imported, and
``tests/test_tensorflow.py`` imports it at collection, before anything has
imported the TensorFlow engine, which sets it too.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import chrombpnet  # noqa: E402,F401  (imported for its import-time side effect)


import pytest

from chrombpnet import fused_kernels


@pytest.fixture(autouse=True)
def _fused_kernels_off():
    """Tests see the shipped default, unless they say otherwise.

    The setting is process-global, so without this a test that turned the
    fused kernels on would leak that into every test that ran after it.
    """
    previous = fused_kernels.enabled()
    fused_kernels.configure(False)
    yield
    fused_kernels.configure(previous)


@pytest.fixture
def predict_fk():
    """Turn the fused-kernel route on for one test: `predict_fk(True)`."""
    return fused_kernels.configure
