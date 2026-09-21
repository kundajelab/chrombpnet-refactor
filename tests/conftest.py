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
"""

import chrombpnet  # noqa: F401  (imported for its import-time side effect)


import pytest

from chrombpnet import optimize
from chrombpnet.parsers import OPTIMIZE_DEFAULT


@pytest.fixture(autouse=True)
def _shipped_optimizations():
    """Tests see what the CLI ships by default, unless they say otherwise.

    `optimize` is process-global, so without this a test that switched
    phases would leak its setting into every test that ran after it.
    """
    previous = optimize.active()
    optimize.configure(OPTIMIZE_DEFAULT)
    yield
    optimize._ACTIVE = previous


@pytest.fixture
def phases():
    """Set the optimization phases for one test: `phases("none")`."""
    def _set(value):
        return optimize.configure(value)
    return _set
