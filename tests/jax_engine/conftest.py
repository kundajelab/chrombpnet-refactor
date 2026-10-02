"""JAX-engine tests: collected only where Keras 3 and JAX are installed.

Importing ``chrombpnet.engines.jax`` first selects the Keras JAX backend
before any test module imports keras.
"""

import importlib.util

import chrombpnet.engines.jax  # noqa: F401  (sets KERAS_BACKEND=jax before keras is imported)

if any(importlib.util.find_spec(name) is None for name in ("jax", "keras")):
    collect_ignore_glob = ["test_*.py"]
