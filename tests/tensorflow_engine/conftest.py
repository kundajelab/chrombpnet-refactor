"""tensorflow engine (Keras 3) tests: collected only where TensorFlow and Keras 3 are installed.

Starts Keras on TensorFlow before any test module imports keras (``select_backend``), as the engine does for
``--engine tensorflow``.
"""

import importlib.metadata
import importlib.util


def _keras3_on_tensorflow():
    if any(importlib.util.find_spec(name) is None for name in ("tensorflow", "keras")):
        return False
    return importlib.metadata.version("keras").startswith("3.")


if _keras3_on_tensorflow():
    from chrombpnet.engines.tensorflow import select_backend

    select_backend()
else:
    collect_ignore_glob = ["test_*.py"]
