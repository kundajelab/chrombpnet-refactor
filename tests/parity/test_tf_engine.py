"""Opt-in pytest entry point for the TensorFlow engine's agreement harness.

Skipped by default: it runs about a dozen TensorFlow processes against the
original repository and takes several minutes. Enable with
``CHROMBPNET_TF_ENGINE_PARITY=1 pytest tests/parity`` in the
``chrombpnet-tensorflow`` environment; the original runs in its own
(``chrombpnet``, or ``CHROMBPNET_ORIGINAL_PYTHON``).
"""

import os
from pathlib import Path

import pytest

from tests.parity import tf_engine


def test_tensorflow_engine_agrees_with_the_original():
    if not os.environ.get(tf_engine.PARITY_OPT_IN):
        pytest.skip("set {}=1 to run the TensorFlow engine agreement harness "
                    "(needs the chrombpnet-tensorflow environment, and the original's "
                    "chrombpnet environment beside it)".format(
                        tf_engine.PARITY_OPT_IN))
    pytest.importorskip("tensorflow")
    arguments = tf_engine.parser().parse_args([])
    if not (Path(arguments.original_repo) / "chrombpnet").is_dir():
        pytest.skip("original repository not found: {}".format(arguments.original_repo))
    assert tf_engine.main(arguments) == 0
