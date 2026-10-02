"""Opt-in pytest entry point for the JAX engine's agreement harness.

Skipped by default: it runs the original implementation in the
chrombpnet-tensorflow environment and the JAX engine in chrombpnet-jax, and
takes several minutes. Enable with
``CHROMBPNET_JAX_ENGINE_PARITY=1 pytest tests/parity/test_jax_engine.py`` from
either environment, with both installed side by side.
"""

import os
from pathlib import Path

import pytest

from tests.parity import jax_engine


def test_jax_engine_agrees_with_the_original():
    if not os.environ.get(jax_engine.PARITY_OPT_IN):
        pytest.skip("set {}=1 to run the JAX engine agreement harness (needs the "
                    "chrombpnet-tensorflow and chrombpnet-jax environments)".format(
                        jax_engine.PARITY_OPT_IN))
    arguments = jax_engine.parser().parse_args([])
    for python in (arguments.tf_python, arguments.jax_python):
        if not Path(python).exists():
            pytest.skip("environment not found: {}".format(python))
    if not (Path(arguments.original_repo) / "chrombpnet").is_dir():
        pytest.skip("original repository not found: {}".format(arguments.original_repo))
    assert jax_engine.main(arguments) == 0
