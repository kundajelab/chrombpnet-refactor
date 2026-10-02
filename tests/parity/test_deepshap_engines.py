"""Opt-in pytest entry point for the cross-engine DeepSHAP harness.

Skipped by default: it runs one worker in each of the chrombpnet-tensorflow,
chrombpnet-jax and chrombpnet-pytorch environments. Enable with
``CHROMBPNET_DEEPSHAP_ENGINES=1 pytest tests/parity/test_deepshap_engines.py``
from any of them, with all three installed side by side.
"""

import os
from pathlib import Path

import pytest

from tests.parity import deepshap_engines


def test_tensorflow_and_jax_deepshap_agree_with_pytorch():
    if not os.environ.get(deepshap_engines.OPT_IN):
        pytest.skip("set {}=1 to run the cross-engine DeepSHAP harness (needs the "
                    "chrombpnet-tensorflow, chrombpnet-jax and chrombpnet-pytorch "
                    "environments)".format(deepshap_engines.OPT_IN))
    arguments = deepshap_engines.parser().parse_args([])
    for python in (arguments.tf_python, arguments.jax_python, arguments.torch_python):
        if not Path(python).exists():
            pytest.skip("environment not found: {}".format(python))
    assert deepshap_engines.main(arguments) == 0
