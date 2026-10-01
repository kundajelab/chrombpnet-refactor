"""The TensorFlow engine, in-process. Runs only in the chrombpnet-tensorflow env.

Byte-identity with the original implementation is tests/parity/tf_engine.py's
job; these pin the engine's own behaviour, including where it deliberately
departs from the original.
"""

from types import SimpleNamespace

import numpy as np
import pytest

# TensorFlow-engine tests: skipped in the PyTorch and JAX environments.
tf = pytest.importorskip("tensorflow")

from chrombpnet import engines  # noqa: E402

INPUTLEN, OUTPUTLEN, FILTERS, N_DIL = 128, 20, 4, 2


@pytest.fixture(scope="module")
def engine():
    return engines.get("tensorflow")


def _args(engine, kind, seed=7, **extra):
    values = dict(seed=seed, learning_rate=0.001,
                  architecture_from_file=engine.default_architecture(kind))
    values.update(extra)
    return SimpleNamespace(**values)


def _parameters(**extra):
    values = {"filters": str(FILTERS), "n_dil_layers": str(N_DIL),
              "counts_loss_weight": "1.0", "inputlen": str(INPUTLEN),
              "outputlen": str(OUTPUTLEN)}
    values.update(extra)
    return values


@pytest.fixture(scope="module")
def bias_path(engine, tmp_path_factory):
    model, _ = engine.get_model(_args(engine, "bias"), _parameters())
    path = str(tmp_path_factory.mktemp("bias") / "bias.h5")
    model.save(path)
    return path


def _tower_weights(engine, bias_path, seed):
    model, _ = engine.get_model(
        _args(engine, "chrombpnet", seed=seed), _parameters(bias_model_path=bias_path))
    return model.get_layer("model_wo_bias").get_weights()


def test_seed_controls_the_bias_factorized_tower(engine, bias_path):
    """--seed must reach the trainable tower's initial weights.

    The original seeded only after building that tower, so two runs with the
    same --seed started from different weights (defect 6 in chrombpnet_bugs.md).
    """
    first = _tower_weights(engine, bias_path, seed=7)
    again = _tower_weights(engine, bias_path, seed=7)
    other = _tower_weights(engine, bias_path, seed=8)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))
