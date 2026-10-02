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


def test_honour_stop_gradient_restores_the_shap_forks_handlers():
    from shap.explainers.deep import deep_tf

    from chrombpnet.engines.tensorflow.attribution import honour_stop_gradient

    before = dict(deep_tf.op_handlers)
    assert "StopGradient" not in before  # commented out in the fork (deep_tf.py:595)
    with honour_stop_gradient():
        assert deep_tf.op_handlers["StopGradient"] is deep_tf.break_dependence
    assert deep_tf.op_handlers == before
    with pytest.raises(RuntimeError):
        with honour_stop_gradient():
            raise RuntimeError
    assert deep_tf.op_handlers == before


# explain() needs graph mode, which must be entered before TensorFlow runs
# anything in the process, so it runs in a fresh interpreter.
_EXPLAIN_PROFILE = """
import contextlib, sys
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from chrombpnet.engines.tensorflow import attribution
from chrombpnet.engines.tensorflow.predictor import load_model
if sys.argv[3] == "without-handler":
    attribution.honour_stop_gradient = contextlib.nullcontext
model = load_model(sys.argv[1], compile=False)
np.save(sys.argv[4], attribution.explain(model, np.load(sys.argv[2]), ["profile"], 3, 11)["profile"])
"""


def test_profile_scores_honour_the_targets_stop_gradient(engine, bias_path, tmp_path):
    """With the handler the profile logits are weighted by softmax(mn_x), as in
    the PyTorch engine; without it, by the midpoint of the input's and the
    reference's softmax, as in the original."""
    import subprocess
    import sys

    seqs = np.eye(4, dtype=np.int8)[np.random.RandomState(3).randint(0, 4, (2, INPUTLEN))]
    np.save(tmp_path / "seqs.npy", seqs)
    scores = {}
    for mode in ("with-handler", "without-handler"):
        out = tmp_path / (mode + ".npy")
        subprocess.run([sys.executable, "-c", _EXPLAIN_PROFILE, bias_path,
                        str(tmp_path / "seqs.npy"), mode, str(out)], check=True)
        scores[mode] = np.load(out)
    honoured, midpoint = scores["with-handler"], scores["without-handler"]
    assert np.linalg.norm(honoured - midpoint) > 1e-3 * np.linalg.norm(midpoint)
    # The CLI's path, a spawned graph-mode child, computes the former.
    handle = engine.load_attribution_model(bias_path, None)
    cli = dict(engine.attribute(handle, seqs, ["profile"], device=None, batch_size=1,
                                num_shuffles=3, seed=11))["profile"]
    np.testing.assert_allclose(cli, honoured, rtol=1e-6, atol=1e-9)
