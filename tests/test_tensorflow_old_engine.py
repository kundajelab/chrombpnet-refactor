"""The tensorflow-old engine (chrombpnet 1.x's Keras 2 code), in-process. Runs only in the chrombpnet-tensorflow env.

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
    return engines.get("tensorflow-old")


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
    from chrombpnet.engines.tensorflow_old import deep_tf

    from chrombpnet.engines.tensorflow_old.attribution import honour_stop_gradient

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
import sys
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from chrombpnet.engines.tensorflow_old import attribution
from chrombpnet.engines.tensorflow_old.predictor import load_model
model = load_model(sys.argv[1], compile=False)
scores = attribution.explain(model, np.load(sys.argv[2]), ["profile"], 3, 11,
                             stop_gradient=sys.argv[3] == "with-handler")
np.save(sys.argv[4], scores["profile"])
"""


def test_profile_scores_follow_the_original_unless_stopgrad_is_asked_for(engine, bias_path, tmp_path):
    """By default the profile logits are weighted by the midpoint of the
    input's and the reference's softmax, as in the original, whose shap fork
    leaves the StopGradient handler off; with --contribs-profile-stopgrad the
    handler is registered and the weights are softmax(mn_x)."""
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
    # The CLI's path, a spawned graph-mode child, computes each on request.
    handle = engine.load_attribution_model(bias_path, None)
    for stop_gradient, expected in ((False, midpoint), (True, honoured)):
        cli = dict(engine.attribute(handle, seqs, ["profile"], device=None, batch_size=1,
                                    num_shuffles=3, seed=11,
                                    profile_stop_gradient=stop_gradient))["profile"]
        np.testing.assert_allclose(cli, expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("first", ["import tensorflow, tensorflow.keras.callbacks",
                                   # tf.keras.Model is then tf-keras's, but the engine's
                                   # `from tensorflow.keras... import` lines would get Keras 3
                                   "import tensorflow"])
def test_require_refuses_keras3_as_tf_keras(first):
    """TensorFlow imported before the engine, without TF_USE_LEGACY_KERAS,
    gives Keras 3 where the engine needs tf-keras; the engine must say so
    instead of failing somewhere inside Keras. With the variable, require
    passes."""
    import os
    import subprocess
    import sys

    code = (first + "\n"
            "import chrombpnet.engines.tensorflow_old as e\n"
            "e.ENGINE.require()\n")
    # KERAS_BACKEND too: the JAX engine sets it to jax, and Keras 3 would
    # then fail on the missing jax before the engine is reached.
    env = {k: v for k, v in os.environ.items() if k not in ("TF_USE_LEGACY_KERAS", "KERAS_BACKEND")}
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    refused = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert refused.returncode != 0
    assert "TF_USE_LEGACY_KERAS" in refused.stderr
    passed = subprocess.run([sys.executable, "-c", code], env=dict(env, TF_USE_LEGACY_KERAS="1"),
                            capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr


def test_saved_models_are_rewritten_for_the_original(tmp_path):
    """make_original_readable: the count head becomes chrombpnet 1.x's bytecode
    Lambda, and the fields Keras 2.8 refuses go, while tf-keras still loads the
    file and predicts the same."""
    import json

    import h5py

    from chrombpnet import legacy_lambda
    from chrombpnet.engines.tensorflow_old import layers
    from chrombpnet.engines.tensorflow_old.predictor import load_model

    inputs = tf.keras.layers.Input(shape=(2,), name="counts")
    model = tf.keras.Model(inputs, layers.count_head()(inputs))
    path = str(tmp_path / "model.h5")
    model.save(path)
    assert layers.make_original_readable(path)
    with h5py.File(path, "r") as handle:
        raw = handle.attrs["model_config"]
    config = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    assert "trainable" not in config["config"]
    by_class = {layer["class_name"]: layer["config"] for layer in config["config"]["layers"]}
    assert "optional" not in by_class["InputLayer"]
    head = by_class["Lambda"]
    assert head["function"][0] == legacy_lambda.LEGACY_LOGSUMEXP_BYTECODE
    assert head["function_type"] == "lambda"
    x = np.array([[0.5, -1.0], [3.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(load_model(path, compile=False).predict(x, verbose=0),
                                  model.predict(x, verbose=0))
    assert not layers.make_original_readable(path)  # nothing left to rewrite
