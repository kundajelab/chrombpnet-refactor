"""The TensorFlow DeepLIFT/DeepSHAP rules of the tensorflow engine, on tiny BPNet-shaped Keras 3 models (CPU).

The JAX engine's rule tests (``tests/jax_engine/test_deeplift_jax.py``) for the TensorFlow port: the rescale rule
and its fallbacks, the legacy profile weighting, summation-to-delta, the paired vs full layout, the same NumPy
float64 brute-force DeepLIFT oracle (``tests/keras3_deeplift``), and the hypothetical projection.
"""
import warnings

import keras
import numpy as np
import pytest
import tensorflow as tf

from chrombpnet.engines.jax.attribution.dinuc_shuffle import make_references
from chrombpnet.engines.jax.layers import LogSumExp
from chrombpnet.engines.tensorflow import deeplift_tf as dl
from chrombpnet.engines.tensorflow.explainer import DeepLiftShap, check_model, resolve_precision
from tests import keras3_deeplift as oracle
from tests.keras3_deeplift import K, L, build_bpnet, np_deeplift, np_weights, onehot, predict, rel_l2, shap_utils


@pytest.fixture(scope="module")
def model():
    return build_bpnet()


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(1)
    x = onehot(rng.randint(0, 4, (5, L)))
    refs = make_references(x, num_shuffles=K, seed=1234)
    return x, refs


@pytest.fixture(scope="module")
def result(model, data):
    x, refs = data
    return DeepLiftShap(model, batch_seqs=2).run(x, references=refs, return_multipliers=True)


def _grad(fn, z):
    z = tf.constant(z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        out = tf.reduce_sum(fn(z))
    return tape.gradient(out, z).numpy()


# ------------------------------------------------------------------ rules
def test_thresholds_are_the_jax_rules():
    assert dl.EPS_RELU == oracle.EPS_RELU and dl.EPS_MUL == oracle.EPS_MUL
    assert dl.PROFILE_WEIGHTINGS == ("chrombpnet", "chrombpnet_tf", "softmax_x", "tangermeme")


def test_rescale_rule_cases():
    zx = np.array([1.0, -1.0, 0.5, -0.5, 0.3, 2e-7, -4e-7, 3.0], np.float32)
    zr = np.array([2.0, -2.0, -0.5, 0.5, 0.3, -3e-7, 1e-7, 3.0 - 2e-6], np.float32)
    z = np.concatenate([zx, zr])
    g = _grad(dl.dl_relu, z)
    expect_x = np.array([1.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0], np.float32)
    np.testing.assert_allclose(g[:8], expect_x, rtol=1e-6)
    np.testing.assert_array_equal(g[8:], 0.0)
    g_full = _grad(dl.dl_relu_full, z)
    np.testing.assert_array_equal(g_full[:8], g[:8])
    expect_r = np.array([1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0], np.float32)
    np.testing.assert_allclose(g_full[8:], expect_r, rtol=1e-6)
    np.testing.assert_array_equal(dl.dl_relu(tf.constant(z)).numpy(), np.maximum(z, 0))


def test_profile_weights_midpoint_rule():
    rng = np.random.RandomState(0)
    mn_x = rng.normal(size=(3, 50)).astype(np.float32)
    mn_r = mn_x.copy()
    mn_r[:, :5] += rng.normal(size=(3, 5)).astype(np.float32)
    mn_r[:, 10] = mn_x[:, 10] + 5e-8  # |delta| < 1e-7 -> zero weight
    s_x, s_r = tf.nn.softmax(mn_x, -1).numpy(), tf.nn.softmax(mn_r, -1).numpy()
    w = dl.profile_weights(tf.constant(mn_x), tf.constant(mn_r), "chrombpnet").numpy()
    expect = np.where(np.abs(mn_x - mn_r) < 1e-7, 0.0, 0.5 * (s_x + s_r))
    np.testing.assert_allclose(w, expect, rtol=1e-6)
    assert np.all(w[:, 10] == 0)
    np.testing.assert_allclose(dl.profile_weights(tf.constant(mn_x), tf.constant(mn_r), "chrombpnet_tf").numpy(),
                               expect, rtol=1e-3, atol=1e-6)
    np.testing.assert_array_equal(dl.profile_weights(tf.constant(mn_x), tf.constant(mn_r), "softmax_x").numpy(), s_x)
    with pytest.raises(ValueError):
        dl.profile_weights(tf.constant(mn_x), tf.constant(mn_r), "nope")


# ------------------------------------------------------------------ explainer
def test_counts_summation_to_delta(model, data, result):
    x, refs = data
    _, cx = predict(model, x)
    _, cr = predict(model, refs.reshape(-1, L, 4))
    cr = cr.reshape(len(x), K)
    lhs = np.sum(result.mult["counts"] * (x[:, None] - refs), axis=(2, 3))
    rhs = cx[:, None] - cr
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-4 * np.abs(rhs).max())
    assert result.additivity["counts"]["max_rel_err"] < 1e-4


def test_profile_summation_to_delta(result):
    assert result.additivity["profile"]["max_rel_err"] < 1e-4
    assert result.additivity["profile"]["n_pairs"] == 5 * K


def test_paired_equals_full_variant(model, data, result):
    x, refs = data
    full = DeepLiftShap(model, batch_seqs=5, variant="full").run(x, references=refs, return_multipliers=True)
    for head in ("counts", "profile"):
        np.testing.assert_allclose(full.mult[head], result.mult[head], rtol=1e-5,
                                   atol=1e-6 * np.abs(result.mult[head]).max())


@pytest.mark.parametrize("head", ["counts", "profile"])
def test_matches_numpy_bruteforce_deeplift(model, data, result, head):
    x, refs = data
    wts = np_weights(model)
    xs = np.repeat(x, K, 0).astype(np.float64)
    rs = refs.reshape(-1, L, 4).astype(np.float64)
    expected = np_deeplift(wts, xs, rs, head).reshape(len(x), K, L, 4)
    assert rel_l2(result.mult[head], expected) < 1e-5


def test_rescale_rule_differs_from_plain_gradient(model, data, result):
    x, _ = data
    v = tf.constant(x.astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(v)
        out = tf.reduce_sum(model(v, training=False)[1])
    plain = tape.gradient(out, v).numpy()
    mult = result.mult["counts"].mean(axis=1)
    assert rel_l2(mult, plain) > 1e-3


def test_hypothetical_matches_combine_mult_and_diffref(data, result):
    x, refs = data
    for head in ("counts", "profile"):
        expected = np.stack([shap_utils.combine_mult_and_diffref([result.mult[head][i]], [x[i].astype(float)],
                                                                 [refs[i]])[0] for i in range(len(x))])
        np.testing.assert_allclose(result.hyp[head], expected, rtol=1e-5, atol=1e-7)


def test_batching_and_padding_invariance(model, data, result):
    x, refs = data
    for batch_seqs in (1, 3, 5):
        other = DeepLiftShap(model, batch_seqs=batch_seqs).run(x, references=refs)
        for head in ("counts", "profile"):
            np.testing.assert_allclose(other.hyp[head], result.hyp[head], rtol=1e-5,
                                       atol=1e-7 * max(1.0, np.abs(result.hyp[head]).max()))


def test_activations_restored_and_model_unchanged(model, data):
    x, refs = data
    before = [layer.activation for layer in model.layers if hasattr(layer, "activation")]
    weights = [w.copy() for w in model.get_weights()]
    out = predict(model, x)
    DeepLiftShap(model, batch_seqs=2).run(x, references=refs)
    assert [layer.activation for layer in model.layers if hasattr(layer, "activation")] == before
    for a, b in zip(weights, model.get_weights()):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(out, predict(model, x)):
        np.testing.assert_array_equal(a, b)


def test_additivity_warning(model, data):
    x, refs = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DeepLiftShap(model, batch_seqs=2, additivity_threshold=0.0).run(x, references=refs)
    assert any("summation-to-delta" in str(w.message) for w in caught)


def test_mixed_precision_model_is_explained_in_float32(model, data, result):
    x, refs = data
    bf16 = build_bpnet(dtype="mixed_bfloat16")
    bf16.set_weights(model.get_weights())
    assert bf16.get_layer("bpnet_1st_conv").compute_dtype == "bfloat16"
    res = DeepLiftShap(bf16, batch_seqs=2).run(x, references=refs)
    assert bf16.get_layer("bpnet_1st_conv").compute_dtype == "bfloat16"
    for head in ("counts", "profile"):
        assert res.hyp[head].dtype == np.float32
        np.testing.assert_allclose(res.hyp[head], result.hyp[head], rtol=1e-5,
                                   atol=1e-6 * np.abs(result.hyp[head]).max())


# ------------------------------------------------------------------ model support
def build_full_model():
    """chrombpnet_with_bias_model structure: no-bias + bias submodels, Add profile, logsumexp counts."""
    nobias = build_bpnet(seed=3, prefix="wo_bias_", name="model_wo_bias")
    bias = build_bpnet(seed=4, filters=4, name="model_bias")
    inp = keras.Input((L, 4), name="sequence")
    b_out, nb_out = bias(inp), nobias(inp)
    profile = keras.layers.Add(name="logits_profile_predictions")([nb_out[0], b_out[0]])
    counts = LogSumExp(name="logcount_predictions")(keras.layers.Concatenate(axis=-1)([nb_out[1], b_out[1]]))
    return keras.Model(inp, [profile, counts]), nobias, bias


def test_full_model_profile_supported_counts_refused(data):
    x, refs = data
    full, nobias, bias = build_full_model()
    with pytest.raises(NotImplementedError, match="_nobias.keras"):
        DeepLiftShap(full, heads=["counts", "profile"])
    res = DeepLiftShap(full, heads=["profile"], batch_seqs=5).run(x, references=refs, return_multipliers=True)
    assert res.additivity["profile"]["max_rel_err"] < 1e-4
    # the swap reaches the nested submodels: summation-to-delta of the full-model profile head
    lx = np.asarray(full(x.astype(np.float32))[0], np.float64)
    lr = np.asarray(full(refs.reshape(-1, L, 4).astype(np.float32))[0], np.float64)
    mnx = np.repeat(lx - lx.mean(1, keepdims=True), K, 0)
    mnr = lr - lr.mean(1, keepdims=True)
    w = dl.profile_weights(tf.constant(mnx), tf.constant(mnr)).numpy()
    rhs = np.sum(w * (mnx - mnr), axis=-1)
    lhs = np.sum(res.mult["profile"] * (x[:, None] - refs), axis=(2, 3)).reshape(-1)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-4 * np.abs(rhs).max())
    check_model(nobias, ("counts", "profile"))  # the no-bias submodel is fine


def test_unsupported_layers_raise():
    inp = keras.Input((L, 4))
    x = keras.layers.Conv1D(4, 3, activation="sigmoid")(inp)
    m = keras.Model(inp, [keras.layers.Flatten()(x), keras.layers.Dense(1)(keras.layers.Flatten()(x))])
    with pytest.raises(NotImplementedError, match="sigmoid"):
        DeepLiftShap(m)


def test_resolve_precision():
    assert resolve_precision("highest") == "highest" and resolve_precision("default") == "default"
    expected = "default" if tf.config.list_logical_devices("GPU") else "highest"
    assert resolve_precision("auto") == expected
    with pytest.raises(ValueError):
        resolve_precision("high")
