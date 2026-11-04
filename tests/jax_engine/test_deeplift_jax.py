# Ported from kundajelab/chrombpnet#284 (NNFC-GMD/chrombpnet@14ca773), tests/test_deeplift_jax.py, with imports
# pointed at chrombpnet.engines.jax; other changes are marked "(chrombpnet-engines)".
"""JAX DeepLIFT/DeepSHAP rules on tiny BPNet-shaped Keras 3 models (CPU).

Checks the rescale rule, the legacy profile weighting, summation-to-delta, the paired vs full (TFDeepExplainer)
layout, an independent NumPy float64 brute-force DeepLIFT, and the hypothetical projection.
"""
import warnings

import chrombpnet.engines.jax  # noqa: F401  (KERAS_BACKEND=jax)
import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest
from keras import layers

from chrombpnet.engines.jax.attribution import deeplift_jax as dl
from chrombpnet.engines.jax.attribution.dinuc_shuffle import make_references
from chrombpnet.engines.jax.attribution.explainer import DeepLiftShap, auto_batch_seqs, check_model
from chrombpnet.engines.jax.layers import LogSumExp
from tests.keras3_deeplift import (  # noqa: F401  (the oracle, shared with tests/tensorflow_engine)
    F, K, L, ND, OUTLEN, build_bpnet, np_conv, np_conv_t, np_deeplift, np_forward, np_rescale, np_weights, onehot,
    predict, rel_l2, shap_utils)


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


# ------------------------------------------------------------------ rules
def test_rescale_rule_cases():
    zx = jnp.array([1.0, -1.0, 0.5, -0.5, 0.3, 2e-7, -4e-7, 3.0], jnp.float32)
    zr = jnp.array([2.0, -2.0, -0.5, 0.5, 0.3, -3e-7, 1e-7, 3.0 - 2e-6], jnp.float32)
    z = jnp.concatenate([zx, zr])
    g = jax.grad(lambda z: jnp.sum(dl.dl_relu(z)))(z)
    expect_x = np.array([1.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0], np.float32)
    np.testing.assert_allclose(np.asarray(g[:8]), expect_x, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(g[8:]), 0.0)
    g_full = jax.grad(lambda z: jnp.sum(dl.dl_relu_full(z)))(z)
    np.testing.assert_array_equal(np.asarray(g_full[:8]), np.asarray(g[:8]))
    # reference half: same secant, its own fallback (zr > 0)
    expect_r = np.array([1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0], np.float32)
    np.testing.assert_allclose(np.asarray(g_full[8:]), expect_r, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(dl.dl_relu(z)), np.maximum(np.asarray(z), 0))


def test_profile_weights_midpoint_rule():
    rng = np.random.RandomState(0)
    mn_x = jnp.asarray(rng.normal(size=(3, 50)), jnp.float32)
    mn_r = mn_x.at[:, :5].add(jnp.asarray(rng.normal(size=(3, 5)), jnp.float32))
    mn_r = mn_r.at[:, 10].set(mn_x[:, 10] + 5e-8)  # |delta| < 1e-7 -> zero weight
    s_x, s_r = jax.nn.softmax(mn_x, -1), jax.nn.softmax(mn_r, -1)
    w = dl.profile_weights(mn_x, mn_r, "chrombpnet")
    expect = np.where(np.abs(np.asarray(mn_x - mn_r)) < 1e-7, 0.0, 0.5 * np.asarray(s_x + s_r))
    np.testing.assert_allclose(np.asarray(w), expect, rtol=1e-6)
    assert np.all(np.asarray(w)[:, 10] == 0)
    np.testing.assert_allclose(np.asarray(dl.profile_weights(mn_x, mn_r, "chrombpnet_tf")), expect,
                               rtol=1e-3, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(dl.profile_weights(mn_x, mn_r, "softmax_x")), np.asarray(s_x))
    with pytest.raises(ValueError):
        dl.profile_weights(mn_x, mn_r, "nope")


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


def test_profile_linearized_identity(model, data, result):
    x, refs = data
    lx, _ = predict(model, x)
    lr, _ = predict(model, refs.reshape(-1, L, 4))
    mnx = lx - lx.mean(1, keepdims=True)
    mnr = (lr - lr.mean(1, keepdims=True)).reshape(len(x), K, -1)
    w = np.asarray(dl.profile_weights(jnp.asarray(np.repeat(mnx, K, 0)), jnp.asarray(mnr.reshape(len(x) * K, -1))))
    rhs = np.sum(w.reshape(len(x), K, -1) * (mnx[:, None] - mnr), axis=-1)
    lhs = np.sum(result.mult["profile"] * (x[:, None] - refs), axis=(2, 3))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-4 * np.abs(rhs).max())
    # it is the linearised delta, not T(x) - T(r): the dropped softmax share is visible
    t = lambda mn: np.sum(np.exp(mn) / np.exp(mn).sum(-1, keepdims=True) * mn, axis=-1)
    assert np.abs(lhs - (t(mnx)[:, None] - t(mnr))).max() > 100 * np.abs(lhs - rhs).max()


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
    oracle = np_deeplift(wts, xs, rs, head).reshape(len(x), K, L, 4)
    assert rel_l2(result.mult[head], oracle) < 1e-5
    lx, cx, _ = np_forward(wts, xs[:3])
    klx, kcx = predict(model, xs[:3])
    np.testing.assert_allclose(klx, lx, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(kcx, cx, rtol=1e-5, atol=1e-6)


def test_rescale_rule_differs_from_plain_gradient(model, data, result):
    x, _ = data
    plain = jax.grad(lambda v: jnp.sum(model(v, training=False)[1]))(jnp.asarray(x, jnp.float32))
    diff = np.abs(result.mult["counts"] - np.asarray(plain)[:, None]).max()
    assert diff > 1e-3 * np.abs(np.asarray(plain)).max()


def test_hypothetical_matches_combine_mult_and_diffref(data, result):
    x, refs = data
    for head in ("counts", "profile"):
        oracle = np.stack([shap_utils.combine_mult_and_diffref([result.mult[head][i]], [x[i].astype(float)],
                                                               [refs[i]])[0] for i in range(len(x))])
        np.testing.assert_allclose(result.hyp[head], oracle, rtol=1e-5, atol=1e-7 * np.abs(oracle).max())


def test_periodic_sequence_has_zero_contributions(model):
    x = onehot(np.tile([0, 1], L // 2))[None]  # (AC)^n: every dinucleotide shuffle equals the input
    explainer = DeepLiftShap(model, batch_seqs=1, num_shuffles=K)
    res = explainer.run(x, return_multipliers=True, return_references=True)
    assert np.all(res.refs == x[:, None])
    for head in ("counts", "profile"):
        assert np.all(x * res.hyp[head] == 0)
    # every delta is 0: counts multipliers fall back to the plain gradient, profile weights are zeroed
    plain = jax.grad(lambda v: jnp.sum(model(v, training=False)[1]))(jnp.asarray(x, jnp.float32))
    np.testing.assert_allclose(res.mult["counts"], np.broadcast_to(np.asarray(plain)[:, None], res.mult["counts"].shape),
                               rtol=1e-5, atol=1e-7)
    assert np.all(res.mult["profile"] == 0)


def test_batching_and_padding_invariance(model, data, result):
    x, refs = data
    for bs in (1, 3, None):
        explainer = DeepLiftShap(model, batch_seqs=bs)
        hyp = explainer.run(x, references=refs).hyp
        assert len(explainer._steps) == 1
        for head in ("counts", "profile"):
            np.testing.assert_allclose(hyp[head], result.hyp[head], rtol=1e-5,
                                       atol=1e-6 * np.abs(result.hyp[head]).max())


def test_generated_references_are_content_seeded(model, data):
    x, _ = data
    explainer = DeepLiftShap(model, heads=["counts"], batch_seqs=2, num_shuffles=K)
    a = explainer.run(x, seed=7, return_references=True)
    np.testing.assert_array_equal(a.refs, make_references(x, num_shuffles=K, seed=7))
    b = explainer.run(x[::-1], seed=7)
    np.testing.assert_allclose(b.hyp["counts"][::-1], a.hyp["counts"], rtol=1e-6, atol=1e-8)


def test_activations_restored_and_model_unchanged(model, data):
    x, _ = data
    before = predict(model, x)
    DeepLiftShap(model, batch_seqs=5, num_shuffles=2).explain(x)
    for layer in model.layers:
        if isinstance(layer, layers.Conv1D):
            assert keras.activations.serialize(layer.activation) in ("relu", "linear")
    after = predict(model, x)
    np.testing.assert_array_equal(before[0], after[0])


def test_profile_weighting_alternatives(model, data):
    x, refs = data
    lx, _ = predict(model, x)
    lr, _ = predict(model, refs.reshape(-1, L, 4))
    mnx = np.repeat(lx - lx.mean(1, keepdims=True), K, 0)
    mnr = lr - lr.mean(1, keepdims=True)
    for mode in ("softmax_x", "tangermeme", "chrombpnet_tf"):
        mult = DeepLiftShap(model, heads=["profile"], profile_weighting=mode, batch_seqs=5).multipliers(
            x, references=refs)["profile"]
        lhs = np.sum(mult * (x[:, None] - refs), axis=(2, 3)).reshape(-1)
        w = np.asarray(dl.profile_weights(jnp.asarray(mnx, jnp.float32), jnp.asarray(mnr, jnp.float32), mode))
        rhs = np.sum(w * (mnx - mnr), axis=-1)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-4 * np.abs(rhs).max())


def test_additivity_warning(model, data):
    x, refs = data
    explainer = DeepLiftShap(model, heads=["counts"], batch_seqs=5, additivity_threshold=1e-12)
    with pytest.warns(UserWarning, match="summation-to-delta"):
        explainer.run(x, references=refs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DeepLiftShap(model, heads=["counts"], batch_seqs=5).run(x, references=refs)


def test_mixed_precision_model_is_explained_in_float32(model, data, result):
    x, refs = data
    bf16 = build_bpnet(dtype="mixed_bfloat16")
    bf16.set_weights(model.get_weights())
    assert bf16.get_layer("bpnet_1st_conv").compute_dtype == "bfloat16"
    res = DeepLiftShap(bf16, batch_seqs=2).run(x, references=refs)
    assert bf16.get_layer("bpnet_1st_conv").compute_dtype == "bfloat16"
    for head in ("counts", "profile"):
        np.testing.assert_allclose(res.hyp[head], result.hyp[head], rtol=1e-5,
                                   atol=1e-6 * np.abs(result.hyp[head]).max())


# ------------------------------------------------------------------ model support
def build_full_model():
    """chrombpnet_with_bias_model structure: no-bias + bias submodels, Add profile, logsumexp counts."""
    nobias = build_bpnet(seed=3, prefix="wo_bias_", name="model_wo_bias")
    bias = build_bpnet(seed=4, filters=4, name="model_bias")
    inp = keras.Input((L, 4), name="sequence")
    b_out, nb_out = bias(inp), nobias(inp)
    profile = layers.Add(name="logits_profile_predictions")([nb_out[0], b_out[0]])
    counts = LogSumExp(name="logcount_predictions")(layers.Concatenate(axis=-1)([nb_out[1], b_out[1]]))
    return keras.Model(inp, [profile, counts]), nobias, bias


def test_full_model_profile_supported_counts_refused(data):
    x, refs = data
    full, nobias, bias = build_full_model()
    # (chrombpnet-engines) .keras checkpoints
    with pytest.raises(NotImplementedError, match="_nobias.keras"):
        DeepLiftShap(full, heads=["counts", "profile"])
    res = DeepLiftShap(full, heads=["profile"], batch_seqs=5).run(x, references=refs, return_multipliers=True)
    assert res.additivity["profile"]["max_rel_err"] < 1e-4
    # the swap reaches the nested submodels: summation-to-delta of the full-model profile head
    lx = np.asarray(full(x.astype(np.float32))[0], np.float64)
    lr = np.asarray(full(refs.reshape(-1, L, 4).astype(np.float32))[0], np.float64)
    mnx = np.repeat(lx - lx.mean(1, keepdims=True), K, 0)
    mnr = lr - lr.mean(1, keepdims=True)
    w = np.asarray(dl.profile_weights(jnp.asarray(mnx), jnp.asarray(mnr)))
    rhs = np.sum(w * (mnx - mnr), axis=-1)
    lhs = np.sum(res.mult["profile"] * (x[:, None] - refs), axis=(2, 3)).reshape(-1)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-4 * np.abs(rhs).max())
    DeepLiftShap(nobias, heads=["counts", "profile"])  # the no-bias submodel is fine


def test_unsupported_layers_raise():
    inp = keras.Input((64, 4))
    h = layers.Conv1D(4, 5, activation="sigmoid")(inp)
    out = layers.Dense(1)(layers.GlobalAveragePooling1D()(h))
    with pytest.raises(NotImplementedError, match="sigmoid"):
        check_model(keras.Model(inp, [layers.Flatten()(h), out]), ["counts"])
    h = layers.BatchNormalization()(layers.Conv1D(4, 5, activation="relu")(inp))
    out = layers.Dense(1)(layers.GlobalAveragePooling1D()(h))
    with pytest.raises(NotImplementedError, match="BatchNormalization"):
        check_model(keras.Model(inp, [layers.Flatten()(h), out]), ["profile"])


def test_auto_batch_seqs(model):
    # small models are capped at 32 sequences per step
    assert auto_batch_seqs(model) == 32
    # chrombpnet_nobias-sized (512 filters, 8 dilated layers, 2114 bp): ~4 GB per sequence (estimate) in 12 GB
    wide = build_bpnet(filters=512, n_dil=8, inputlen=2114)
    assert auto_batch_seqs(wide) == 2
    assert auto_batch_seqs(wide, n_heads=2) == 1

    class SmallDevice:
        def memory_stats(self):
            return {"bytes_limit": 3e9, "bytes_in_use": 1e9}

    # only 60% of the 2 GB still free may be used
    assert auto_batch_seqs(wide, device=SmallDevice()) == 1


def test_resolve_precision():
    from chrombpnet.engines.jax.attribution.explainer import resolve_precision
    assert resolve_precision("auto", backend="cpu") == "highest"
    assert resolve_precision("auto", backend="gpu") == "default"
    assert resolve_precision("highest", backend="gpu") == "highest"