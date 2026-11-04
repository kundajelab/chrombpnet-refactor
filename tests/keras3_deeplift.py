"""Backend-neutral DeepLIFT test helpers for the Keras 3 engines: a tiny BPNet-shaped Keras 3 model and an
independent NumPy float64 brute-force DeepLIFT.

Moved unchanged from tests/jax_engine/test_deeplift_jax.py (ported from kundajelab/chrombpnet#284,
tests/test_deeplift_jax.py) so that the tensorflow (Keras 3) engine's tests check its TensorFlow rules against the
same oracle; only the rule thresholds are named here instead of read from deeplift_jax. Keras must already be
on its backend when this is imported.
"""
import keras
import numpy as np
from keras import layers

# deeplift_jax.EPS_RELU and EPS_MUL (deeplift_tf's are the same; the tests check it)
EPS_RELU = 1e-6
EPS_MUL = 1e-7


class shap_utils:  # (chrombpnet-engines) the oracle, inlined: shap_utils needs TensorFlow here
    @staticmethod
    def combine_mult_and_diffref(mult, orig_inp, bg_data):
        """kundajelab/chrombpnet@09938fd evaluation/interpret/shap_utils.py, verbatim."""
        to_return = []
        for l in [0]:
            projected_hypothetical_contribs = \
                np.zeros_like(bg_data[l]).astype("float")
            assert len(orig_inp[l].shape)==2
            for i in range(orig_inp[l].shape[-1]):
                hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
                hypothetical_input[:, i] = 1.0
                hypothetical_difference_from_reference = \
                    (hypothetical_input[None, :, :] - bg_data[l])
                hypothetical_contribs = hypothetical_difference_from_reference * \
                                        mult[l]
                projected_hypothetical_contribs[:, :, i] = \
                    np.sum(hypothetical_contribs, axis=-1)
            to_return.append(np.mean(projected_hypothetical_contribs,axis=0))
        return to_return


L, F, ND, OUTLEN, K = 256, 8, 2, 100, 4


def build_bpnet(inputlen=L, filters=F, n_dil=ND, outlen=OUTLEN, seed=0, prefix="", dtype=None, name=None):
    """Same layer structure as chrombpnet's bpnet_model, small."""
    keras.utils.set_random_seed(seed)
    kw = {"dtype": dtype} if dtype else {}
    inp = keras.Input((inputlen, 4), name="sequence")
    x = layers.Conv1D(filters, 21, activation="relu", name=prefix + "bpnet_1st_conv", **kw)(inp)
    for i in range(1, n_dil + 1):
        conv_x = layers.Conv1D(filters, 3, activation="relu", dilation_rate=2 ** i, name=prefix + "bpnet_{}conv".format(i),
                               **kw)(x)
        crop = (x.shape[1] - conv_x.shape[1]) // 2
        x = layers.Cropping1D(crop, name=prefix + "bpnet_{}crop".format(i), **kw)(x)
        x = layers.add([conv_x, x], **kw)
    prof = layers.Conv1D(1, 75, name=prefix + "prof_out_precrop", **kw)(x)
    cropsize = prof.shape[1] // 2 - outlen // 2
    prof = layers.Cropping1D(cropsize, name=prefix + "logits_profile_predictions_preflatten", **kw)(prof)
    profile_out = layers.Flatten(name=prefix + "logits_profile_predictions", **kw)(prof)
    gap = layers.GlobalAveragePooling1D(name=prefix + "gap", **kw)(x)
    count_out = layers.Dense(1, name=prefix + "logcount_predictions", **kw)(gap)
    model = keras.Model(inp, [profile_out, count_out], name=name)
    # non-zero biases so ReLUs switch inside the sequence
    rng = np.random.RandomState(seed + 100)
    model.set_weights([w if w.ndim > 1 else rng.normal(0, 0.05, w.shape).astype(w.dtype)
                       for w in model.get_weights()])
    return model


def onehot(tokens):
    return np.eye(4, dtype=np.int8)[tokens]


def predict(model, x):
    logits, logcounts = model(np.asarray(x, np.float32), training=False)
    return np.asarray(logits, np.float64), np.asarray(logcounts, np.float64)[:, 0]


def rel_l2(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


# ------------------------------------------------------------------ NumPy float64 brute-force DeepLIFT
def np_conv(x, w, b, d):
    k = w.shape[0]
    lout = x.shape[1] - d * (k - 1)
    out = np.zeros((x.shape[0], lout, w.shape[2]))
    for j in range(k):
        out += x[:, j * d:j * d + lout] @ w[j]
    return out + b


def np_conv_t(g, w, d, lin):
    dx = np.zeros((g.shape[0], lin, w.shape[1]))
    for j in range(w.shape[0]):
        dx[:, j * d:j * d + g.shape[1]] += g @ w[j].T
    return dx


def np_weights(model, prefix=""):
    get = lambda n: [np.asarray(v, np.float64) for v in model.get_layer(prefix + n).get_weights()]
    convs = [get("bpnet_1st_conv") + [1]] + [get("bpnet_{}conv".format(i)) + [2 ** i] for i in range(1, ND + 1)]
    return {"convs": convs, "prof": get("prof_out_precrop"), "dense": get("logcount_predictions")}


def np_forward(wts, x):
    zs, lens = [], []
    (w1, b1, _) = wts["convs"][0]
    z = np_conv(x, w1, b1, 1)
    zs.append(z)
    h = np.maximum(z, 0)
    for (w, b, d) in wts["convs"][1:]:
        lens.append(h.shape[1])
        z = np_conv(h, w, b, d)
        zs.append(z)
        c = np.maximum(z, 0)
        crop = (h.shape[1] - c.shape[1]) // 2
        h = c + h[:, crop:h.shape[1] - crop]
    wp, bp = wts["prof"]
    p = np_conv(h, wp, bp, 1)
    cs = p.shape[1] // 2 - OUTLEN // 2
    logits = p[:, cs:cs + OUTLEN, 0]
    wd, bd = wts["dense"]
    logcounts = (h.mean(1) @ wd + bd)[:, 0]
    return logits, logcounts, {"zs": zs, "lens": lens, "h_len": h.shape[1], "p_len": p.shape[1], "cs": cs,
                               "in_len": x.shape[1]}


def np_rescale(zx, zr):
    d = zx - zr
    small = np.abs(d) < EPS_RELU
    return np.where(small, (zx > 0).astype(float), (np.maximum(zx, 0) - np.maximum(zr, 0)) / np.where(small, 1, d))


def np_deeplift(wts, x, r, head):
    """Multipliers of the input rows of pairs (x[i], r[i]) by explicit backprop with the rescale rule."""
    lx, cx, cache_x = np_forward(wts, x)
    lr, cr, cache_r = np_forward(wts, r)
    ms = [np_rescale(zx, zr) for zx, zr in zip(cache_x["zs"], cache_r["zs"])]
    n = x.shape[0]
    if head == "counts":
        g_logits, g_counts = np.zeros_like(lx), np.ones((n, 1))
    else:
        mnx = lx - lx.mean(1, keepdims=True)
        mnr = lr - lr.mean(1, keepdims=True)
        sx = np.exp(mnx) / np.exp(mnx).sum(1, keepdims=True)
        sr = np.exp(mnr) / np.exp(mnr).sum(1, keepdims=True)
        w = np.where(np.abs(mnx - mnr) < EPS_MUL, 0.0, 0.5 * (sx + sr))
        g_logits, g_counts = w - w.mean(1, keepdims=True), np.zeros((n, 1))
    wp, _ = wts["prof"]
    gp = np.zeros((n, cache_x["p_len"], 1))
    gp[:, cache_x["cs"]:cache_x["cs"] + OUTLEN, 0] = g_logits
    gh = np_conv_t(gp, wp, 1, cache_x["h_len"])
    wd, _ = wts["dense"]
    gh += (g_counts @ wd.T)[:, None, :] / cache_x["h_len"]
    for i in range(len(wts["convs"]) - 1, 0, -1):
        w, _, d = wts["convs"][i]
        lin = cache_x["lens"][i - 1]
        gz = gh * ms[i]
        g_in = np_conv_t(gz, w, d, lin)
        crop = (lin - gz.shape[1]) // 2
        g_in[:, crop:lin - crop] += gh
        gh = g_in
    w1, _, _ = wts["convs"][0]
    return np_conv_t(gh * ms[0], w1, 1, cache_x["in_len"])
