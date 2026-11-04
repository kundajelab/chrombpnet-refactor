"""DeepLIFT / DeepSHAP rules in TensorFlow: the JAX engine's ``deeplift_jax``, op for op.

The same batch layout (a joint batch ``[x_1 .. x_P, r_1 .. r_P]``, row p paired with row P + p), the same
ReLU rescale rule with the same fallback, the same profile weightings and targets, and the same hypothetical
projection; see ``chrombpnet.engines.jax.attribution.deeplift_jax`` for what each reproduces in chrombpnet
1.x. ``jax.custom_vjp`` becomes ``tf.custom_gradient`` and ``jax.lax.stop_gradient`` ``tf.stop_gradient``.
"""
import tensorflow as tf

# deep_tf.py:494 (nonlinearity_1d_handler): |delta_in| < 1e-6 (strict) -> plain gradient
EPS_RELU = 1e-6
# deep_tf.py:525-526 (nonlinearity_2d_handler): |delta_in| < 1e-7 (strict) -> zero
EPS_MUL = 1e-7

PROFILE_WEIGHTINGS = ("chrombpnet", "chrombpnet_tf", "softmax_x", "tangermeme")


def _half(a):
    rows = a.shape[0]
    return rows // 2 if rows is not None else tf.shape(a)[0] // 2


def split_pairs(a):
    """Split a joint batch into its input half and its reference half."""
    p = _half(a)
    return a[:p], a[p:]


def rescale_multipliers(zx, zr):
    """ReLU rescale-rule multipliers for the input half and the reference half (deeplift_jax's)."""
    delta = zx - zr
    small = tf.abs(delta) < EPS_RELU
    secant = (tf.maximum(zx, 0.0) - tf.maximum(zr, 0.0)) / tf.where(small, tf.ones_like(delta), delta)
    mx = tf.where(small, tf.cast(zx > 0, zx.dtype), secant)
    mr = tf.where(small, tf.cast(zr > 0, zr.dtype), secant)
    return mx, mr


@tf.custom_gradient
def dl_relu(z):
    """ReLU whose gradient is the DeepLIFT rescale multiplier on the input half and zero on the reference half
    (exact for chrombpnet targets, which read reference rows only through stop_gradient)."""
    zx, zr = split_pairs(z)
    mx, _ = rescale_multipliers(zx, zr)

    def grad(g):
        gx, gr = split_pairs(g)
        return tf.concat([gx * mx, tf.zeros_like(gr)], axis=0)

    return tf.maximum(z, 0.0), grad


@tf.custom_gradient
def dl_relu_full(z):
    """Both-halves variant, as TFDeepExplainer computes it (reference half gets its own fallback). For tests."""
    zx, zr = split_pairs(z)
    mx, mr = rescale_multipliers(zx, zr)
    m = tf.concat([mx, mr], axis=0)

    def grad(g):
        return g * m

    return tf.maximum(z, 0.0), grad


def mean_normalize(logits):
    """Mean-normalised profile logits, (rows, outlen)."""
    logits = tf.reshape(logits, (tf.shape(logits)[0], -1))
    return logits - tf.reduce_mean(logits, axis=1, keepdims=True)


def profile_weights(mn_x, mn_r, mode="chrombpnet"):
    """Per-logit weights w (pairs, outlen); deeplift_jax.profile_weights."""
    s_x = tf.nn.softmax(mn_x, axis=-1)
    s_r = tf.nn.softmax(mn_r, axis=-1)
    delta = mn_x - mn_r
    if mode == "chrombpnet":
        return tf.where(tf.abs(delta) < EPS_MUL, tf.zeros_like(s_x), 0.5 * (s_x + s_r))
    if mode == "chrombpnet_tf":
        small = tf.abs(delta) < EPS_MUL
        out1 = 0.5 * (s_x * mn_x - s_x * mn_r + s_r * mn_x - s_r * mn_r)
        return tf.where(small, tf.zeros_like(s_x), out1 / tf.where(small, tf.ones_like(delta), delta))
    if mode == "softmax_x":
        return s_x
    if mode == "tangermeme":
        small = tf.abs(delta) < EPS_RELU
        secant = (mn_x * s_x - mn_r * s_r) / tf.where(small, tf.ones_like(delta), delta)
        return tf.where(small, s_x, secant)
    raise ValueError("profile_weighting must be one of {}, got {!r}".format(PROFILE_WEIGHTINGS, mode))


def targets(logits, logcounts, mode="chrombpnet", full=False):
    """((T_counts, T_profile), {"counts": c(x) - c(r), "profile": sum(w * (mn_x - mn_r))}); deeplift_jax.targets."""
    counts = tf.reduce_sum(tf.reshape(logcounts, (tf.shape(logcounts)[0], -1)), axis=-1)
    mn = mean_normalize(logits)
    mn_x, mn_r = split_pairs(tf.stop_gradient(mn))
    w = tf.stop_gradient(profile_weights(mn_x, mn_r, mode))
    mn_xg, mn_rg = split_pairs(mn)
    c_x, c_r = split_pairs(counts)
    if full:
        t_counts = tf.reduce_sum(counts)
        t_profile = tf.reduce_sum(w * mn_xg) + tf.reduce_sum(w * mn_rg)
    else:
        t_counts = tf.reduce_sum(c_x)
        t_profile = tf.reduce_sum(w * mn_xg)
    aux = {
        "counts": tf.stop_gradient(c_x - c_r),
        "profile": tf.reduce_sum(w * (mn_x - mn_r), axis=-1),
    }
    return (t_counts, t_profile), aux


def hypothetical(mult, refs):
    """Hypothetical contributions, (S, K, L, 4) multipliers and references -> (S, L, 4)."""
    return tf.reduce_mean(mult - tf.reduce_sum(refs * mult, axis=-1, keepdims=True), axis=1)


def summation_delta(mult, x, refs):
    """sum_{l,c} m * (x - r) per pair: (P, L, 4) arrays -> (P,)."""
    return tf.reduce_sum(mult * (x - refs), axis=(1, 2))
