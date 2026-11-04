# JAX engine. The backend-neutral parts of explainer.py (kundajelab/chrombpnet#284,
# NNFC-GMD/chrombpnet@14ca773, chrombpnet/evaluation/interpret/explainer.py), moved here unchanged
# so that the tensorflow engine's (Keras 3) explainer shares them: which layers have a DeepLIFT rule,
# the result container, the summation-to-delta bookkeeping and the progress printer.
"""DeepSHAP helpers shared by the JAX and TensorFlow (Keras 3) explainers. NumPy and Keras only."""
import time
import warnings

import keras
import numpy as np

from chrombpnet.engines.jax.layers import LogSumExp

HEADS = ("counts", "profile")
ADDITIVITY_THRESHOLD = 1e-3

_LINEAR_LAYERS = (
    keras.layers.InputLayer,
    keras.layers.Cropping1D,
    keras.layers.Add,
    keras.layers.Flatten,
    keras.layers.GlobalAveragePooling1D,
    keras.layers.Concatenate,
)
_ACTIVATION_LAYERS = (keras.layers.Conv1D, keras.layers.Dense)


def iter_layers(model):
    """All layers of a model, recursing into nested models (e.g. the two submodels of a full chrombpnet.h5)."""
    for layer in model.layers:
        yield layer
        if isinstance(layer, keras.Model):
            yield from iter_layers(layer)


def activation_name(layer):
    act = getattr(layer, "activation", None)
    if act is None:
        return "linear"
    try:
        name = keras.activations.serialize(act)
    except Exception:
        name = getattr(act, "__name__", None)
    return name if isinstance(name, str) else repr(name)


def check_model(model, heads):
    """Raise NotImplementedError unless every layer has a DeepLIFT rule here (ReLU rescale, or linear)."""
    if len(model.inputs) != 1:
        raise NotImplementedError("DeepSHAP supports single-input models only ({} has {} inputs)".format(
            model.name, len(model.inputs)))
    n_out = len(model.outputs)
    for head in heads:
        if head not in HEADS:
            raise ValueError("unknown head {!r}; expected one of {}".format(head, HEADS))
        if head == "counts" and n_out < 2:
            raise ValueError("counts head requested but the model has a single output")
    has_logsumexp = False
    for layer in iter_layers(model):
        if isinstance(layer, keras.Model):
            continue
        if isinstance(layer, LogSumExp):
            has_logsumexp = True
        elif isinstance(layer, _ACTIVATION_LAYERS):
            act = activation_name(layer)
            if act not in ("relu", "linear"):
                raise NotImplementedError(
                    "layer {!r} ({}) uses activation {!r}; DeepSHAP here only has rules for relu and linear".format(
                        layer.name, type(layer).__name__, act))
        elif not isinstance(layer, _LINEAR_LAYERS):
            raise NotImplementedError(
                "layer {!r} ({}) has no DeepLIFT rule; supported layers: InputLayer, Conv1D and Dense (relu or "
                "linear), Cropping1D, Add, Concatenate, Flatten, GlobalAveragePooling1D, nested models".format(
                    layer.name, type(layer).__name__))
    if has_logsumexp and "counts" in heads:
        raise NotImplementedError(
            "counts-head DeepSHAP of a full chrombpnet model (logsumexp of the bias and no-bias counts) is not "
            "supported: interpret the bias-corrected *_nobias.keras model instead (as the chrombpnet pipeline does). "
            "The profile head of the full model is supported.")


class ExplainResult:
    """hyp[head]: (N, L, 4) float32; mult[head]: (N, K, L, 4) float32 (if requested); refs: (N, K, L, 4) int8
    (if requested); additivity[head]: summary of the per-pair summation-to-delta check."""

    def __init__(self):
        self.hyp, self.mult, self.additivity = {}, {}, {}
        self.refs = None


def update_additivity(stat, total, delta):
    total = np.asarray(total, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    abs_err = np.abs(total - delta)
    rel_err = abs_err / np.maximum(np.abs(delta), 1.0)
    stat["n_pairs"] += int(rel_err.size)
    if rel_err.size:
        stat["max_rel_err"] = max(stat["max_rel_err"], float(rel_err.max()))
        stat["max_abs_err"] = max(stat["max_abs_err"], float(abs_err.max()))
        stat["n_over_threshold"] += int((rel_err > stat["threshold"]).sum())


def report_additivity(stats):
    for head, stat in stats.items():
        if stat["n_over_threshold"]:
            warnings.warn(
                "DeepSHAP {} head: {} of {} (input, reference) pairs violate summation-to-delta by more than "
                "{:g} (max relative error {:.3g}); expected only with precision='default' on GPUs (TF32) or "
                "for layers without a DeepLIFT rule".format(head, stat["n_over_threshold"], stat["n_pairs"],
                                                             stat["threshold"], stat["max_rel_err"]))


def timed_iter(iterator, total, every=60.0, label="DeepSHAP"):
    """Pass chunks through, printing progress about every `every` seconds."""
    t0 = last = time.time()
    for item in iterator:
        yield item
        now = time.time()
        done = item[1]
        if now - last >= every or done == total:
            rate = done / max(now - t0, 1e-9)
            print("{}: {}/{} sequences ({:.1f} seq/s)".format(label, done, total, rate), flush=True)
            last = now
