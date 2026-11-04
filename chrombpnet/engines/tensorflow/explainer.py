"""DeepSHAP for ChromBPNet / BPNet Keras 3 models on the TensorFlow backend.

The JAX engine's explainer (``chrombpnet.engines.jax.attribution.explainer``), ported to TensorFlow: the same
DeepLIFT rules (``deeplift_tf``, op for op ``deeplift_jax``), the same joint batch of S sequences x K
references per step shared by the requested heads, the same padding of a short last chunk, the same
out-of-memory back-off and summation-to-delta check (``common``, shared with it). Where the JAX version jits
``jax.vjp`` of a ``stateless_call``, this one traces the model call under a persistent ``tf.GradientTape``
inside a ``tf.function``, one per chunk shape; the ReLU swap happens while tracing, as there.

Usage::

    explainer = DeepLiftShap(model, heads=("counts", "profile"))
    hyp = explainer.explain(onehot_seqs)          # {"counts": (N, L, 4), "profile": (N, L, 4)} float32
"""
import contextlib
import warnings

import keras
import numpy as np
import tensorflow as tf

from chrombpnet.engines.jax.attribution.common import (
    ADDITIVITY_THRESHOLD, HEADS, ExplainResult, _ACTIVATION_LAYERS, activation_name, check_model, iter_layers,
    report_additivity, timed_iter, update_additivity)  # noqa: F401  (timed_iter: re-exported)
from chrombpnet.engines.jax.attribution.dinuc_shuffle import NUM_SHUFFLES, make_references
from chrombpnet.engines.tensorflow import deeplift_tf

#: "auto" is full float32 on the CPU (always, on TensorFlow) and TensorFlow's default on a GPU (TF32 where the
#: GPU has it, as chrombpnet 1.x ran DeepSHAP); "highest" turns TF32 off for the steps; "default" leaves it.
PRECISIONS = ("auto", "highest", "default")
#: Sequences per step when none is given (the JAX explainer sizes it from free device memory).
DEFAULT_BATCH_SEQS = 4


@contextlib.contextmanager
def deeplift_rules(model, relu=deeplift_tf.dl_relu):
    """Temporarily give every relu Conv1D/Dense the DeepLIFT rule, and run every layer in float32.

    Entered while the step is traced, so a retrace (a new chunk shape) sees the rules too.
    """
    swapped, policies = [], []
    try:
        for layer in [model] + list(iter_layers(model)):
            if isinstance(layer, _ACTIVATION_LAYERS) and activation_name(layer) == "relu":
                swapped.append((layer, layer.activation))
                layer.activation = relu
            if layer.compute_dtype in ("bfloat16", "float16"):
                policies.append((layer, layer.dtype_policy))
                layer.dtype_policy = "float32"
        yield
    finally:
        for layer, act in reversed(swapped):
            layer.activation = act
        for layer, policy in reversed(policies):
            layer.dtype_policy = policy


def resolve_precision(precision):
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of {} for DeepSHAP, got {!r}".format(PRECISIONS, precision))
    if precision == "auto":
        return "default" if tf.config.list_logical_devices("GPU") else "highest"
    return precision


@contextlib.contextmanager
def precision_context(precision):
    """TF32 off inside the block for 'highest' (TensorFlow reads the setting when a kernel runs)."""
    if precision != "highest":
        yield
        return
    previous = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        yield
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(previous)


class DeepLiftShap:
    """DeepSHAP (DeepLIFT rescale rule averaged over dinucleotide-shuffled references) for BPNet-style models.

    The arguments are the JAX explainer's (``chrombpnet.engines.jax.attribution.explainer.DeepLiftShap``), with
    ``precision`` one of PRECISIONS.
    """

    def __init__(self, model, heads=HEADS, profile_weighting="chrombpnet", precision="auto", batch_seqs=None,
                 num_shuffles=NUM_SHUFFLES, additivity_threshold=ADDITIVITY_THRESHOLD, variant="paired"):
        heads = tuple(h for h in HEADS if h in tuple(heads))
        if not heads:
            raise ValueError("no heads requested")
        check_model(model, heads)
        if profile_weighting not in deeplift_tf.PROFILE_WEIGHTINGS:
            raise ValueError("profile_weighting must be one of {}, got {!r}".format(
                deeplift_tf.PROFILE_WEIGHTINGS, profile_weighting))
        if variant not in ("paired", "full"):
            raise ValueError("variant must be 'paired' or 'full'")
        self.model = model
        self.heads = heads
        self.profile_weighting = profile_weighting
        self.precision = resolve_precision(precision)
        self.batch_seqs = int(batch_seqs) if batch_seqs else None
        self.num_shuffles = int(num_shuffles)
        self.additivity_threshold = additivity_threshold
        self.variant = variant
        self.input_length = model.inputs[0].shape[1]
        self._steps = {}

    # ------------------------------------------------------------------ traced step
    def _step(self, n_seqs, n_refs, seq_len, return_mult):
        key = (n_seqs, n_refs, seq_len, return_mult)
        if key not in self._steps:
            signature = (tf.TensorSpec((n_seqs, seq_len, 4), tf.int8),
                         tf.TensorSpec((n_seqs, n_refs, seq_len, 4), tf.int8))
            self._steps[key] = tf.function(self._make_step(n_seqs, n_refs, seq_len, return_mult),
                                           input_signature=signature)
        return self._steps[key]

    def _make_step(self, n_seqs, n_refs, seq_len, return_mult):
        model, heads, mode = self.model, self.heads, self.profile_weighting
        full = self.variant == "full"
        relu = deeplift_tf.dl_relu_full if full else deeplift_tf.dl_relu
        n_pairs = n_seqs * n_refs

        def step(x, refs):
            xf = tf.cast(x, tf.float32)
            rf = tf.cast(refs, tf.float32)
            x_rows = tf.repeat(xf, n_refs, axis=0)
            r_rows = tf.reshape(rf, (n_pairs, seq_len, 4))
            joint = tf.concat([x_rows, r_rows], axis=0)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(joint)
                with deeplift_rules(model, relu):
                    outputs = model(joint, training=False)
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                logits = tf.cast(outputs[0], tf.float32)
                logcounts = (tf.cast(outputs[1], tf.float32) if len(outputs) > 1
                             else tf.zeros((2 * n_pairs, 1), tf.float32))
                (t_counts, t_profile), deltas = deeplift_tf.targets(logits, logcounts, mode=mode, full=full)
            out = {}
            for head in heads:
                target = t_counts if head == "counts" else t_profile
                m = tape.gradient(target, joint)[:n_pairs]
                m4 = tf.reshape(m, (n_seqs, n_refs, seq_len, 4))
                res = {
                    "hyp": deeplift_tf.hypothetical(m4, rf),
                    "sum": tf.reshape(deeplift_tf.summation_delta(m, x_rows, r_rows), (n_seqs, n_refs)),
                    "delta": tf.reshape(deltas[head], (n_seqs, n_refs)),
                }
                if return_mult:
                    res["mult"] = m4
                out[head] = res
            del tape
            return out

        return step

    # ------------------------------------------------------------------ driver
    def _reference_fn(self, references, seed):
        if references is None:
            return lambda x, start, stop: make_references(x, num_shuffles=self.num_shuffles, seed=seed)
        if callable(references):
            return references
        refs = np.asarray(references)
        return lambda x, start, stop: refs[start:stop]

    def iter_explain(self, seqs, references=None, seed=1234, return_multipliers=False, additivity=None):
        """Yield (start, stop, refs, {head: {"hyp", ["mult"]}}) chunk by chunk, in input order; the JAX
        explainer's ``iter_explain``."""
        seqs = np.asarray(seqs)
        if seqs.ndim != 3 or seqs.shape[2] != 4:
            raise ValueError("expected one-hot sequences of shape (N, L, 4), got {}".format(seqs.shape))
        if self.input_length is not None and seqs.shape[1] != self.input_length:
            raise ValueError("sequences have length {} but the model expects {}".format(
                seqs.shape[1], self.input_length))
        n = seqs.shape[0]
        stats = additivity if additivity is not None else {}
        for head in self.heads:
            stats[head] = {"max_rel_err": 0.0, "max_abs_err": 0.0, "n_pairs": 0, "n_over_threshold": 0,
                           "threshold": self.additivity_threshold}
        if n == 0:
            return
        get_refs = self._reference_fn(references, seed)
        chunk = min(self.batch_seqs or DEFAULT_BATCH_SEQS, n)

        def load(start):
            stop = min(start + chunk, n)
            x = np.ascontiguousarray(seqs[start:stop], dtype=np.int8)
            refs = np.ascontiguousarray(get_refs(x, start, stop), dtype=np.int8)
            if refs.ndim != 4 or refs.shape[0] != stop - start or refs.shape[2:] != x.shape[1:]:
                raise ValueError("references for sequences {}:{} have shape {}, expected ({}, K, {}, {})".format(
                    start, stop, refs.shape, stop - start, x.shape[1], x.shape[2]))
            return start, stop, x, refs

        start = 0
        nxt = load(0)
        while start < n:
            if nxt is None or nxt[0] != start or nxt[1] != min(start + chunk, n):
                nxt = load(start)
            _, stop, x, refs = nxt
            n_refs = refs.shape[1]
            pad = chunk - (stop - start)
            if pad:
                x_in = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
                r_in = np.concatenate([refs, np.repeat(refs[-1:], pad, axis=0)])
            else:
                x_in, r_in = x, refs
            try:
                with precision_context(self.precision):
                    out = self._step(chunk, n_refs, x.shape[1], return_multipliers)(x_in, r_in)
                    out = {head: {k: v.numpy() for k, v in res.items()} for head, res in out.items()}
                nxt = load(stop) if stop < n else None
            except tf.errors.ResourceExhaustedError:
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                warnings.warn("DeepSHAP ran out of device memory; retrying with {} sequences per step".format(chunk))
                nxt = None
                continue
            valid = stop - start
            results = {}
            for head in self.heads:
                res = out[head]
                update_additivity(stats[head], res["sum"][:valid], res["delta"][:valid])
                results[head] = {"hyp": np.asarray(res["hyp"][:valid])}
                if return_multipliers:
                    results[head]["mult"] = np.asarray(res["mult"][:valid])
            yield start, stop, refs, results
            start = stop
        report_additivity(stats)

    def run(self, seqs, references=None, seed=1234, return_multipliers=False, return_references=False):
        """Explain all sequences at once; returns an ExplainResult."""
        seqs = np.asarray(seqs)
        result = ExplainResult()
        parts = {head: [] for head in self.heads}
        mults = {head: [] for head in self.heads}
        refs_parts = []
        for _, _, refs, res in self.iter_explain(seqs, references, seed, return_multipliers, result.additivity):
            for head in self.heads:
                parts[head].append(res[head]["hyp"])
                if return_multipliers:
                    mults[head].append(res[head]["mult"])
            if return_references:
                refs_parts.append(refs)
        for head in self.heads:
            result.hyp[head] = np.concatenate(parts[head]) if parts[head] else \
                np.zeros((0,) + seqs.shape[1:], np.float32)
            if return_multipliers:
                result.mult[head] = np.concatenate(mults[head]) if mults[head] else \
                    np.zeros((0, self.num_shuffles) + seqs.shape[1:], np.float32)
        if return_references:
            result.refs = np.concatenate(refs_parts) if refs_parts else \
                np.zeros((0, self.num_shuffles) + seqs.shape[1:], np.int8)
        return result

    def explain(self, seqs, references=None, seed=1234):
        """Hypothetical contributions {head: (N, L, 4) float32}; multiply by the one-hot input to project."""
        return self.run(seqs, references, seed).hyp

    def multipliers(self, seqs, references=None, seed=1234):
        """Raw input-half multipliers {head: (N, K, L, 4) float32} (the TFDeepExplainer `mult`)."""
        return self.run(seqs, references, seed, return_multipliers=True).mult
