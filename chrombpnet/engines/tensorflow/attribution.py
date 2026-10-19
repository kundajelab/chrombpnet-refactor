"""DeepSHAP contribution scores for the TensorFlow engine.

``explain`` is adapted from kundajelab/chrombpnet@09938fd,
chrombpnet/evaluation/interpret/interpret.py, ``interpret()``: the same two
targets of the kundajelab shap fork's ``TFDeepExplainer`` (copied into
``deep_tf``): summed log-counts and the softmax-weighted mean-normalized
profile logits; the same reference generator and ``combine_mult_and_diffref``
(``shap_utils``, verbatim), and the same ``shap_values(...,
progress_message=100)`` call. It returns the
hypothetical scores instead of writing them, because the output schema and
writer are shared by every engine (``chrombpnet/evaluation/interpret``).

It runs in a spawned child process. The kundajelab shap fork's DeepExplainer
works on TF1 graphs, and ``tf.compat.v1.disable_eager_execution()`` is
process-global and only valid before any graph exists. The original got away
with calling it at import time because ``interpret`` ran in its own process;
here, in a pipeline, prediction and footprinting have already used TensorFlow
eagerly in this process by the time attribution starts.

``--interpret-batch-size`` has no effect: the fork explains one example at a
time, against its own references.

One deliberate departure from the original: the dinucleotide-shuffled
references are drawn from ``np.random.RandomState(seed)``, fresh for each head,
where the original drew them from an unseeded stream. Contribution scores are
therefore reproducible run to run, and ``--seed`` means what it says.

The profile target weights each mean-normalized logit by its softmax under
``tf.stop_gradient``, meaning the softmax to act as fixed weights. The shap
fork has its ``StopGradient`` handler commented out (its ``deep_tf.py:595``,
kept so in ``deep_tf``), so
by default -- as in the original -- the weights count as a second varying
input, and the fork's two-input rule weights the logits by the midpoint
``0.5 * (softmax(mn_x) + softmax(mn_r))``. With ``stop_gradient``
(``--contribs-profile-stopgrad``) the handler is registered for the profile
explainer (``honour_stop_gradient``) and the weights are ``softmax(mn_x)``.
The counts head is the same either way.
"""

from __future__ import annotations

import contextlib
import functools
import multiprocessing
import os
import shutil
import tempfile

import numpy as np

#: Order in which heads are computed and yielded; the original's order.
HEADS = ("counts", "profile")


@contextlib.contextmanager
def honour_stop_gradient():
    """Have the shap fork stop attributions at ``tf.stop_gradient``.

    Registers ``break_dependence`` for ``StopGradient``, the line the fork
    has commented out (its ``deep_tf.py:595``), and restores the previous entry
    on exit. It must cover both the explainer's construction, which reads
    the dependence breakers, and ``shap_values``, which builds the gradient
    graph from ``op_handlers`` on its first call.
    """
    from chrombpnet.engines.tensorflow import deep_tf

    missing = object()
    previous = deep_tf.op_handlers.get("StopGradient", missing)
    deep_tf.op_handlers["StopGradient"] = deep_tf.break_dependence
    try:
        yield
    finally:
        if previous is missing:
            del deep_tf.op_handlers["StopGradient"]
        else:
            deep_tf.op_handlers["StopGradient"] = previous


def _references(num_shuffles, seed):
    """The reference generator for one head, on its own seeded stream."""
    from chrombpnet.engines.tensorflow import shap_utils

    return functools.partial(shap_utils.shuffle_several_times,
                             numshuffles=num_shuffles,
                             rng=np.random.RandomState(seed))


def explain(model, seqs, profile_or_counts, num_shuffles, seed, stop_gradient=False):
    """Hypothetical scores per head. Needs graph mode and a graph-mode model.

    ``stop_gradient`` honours the profile target's ``tf.stop_gradient``; by
    default the fork's midpoint weighting applies, as in the original.
    """
    import tensorflow as tf

    from chrombpnet.engines.tensorflow import deep_tf, shap_utils

    print("Seqs dimension : {}".format(seqs.shape))

    profile_model_input = model.input
    profile_input = seqs
    counts_model_input = model.input
    counts_input = seqs

    scores = {}
    if "counts" in profile_or_counts:
        profile_model_counts_explainer = deep_tf.TFDeepExplainer(
            (counts_model_input, tf.reduce_sum(model.outputs[1], axis=-1)),
            _references(num_shuffles, seed),
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)

        print("Generating 'counts' shap scores")
        scores["counts"] = profile_model_counts_explainer.shap_values(
            counts_input, progress_message=100)

    if "profile" in profile_or_counts:
        weightedsum_meannormed_logits = shap_utils.get_weightedsum_meannormed_logits(model)
        handler = honour_stop_gradient() if stop_gradient else contextlib.nullcontext()
        with handler:
            profile_model_profile_explainer = deep_tf.TFDeepExplainer(
                (profile_model_input, weightedsum_meannormed_logits),
                _references(num_shuffles, seed),
                combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)

            print("Generating 'profile' shap scores")
            scores["profile"] = profile_model_profile_explainer.shap_values(
                profile_input, progress_message=100)
    return scores


def _worker(model_path, workdir, profile_or_counts, num_shuffles, seed, device,
            stop_gradient=False):
    """Child-process entry point: graph mode, load, explain, write ``.npy``."""
    from chrombpnet.engines.tensorflow.predictor import configure_device, load_model

    configure_device(device)
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    # evaluation/interpret/input_utils.load_model_wrapper
    model = load_model(model_path, compile=False)
    print("got the model")
    model.summary()

    seqs = np.load(os.path.join(workdir, "seqs.npy"))
    for head, values in explain(model, seqs, profile_or_counts, num_shuffles, seed,
                                stop_gradient).items():
        np.save(os.path.join(workdir, f"{head}.npy"), values)


def attribute(model_path, seqs, profile_or_counts, *, device, num_shuffles, seed,
              profile_stop_gradient=False):
    """Yield ``(head, hypothetical_scores)``, counts before profile.

    The scores travel back through ``.npy`` files in a temporary directory
    (``$TMPDIR``) rather than a pipe: at the pipelines' 30,000 regions each
    head is about 2 GB of float64.
    """
    workdir = tempfile.mkdtemp(prefix="chrombpnet-deepshap-")
    try:
        np.save(os.path.join(workdir, "seqs.npy"), np.asarray(seqs))
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=_worker,
            args=(model_path, workdir, list(profile_or_counts), int(num_shuffles),
                  int(seed), device, bool(profile_stop_gradient)))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"TensorFlow DeepSHAP worker failed with exit code {process.exitcode}")
        for head in HEADS:
            if head in profile_or_counts:
                yield head, np.load(os.path.join(workdir, f"{head}.npy"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
