"""DeepSHAP contribution scores for the TensorFlow engine.

``explain`` is adapted from kundajelab/chrombpnet@09938fd,
chrombpnet/evaluation/interpret/interpret.py, ``interpret()``: the same two
``shap.explainers.deep.TFDeepExplainer`` targets (summed log-counts; the
softmax-weighted mean-normalized profile logits), the same reference
generator and ``combine_mult_and_diffref`` (``shap_utils``, verbatim), and the
same ``shap_values(..., progress_message=100)`` call. It returns the
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
"""

from __future__ import annotations

import functools
import multiprocessing
import os
import shutil
import tempfile

import numpy as np

#: Order in which heads are computed and yielded; the original's order.
HEADS = ("counts", "profile")


def explain(model, seqs, profile_or_counts, num_shuffles):
    """Hypothetical scores per head. Needs graph mode and a graph-mode model."""
    import shap
    import tensorflow as tf

    from chrombpnet.engines.tensorflow import shap_utils

    print("Seqs dimension : {}".format(seqs.shape))

    profile_model_input = model.input
    profile_input = seqs
    counts_model_input = model.input
    counts_input = seqs

    references = shap_utils.shuffle_several_times
    if num_shuffles != 20:  # shuffle_several_times's own default
        references = functools.partial(shap_utils.shuffle_several_times,
                                       numshuffles=num_shuffles)

    scores = {}
    if "counts" in profile_or_counts:
        profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
            (counts_model_input, tf.reduce_sum(model.outputs[1], axis=-1)),
            references,
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)

        print("Generating 'counts' shap scores")
        scores["counts"] = profile_model_counts_explainer.shap_values(
            counts_input, progress_message=100)

    if "profile" in profile_or_counts:
        weightedsum_meannormed_logits = shap_utils.get_weightedsum_meannormed_logits(model)
        profile_model_profile_explainer = shap.explainers.deep.TFDeepExplainer(
            (profile_model_input, weightedsum_meannormed_logits),
            references,
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)

        print("Generating 'profile' shap scores")
        scores["profile"] = profile_model_profile_explainer.shap_values(
            profile_input, progress_message=100)
    return scores


def _worker(model_path, workdir, profile_or_counts, num_shuffles, device):
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
    for head, values in explain(model, seqs, profile_or_counts, num_shuffles).items():
        np.save(os.path.join(workdir, f"{head}.npy"), values)


def attribute(model_path, seqs, profile_or_counts, *, device, num_shuffles):
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
            args=(model_path, workdir, list(profile_or_counts), int(num_shuffles), device))
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
