"""DeepSHAP contribution scores for the tensorflow engine (Keras 3).

``chrombpnet.engines.jax.attribution.attribute`` with the TensorFlow explainer: the same profile weighting
(the original ChromBPNet's midpoint weights by default, ``softmax_x`` with ``--contribs-profile-stopgrad``),
the same reference stream (one ``np.random.RandomState(seed)`` in region order, as the TensorFlow and JAX
engines draw it), and both heads from one pass.
"""

from __future__ import annotations

import numpy as np

from chrombpnet.engines.jax.attribution import HEADS, SequentialReferences, sequences_per_step


def attribute(model, seqs, profile_or_counts, *, batch_size, num_shuffles, seed,
              profile_stop_gradient=False):
    """Yield ``(head, hypothetical_scores)``, counts before profile."""
    from chrombpnet.engines.tensorflow.explainer import DeepLiftShap, timed_iter

    heads = tuple(head for head in HEADS if head in profile_or_counts)
    if not heads:
        return
    seqs = np.asarray(seqs)
    explainer = DeepLiftShap(
        model, heads=heads,
        profile_weighting="softmax_x" if profile_stop_gradient else "chrombpnet",
        batch_seqs=sequences_per_step(batch_size, num_shuffles),
        num_shuffles=num_shuffles)
    print("Seqs dimension : {}".format(seqs.shape))
    print("Generating {} shap scores".format(" and ".join(repr(h) for h in heads)))
    references = SequentialReferences(num_shuffles, seed)
    parts = {head: [] for head in heads}
    chunks = explainer.iter_explain(seqs, references=references)
    for _, _, _, results in timed_iter(chunks, len(seqs)):
        for head in heads:
            parts[head].append(results[head]["hyp"])
    for head in heads:
        if parts[head]:
            yield head, np.concatenate(parts[head])
        else:
            yield head, np.zeros(seqs.shape, dtype=np.float32)
