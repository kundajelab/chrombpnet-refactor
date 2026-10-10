"""DeepSHAP contribution scores for the JAX engine.

The DeepLIFT rules (``deeplift_jax``), the explainer (``explainer``) and the
dinucleotide shuffle (``dinuc_shuffle``) are kundajelab/chrombpnet#284's native
JAX DeepSHAP: the multipliers of the kundajelab ``shap`` fork's
``TFDeepExplainer`` that the original ChromBPNet used, computed with one jitted
forward per chunk for both heads, and projected on the device.

This module adds two things. The first is the choice of the profile target's
weights. By default the engine uses the PR's ``profile_weighting="chrombpnet"``,
the shap fork's midpoint weights ``0.5 * (softmax(mn_x) + softmax(mn_r))``,
which chrombpnet 1.x computes because the fork has its ``StopGradient`` handler
commented out (``deep_tf.py:595``); the PyTorch and TensorFlow engines compute
the same by default. ``profile_stop_gradient`` (``--contribs-profile-stopgrad``)
selects ``"softmax_x"`` instead: the target's ``stop_gradient`` honoured, so
each mean-normalized logit of the input is weighted by its own softmax.

The second is the reference stream. The PR seeds each sequence's
references from its own content; here they are drawn exactly as the
TensorFlow engine draws them: one ``np.random.RandomState(seed)``, consumed in
region order, ``num_shuffles`` single ``dinuc_shuffle`` calls per sequence --
the same stream, call for call, as ``shap_utils.shuffle_several_times`` with
``rng=RandomState(seed)``. The TensorFlow engine restarts that stream for each
head, so both heads see identical references, which is what lets one forward
pass serve both here.
"""

from __future__ import annotations

import numpy as np

#: Order in which heads are yielded; the TensorFlow and PyTorch engines' order.
HEADS = ("counts", "profile")


class SequentialReferences:
    """``references`` callable for ``DeepLiftShap.iter_explain``.

    ``iter_explain`` asks for each chunk's references once, prefetches the
    next chunk's while the device works, and after an out-of-memory error
    asks again for the current chunk with fewer sequences. Each sequence's
    references are therefore drawn once, in order, and kept until no request
    can reach back to them: a request never starts before the start of the
    request before it.
    """

    def __init__(self, num_shuffles, seed):
        self.num_shuffles = int(num_shuffles)
        self.rng = np.random.RandomState(seed)
        self.cache = {}
        self.next_index = 0
        self.previous_start = 0

    def __call__(self, x, start, stop):
        from chrombpnet.engines.jax.attribution.dinuc_shuffle import dinuc_shuffle

        low = min(start, self.previous_start)
        for index in [i for i in self.cache if i < low]:
            del self.cache[index]
        self.previous_start = start
        if start > self.next_index:
            raise RuntimeError(
                "references requested for sequences {}:{} before sequence {}".format(
                    start, stop, self.next_index))
        while self.next_index < stop:
            seq = x[self.next_index - start]
            self.cache[self.next_index] = np.stack(
                [dinuc_shuffle(seq, rng=self.rng) for _ in range(self.num_shuffles)]
            ).astype(np.int8)
            self.next_index += 1
        missing = [i for i in range(start, stop) if i not in self.cache]
        if missing:
            raise RuntimeError(
                "references for sequences {} were already discarded".format(missing[:4]))
        return np.stack([self.cache[i] for i in range(start, stop)])


def sequences_per_step(batch_size, num_shuffles):
    """``--interpret-batch-size`` counts (sequence, reference) pairs, as for
    the PyTorch engine; a step here holds whole sequences with all their
    references, so this is how many sequences fit in that many pairs."""
    return max(1, int(batch_size) // max(1, int(num_shuffles)))


def attribute(model, seqs, profile_or_counts, *, batch_size, num_shuffles, seed,
              profile_stop_gradient=False):
    """Yield ``(head, hypothetical_scores)``, counts before profile.

    Both heads come from the same chunked pass, so both are computed before
    the first is yielded.
    """
    from chrombpnet.engines.jax.attribution.explainer import DeepLiftShap, timed_iter

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
