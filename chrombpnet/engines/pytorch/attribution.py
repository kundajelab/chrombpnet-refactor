"""DeepLIFT-SHAP contribution scores for the PyTorch engine, via Tangermeme.

Moved from ``chrombpnet/evaluation/interpret/interpret.py``, which keeps
everything framework-neutral about contribution scoring: region loading, the
ACGT check, the output schema and the HDF5 writer.

The profile target's weights follow chrombpnet 1.x by default (see
``ProfileWrapper``); ``profile_stop_gradient`` gives the softmax of the input
alone, which is what this engine computed before the default changed.
"""

from __future__ import annotations

import torch
from torch import nn


class CountsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sequence):
        return self.model(sequence)[1]


#: Below this |mn_x - mn_r| the shap fork's two-input rule gives a logit no
#: weight (deep_tf.py:525-526); deeplift_jax.EPS_MUL is the same constant.
MIDPOINT_EPS = 1e-7


class ProfileWrapper(nn.Module):
    """The profile target ``sum(w * mn)`` over mean-normalized logits ``mn``.

    With ``stop_gradient`` the weights are ``softmax(mn)`` of each row,
    detached: the target's ``stop_gradient`` honoured. Without it (the
    default) they reproduce the kundajelab shap fork, whose ``StopGradient``
    handler is commented out (``deep_tf.py:595``): its two-input rule for
    ``softmax * mn`` weights each logit of a pair by the midpoint
    ``0.5 * (softmax(mn_x) + softmax(mn_r))``, zero where the logits barely
    differ, and the softmax's own change dies at the stop-gradient.

    The midpoint pairs rows: Tangermeme's ``deep_lift_shap`` runs each batch as
    ``[inputs; references]``, row ``i`` of the first half paired with row ``i``
    of the second. Both halves get the pair's weights, so the target's
    input-reference difference is exactly what the attributions sum to.
    """

    def __init__(self, model, stop_gradient=False):
        super().__init__()
        self.model = model
        self.stop_gradient = stop_gradient

    def forward(self, sequence):
        logits = self.model(sequence)[0]
        mean_normalized = logits - logits.mean(dim=1, keepdim=True)
        weights = torch.softmax(mean_normalized.detach(), dim=1)
        if not self.stop_gradient:
            mn_x, mn_r = mean_normalized.detach().chunk(2)
            s_x, s_r = weights.chunk(2)
            midpoint = torch.where((mn_x - mn_r).abs() < MIDPOINT_EPS,
                                   torch.zeros_like(s_x), 0.5 * (s_x + s_r))
            weights = torch.cat([midpoint, midpoint])
        return (weights * mean_normalized).sum(dim=1, keepdim=True)


def attribute(model, seqs, profile_or_counts, *, device, batch_size,
              num_shuffles, seed, profile_stop_gradient=False):
    """Yield ``(head, hypothetical_scores_nlc)``, counts before profile."""
    # Tangermeme is intentionally lazy-loaded so prediction/preprocessing users
    # do not pay its import and numba compilation cost.
    from tangermeme.deep_lift_shap import deep_lift_shap

    sequence = torch.as_tensor(seqs, dtype=torch.float32).permute(0, 2, 1)
    targets = []
    if "counts" in profile_or_counts:
        targets.append(("counts", CountsWrapper(model)))
    if "profile" in profile_or_counts:
        targets.append(("profile", ProfileWrapper(model, profile_stop_gradient)))

    for name, wrapper in targets:
        wrapper.to(device).eval()
        scores = deep_lift_shap(
            wrapper,
            sequence,
            hypothetical=True,
            n_shuffles=num_shuffles,
            random_state=seed,
            batch_size=batch_size,
            device=device,
            verbose=True,
        )
        yield name, scores.detach().cpu().permute(0, 2, 1).numpy()
