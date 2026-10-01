"""DeepLIFT-SHAP contribution scores for the PyTorch engine, via Tangermeme.

Moved verbatim from ``chrombpnet/evaluation/interpret/interpret.py``, which
keeps everything framework-neutral about contribution scoring: region loading,
the ACGT check, the output schema and the HDF5 writer.
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


class ProfileWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sequence):
        logits = self.model(sequence)[0]
        mean_normalized = logits - logits.mean(dim=1, keepdim=True)
        weights = torch.softmax(mean_normalized.detach(), dim=1)
        return (weights * mean_normalized).sum(dim=1, keepdim=True)


def attribute(model, seqs, profile_or_counts, *, device, batch_size,
              num_shuffles, seed):
    """Yield ``(head, hypothetical_scores_nlc)``, counts before profile."""
    # Tangermeme is intentionally lazy-loaded so prediction/preprocessing users
    # do not pay its import and numba compilation cost.
    from tangermeme.deep_lift_shap import deep_lift_shap

    sequence = torch.as_tensor(seqs, dtype=torch.float32).permute(0, 2, 1)
    targets = []
    if "counts" in profile_or_counts:
        targets.append(("counts", CountsWrapper(model)))
    if "profile" in profile_or_counts:
        targets.append(("profile", ProfileWrapper(model)))

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
