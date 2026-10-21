"""PyTorch loss functions matching the TensorFlow 2.8 reference."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def multinomial_nll(true_counts: Tensor, logits: Tensor) -> Tensor:
    true_counts = true_counts.float()
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    counts_per_example = true_counts.sum(dim=-1)
    log_probability = (
        torch.lgamma(counts_per_example + 1.0)
        - torch.lgamma(true_counts + 1.0).sum(dim=-1)
        + (true_counts * log_probs).sum(dim=-1)
    )
    return -log_probability.sum() / true_counts.shape[0]


def chrombpnet_loss(
    profile_logits: Tensor,
    logcount_predictions: Tensor,
    true_profile_counts: Tensor,
    true_logcounts: Tensor,
    counts_loss_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    profile_loss = multinomial_nll(true_profile_counts, profile_logits)
    if logcount_predictions is None:
        # A profile-only bias model: the profile NLL is the whole loss, and
        # the counts term is reported as zero.
        return profile_loss, profile_loss, torch.zeros_like(profile_loss)
    counts_loss = F.mse_loss(logcount_predictions, true_logcounts.float())
    total = profile_loss + float(counts_loss_weight) * counts_loss
    return total, profile_loss, counts_loss
