"""PyTorch attribution utilities retained under the historical module path."""

import numpy as np
import torch


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """Project multipliers onto every hypothetical nucleotide."""
    projected = np.zeros_like(bg_data[0], dtype=float)
    assert len(orig_inp[0].shape) == 2
    for channel in range(orig_inp[0].shape[-1]):
        hypothetical = np.zeros_like(orig_inp[0], dtype=float)
        hypothetical[:, channel] = 1.0
        difference = hypothetical[None, :, :] - bg_data[0]
        projected[:, :, channel] = np.sum(difference * mult[0], axis=-1)
    result = [np.mean(projected, axis=0)]
    if len(orig_inp) > 1:
        result.append(np.zeros_like(orig_inp[1]))
    return result


def get_weightedsum_meannormed_logits(profile_logits):
    """Return the BPNet profile attribution target for N x L logits."""
    centered = profile_logits - profile_logits.mean(dim=1, keepdim=True)
    weights = torch.softmax(centered.detach(), dim=1)
    return torch.sum(weights * centered, dim=1)
