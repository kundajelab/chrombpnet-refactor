from chrombpnet.regions import window_bounds
from chrombpnet.sequences import fetch_windows
import numpy as np
from chrombpnet.training.utils.data_utils import one_hot


VALID_BASES = frozenset("ACGT")


def get_seq(peaks_df, genome, width):
    """One-hot windows that are both in bounds and ACGT-only.

    Tangermeme's dinucleotide shuffler requires exactly one A/C/G/T at every
    position. Unknown/IUPAC bases become all-zero one-hot columns, so those
    regions are excluded along with truncated edge regions.
    """
    seqs, in_bounds = fetch_windows(peaks_df, genome, width)
    # An ACGT position contributes exactly 1 to its row's sum and anything
    # else contributes 0, so this is the vectorized form of the old
    # set(sequence).issubset(VALID_BASES) test.
    unambiguous = seqs.sum(axis=(1, 2)) == width

    peaks_used = in_bounds.copy()
    peaks_used[np.flatnonzero(in_bounds)] = unambiguous
    if not peaks_used.any():
        raise ValueError(
            "No full-length ACGT-only sequences are available for contribution scoring"
        )
    return seqs[unambiguous], peaks_used


