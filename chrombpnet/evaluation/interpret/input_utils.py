from chrombpnet.regions import window_bounds
import numpy as np
from chrombpnet.training.utils.data_utils import one_hot


VALID_BASES = frozenset("ACGT")


def get_seq(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    """
    vals = []
    peaks_used = []
    for i, r in peaks_df.iterrows():
        start, end = window_bounds(r, width)
        sequence = str(genome[r['chr']][start:end]).upper()
        # Tangermeme's dinucleotide shuffler requires exactly one A/C/G/T at
        # every position. Unknown/IUPAC bases become all-zero one-hot columns,
        # so exclude those regions together with truncated edge regions.
        if len(sequence) == width and set(sequence).issubset(VALID_BASES):
            vals.append(sequence)
            peaks_used.append(True)
        else:
            peaks_used.append(False)

    if not vals:
        raise ValueError(
            "No full-length ACGT-only sequences are available for contribution scoring"
        )
    return one_hot.dna_to_one_hot(vals), np.array(peaks_used)


