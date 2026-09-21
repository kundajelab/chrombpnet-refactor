"""
Written by Alex Tseng

https://gist.github.com/amtseng/010dd522daaabc92b014f075a34a0a0b
"""

import numpy as np

from chrombpnet import optimize

#: ASCII byte -> one-hot row. Everything that is not ACGT/acgt stays all-zero,
#: which is what the docstring promises. Lowercase is mapped as well as
#: uppercase so callers that already hold raw FASTA bytes can index this table
#: directly without paying for an upper-casing pass.
BASE_TO_ONE_HOT = np.zeros((256, 4), dtype=np.int8)
for _index, _base in enumerate("ACGT"):
    BASE_TO_ONE_HOT[ord(_base), _index] = 1
    BASE_TO_ONE_HOT[ord(_base.lower()), _index] = 1
del _index, _base


def dna_to_one_hot(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.
    """
    if len(seqs) == 0:
        return np.empty((0, 0, 4), dtype=np.int8)
    seq_len = len(seqs[0])
    if not np.all(np.array([len(s) for s in seqs]) == seq_len):
        raise ValueError("All DNA sequences must have the same length")

    if not optimize.active().prefetch:
        return _dna_to_one_hot_unoptimized(seqs, seq_len)

    # One gather through BASE_TO_ONE_HOT, in place of the original's
    # isin + unique(return_inverse=True). That pair sorted every one of the
    # N*L bytes to recover indices it could only ever assign one way: after
    # the sanitizing step the byte set was exactly {A, C, G, T, other}, and
    # the "ACGT" sentinel guaranteed all four letters were present, so the
    # sorted uniques were always A->0, C->1, G->2, T->3, other->4 against
    # np.identity(5)[:, :-1], whose row 4 is zeros. That is this table.
    # Same array, same dtype, no sort: ~3.8x at 20,000 x 2,114.
    encoded = "".join(seqs).upper().encode("utf8")
    base_vals = np.frombuffer(encoded, dtype=np.uint8)
    return BASE_TO_ONE_HOT[base_vals].reshape((len(seqs), seq_len, 4))


def _dna_to_one_hot_unoptimized(seqs, seq_len):
    """The pre-lookup-table encoder, kept so `--optimize none` can run it.

    It recovers the base indices with `np.unique(..., return_inverse=True)`,
    i.e. a sort over every one of the N*L bytes. Retained verbatim rather
    than approximated: it is both the comparison baseline and the oracle the
    table is tested against.
    """
    seq_concat = "".join(seqs).upper() + "ACGT"
    one_hot_map = np.identity(5)[:, :-1].astype(np.int8)
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85
    _, base_inds = np.unique(base_vals, return_inverse=True)
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))


def one_hot_to_dna(one_hot):
    """
    Converts a one-hot encoding into a list of DNA ("ACGT") sequences, where the
    position of 1s is ordered alphabetically by "ACGT". `one_hot` must be an
    N x L x 4 array of one-hot encodings. Returns a lits of N "ACGT" strings,
    each of length L, in the same order as the input array. The returned
    sequences will only consist of letters "A", "C", "G", "T", or "N" (all
    upper-case). Any encodings that are all 0s will be translated to "N".
    """
    bases = np.array(["A", "C", "G", "T", "N"])
    # Create N x L array of all 5s
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]
