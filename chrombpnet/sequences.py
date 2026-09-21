"""Reading one-hot sequence windows out of an indexed FASTA.

Three call sites fetch ``width``-wide windows centred on a region's summit and
one-hot encode them, differing only in which windows they *keep*:

* ``evaluation/make_bigwigs/bigwig_helper.get_seq`` keeps in-bounds windows,
* ``evaluation/interpret/input_utils.get_seq`` additionally requires ACGT-only,
* ``training/utils/data_utils.get_seq`` keeps everything and lets the one-hot
  encoder raise on the resulting length mismatch.

The first two share :func:`fetch_windows` below; the third deliberately does
not, because "no mask at all" cannot be recovered from a mask when every row
happens to be truncated to the same length.

Each of those loops ran one ``pyfaidx`` slice per region, which on a real hg38
on network storage costs about 4.7 ms a region -- 94 s for 20,000 regions,
several times the GPU forward it feeds. :func:`fetch_windows` reads the same
bytes through a memory map and one gather per chromosome: 1.2 s for the same
20,000 regions, byte-identical output and an identical keep-mask.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from chrombpnet.regions import window_bounds
from chrombpnet.training.utils.one_hot import BASE_TO_ONE_HOT, dna_to_one_hot


def _fast_path_blocker(genome, peaks_df):
    """Why the memory map cannot serve this request, or None if it can.

    Every condition here is one where ``pyfaidx`` does something a raw memory
    map does not, so we hand the whole request back to the per-region loop
    rather than reimplement it. In particular a missing chromosome must reach
    pyfaidx, so the caller sees pyfaidx's own KeyError rather than ours.
    """
    faidx = getattr(genome, "faidx", None)
    if faidx is None:
        return "the genome handle exposes no faidx index"
    if getattr(faidx, "_bgzf", False):
        return "the FASTA is BGZF-compressed"
    if getattr(faidx, "default_seq", None) is not None:
        return "faidx.default_seq pads out-of-bounds reads"
    filename = getattr(faidx, "filename", None)
    if filename is None or not os.path.isfile(filename):
        return "the FASTA is not a local file"
    missing = set(peaks_df["chr"].unique()) - set(faidx.index)
    if missing:
        return f"{len(missing)} chromosome(s) are absent from the index"
    return None


def _window_edges(peaks_df, width):
    """Half-open ``[lo, hi)`` bounds per region, vectorized over the frame.

    Matches :func:`chrombpnet.regions.window_bounds` exactly, including its
    asymmetry for odd ``width`` (it spans ``2 * (width // 2)``, not ``width``).
    """
    centre = peaks_df["start"].to_numpy(np.int64) + peaks_df["summit"].to_numpy(np.int64)
    return centre - width // 2, centre + width // 2


def in_bounds_mask(peaks_df, genome, width):
    """Which regions ``pyfaidx`` would return a full ``width``-long slice for.

    Reproducing this exactly is the whole correctness problem, so the rules are
    spelled out rather than inferred. With ``strict_bounds=False`` (pyfaidx's
    default, and what every call site uses):

    * a negative start is rewritten as ``len(chromosome) + start``, i.e. it
      *wraps to the far end* of the chromosome, and the resulting backwards
      interval yields the empty string;
    * an end past the chromosome is clamped, returning fewer than ``width``
      bases;
    * otherwise exactly ``hi - lo`` bases come back.

    So a window survives iff it is non-negative, within the chromosome, and
    ``width`` wide -- and that last clause is load-bearing rather than
    tautological, because ``window_bounds`` spans ``width - 1`` for odd
    ``width`` and every region is therefore dropped.
    """
    lo, hi = _window_edges(peaks_df, width)
    lengths = peaks_df["chr"].map(
        {name: record.rlen for name, record in genome.faidx.index.items()}
    ).to_numpy(np.int64)
    return (lo >= 0) & (hi <= lengths) & ((hi - lo) == width)


def _fetch_windows_mapped(peaks_df, genome, width):
    """The memory-mapped fast path. Assumes :func:`_fast_path_blocker` passed."""
    faidx = genome.faidx
    keep = in_bounds_mask(peaks_df, genome, width)
    kept = int(keep.sum())
    one_hot = np.empty((kept, width, 4), dtype=np.int8)
    if kept == 0:
        return one_hot, keep

    lo, _ = _window_edges(peaks_df, width)
    lo = lo[keep]
    chroms = peaks_df["chr"].to_numpy()[keep]
    mapped = np.memmap(faidx.filename, dtype=np.uint8, mode="r")
    offsets = np.arange(width, dtype=np.int64)
    try:
        for chrom in pd.unique(chroms):
            rows = np.flatnonzero(chroms == chrom)
            record = faidx.index[chrom]
            # lenc = bases per line, lenb = bytes per line. Their difference is
            # the line terminator, so this arithmetic handles CRLF for free --
            # it is the same mapping pyfaidx itself uses.
            position = lo[rows][:, None] + offsets[None, :]
            byte = record.offset + (position // record.lenc) * record.lenb + (position % record.lenc)
            one_hot[rows] = BASE_TO_ONE_HOT[mapped[byte]]
    finally:
        del mapped
    return one_hot, keep


def _fetch_windows_pyfaidx(peaks_df, genome, width):
    """The original per-region loop, kept as the reference implementation."""
    sequences = []
    keep = []
    for _, row in peaks_df.iterrows():
        start, end = window_bounds(row, width)
        sequence = str(genome[row["chr"]][start:end])
        keep.append(len(sequence) == width)
        if keep[-1]:
            sequences.append(sequence)
    keep = np.array(keep, dtype=bool)
    if not sequences:
        return np.empty((0, width, 4), dtype=np.int8), keep
    return dna_to_one_hot(sequences), keep


def fetch_windows(peaks_df, genome, width):
    """One-hot windows for every in-bounds region, plus the keep-mask.

    Returns ``(one_hot, kept)`` where ``one_hot`` has shape
    ``(kept.sum(), width, 4)`` in BED order and ``kept`` has one entry per row
    of ``peaks_df``. Callers layer their own additional filtering on top.

    Falls back to the per-region ``pyfaidx`` loop whenever the memory map
    cannot faithfully stand in for it; the two are asserted byte-identical by
    ``tests/test_sequences.py``.
    """
    if _fast_path_blocker(genome, peaks_df) is None:
        return _fetch_windows_mapped(peaks_df, genome, width)
    return _fetch_windows_pyfaidx(peaks_df, genome, width)
