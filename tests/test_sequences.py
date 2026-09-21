"""The memory-mapped window reader must match the pyfaidx loop it replaces.

The golden-artifact harness cannot carry this on its own: its fixture
chromosomes need not produce a single out-of-bounds window, and the mask is
exactly where a silent one-region discrepancy would shift every downstream
array. So the edge cases are injected here by hand and both paths are run
against each other on the same input.
"""

import numpy as np
import pandas as pd
import pyfaidx
import pytest

from chrombpnet.regions import NARROWPEAK_SCHEMA
from chrombpnet.sequences import (
    _fast_path_blocker,
    _fetch_windows_mapped,
    _fetch_windows_pyfaidx,
    fetch_windows,
)

WIDTH = 20
CHROM_LENGTH = 200


def _write_fasta(directory, line_width=60, newline="\n"):
    """A two-chromosome FASTA with a chosen line width and line terminator."""
    rng = np.random.default_rng(0)
    sequences = {
        "chr1": "".join(rng.choice(list("ACGT"), CHROM_LENGTH)),
        # chr2 carries lowercase and IUPAC codes so the ACGT policy has
        # something to reject and the LUT's lowercase rows get used.
        "chr2": "".join(rng.choice(list("acgtNRYn"), CHROM_LENGTH)),
    }
    path = directory / "genome.fa"
    with open(path, "w", newline="") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name}{newline}")
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start:start + line_width] + newline)
    return path, sequences


def _frame(rows):
    frame = pd.DataFrame(
        [(chrom, start, start + 100, ".", 0, ".", 0.0, 0.0, 0.0, summit)
         for chrom, start, summit in rows],
        columns=NARROWPEAK_SCHEMA,
    )
    return frame


#: summit is an offset from start, so the window centres on start + summit.
EDGE_ROWS = [
    ("chr1", 100, 0),               # comfortably interior
    ("chr1", 0, 0),                 # centre 0 -> negative start, must drop
    ("chr1", 0, WIDTH // 2),        # window starts at exactly 0, must keep
    ("chr1", CHROM_LENGTH, 0),      # centre at the end -> overruns, must drop
    ("chr1", CHROM_LENGTH - WIDTH // 2, 0),   # ends exactly at the end, keep
    ("chr1", 5, 0),                 # centre 5 -> start -5, must drop
    ("chr2", 100, 0),               # lowercase and IUPAC
    ("chr2", 50, 10),
]


@pytest.fixture
def genome(tmp_path):
    path, _ = _write_fasta(tmp_path)
    handle = pyfaidx.Fasta(str(path))
    yield handle
    handle.close()


def test_mapped_path_matches_pyfaidx_on_edge_regions(genome):
    frame = _frame(EDGE_ROWS)
    fast_seqs, fast_mask = _fetch_windows_mapped(frame, genome, WIDTH)
    slow_seqs, slow_mask = _fetch_windows_pyfaidx(frame, genome, WIDTH)

    assert np.array_equal(fast_mask, slow_mask), "keep-mask differs"
    assert fast_seqs.shape == slow_seqs.shape
    assert fast_seqs.dtype == slow_seqs.dtype == np.int8
    assert np.array_equal(fast_seqs, slow_seqs), "one-hot bytes differ"


def test_the_edge_rows_actually_exercise_both_outcomes(genome):
    """Guards the test itself: a mask of all-True would prove nothing."""
    _, mask = _fetch_windows_pyfaidx(_frame(EDGE_ROWS), genome, WIDTH)
    assert mask.any() and not mask.all()


@pytest.mark.parametrize("line_width", [7, 60, 61, CHROM_LENGTH])
def test_line_geometry_is_handled(tmp_path, line_width):
    """Byte offsets depend on bases-per-line; check several, including one
    where the whole chromosome is a single line."""
    path, _ = _write_fasta(tmp_path, line_width=line_width)
    with pyfaidx.Fasta(str(path)) as handle:
        frame = _frame(EDGE_ROWS)
        fast, fast_mask = _fetch_windows_mapped(frame, handle, WIDTH)
        slow, slow_mask = _fetch_windows_pyfaidx(frame, handle, WIDTH)
    assert np.array_equal(fast_mask, slow_mask)
    assert np.array_equal(fast, slow)


def test_crlf_line_endings_are_handled(tmp_path):
    """lenb - lenc is the terminator width, so CRLF must work unchanged."""
    path, _ = _write_fasta(tmp_path, newline="\r\n")
    with pyfaidx.Fasta(str(path)) as handle:
        frame = _frame(EDGE_ROWS)
        fast, fast_mask = _fetch_windows_mapped(frame, handle, WIDTH)
        slow, slow_mask = _fetch_windows_pyfaidx(frame, handle, WIDTH)
    assert np.array_equal(fast_mask, slow_mask)
    assert np.array_equal(fast, slow)


def test_odd_width_drops_everything_in_both_paths(genome):
    """window_bounds spans 2 * (width // 2), so an odd width can never fill
    the window. Surprising, pre-existing, and the fast path must agree."""
    frame = _frame(EDGE_ROWS)
    fast, fast_mask = _fetch_windows_mapped(frame, genome, 21)
    slow, slow_mask = _fetch_windows_pyfaidx(frame, genome, 21)
    assert not fast_mask.any()
    assert np.array_equal(fast_mask, slow_mask)
    assert fast.shape == slow.shape == (0, 21, 4)


def test_missing_chromosome_defers_to_pyfaidx(genome):
    """An absent contig must reach pyfaidx so its own error text is raised."""
    frame = _frame([("chrZ", 100, 0)])
    assert _fast_path_blocker(genome, frame) is not None
    with pytest.raises(KeyError):
        fetch_windows(frame, genome, WIDTH)


def test_all_regions_out_of_bounds_returns_empty_not_error(genome):
    frame = _frame([("chr1", 0, 0), ("chr1", CHROM_LENGTH, 0)])
    seqs, mask = fetch_windows(frame, genome, WIDTH)
    assert not mask.any()
    assert seqs.shape == (0, WIDTH, 4)


def test_interpret_get_seq_keeps_only_acgt_windows(genome):
    """chr2 is lowercase plus IUPAC, so the ACGT policy must reject some of it
    while the bigwig policy keeps it."""
    from chrombpnet.evaluation.interpret.input_utils import get_seq as acgt_get_seq
    from chrombpnet.evaluation.make_bigwigs.bigwig_helper import get_seq as bw_get_seq

    frame = _frame([("chr1", 100, 0), ("chr2", 100, 0), ("chr1", 50, 0)])
    bw_seqs, bw_mask = bw_get_seq(frame, genome, WIDTH)
    acgt_seqs, acgt_mask = acgt_get_seq(frame, genome, WIDTH)

    assert bw_mask.all()
    # Every kept ACGT row is saturated; the rejected chr2 row was not.
    assert (acgt_seqs.sum(axis=(1, 2)) == WIDTH).all()
    assert acgt_mask.sum() < bw_mask.sum()
    assert len(acgt_mask) == len(frame)
