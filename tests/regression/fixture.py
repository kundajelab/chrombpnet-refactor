"""Deterministic synthetic inputs for the CLI regression harness.

Everything here is generated from fixed seeds rather than committed, so the
repository stays small and the fixture can be regenerated anywhere. The inputs
are shaped to drive the real command line end to end -- including the
preprocessing path that shells out to bedtools and bedGraphToBigWig -- at sizes
that run in seconds.

Three properties are load-bearing and easy to break by accident:

1. ``auto_shift_detect.compute_shift`` requires every reference motif to agree
   on one shift, and that shift must be one of the standard values. Random
   sequence does not satisfy this, so the Tn5 consensus is planted at the
   offsets that make every ATAC reference motif report +0/-0. See
   ``PLUS_OFFSET`` / ``MINUS_OFFSET``.
2. ``auto_shift_detect.sample_reads`` pipes reads through ``shuf`` with no
   ``--random-source``, so the shift estimate is only reproducible when every
   read survives sampling. ``shuf -n`` is called with ``2 * num_samples``
   (default 10000), so the fixture must stay under 20000 reads. See
   :data:`MAX_REPRODUCIBLE_READS`.
3. ``find_bias_hyperparams`` keeps nonpeaks below
   ``quantile(peak_counts, 0.01) * bias_threshold_factor``. That threshold is
   degenerate unless peaks are genuinely enriched over nonpeaks, so peak
   windows repeat their read anchors -- see ``PEAK_PILEUP_RANGE``. Repeating
   existing anchors rather than introducing new ones keeps every read flank on
   a planted consensus, which property 1 depends on.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

# Chromosome layout. chr1/chr2 train, chr3 validates, chr4 tests.
#
# The chromosomes are long but reads and training regions occupy only the first
# REGION_SPAN bases of each. These two needs pull in opposite directions:
# `prep nonpeaks` slops every peak by inputlen // 2 and excludes it, so it needs
# a lot of unexcluded sequence to draw GC-matched candidates from, while
# auto_shift_detect's unseeded shuf caps the usable read count (see
# MAX_REPRODUCIBLE_READS). Confining the reads keeps both satisfiable, since GC
# matching reads sequence only and never touches the reads file.
CHROM_LENGTH = 200_000
REGION_SPAN = 40_000
CHROMS = ("chr1", "chr2", "chr3", "chr4")
TRAIN_CHROMS = ("chr1", "chr2")
VALID_CHROMS = ("chr3",)
TEST_CHROMS = ("chr4",)

# Model geometry. Small enough to train in seconds, even enough to satisfy
# ModelConfig.validate().
# Both lengths are pinned by hardcoded constants elsewhere in the package:
#
# * OUTPUTLEN >= 200, because marginal_footprinting.py:145 plots
#   footprint[outputlen//2 - 100 : outputlen//2 + 100] against range(200) and
#   raises "x and y must have same first dimension" for anything smaller.
# * INPUTLEN >= 500, because the qc pipelines call modisco through
#   pipelines._run_modisco_motifs, whose window defaults to 500 and is not
#   configurable from the command line; centered_slice then rejects any
#   contribution track shorter than the window.
#
# INPUTLEN must also cover the receptive field: a 1-dilation BPNet loses
# 20 (first conv) + 4 (dilated) + 74 (profile conv) = 98 bases, and the
# remaining profile must exceed OUTPUTLEN by an even margin.
INPUTLEN = 600
OUTPUTLEN = 200
MAX_JITTER = 10

GENOME_SEED = 20260918
READS_SEED = 4711
COUNTS_SEED = 99

# Tn5 consensus from chrombpnet/data/ATAC.ref.motifs.txt (scATAC_fibroblast).
# auto_shift_detect examines genome[pos-20:pos+20]; planting the 20 bp
# consensus at these offsets inside that window yields a +0/-0 estimate from
# every reference motif.
PLUS_MOTIF = "TGCACAGTCCAGGACTGTGC"
MINUS_MOTIF = "TGCACAGTCCTGGACTGTGC"
PLUS_OFFSET = 14
MINUS_OFFSET = 5

MAX_REPRODUCIBLE_READS = 20_000

# Plus anchors sit every READ_SPACING bases; the minus anchor for each one sits
# MINUS_ANCHOR_DELTA further along. Both values are constrained by the planted
# consensus windows, which must not overlap or shift estimation breaks -- see
# _validate_anchor_spacing(), which enforces the arithmetic.
READ_SPACING = 80
MINUS_ANCHOR_DELTA = 34
READ_LENGTH = 40
READ_MARGIN = 200
PEAK_PILEUP_RANGE = (8, 17)
# Background anchors also vary in depth. Without this the nonpeak counts are
# near-constant, the 0.9999 outlier quantiles in find_bias_hyperparams
# collapse onto one value, and its strict < / > bounds discard every region.
BACKGROUND_PILEUP_RANGE = (1, 4)

# Region layout, shared by the peak/nonpeak writers and by the pileup.
# Must exceed INPUTLEN // 2 + MAX_JITTER so jittered windows stay in bounds.
REGION_MARGIN = 800
PEAKS_PER_CHROM = 40
NONPEAK_SUBOFFSETS = (320, 485)


def _region_step() -> int:
    return (REGION_SPAN - 2 * REGION_MARGIN) // PEAKS_PER_CHROM


def peak_centers() -> list[int]:
    step = _region_step()
    return [REGION_MARGIN + index * step for index in range(PEAKS_PER_CHROM)]


def nonpeak_centers() -> list[int]:
    """Nonpeak centres, offset far enough from peak centres not to overlap.

    Peak output windows span centre +/- OUTPUTLEN // 2, so suboffsets well
    above that keep the two count distributions separate.
    """
    step = _region_step()
    limit = REGION_SPAN - REGION_MARGIN
    return sorted(
        centre + suboffset
        for centre in peak_centers()
        for suboffset in NONPEAK_SUBOFFSETS
        if centre + suboffset < limit
    )


def _validate_anchor_spacing() -> None:
    """Check that no two planted consensus windows overlap.

    A plus anchor at ``p`` plants over ``[p - 20 + PLUS_OFFSET, ... + 20)`` and a
    minus anchor at ``m`` over ``[m - 20 + MINUS_OFFSET, ... + 20)``. If any two
    plantings overlap they corrupt each other, the averaged PWM stops matching
    the reference, and compute_shift raises "Input shift is non-standard".
    """
    width = len(PLUS_MOTIF)
    plus_start = -20 + PLUS_OFFSET
    plus_end = plus_start + width
    minus_start = MINUS_ANCHOR_DELTA - 20 + MINUS_OFFSET
    minus_end = minus_start + width
    if plus_end > minus_start:
        raise AssertionError(
            f"plus planting ends at {plus_end} but the minus planting starts at "
            f"{minus_start}; raise MINUS_ANCHOR_DELTA to at least "
            f"{plus_end + 20 - MINUS_OFFSET}"
        )
    next_plus_start = READ_SPACING + plus_start
    if minus_end > next_plus_start:
        raise AssertionError(
            f"minus planting ends at {minus_end} but the next plus planting "
            f"starts at {next_plus_start}; raise READ_SPACING to at least "
            f"{minus_end - plus_start}"
        )


def _read_anchors() -> tuple[list[int], list[int]]:
    """Plus-read starts and minus-read ends, spaced so plantings never overlap."""
    _validate_anchor_spacing()
    plus_starts = list(range(READ_MARGIN, REGION_SPAN - READ_MARGIN, READ_SPACING))
    minus_ends = [start + MINUS_ANCHOR_DELTA for start in plus_starts]
    return plus_starts, minus_ends


def build_genome() -> dict[str, list[str]]:
    """Random sequence with the Tn5 consensus planted at every read anchor."""
    rng = np.random.RandomState(GENOME_SEED)
    plus_starts, minus_ends = _read_anchors()
    genome = {}
    for index, chrom in enumerate(CHROMS):
        # A distinct GC target per chromosome, so GC-matched negative selection
        # has a non-degenerate distribution to match against.
        gc = 0.35 + 0.08 * index
        probabilities = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        bases = list(rng.choice(list("ACGT"), size=CHROM_LENGTH, p=probabilities))
        for start in plus_starts:
            planted = start - 20 + PLUS_OFFSET
            bases[planted:planted + len(PLUS_MOTIF)] = list(PLUS_MOTIF)
        for end in minus_ends:
            planted = end - 20 + MINUS_OFFSET
            bases[planted:planted + len(MINUS_MOTIF)] = list(MINUS_MOTIF)
        genome[chrom] = bases
    return genome


def write_genome(directory: Path) -> Path:
    """Write genome.fa wrapped at 60 columns."""
    genome = build_genome()
    path = directory / "genome.fa"
    with path.open("w") as handle:
        for chrom in CHROMS:
            handle.write(f">{chrom}\n")
            sequence = "".join(genome[chrom])
            for offset in range(0, len(sequence), 60):
                handle.write(sequence[offset:offset + 60] + "\n")
    return path


def write_chrom_sizes(directory: Path) -> Path:
    path = directory / "chrom.sizes"
    path.write_text("".join(f"{chrom}\t{CHROM_LENGTH}\n" for chrom in CHROMS))
    return path


def write_reads(directory: Path) -> Path:
    """A gzipped tagAlign whose flanks sit on the planted Tn5 consensus.

    The 5' end of a plus read is its start and of a minus read its end, which
    is what ``bedtools genomecov -5`` counts, so peak enrichment is produced by
    repeating the anchors that fall inside a peak's output window.
    """
    rng = np.random.RandomState(READS_SEED)
    plus_starts, minus_ends = _read_anchors()
    half_output = OUTPUTLEN // 2

    # Per-peak pileup depth, drawn once so peaks differ from one another and
    # the count metrics have variance to work with.
    depths = {
        chrom: rng.randint(*PEAK_PILEUP_RANGE, size=PEAKS_PER_CHROM)
        for chrom in CHROMS
    }

    centres = peak_centers()

    def depth_at(chrom: str, position: int) -> int:
        for index, centre in enumerate(centres):
            if centre - half_output <= position < centre + half_output:
                return int(depths[chrom][index])
        return int(rng.randint(*BACKGROUND_PILEUP_RANGE))

    rows = []
    for chrom in CHROMS:
        for start in plus_starts:
            row = (chrom, start, start + READ_LENGTH, "N", 1000, "+")
            rows.extend([row] * depth_at(chrom, start))
        for end in minus_ends:
            row = (chrom, max(end - READ_LENGTH, 0), end, "N", 1000, "-")
            rows.extend([row] * depth_at(chrom, end))

    if len(rows) > MAX_REPRODUCIBLE_READS:
        raise AssertionError(
            f"{len(rows)} reads exceeds the {MAX_REPRODUCIBLE_READS}-read ceiling "
            "above which auto_shift_detect's unseeded shuf subsamples and the "
            "estimated shift stops being reproducible"
        )

    # A fixed shuffle, so the file is not already coordinate-sorted.
    order = rng.permutation(len(rows))
    path = directory / "reads.tagAlign.gz"
    with gzip.open(path, "wt") as handle:
        for index in order:
            handle.write("\t".join(str(field) for field in rows[index]) + "\n")
    return path


def _write_narrowpeak(path: Path, centers: list[int]) -> Path:
    """10-column narrowPeak with the summit offset in column 10."""
    summit = INPUTLEN // 2
    with path.open("w") as handle:
        for chrom in CHROMS:
            for centre in centers:
                start = centre - summit
                row = (chrom, start, start + 2 * summit,
                       ".", ".", ".", ".", ".", ".", summit)
                handle.write("\t".join(str(field) for field in row) + "\n")
    return path


def write_peaks(directory: Path) -> Path:
    return _write_narrowpeak(directory / "peaks.bed", peak_centers())


def write_nonpeaks(directory: Path) -> Path:
    return _write_narrowpeak(directory / "nonpeaks.bed", nonpeak_centers())


def write_bigwig(directory: Path) -> Path:
    """An independent observed-counts bigwig, enriched over the peak centres.

    Scenarios that need observed data but do not run preprocessing (``qc``,
    ``pred_bw``) use this instead of depending on another scenario's output.
    """
    import pyBigWig

    rng = np.random.RandomState(COUNTS_SEED)
    centers = set(peak_centers())
    path = directory / "data.bw"
    handle = pyBigWig.open(str(path), "w")
    handle.addHeader([(chrom, CHROM_LENGTH) for chrom in CHROMS])
    for chrom in CHROMS:
        values = rng.poisson(0.5, size=CHROM_LENGTH).astype(np.float64)
        for centre in centers:
            width = OUTPUTLEN // 2
            values[centre - width:centre + width] += rng.poisson(
                8.0, size=2 * width)
        handle.addEntries(chrom, 0, values=values, span=1, step=1)
    handle.close()
    return path


def write_folds(directory: Path) -> Path:
    """The fold JSON.

    Scenario 1 also produces this via ``chrombpnet prep splits``; writing it
    here keeps every other scenario independent of that one.
    """
    path = directory / "folds.json"
    path.write_text(json.dumps({
        "train": list(TRAIN_CHROMS),
        "valid": list(VALID_CHROMS),
        "test": list(TEST_CHROMS),
    }))
    return path


def build(directory: Path) -> dict[str, Path]:
    """Materialize the whole fixture and return its paths by role."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": directory,
        "genome": write_genome(directory),
        "chrom_sizes": write_chrom_sizes(directory),
        "reads": write_reads(directory),
        "peaks": write_peaks(directory),
        "nonpeaks": write_nonpeaks(directory),
        "bigwig": write_bigwig(directory),
        "folds": write_folds(directory),
    }
    # pyfaidx writes genome.fa.fai on first use. Build it now so the index is
    # part of the fixture rather than a side effect of whichever command runs
    # first, which would otherwise show up as a spurious artifact diff.
    import pyfaidx

    with pyfaidx.Fasta(str(paths["genome"])):
        pass
    return paths
