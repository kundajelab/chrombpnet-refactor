"""What each --dnase-convention actually writes into the bigwig.

`test_conventions.py` covers the augmentation algebra. This covers the other
half: that `reads_to_bigwig` deposits where the convention says it should.

The fixture is two reads that report the *same* nick from opposite strands,
which is the case the 0/+1 convention exists to make agree:

    plus  read [100, 140)   5' base 100   nick between 99 and 100
    minus read [ 60, 100)   5' base  99   nick between 99 and 100

So every convention has one exact expected answer, and the plus/minus convergence is
asserted rather than assumed.
"""

import gzip
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

pyBigWig = pytest.importorskip("pyBigWig")

CHROM_LEN = 400
TOOLS = ("bedtools", "bedGraphToBigWig", "awk", "sort")


def _require_tools():
    missing = [t for t in TOOLS if shutil.which(t) is None]
    if missing:
        pytest.skip(f"preprocessing needs {', '.join(missing)} on PATH")


@pytest.fixture
def workspace(tmp_path):
    _require_tools()
    # Non-palindromic, so a mis-registered PWM could not look symmetric by
    # accident if this fixture is reused for a motif check.
    seq = ("ACGGTTACAG" * (CHROM_LEN // 10))[:CHROM_LEN]
    (tmp_path / "genome.fa").write_text(f">chr1\n{seq}\n")
    (tmp_path / "sizes.tsv").write_text(f"chr1\t{CHROM_LEN}\n")

    reads = ["chr1\t100\t140\tN\t1000\t+", "chr1\t60\t100\tN\t1000\t-"]
    with gzip.open(tmp_path / "reads.tagAlign.gz", "wt") as handle:
        handle.write("\n".join(reads) + "\n")
    return tmp_path


def _run(workspace, data_type, dnase_convention, data_aug=None):
    import chrombpnet.helpers.preprocessing.reads_to_bigwig as reads_to_bigwig

    prefix = str(workspace / f"out_{dnase_convention}_{data_aug}")
    reads_to_bigwig.main(SimpleNamespace(
        genome=str(workspace / "genome.fa"),
        input_bam_file=None,
        input_fragment_file=None,
        input_tagalign_file=str(workspace / "reads.tagAlign.gz"),
        chrom_sizes=str(workspace / "sizes.tsv"),
        output_prefix=prefix,
        data_type=data_type,
        bsort=False,
        no_st=False,
        tmpdir=None,
        plus_shift=0,          # given, so no estimation from two reads
        minus_shift=0,
        ATAC_ref_path=None,
        DNASE_ref_path=None,
        num_samples=10,
        data_aug=data_aug,
        dnase_convention=dnase_convention,
    ))
    with pyBigWig.open(prefix + "_unstranded.bw") as bw:
        values = np.nan_to_num(np.array(bw.values("chr1", 0, CHROM_LEN)))
    return {int(i): float(v) for i, v in enumerate(values) if v != 0}


@pytest.mark.parametrize("dnase_convention,expected", [
    # Both reads name the bond (99, 100) by its right-hand base.
    ("right-deprecated", {100: 2.0}),
    ("right",     {100: 2.0}),
    # ...and by its left-hand base.
    ("left",      {99: 2.0}),
    # ...and on both, which is what makes plain reversal correct.
    ("both-full", {99: 2.0, 100: 2.0}),
    ("both-half", {99: 1.0, 100: 1.0}),
    # An omitted flag is both-half on DNASE, so it must land in the same
    # place. Tested end to end rather than only in resolve(), because the
    # regression fixture is ATAC-only and so cannot see a DNase default move.
    (None,        {99: 1.0, 100: 1.0}),
])
def test_each_convention_deposits_where_it_says(workspace, dnase_convention, expected):
    assert _run(workspace, "DNASE", dnase_convention) == expected


@pytest.mark.parametrize("data_aug", ["augment", "forward-only", "revcomp-only"])
def test_the_bigwig_does_not_depend_on_data_aug(workspace, data_aug):
    """--data-aug picks which examples are flipped in training; the track
    is the convention's alone."""
    assert _run(workspace, "DNASE", "right", data_aug) == {100: 2.0}
    assert _run(workspace, "ATAC", None, data_aug) == {95: 1.0, 104: 1.0}


def test_the_dnase_default_conserves_the_track_total(workspace):
    """The default may change *where* signal sits, never *how much*."""
    omitted = sum(_run(workspace, "DNASE", None).values())
    single = sum(_run(workspace, "DNASE", "right-deprecated").values())
    assert omitted == single


def test_both_half_conserves_the_track_total(workspace):
    """The gate that makes `both-half` comparable to the other conventions.

    Spreading a cut over two bases must not change how much signal there is,
    or the counts head is being trained against a different target scale for
    reasons that have nothing to do with the bond.
    """
    single = sum(_run(workspace, "DNASE", "right-deprecated").values())
    half = sum(_run(workspace, "DNASE", "both-half").values())
    full = sum(_run(workspace, "DNASE", "both-full").values())
    assert half == pytest.approx(single)
    assert full == pytest.approx(2 * single)


def test_half_counts_survive_the_bigwig_round_trip(workspace):
    """0.5 is binary-exact in float32, so `-scale 0.5` is lossless."""
    assert set(_run(workspace, "DNASE", "both-half").values()) == {1.0}
    # ...and genomecov still omits zero intervals under -scale.
    assert 0.0 not in _run(workspace, "DNASE", "both-half").values()


def test_atac_is_unchanged_and_refuses_a_dnase_convention(workspace):
    """+4/-4: plus 5' + 4, minus 5' - 4. Not a Tn5 pair here, so two bases."""
    assert _run(workspace, "ATAC", None) == {95: 1.0, 104: 1.0}
    for dnase_convention in ("right", "left", "both-half", "both-full", "right-deprecated"):
        with pytest.raises(ValueError, match="defined only for DNASE"):
            _run(workspace, "ATAC", dnase_convention)


@pytest.mark.parametrize("n_deposits", [1, 2])
def test_every_deposit_is_guarded(n_deposits):
    """bedtools aborts the run on a record it cannot parse, so each emitted
    deposit is bounds-checked -- including the single-deposit form, because
    `left` shifts plus reads to `start - 1`."""
    from chrombpnet.helpers.preprocessing.reads_to_bigwig import _awk_program

    program = _awk_program(n_deposits)
    for i in range(n_deposits):
        assert f"$2+plus{i} >= 0" in program
        # >= and not >: bedtools accepts a zero-width record and emits a
        # position for it, so a strict test would silently drop real signal.
        assert f"$3+minus{i} >= $2" in program


def test_a_read_at_coordinate_zero_survives_the_left_convention(workspace):
    """`left` deposits a plus read at `start - 1`. Without the guard the
    negative coordinate makes bedtools exit 1 and the whole run dies."""
    import gzip

    reads = ["chr1\t0\t40\tN\t1000\t+",          # would become -1
             "chr1\t100\t140\tN\t1000\t+"]
    with gzip.open(workspace / "reads.tagAlign.gz", "wt") as handle:
        handle.write("\n".join(reads) + "\n")

    # The out-of-bounds deposit is dropped; the in-bounds one survives.
    assert _run(workspace, "DNASE", "left") == {99: 1.0}
