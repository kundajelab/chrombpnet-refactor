"""The cut convention, the reverse-complement shift, and the arms.

The geometric claim -- DNase records a *bond* and ATAC records a *base* -- is
derived in `chrombpnet/conventions.py`. Derivations can be wrong, so
`test_the_reference_pwms_agree_with_the_resolved_shift` checks it against
measured data shipped in the repository instead of against the derivation.
"""

import numpy as np
import pytest

from chrombpnet import conventions
from chrombpnet.conventions import (BASE, BOND_BOTH, BOND_LEFT, BOND_RIGHT,
                                    resolve, shift_delta_pairs)
from chrombpnet.training.utils.augment import random_rev_comp


def _one_hot(width, count=1):
    seqs = np.zeros((count, width, 4), dtype=np.int8)
    seqs[:, :, 0] = 1
    return seqs


def _flip_one(index, width, rc_shift, pad_left, pad_right):
    """Reverse-complement a profile holding a single count at `index`.

    The array handed to the augmenter is the *widened* one when the arm
    shifts, exactly as the generator will hand it over.
    """
    stored = np.zeros((1, width + pad_left + pad_right))
    stored[0, index + pad_left] = 1.0
    coords = np.array([["chr1", "100", "f", "1"]])
    _, labels, _ = random_rev_comp(
        _one_hot(width), stored, coords, frac=1.0,
        rng=np.random.RandomState(0), rc_shift=rc_shift, output_width=width)
    assert labels.shape == (1, width)
    return labels[0]


@pytest.mark.parametrize("width", [8, 12])
@pytest.mark.parametrize("index", [1, 2, 3, 4, 5])
def test_the_right_convention_lands_the_bond_on_its_mirrored_right_base(width, index):
    """`i -> W-i`: reverse, then shift one right.

    A count at slot `i` means a nick in the bond `(i-1, i)`. Reversal maps
    those two bases to `(W-1-i, W-i)`, and the right-hand base of the
    mirrored bond is `W-i`. Plain reversal would give `W-1-i` -- one short.
    """
    out = _flip_one(index, width, rc_shift=+1, pad_left=0, pad_right=1)
    assert out.argmax() == width - index
    assert out.sum() == 1.0


@pytest.mark.parametrize("width", [8, 12])
@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_the_left_convention_lands_on_the_mirrored_left_base(width, index):
    """The mirror of the above: `i -> W-2-i`."""
    out = _flip_one(index, width, rc_shift=-1, pad_left=1, pad_right=0)
    assert out.argmax() == width - 2 - index
    assert out.sum() == 1.0


@pytest.mark.parametrize("index", [0, 3, 7])
def test_no_shift_is_plain_reversal(index):
    """standard, both-* and every ATAC arm: `i -> W-1-i`."""
    out = _flip_one(index, 8, rc_shift=0, pad_left=0, pad_right=0)
    assert out.argmax() == 8 - 1 - index
    assert out.sum() == 1.0


def test_the_widened_column_is_what_fills_the_edge_slot():
    """Slot 0 of a +1 flip comes from the extra base, not from a zero.

    Zero-filling would plant the same systematic zero in every
    reverse-complemented example. The widened fetch supplies real signal.
    """
    stored = np.zeros((1, 9))
    stored[0, 8] = 5.0                     # the extra right-hand column
    coords = np.array([["chr1", "100", "f", "1"]])
    _, labels, _ = random_rev_comp(
        _one_hot(8), stored, coords, frac=1.0,
        rng=np.random.RandomState(0), rc_shift=+1, output_width=8)
    assert labels[0, 0] == 5.0
    assert labels[0].sum() == 5.0


def test_a_shifted_flip_is_not_an_involution():
    """Why the aliasing fix had to land first.

    Plain reversal is its own inverse, so in-place double-flips were
    survivable. With a shift, slot 0 is destroyed on every round trip.
    """
    coords = np.array([["chr1", "100", "f", "1"]])
    once = np.arange(1.0, 10.0).reshape(1, 9)
    _, flipped, _ = random_rev_comp(
        _one_hot(8), once.copy(), coords.copy(), frac=1.0,
        rng=np.random.RandomState(0), rc_shift=+1, output_width=8)
    padded = np.concatenate([flipped, np.zeros((1, 1))], axis=1)
    _, twice, _ = random_rev_comp(
        _one_hot(8), padded, coords.copy(), frac=1.0,
        rng=np.random.RandomState(0), rc_shift=+1, output_width=8)
    assert not np.array_equal(twice[0], once[0, :8])


# -- The arms ---------------------------------------------------------------

@pytest.mark.parametrize("arm,shift", [
    ("standard", 0), ("fr-only", 0), ("rc-only", 0),
    ("right", +1), ("left", -1), ("both-half", 0), ("both-full", 0),
])
def test_dnase_shifts(arm, shift):
    assert resolve("DNASE", arm).rc_shift == shift


@pytest.mark.parametrize("arm", conventions.ASSAY_AGNOSTIC)
def test_atac_never_shifts(arm):
    """+4/-4 centres the cut on a base, whose mirror is itself."""
    convention = resolve("ATAC", arm)
    assert convention.rc_shift == 0
    assert convention.deposit == BASE


@pytest.mark.parametrize("arm", ["right", "left", "both-half", "both-full"])
def test_the_bond_arms_are_refused_for_atac(arm):
    """A silent no-op would return five identical ATAC numbers with nothing
    to say why."""
    with pytest.raises(ValueError, match="defined only for DNASE"):
        resolve("ATAC", arm)


def test_which_splits_are_augmented():
    assert resolve("DNASE", "standard").rc_modes == frozenset({"train"})
    assert resolve("DNASE", "fr-only").rc_modes == frozenset()
    assert resolve("DNASE", "rc-only").rc_modes == frozenset(
        {"train", "valid", "test"})
    assert resolve("DNASE", "rc-only").rc_frac == 1.0
    assert resolve("DNASE", "standard").rc_frac == 0.5


def test_only_the_shifted_arms_widen_the_label_window():
    """Everything else stays on the original get_cts code path."""
    widen = {a for a in conventions.DATA_AUG_CHOICES
             if resolve("DNASE", a).widens_label_window}
    assert widen == {"right", "left"}


def test_the_default_is_todays_behaviour():
    for data_type in ("ATAC", "DNASE"):
        assert resolve(data_type, None) == resolve(data_type, "standard")
    convention = resolve("DNASE", "standard")
    assert convention.deposit == BOND_RIGHT and convention.scale == 1.0
    assert shift_delta_pairs(convention, 0, 0) == [(0, 1)]
    assert shift_delta_pairs(resolve("ATAC", "standard"), 0, 0) == [(4, -4)]


def test_both_is_the_union_of_left_and_right():
    """Depositing on both flanking bases *is* applying both single-base
    conventions at once, which is why the awk emits two rows."""
    left = shift_delta_pairs(resolve("DNASE", "left"), 0, 0)
    right = shift_delta_pairs(resolve("DNASE", "right"), 0, 0)
    both = shift_delta_pairs(resolve("DNASE", "both-half"), 0, 0)
    assert both == left + right
    assert resolve("DNASE", "both-half").scale == 0.5
    assert resolve("DNASE", "both-full").scale == 1.0


@pytest.mark.parametrize("plus_shift,minus_shift", [(0, 0), (4, -5), (-2, 3)])
def test_the_deltas_cancel_a_pre_applied_enzyme_shift(plus_shift, minus_shift):
    """An already-shifted input must land in the same place as a raw one."""
    for arm in ("standard", "left", "both-full"):
        raw = shift_delta_pairs(resolve("DNASE", arm), 0, 0)
        pre = shift_delta_pairs(resolve("DNASE", arm), plus_shift, minus_shift)
        assert [(p - plus_shift, m - minus_shift) for p, m in raw] == pre


# -- The premise, against data rather than the derivation -------------------

def _reference_pwms(assay):
    from chrombpnet.data import DefaultDataFile, get_default_data_path
    from chrombpnet.helpers.preprocessing.auto_shift_detect import get_ref_pwms

    handle = {"DNASE": DefaultDataFile.dnase_ref_motifs,
              "ATAC": DefaultDataFile.atac_ref_motifs}[assay]
    return get_ref_pwms(get_default_data_path(handle))


def _best_lag(a, b, lags=range(-3, 4)):
    scores = {}
    n = len(a)
    for s in lags:
        lo, hi = max(0, -s), min(n, n - s)
        if hi - lo < n - 3:
            continue
        x = a[lo:hi].ravel() - a[lo:hi].mean()
        y = b[lo + s:hi + s].ravel() - b[lo + s:hi + s].mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        scores[s] = float(x @ y / denom) if denom else 0.0
    best = max(scores, key=scores.get)
    runner = max(v for k, v in scores.items() if k != best)
    return best, scores[best], runner


@pytest.mark.parametrize("assay,expected_lag", [("DNASE", 0), ("ATAC", -1)])
def test_the_reference_pwms_agree_with_the_resolved_shift(assay, expected_lag):
    """Measured confirmation that DNase is bond-registered and ATAC is not.

    `auto_shift_detect.get_pwms` builds both PWMs in plus-strand genomic
    orientation and centres them on the deposit base -- plus on `start`,
    minus on `end`. DNase cuts one strand at a time, so for one physical nick
    the minus-strand context is the reverse complement of the plus-strand
    one; the lag at which they align says where the symmetry centre sits.
    Bond-registered gives lag 0, base-registered gives lag -1, and the label
    shift is `lag + 1` either way.

    This fails loudly if anyone changes the deltas in `reads_to_bigwig`
    without changing `rc_shift` to match.
    """
    plus, minus = _reference_pwms(assay)
    lags = set()
    for name, plus_pwm in plus.items():
        minus_pwm = minus[name[:-len("_plus")] + "_minus"]
        rc_plus = plus_pwm[::-1, ::-1]
        lag, best, runner = _best_lag(minus_pwm, rc_plus)
        assert best - runner > 0.3, f"{name}: lag {lag} is not decisive"
        lags.add(lag)
    assert lags == {expected_lag}
    assert resolve(assay, "right" if assay == "DNASE" else "standard").rc_shift \
        == expected_lag + 1


# -- End to end through the generator ---------------------------------------

def _tiny_workspace(tmp_path, cut_at, chrom_len=2000):
    """A bigwig holding a single count at `cut_at`, plus a matching peak."""
    import pyBigWig

    (tmp_path / "g.fa").write_text(">chr1\n" + "ACGT" * (chrom_len // 4) + "\n")
    (tmp_path / "sizes.tsv").write_text(f"chr1\t{chrom_len}\n")
    with pyBigWig.open(str(tmp_path / "obs.bw"), "w") as bw:
        bw.addHeader([("chr1", chrom_len)])
        bw.addEntries(["chr1"], [cut_at], ends=[cut_at + 1], values=[1.0])
    # summit at column 10 of a narrowPeak; the window centres on start+summit
    (tmp_path / "peaks.bed").write_text(
        "chr1\t900\t1100\t.\t0\t.\t0\t0\t0\t100\n")
    return tmp_path


@pytest.mark.parametrize("arm,expected_shift", [
    ("standard", 0), ("right", +1), ("left", -1),
])
def test_the_generator_applies_the_arms_shift_end_to_end(tmp_path, arm, expected_shift):
    """The whole chain: padded get_cts -> random_crop -> trim -> flip.

    The unit tests above pin `random_rev_comp` alone. This pins the plumbing
    around it -- the extra column has to be fetched, survive the crop, and be
    trimmed off again, and getting any of those wrong shifts the profile
    silently.
    """
    import pandas as pd
    from chrombpnet.regions import NARROWPEAK_SCHEMA
    from chrombpnet.training.data_generators.batchgen_generator import (
        ChromBPNetBatchGenerator)

    outputlen, inputlen = 20, 100
    centre = 1000                                  # start 900 + summit 100
    slot = 7                                       # where the count should land
    cut_at = centre - outputlen // 2 + slot
    workspace = _tiny_workspace(tmp_path, cut_at)
    peaks = pd.read_csv(workspace / "peaks.bed", sep="\t", header=None,
                        names=NARROWPEAK_SCHEMA)

    convention = resolve("DNASE", arm)
    assert convention.rc_shift == expected_shift

    generator = ChromBPNetBatchGenerator(
        peak_regions=peaks, nonpeak_regions=None,
        genome_fasta=str(workspace / "g.fa"), batch_size=1,
        inputlen=inputlen, outputlen=outputlen, max_jitter=0,
        negative_sampling_ratio=1.0, cts_bw_file=str(workspace / "obs.bw"),
        add_revcomp=True, return_coords=True, shuffle_at_epoch_start=False,
        mode="train", rc_shift=convention.rc_shift, rc_frac=1.0)

    assert generator.cur_cts.shape == (1, outputlen), "the trim did not happen"
    assert generator.cur_cts.sum() == 1.0, "the count was dropped or duplicated"
    landed = int(generator.cur_cts[0].argmax())
    # reverse (W-1-i), then the arm's shift
    assert landed == outputlen - 1 - slot + expected_shift
    assert generator.cur_coords[0][2] == "r"


def test_an_unshifted_arm_fetches_exactly_the_old_window(tmp_path):
    """`standard` must not read the extra base at all."""
    import pandas as pd
    from chrombpnet.regions import NARROWPEAK_SCHEMA
    from chrombpnet.training.data_generators.batchgen_generator import (
        ChromBPNetBatchGenerator)

    workspace = _tiny_workspace(tmp_path, 1000)
    peaks = pd.read_csv(workspace / "peaks.bed", sep="\t", header=None,
                        names=NARROWPEAK_SCHEMA)
    generator = ChromBPNetBatchGenerator(
        peak_regions=peaks, nonpeak_regions=None,
        genome_fasta=str(workspace / "g.fa"), batch_size=1,
        inputlen=100, outputlen=20, max_jitter=0, negative_sampling_ratio=1.0,
        cts_bw_file=str(workspace / "obs.bw"), add_revcomp=False,
        return_coords=True, shuffle_at_epoch_start=False, mode="train",
        rc_shift=0, rc_frac=0.5)
    assert generator.label_pad == (0, 0)
    assert generator.peak_cts.shape == (1, 20)


# -- Half-integer counts, which the both-half arm produces ------------------

def test_half_integer_counts_train_on_the_same_objective():
    """`both-half` deposits 0.5 per base, so counts stop being integers.

    The multinomial NLL's coefficient term `lgamma(S+1) - sum lgamma(c+1)` is
    the only place integrality could matter, and it does not depend on the
    logits -- so it is a constant offset that shifts the *reported* loss
    without touching gradients. The data term is linear in the counts, so
    halving them halves the gradient and leaves its direction identical.

    That is why `both-half` and `both-full` converge to the same place: they
    optimize the same objective, and Adam normalizes per parameter, so the
    factor of two is largely absorbed. If this ever stops holding, the two
    arms stop being each other's control.
    """
    import torch

    from chrombpnet.training.utils.losses import multinomial_nll

    torch.manual_seed(1)
    logits = torch.randn(3, 24)
    full = torch.poisson(torch.full((3, 24), 4.0))      # integers
    half = full / 2                                     # half-integers

    grads = []
    for counts in (full, half):
        leaf = logits.clone().requires_grad_(True)
        loss = multinomial_nll(counts, leaf)
        assert torch.isfinite(loss), "half-integer counts must not produce NaN"
        loss.backward()
        grads.append(leaf.grad.clone())

    assert torch.isfinite(grads[1]).all()
    torch.testing.assert_close(grads[1], 0.5 * grads[0], atol=1e-6, rtol=0)


def test_half_integer_counts_give_defined_profile_metrics():
    """The metrics normalize to probabilities, so 0.5 counts are not special.

    Guards against a future change that reintroduces an integer assumption
    (a cast, or `np.random.multinomial`, which requires an integer n).
    """
    import numpy as np

    from chrombpnet.training.metrics import profile_metrics, profile_pearsonr

    rng = np.random.default_rng(3)
    integer = rng.poisson(6, size=(12, 40)).astype(float)
    halved = integer / 2

    probabilities = rng.dirichlet(np.ones(40), size=12)

    # Scaling a profile leaves its shape, hence every shape metric, unchanged.
    for index in (2, 3, 4, 5):
        np.testing.assert_allclose(
            profile_metrics(halved, probabilities)[index],
            profile_metrics(integer, probabilities)[index],
            atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(profile_pearsonr(halved, probabilities),
                               profile_pearsonr(integer, probabilities),
                               atol=1e-12, equal_nan=True)

    # And the counts target differs only by the log of that scale factor.
    assert np.all(np.isfinite(np.log1p(halved.sum(axis=1))))
