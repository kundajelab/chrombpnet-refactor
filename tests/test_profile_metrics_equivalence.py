"""`profile_metrics` was vectorized; it must return the identical arrays.

Two things make this delicate. The per-region loop drew one
`rng.permutation` per *surviving* region and skipped the draw entirely for
regions it discarded, so the RNG stream depends on which rows are valid; and
the JSD values themselves must not move. Both are pinned here against the
pre-vectorization implementation, kept verbatim as the oracle.
"""

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon

from chrombpnet.training.metrics import profile_metrics
from chrombpnet.training.utils.metrics_utils import (
    JSD_BASE,
    get_min_max_normalized_value,
    jsd_min_max_bounds,
    normalize_profile,
)


def _profile_metrics_loop(true_counts, pred_probs, seed=1234):
    """The pre-vectorization body, verbatim apart from the dead mNLL lines
    and the base-2 JSD."""
    jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm = [], [], [], []
    rng = np.random.default_rng(seed)
    for idx in range(true_counts.shape[0]):
        observed_prob = normalize_profile(true_counts[idx, :])
        predicted_prob = normalize_profile(pred_probs[idx, :])
        if observed_prob is None or predicted_prob is None:
            jsd_pw.append(np.nan)
            jsd_norm.append(np.nan)
            jsd_rnd.append(np.nan)
            jsd_rnd_norm.append(np.nan)
            continue
        cur_jsd = jensenshannon(observed_prob, predicted_prob, base=JSD_BASE)
        jsd_pw.append(cur_jsd)
        min_jsd, max_jsd = jsd_min_max_bounds(true_counts[idx, :])
        jsd_norm.append(get_min_max_normalized_value(cur_jsd, min_jsd, max_jsd))
        shuffled = rng.permutation(true_counts[idx, :])
        shuffled_prob = normalize_profile(shuffled)
        cur_rnd = jensenshannon(observed_prob, shuffled_prob, base=JSD_BASE)
        jsd_rnd.append(cur_rnd)
        jsd_rnd_norm.append(get_min_max_normalized_value(cur_rnd, min_jsd, max_jsd))
    return (np.array(jsd_pw), np.array(jsd_norm),
            np.array(jsd_rnd), np.array(jsd_rnd_norm))


def _inputs(case, regions=120, width=200):
    rng = np.random.default_rng(0)
    counts = rng.poisson(5, (regions, width)).astype(np.float64)
    probs = rng.random((regions, width))
    probs /= probs.sum(axis=1, keepdims=True)
    if case == "clean":
        pass
    elif case == "zero_observed":
        counts[[3, 17, 99]] = 0.0            # normalize_profile -> None
    elif case == "nan_predicted":
        probs[[5, 60]] = np.nan
    elif case == "mixed_skips":
        # Interleaved so the RNG skip pattern is not a simple prefix.
        counts[[1, 50]] = 0.0
        probs[[2, 51]] = np.inf
    elif case == "uniform_observed":
        counts[7] = 1.0                      # min and max bounds coincide
    elif case == "all_invalid":
        counts[:] = 0.0
    return counts, probs


@pytest.mark.parametrize("case", ["clean", "zero_observed", "nan_predicted",
                                  "mixed_skips", "uniform_observed",
                                  "all_invalid"])
def test_vectorized_profile_metrics_matches_the_loop(case):
    counts, probs = _inputs(case)
    produced = profile_metrics(counts, probs)
    expected = _profile_metrics_loop(counts, probs)
    for name, got, want in zip(("jsd", "jsd_norm", "jsd_rnd", "jsd_rnd_norm"),
                               produced[2:6], expected):
        assert np.array_equal(got, want, equal_nan=True), f"{name} differs"


def test_the_skip_cases_actually_skip():
    """Guards the test: if nothing were skipped the RNG path is untested."""
    counts, probs = _inputs("mixed_skips")
    assert np.isnan(profile_metrics(counts, probs)[2]).sum() == 4


def test_mnll_slots_are_still_empty_arrays():
    """Both call sites unpack eight values and discard the four mNLL ones."""
    counts, probs = _inputs("clean")
    result = profile_metrics(counts, probs)
    assert len(result) == 8
    for index in (0, 1, 6, 7):
        assert result[index].size == 0


def test_jsd_is_base_2():
    """Bounded by [0, 1]: disjoint profiles are 1 apart. The base-e value
    the original reported is this times sqrt(ln 2); the min-max normalized
    JSD compares two distances in the same base, so it does not change."""
    counts = np.array([[5.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    probs = np.array([[0.0, 0.0, 0.0, 1.0], [0.12, 0.2, 0.3, 0.38]])
    _, _, jsd, jsd_norm, _, _, _, _ = profile_metrics(counts, probs)
    assert jsd[0] == pytest.approx(1.0)
    observed = counts[1] / counts[1].sum()
    natural = jensenshannon(observed, probs[1])
    assert jsd[1] == pytest.approx(natural / np.sqrt(np.log(2)))
    natural_max = jensenshannon(observed, np.full(4, 0.25))
    expected = get_min_max_normalized_value(natural, 0.0, natural_max)
    assert 0.0 < expected < 1.0
    assert jsd_norm[1] == pytest.approx(expected)
