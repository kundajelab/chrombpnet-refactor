"""The observed half of profile_metrics, computed ahead of the predictions,
must give exactly what the single call gives."""

import numpy as np
import pytest

from chrombpnet.predictions import softmax
from chrombpnet.training.metrics import ProfileBaseline, profile_metrics


def _inputs(rows=200, width=300):
    rng = np.random.default_rng(7)
    counts = rng.poisson(1.5, size=(rows, width)).astype(float)
    counts[[4, 50]] = 0.0                       # observed profiles that cannot normalize
    probs = softmax(rng.normal(size=(rows, width)).astype(np.float32))
    return counts, probs


def _same(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)


def test_a_precomputed_baseline_gives_the_same_metrics():
    counts, probs = _inputs()
    _same(profile_metrics(counts, probs, seed=11),
          profile_metrics(counts, probs, seed=11, baseline=ProfileBaseline(counts, seed=11)))


def test_a_prediction_that_does_not_normalize_redraws_the_baseline():
    """The draws skip regions whose predicted profile fails to normalize, so a
    baseline drawn before the predictions existed is wrong for them and must
    be drawn again."""
    counts, probs = _inputs()
    probs = probs.astype(np.float64)
    probs[[9, 120]] = np.nan
    baseline = ProfileBaseline(counts, seed=11)
    _same(profile_metrics(counts, probs, seed=11),
          profile_metrics(counts, probs, seed=11, baseline=baseline))


def test_a_baseline_serves_several_predictions_unchanged():
    counts, probs = _inputs()
    baseline = ProfileBaseline(counts, seed=11)
    first = profile_metrics(counts, probs, seed=11, baseline=baseline)
    first[4][:] = -1.0                          # a caller writing into jsd_rnd
    _same(profile_metrics(counts, probs[::-1].copy(), seed=11),
          profile_metrics(counts, probs[::-1].copy(), seed=11, baseline=baseline))


def test_the_baseline_must_have_the_same_seed():
    counts, probs = _inputs()
    with pytest.raises(ValueError, match="seed"):
        profile_metrics(counts, probs, seed=12, baseline=ProfileBaseline(counts, seed=11))


@pytest.mark.parametrize("chunk", [1, 7, 256, 4096])
def test_softmax_rows_do_not_depend_on_the_chunk(chunk):
    """pred_bw now takes the softmax chunk by chunk."""
    logits = np.random.default_rng(2).normal(scale=4, size=(5000, 1000)).astype(np.float32)
    whole = softmax(logits)
    pieces = np.concatenate([softmax(logits[start:start + chunk])
                             for start in range(0, len(logits), chunk)])
    assert pieces.dtype == whole.dtype
    assert pieces.tobytes() == whole.tobytes()


@pytest.mark.parametrize("chunk", [1, 13, 64])
@pytest.mark.parametrize("broken", [False, True])
def test_profile_metrics_fed_in_chunks_is_the_single_call(chunk, broken):
    """Per-region JSD row by row, in any arrival order, and the redraw when a
    predicted profile fails to normalize."""
    from chrombpnet.training.metrics import ProfileMetrics

    counts, probs = _inputs()
    if broken:
        probs = probs.astype(np.float64)
        probs[[9, 120]] = np.nan
    stream = ProfileMetrics(counts, seed=11, baseline=ProfileBaseline(counts, seed=11))
    starts = list(range(0, len(probs), chunk))
    np.random.default_rng(0).shuffle(starts)
    for start in starts:
        stream.add(start, probs[start:start + chunk])
    _same(stream.result(), profile_metrics(counts, probs, seed=11))


@pytest.mark.parametrize("chunk", [1, 13, 64])
def test_profile_pearsonr_rows_do_not_depend_on_the_chunk(chunk):
    from chrombpnet.training.metrics import profile_pearsonr

    counts, probs = _inputs()
    pieces = np.concatenate([profile_pearsonr(counts[start:start + chunk], probs[start:start + chunk])
                             for start in range(0, len(probs), chunk)])
    assert pieces.tobytes() == profile_pearsonr(counts, probs).tobytes()
