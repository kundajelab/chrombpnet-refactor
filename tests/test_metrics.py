import numpy as np

from chrombpnet.training.metrics import profile_metrics


def test_profile_random_baseline_is_reproducible():
    counts = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, 2.0]])
    probabilities = np.full_like(counts, 0.25)
    first = profile_metrics(counts, probabilities, seed=12)
    second = profile_metrics(counts, probabilities, seed=12)
    for left, right in zip(first, second):
        np.testing.assert_equal(left, right)


def test_zero_count_profiles_are_reported_as_nan_without_warnings():
    counts = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 1.0]])
    probabilities = np.full_like(counts, 0.25)
    with np.errstate(all="raise"):
        result = profile_metrics(counts, probabilities, seed=4)
    jsd, normalized_jsd, random_jsd, normalized_random_jsd = result[2:6]
    assert np.isnan([jsd[0], normalized_jsd[0], random_jsd[0], normalized_random_jsd[0]]).all()
    assert np.isfinite([jsd[1], normalized_jsd[1], random_jsd[1], normalized_random_jsd[1]]).all()


def test_uniform_profile_has_undefined_normalized_jsd_without_warnings():
    counts = np.ones((1, 4))
    probabilities = np.full_like(counts, 0.25)
    with np.errstate(all="raise"):
        result = profile_metrics(counts, probabilities, seed=4)
    assert result[2][0] == 0.0
    assert np.isnan(result[3][0])
