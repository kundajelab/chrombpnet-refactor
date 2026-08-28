import numpy as np

from chrombpnet.training.metrics import profile_metrics


def test_profile_random_baseline_is_reproducible():
    counts = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, 2.0]])
    probabilities = np.full_like(counts, 0.25)
    first = profile_metrics(counts, probabilities, seed=12)
    second = profile_metrics(counts, probabilities, seed=12)
    for left, right in zip(first, second):
        np.testing.assert_equal(left, right)
