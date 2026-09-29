import numpy as np

from chrombpnet.training.metrics import profile_metrics, profile_pearsonr


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


def test_profile_pearsonr_matches_scipy_and_is_invariant_to_read_depth():
    from scipy.stats import pearsonr

    rng = np.random.default_rng(12)
    counts = rng.poisson(3, size=(20, 100))
    probabilities = rng.dirichlet(np.ones(100), size=20)
    expected = [pearsonr(x, y)[0] for x, y in zip(counts, probabilities)]
    np.testing.assert_allclose(profile_pearsonr(counts, probabilities), expected,
                               atol=1e-14)
    scaled = counts * np.arange(1, 21)[:, None]
    np.testing.assert_allclose(profile_pearsonr(scaled, probabilities), expected,
                               atol=1e-14)


def test_profile_pearsonr_perfect_and_reversed_profiles():
    counts = np.array([[0., 1., 2., 3.], [0., 1., 2., 3.]])
    probabilities = np.array([[0., 1., 2., 3.], [3., 2., 1., 0.]]) / 6
    np.testing.assert_allclose(profile_pearsonr(counts, probabilities), [1., -1.])


def test_profile_pearsonr_undefined_profiles_without_warnings():
    counts = np.array([[0., 0., 0.], [2., 2., 2.], [0., 1., 2.],
                       [np.nan, 1., 2.], [0., 1., 2.]])
    probabilities = np.array([[0., 0.5, 0.5], [0., 0.5, 0.5], [1., 1., 1.],
                              [0., 0.5, 0.5], [0., np.inf, 1.]])
    with np.errstate(all="raise"):
        assert np.isnan(profile_pearsonr(counts, probabilities)).all()
        # Mean subtraction can leave roundoff even for a constant profile.
        assert np.isnan(profile_pearsonr(np.ones((1, 7)), np.ones((1, 7)))).all()
        assert np.isnan(profile_pearsonr(np.ones((2, 1)), np.ones((2, 1)))).all()
        assert profile_pearsonr(np.empty((0, 3)), np.empty((0, 3))).size == 0
