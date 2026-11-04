import numpy as np

from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import jensenshannon
from chrombpnet.seed import DEFAULT_SEED
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Explicit rather than `import *`: the wildcard also re-bound numpy,
# matplotlib, plt and four scipy names over this module's own imports, so which
# definition won depended on import order.
from chrombpnet.training.utils.metrics_utils import (
    JSD_BASE,
    density_scatter,
    get_min_max_normalized_value,
    jsd_min_max_bounds,
    normalize_profile,
)

   
def counts_metrics(labels,preds,outf,title):
    '''
    Get count metrics
    '''
    spearman_cor=spearmanr(labels,preds)[0]
    pearson_cor=pearsonr(labels,preds)[0]  
    mse=((labels - preds)**2).mean(axis=0)

    plt.rcParams["figure.figsize"]=8,8
    # density_scatter draws its own figure (plt.subplots) and that is the one
    # saved; the empty plt.figure() that used to precede it only leaked.
    ax = density_scatter(labels,
                    preds,
                    xlab='Log Count Labels',
                    ylab='Log Count Predictions')
    plt.suptitle(title+" count: spearman R="+str(round(spearman_cor,3))+", Pearson R="+str(round(pearson_cor,3))+", mse="+str(round(mse,3)))
    plt.savefig(outf+'.counts_pearsonr.png',format='png',dpi=300)
    plt.close(ax.figure)
    
    return spearman_cor, pearson_cor, mse

def _normalize_rows(profiles):
    """Row-wise `normalize_profile`, with a validity mask instead of None.

    `normalize_profile` returns None for a row it cannot turn into a finite
    probability vector. Vectorized, that becomes a boolean mask; invalid rows
    are left as zeros and excluded by the caller, never divided.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    totals = np.nansum(profiles, axis=1)
    valid = np.isfinite(totals) & (totals > 0) & np.all(np.isfinite(profiles), axis=1)
    normalized = np.zeros_like(profiles)
    np.divide(profiles, totals[:, None], out=normalized, where=valid[:, None])
    return normalized, valid


def profile_pearsonr(true_counts, pred_probs):
    """Pearson R across bases for each region's observed/predicted profile.

    Correlate profile shape using probabilities, without smoothing or log
    transforms. Positive rescaling of either profile leaves R unchanged.
    Constant, zero-count, or non-finite profiles have undefined R (NaN).
    """
    observed, observed_ok = _normalize_rows(true_counts)
    predicted, predicted_ok = _normalize_rows(pred_probs)
    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted profiles must have the same shape")
    result = np.full(observed.shape[0], np.nan)
    if observed.shape[1] < 2:
        return result
    variable = (np.ptp(observed, axis=1) > 0) & (np.ptp(predicted, axis=1) > 0)
    observed -= observed.mean(axis=1, keepdims=True)
    predicted -= predicted.mean(axis=1, keepdims=True)
    denominator = (np.linalg.norm(observed, axis=1)
                   * np.linalg.norm(predicted, axis=1))
    valid = observed_ok & predicted_ok & variable & (denominator > 0)
    np.divide(np.sum(observed * predicted, axis=1), denominator,
              out=result, where=valid)
    return np.clip(result, -1.0, 1.0)


def _min_max_normalize(values, minimum, maximum):
    """Row-wise `get_min_max_normalized_value`, clipped to [0, 1].

    The scalar version returns NaN when any input is non-finite or when the
    bounds coincide, and otherwise clips; `np.isclose` is the same predicate
    it used.
    """
    values = np.asarray(values, dtype=np.float64)
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(minimum) & np.isfinite(maximum)
    usable = finite & ~np.isclose(minimum, maximum)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(values - maximum, minimum - maximum, out=out, where=usable)
    return np.clip(out, 0.0, 1.0, out=out, where=usable)


def _profile_metrics_looped(true_counts, pred_probs, seed):
    """Legacy: the pre-vectorization body. Unreachable in normal operation."""
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
        shuffled_prob = normalize_profile(rng.permutation(true_counts[idx, :]))
        cur_rnd = jensenshannon(observed_prob, shuffled_prob, base=JSD_BASE)
        jsd_rnd.append(cur_rnd)
        jsd_rnd_norm.append(get_min_max_normalized_value(cur_rnd, min_jsd, max_jsd))
    empty = np.array([])
    return (empty, empty, np.array(jsd_pw), np.array(jsd_norm),
            np.array(jsd_rnd), np.array(jsd_rnd_norm), empty, empty)


class ProfileBaseline:
    """The half of `profile_metrics` that needs only the observed counts.

    The shuffled-profile baseline -- one seeded permutation per region, in
    region order -- depends on the predictions only through which regions it
    skips: `profile_metrics` draws for the regions whose observed *and*
    predicted profiles normalize, and a softmax row always normalizes unless
    the model produced a non-finite value. So the draws, `jsd_rnd` and the
    `max_jsd` bound can be computed before the predictions exist, for the
    regions whose observed profile normalizes, and `profile_metrics` keeps
    them only if its predictions leave exactly that set; otherwise it draws
    again over the right set. Either way the values are the ones the single
    call computes.

    That is the optimization kit's `metrics_overlap` phase A
    (github.com/anthropics/uplifting-biomolecular-modeling, Apache-2.0),
    which runs the observed side and the RNG ahead of the forward. The kit
    assumes every row is valid; the check and redraw are this package's.
    """

    def __init__(self, true_counts, seed=DEFAULT_SEED, rows=None):
        true_counts = np.asarray(true_counts)
        self.seed = seed
        self.observed_prob, self.observed_ok = _normalize_rows(true_counts)
        # The regions drawn for: every normalizable observed profile, or
        # those of them that `rows` (the predicted profiles' mask) also keeps.
        valid = self.observed_ok if rows is None else self.observed_ok & rows
        self.valid = valid

        # The old loop drew one permutation per *surviving* region, in order, and
        # skipped the draw entirely for regions it discarded. `rng.permutation(x)`
        # on a 1-D array consumes the same state as `x[rng.permutation(len(x))]`
        # and returns the same values, so drawing an index permutation per valid
        # region row reproduces the stream exactly.
        rng = np.random.default_rng(seed)
        num_regions, width = true_counts.shape
        shuffled = np.zeros_like(self.observed_prob)
        for idx in np.flatnonzero(valid):
            shuffled[idx] = true_counts[idx][rng.permutation(width)]
        shuffled_prob, _ = _normalize_rows(shuffled)

        self.jsd_rnd = np.full(num_regions, np.nan)
        self.max_jsd = np.full(num_regions, np.nan)
        if valid.any():
            self.jsd_rnd[valid] = jensenshannon(self.observed_prob[valid], shuffled_prob[valid],
                                                axis=1, base=JSD_BASE)
            # jsd_min_max_bounds: (0, jsd(profile, uniform)). The profile it
            # normalizes is true_counts, i.e. observed_prob again.
            uniform = np.full((int(valid.sum()), width), 1.0 / width)
            self.max_jsd[valid] = jensenshannon(self.observed_prob[valid], uniform,
                                                axis=1, base=JSD_BASE)


class ProfileMetrics:
    """`profile_metrics`, fed the predicted profiles as they arrive.

    Each arriving block of rows is normalized and its per-region JSD taken
    against the observed profiles right away; `result()` then only assembles.
    The JSD is computed row by row (`jensenshannon(..., axis=1)` reduces each
    row on its own), so a row's value does not depend on which rows arrive
    with it, and the result is the single call's.
    """

    def __init__(self, true_counts, seed=DEFAULT_SEED, baseline=None):
        self._true_counts = np.asarray(true_counts)
        if baseline is not None and baseline.seed != seed:
            raise ValueError(f"the baseline was drawn with seed {baseline.seed}, not {seed}")
        self._seed = seed
        self._baseline = baseline if baseline is not None else ProfileBaseline(true_counts, seed=seed)
        num_regions = self._true_counts.shape[0]
        self._jsd_pw = np.full(num_regions, np.nan)
        self._predicted_ok = np.zeros(num_regions, dtype=bool)
        self._arrived = np.zeros(num_regions, dtype=bool)

    def add(self, start, pred_probs):
        """Predicted profiles for regions ``start`` to ``start + len(pred_probs)``."""
        rows = slice(start, start + len(pred_probs))
        predicted_prob, predicted_ok = _normalize_rows(pred_probs)
        valid = self._baseline.observed_ok[rows] & predicted_ok
        jsd_pw = np.full(len(predicted_ok), np.nan)
        if valid.any():
            jsd_pw[valid] = jensenshannon(self._baseline.observed_prob[rows][valid],
                                          predicted_prob[valid], axis=1, base=JSD_BASE)
        self._jsd_pw[rows] = jsd_pw
        self._predicted_ok[rows] = predicted_ok
        self._arrived[rows] = True

    def result(self):
        """The eight arrays `profile_metrics` returns."""
        if not self._arrived.all():
            raise ValueError(f"{int((~self._arrived).sum())} regions have no predicted profile")
        num_regions = len(self._jsd_pw)
        baseline = self._baseline
        if not np.array_equal(baseline.valid, baseline.observed_ok & self._predicted_ok):
            baseline = ProfileBaseline(self._true_counts, seed=self._seed, rows=self._predicted_ok)
        valid = baseline.valid
        # A copy: a baseline can serve several calls, and jsd_rnd is returned.
        jsd_pw, jsd_rnd, max_jsd = self._jsd_pw.copy(), baseline.jsd_rnd.copy(), baseline.max_jsd
        min_jsd = np.where(valid, 0.0, np.nan)

        jsd_norm = np.full(num_regions, np.nan)
        jsd_rnd_norm = np.full(num_regions, np.nan)
        jsd_norm[valid] = _min_max_normalize(jsd_pw[valid], min_jsd[valid], max_jsd[valid])
        jsd_rnd_norm[valid] = _min_max_normalize(jsd_rnd[valid], min_jsd[valid], max_jsd[valid])

        # The four mNLL arrays have been empty since the dead mNLL chain was
        # removed; both call sites unpack and discard them.
        empty = np.array([])
        return empty, empty, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, empty, empty


def profile_metrics(true_counts, pred_probs, pseudocount=0.001, seed=DEFAULT_SEED,
                    baseline=None):
    '''
    Get profile metrics

    ``baseline`` is a `ProfileBaseline` already computed for these observed
    counts and this seed; it is reused when the predictions keep the regions
    it drew for, and drawn again otherwise.
    '''
    # Vectorized over regions. This was a Python loop calling
    # scipy.spatial.distance.jensenshannon three times per region, which
    # profiled as the largest single cost of `pred_bw`'s finish stage.
    # `jensenshannon(..., axis=1)` is bit-identical to the per-row calls and
    # the shuffled draws reproduce the old RNG stream exactly, so every
    # returned value is unchanged; a test pins both claims.
    metrics = ProfileMetrics(true_counts, seed=seed, baseline=baseline)
    metrics.add(0, pred_probs)
    return metrics.result()

def plot_histogram(region_jsd, shuffled_labels_jsd, output_prefix, title):

    #generate histogram distributions 
    num_bins=100
    plt.rcParams["figure.figsize"]=8,8

    #plot jsd histogram
    plt.figure()
    n,bins,patches=plt.hist(region_jsd,num_bins,facecolor='blue',alpha=0.5,label="Predicted vs Labels")
    n1,bins1,patches1=plt.hist(shuffled_labels_jsd,num_bins,facecolor='black',alpha=0.5,label='Shuffled Labels vs Labels')
    plt.xlabel('Jensen Shannon Distance Profile Labels and Predictions in Probability Space')
    plt.title("JSD Dist: "+title)
    plt.legend(loc='best')
    plt.savefig(output_prefix+".profile_jsd.png",format='png',dpi=300)
    plt.close()
