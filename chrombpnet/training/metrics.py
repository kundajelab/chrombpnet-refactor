import numpy as np

from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Explicit rather than `import *`: the wildcard also re-bound numpy,
# matplotlib, plt and four scipy names over this module's own imports, so which
# definition won depended on import order.
from chrombpnet.training.utils.metrics_utils import (
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
        cur_jsd = jensenshannon(observed_prob, predicted_prob)
        jsd_pw.append(cur_jsd)
        min_jsd, max_jsd = jsd_min_max_bounds(true_counts[idx, :])
        jsd_norm.append(get_min_max_normalized_value(cur_jsd, min_jsd, max_jsd))
        shuffled_prob = normalize_profile(rng.permutation(true_counts[idx, :]))
        cur_rnd = jensenshannon(observed_prob, shuffled_prob)
        jsd_rnd.append(cur_rnd)
        jsd_rnd_norm.append(get_min_max_normalized_value(cur_rnd, min_jsd, max_jsd))
    empty = np.array([])
    return (empty, empty, np.array(jsd_pw), np.array(jsd_norm),
            np.array(jsd_rnd), np.array(jsd_rnd_norm), empty, empty)


def profile_metrics(true_counts, pred_probs, pseudocount=0.001, seed=1234):
    '''
    Get profile metrics
    '''
    # Vectorized over regions. This was a Python loop calling
    # scipy.spatial.distance.jensenshannon three times per region, which
    # profiled as the largest single cost of `pred_bw`'s finish stage.
    # `jensenshannon(..., axis=1)` is bit-identical to the per-row calls and
    # the shuffled draws below reproduce the old RNG stream exactly, so every
    # returned value is unchanged; a test pins both claims.
    true_counts = np.asarray(true_counts)
    num_regions = true_counts.shape[0]

    observed_prob, observed_ok = _normalize_rows(true_counts)
    predicted_prob, predicted_ok = _normalize_rows(pred_probs)
    valid = observed_ok & predicted_ok

    # The old loop drew one permutation per *surviving* region, in order, and
    # skipped the draw entirely for regions it discarded. `rng.permutation(x)`
    # on a 1-D array consumes the same state as `x[rng.permutation(len(x))]`
    # and returns the same values, so drawing an index permutation per valid
    # region row reproduces the stream exactly.
    rng = np.random.default_rng(seed)
    width = true_counts.shape[1]
    shuffled = np.zeros_like(observed_prob)
    for idx in np.flatnonzero(valid):
        shuffled[idx] = true_counts[idx][rng.permutation(width)]
    shuffled_prob, _ = _normalize_rows(shuffled)

    jsd_pw = np.full(num_regions, np.nan)
    jsd_rnd = np.full(num_regions, np.nan)
    max_jsd = np.full(num_regions, np.nan)
    if valid.any():
        jsd_pw[valid] = jensenshannon(observed_prob[valid], predicted_prob[valid], axis=1)
        jsd_rnd[valid] = jensenshannon(observed_prob[valid], shuffled_prob[valid], axis=1)
        # jsd_min_max_bounds: (0, jsd(profile, uniform)). The profile it
        # normalizes is true_counts, i.e. observed_prob again.
        uniform = np.full((int(valid.sum()), width), 1.0 / width)
        max_jsd[valid] = jensenshannon(observed_prob[valid], uniform, axis=1)
    min_jsd = np.where(valid, 0.0, np.nan)

    jsd_norm = np.full(num_regions, np.nan)
    jsd_rnd_norm = np.full(num_regions, np.nan)
    jsd_norm[valid] = _min_max_normalize(jsd_pw[valid], min_jsd[valid], max_jsd[valid])
    jsd_rnd_norm[valid] = _min_max_normalize(jsd_rnd[valid], min_jsd[valid], max_jsd[valid])

    # The four mNLL arrays have been empty since the dead mNLL chain was
    # removed; both call sites unpack and discard them.
    empty = np.array([])
    return empty, empty, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, empty, empty

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
