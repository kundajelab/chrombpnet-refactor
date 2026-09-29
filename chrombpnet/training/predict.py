import numpy as np
import json
import h5py
import chrombpnet.training.metrics as metrics
import chrombpnet.training.data_generators.initializers as initializers
from chrombpnet.model import load_model_wrapper
from chrombpnet.predictions import softmax
from chrombpnet.regions import is_disabled
from chrombpnet.predictions import (
    write_predictions_h5py as _write_predictions_h5py,
)

def write_predictions_h5py(output_prefix, profile, logcts, coords):
    """Write test-set predictions, including the peak/nonpeak flag.

    Kept as a module-level function with this exact signature because it is
    public API: the cross-framework parity harness in test.py imports it for
    both the TensorFlow and PyTorch workers, and two test modules call it
    directly. Test-set coordinates always carry the peak indicator in column 4,
    so this path always writes coords_peak.
    """
    _write_predictions_h5py(output_prefix, profile, logcts, coords,
                            include_peak_flag=True)


def predict_on_batch_wrapper(model,test_generator):
    num_batches=len(test_generator)
    profile_probs_predictions = []
    true_counts = []
    counts_sum_predictions = []
    true_counts_sum = []
    coordinates = []

    for idx in range(num_batches):
        if idx%100==0:
            print(str(idx)+'/'+str(num_batches))
        
        X,y,coords=test_generator[idx]

        #get the model predictions            
        preds=model.predict_on_batch(X)

        # get counts predictions
        true_counts.extend(y[0])
        profile_probs_predictions.extend(softmax(preds[0]))

        # get profile predictions
        true_counts_sum.extend(y[1][:,0])
        counts_sum_predictions.extend(preds[1][:,0])
        coordinates.extend(coords)

    return np.array(true_counts), np.array(profile_probs_predictions), np.array(true_counts_sum), np.array(counts_sum_predictions), np.array(coordinates)


def _record_subset_metrics(metrics_dictionary, name, suffix, label, mask,
                           true_counts_sum, counts_sum_predictions,
                           jsd_pw, jsd_norm, jsd_rnd, output_prefix,
                           profile_cor):
    """Score one subset of the test set and add it to the metrics dictionary.

    Replaces three copies of the same twelve lines, which differed only in the
    subset mask, the output-file suffix and the plot label. Each subset also
    writes its own JSD histogram, so the caller's order of subsets determines
    which PNGs appear -- it is preserved.
    """
    spearman_cor, pearson_cor, mse = metrics.counts_metrics(
        true_counts_sum[mask], counts_sum_predictions[mask],
        output_prefix + suffix, label)
    metrics_dictionary["counts_metrics"][name] = {
        "spearmanr": spearman_cor,
        "pearsonr": pearson_cor,
        "mse": mse,
    }
    metrics_dictionary["profile_metrics"][name] = {
        "median_jsd": np.nanmedian(jsd_pw[mask]),
        "median_norm_jsd": np.nanmedian(jsd_norm[mask]),
    }
    valid_cor = profile_cor[mask]
    valid_cor = valid_cor[np.isfinite(valid_cor)]
    metrics_dictionary["profile_metrics"][name]["median_pearsonr"] = (
        float(np.median(valid_cor)) if valid_cor.size else None)
    metrics.plot_histogram(jsd_pw[mask], jsd_rnd[mask],
                           output_prefix + suffix, label)


def main(args):
    metrics_dictionary = {"counts_metrics": {}, "profile_metrics": {}}

    model = load_model_wrapper(
        args.model_h5, getattr(args, "device", None), summary=False)

    test_generator = initializers.initialize_generators(
        args, mode="test", parameters=None, return_coords=True)
    (true_counts, profile_probs_predictions, true_counts_sum,
     counts_sum_predictions, coordinates) = predict_on_batch_wrapper(
        model, test_generator)

    write_predictions_h5py(args.output_prefix, profile_probs_predictions,
                           counts_sum_predictions, coordinates)

    _, _, jsd_pw, jsd_norm, jsd_rnd, _, _, _ = metrics.profile_metrics(
        true_counts, profile_probs_predictions)
    profile_cor = metrics.profile_pearsonr(true_counts, profile_probs_predictions)

    # Same sentinel test the generator uses. These used to compare
    # case-sensitively against "None" while initializers lowercased, so a
    # lowercase "none" made predict score peak metrics over regions the
    # generator had excluded.
    has_peaks = not is_disabled(args.peaks)
    has_nonpeaks = not is_disabled(args.nonpeaks)
    everything = np.ones(len(coordinates), dtype=bool)

    # Order matters: each subset writes its own histogram PNG.
    subsets = []
    if has_peaks and has_nonpeaks:
        subsets.append(("peaks_and_nonpeaks", "_peaks_and_nonpeaks",
                        "Both peaks and non peaks", everything))
    if has_nonpeaks:
        subsets.append(("nonpeaks", "_only_nonpeaks", "Only non peaks",
                        coordinates[:, 3] == '0'))
    if has_peaks:
        subsets.append(("peaks", "_only_peaks", "Only peaks",
                        coordinates[:, 3] == '1'))

    for name, suffix, label, mask in subsets:
        _record_subset_metrics(
            metrics_dictionary, name, suffix, label, mask,
            true_counts_sum, counts_sum_predictions,
            jsd_pw, jsd_norm, jsd_rnd, args.output_prefix, profile_cor)

    with open(args.output_prefix + '_metrics.json', 'w') as fp:
        json.dump(metrics_dictionary, fp, indent=4)
