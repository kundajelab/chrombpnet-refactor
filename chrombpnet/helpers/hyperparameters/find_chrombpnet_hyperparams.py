import argparse
import pandas as pd
import numpy as np
from chrombpnet.training.utils.benchmark_utils import benchmark_enables
from chrombpnet.helpers.hyperparameters import param_utils as param_utils
from chrombpnet.helpers.hyperparameters import common
from chrombpnet import engines

def parse_data_args():
    parser=argparse.ArgumentParser(description="find hyper-parameters for the architecture in chrombpnet/training/models/chrombpnet_with_bias_model.py")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-i", "--bigwig", type=str, required=True, help="Bigwig of tn5 insertions. Ensure it is +4/-4 shifted")
    parser.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    parser.add_argument("-n", "--nonpeaks", type=str, required=True, help="10 column bed file of non-peak regions, centered at summit (10th column)")
    parser.add_argument("-sr", "--negative_sampling_ratio", type=float, default=0.1, help="Ratio of negatives to positive samples per epoch")
    parser.add_argument("-oth", "--outlier-threshold", type=float, default=0.9999, help="threshold to use to filter outlies")
    parser.add_argument("-j", "--max-jitter", type=int, required=True, default=500, help="Maximum jitter applied on either side of region (default 500 for chrombpnet model)")
    parser.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    return parser

def parse_model_args(parser):
    # arguments here define the model in chrombpnet/training/models/chrombpnet_with_bias_model.py
    parser.add_argument("-il", "--inputlen", type=int, required=True, help="Sequence input length")
    parser.add_argument("-ol", "--outputlen", type=int, required=True, help="Prediction output length")
    parser.add_argument("-fil", "--filters", type=int, default=512, help="Number of filters to use in chrombpnet mode")
    parser.add_argument("-dil", "--n-dilation-layers", type=int, default=8, help="Number of dilation layers to use in chrombpnet model")
    parser.add_argument("-b", "--bias-model-path", type=str, required=True, help="path of bias model")
    parser.add_argument("-op", "--output-prefix", help="output prefix for storing hyper-param TSV for chrombpnet")
    args = parser.parse_args()
    return args

def main(args): 

    # read the fold information - we will evaluate hyperparams on the train+valid set and do nothing on the test set 
    inputs = common.FoldInputs(args)
    bw, genome = inputs.bw, inputs.genome
    peaks, test_peaks = inputs.peaks, inputs.test_peaks
    nonpeaks, test_nonpeaks = inputs.nonpeaks, inputs.test_nonpeaks

    # step 1 filtering: filter peaks that are in the edges - which prevents us from making the inputlen regions - do this for all splits
    peaks = param_utils.filter_edge_regions(peaks, bw, args.inputlen+2*args.max_jitter, peaks_bool=1)
    test_peaks = param_utils.filter_edge_regions(test_peaks, bw, args.inputlen, peaks_bool=1)
   
    nonpeaks = param_utils.filter_edge_regions(nonpeaks, bw, args.inputlen, peaks_bool=0)
    test_nonpeaks = param_utils.filter_edge_regions(test_nonpeaks, bw, args.inputlen, peaks_bool=0)

    # step 2 filtering: filter peaks that are outliers in train and valid set - no filtering on test set
    peak_cnts, _ = param_utils.get_seqs_cts(genome, bw, peaks, args.inputlen, args.outputlen)
    nonpeak_cnts, nonpeak_seqs = param_utils.get_seqs_cts(genome, bw, nonpeaks, args.inputlen, args.outputlen)    
    if len(peak_cnts) != peaks.shape[0] or len(nonpeak_cnts) != nonpeaks.shape[0]:
        raise RuntimeError("Extracted count arrays do not match their BED regions")


    if args.negative_sampling_ratio > 0:
        sample_size = min(int(args.negative_sampling_ratio * len(peak_cnts)), len(nonpeak_cnts))
        # A seeded draw keeps the count threshold, counts_loss_weight and the
        # filtered BEDs byte-identical between the two frameworks.
        rng = np.random.RandomState(args.seed) if benchmark_enables(args, "seeded_threshold_draw") else np.random
        final_cnts = np.concatenate((peak_cnts, rng.choice(nonpeak_cnts, replace=False, size=sample_size)))
    else:
        final_cnts = peak_cnts

    upper_thresh = np.quantile(final_cnts, args.outlier_threshold)
    lower_thresh = np.quantile(final_cnts, 1-args.outlier_threshold)

    peaks = peaks[(peak_cnts< upper_thresh) & (peak_cnts>lower_thresh)]
    nonpeaks = nonpeaks[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)]

    print("Number of peaks after removing outliers: ", peaks.shape[0])
    print("Number of nonpeaks after removing outliers: ", nonpeaks.shape[0])

    # combine train valid and test peak set and store them in a new file
    if nonpeaks.shape[0] > peaks.shape[0]:
        train_nonpeaks = nonpeaks.sample(n=peaks.shape[0], random_state=1)
    else:
        train_nonpeaks = nonpeaks
        
    frames = [peaks, test_peaks]
    all_peaks = pd.concat(frames)
    all_peaks.to_csv("{}filtered.peaks.bed".format(args.output_prefix), sep="\t",  header=False, index=False)
    frames = [train_nonpeaks, test_nonpeaks]
    all_nonpeaks = pd.concat(frames)
    all_nonpeaks.to_csv("{}filtered.nonpeaks.bed".format(args.output_prefix), sep="\t", header=False, index=False)

    # find counts loss weight for model training - using train and validation set
    # Inclusive bounds here, strict in find_bias_hyperparams; see
    # common.counts_loss_weight.
    counts_loss_weight = common.counts_loss_weight(
        final_cnts, lower_thresh, upper_thresh, inclusive=True)

    # adjust bias model for training  - using train and validation set
    # the bias model might be trained on a difference read depth compared to the given data - so this step scales the bias model to account for that
    # The weight surgery is framework-specific, so the engine that wrote the
    # bias model does it and saves the result in its own checkpoint format.
    engine = engines.for_model(args.bias_model_path)
    scaled_bias_model_path = "{}bias_model_scaled{}".format(
        args.output_prefix, engine.checkpoint_suffix)
    engine.rescale_bias_logcounts(
        args.bias_model_path, scaled_bias_model_path,
        nonpeak_seqs[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)],
        nonpeak_cnts[(nonpeak_cnts< upper_thresh) & (nonpeak_cnts>lower_thresh)])

    # store the parameters being used  - in a TSV file
    common.write_params_tsv(
        "{}chrombpnet_data_params.tsv".format(args.output_prefix),
        [("counts_sum_min_thresh", round(lower_thresh, 2)),
         ("counts_sum_max_thresh", round(upper_thresh, 2)),
         ("trainings_pts_post_thresh",
          sum((final_cnts < upper_thresh) & (final_cnts > lower_thresh)))],
        trailing_newline=True)

    common.write_params_tsv(
        "{}chrombpnet_model_params.tsv".format(args.output_prefix),
        [("counts_loss_weight", round(counts_loss_weight, 2)),
         ("filters", args.filters),
         ("n_dil_layers", args.n_dilation_layers),
         ("bias_model_path", scaled_bias_model_path),
         ("inputlen", args.inputlen),
         ("outputlen", args.outputlen),
         ("max_jitter", args.max_jitter),
         ("chr_fold_path", args.chr_fold_path),
         ("negative_sampling_ratio", args.negative_sampling_ratio)],
        trailing_newline=False)


if __name__=="__main__":
    # read the arguments
    parser = parse_data_args()
    args = parse_model_args(parser)

    main(args)

    
