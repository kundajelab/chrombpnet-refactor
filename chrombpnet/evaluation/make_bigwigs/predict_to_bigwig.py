import argparse
import pyBigWig
import numpy as np
import pandas as pd
import pyfaidx
import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
import chrombpnet.training.utils.data_utils as data_utils 
from chrombpnet.engines import load_model_wrapper
from chrombpnet.predictions import softmax, write_predictions_h5py
from chrombpnet import finish_stage, preimport
from chrombpnet.regions import NARROWPEAK_SCHEMA
import contextlib
import h5py
import json


def read_observed_counts(bigwig, regions_df, outputlen):
	"""Per-base observed counts for each region, from the observed bigwig."""
	with contextlib.closing(pyBigWig.open(bigwig)) as obs_bw:
		return data_utils.get_cts(regions_df, obs_bw, outputlen)


def compare_with_observed(bigwig, regions_df, regions, outputlen, pred_logits, pred_logcts, output_prefix, obs_data=None):

	import chrombpnet.training.metrics as metrics 
	
	if obs_data is None:
		obs_data = read_observed_counts(bigwig, regions_df, outputlen)
	
	true_counts = obs_data
	true_counts_sum = np.log(np.sum(true_counts, axis=-1)+1)
	profile_probs_predictions = softmax(pred_logits) ##
	counts_sum_predictions = np.asarray(pred_logcts).reshape(-1)
	coordinates =  [[r[0], r[-1]] for r in regions]
	
	write_predictions_h5py(output_prefix, profile_probs_predictions,
		counts_sum_predictions, coordinates, include_peak_flag=False)
	
	# store regions, their predictions and corresponding pointwise metrics
	mnll_pw, mnll_norm, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, mnll_rnd, mnll_rnd_norm =  metrics.profile_metrics(true_counts,profile_probs_predictions)

	spearman_cor, pearson_cor, mse = metrics.counts_metrics(true_counts_sum, counts_sum_predictions, output_prefix, "All regions provided")
	
	metrics_dictionary={}
	metrics_dictionary["counts_metrics"] = {}
	metrics_dictionary["profile_metrics"] = {}
	metrics_dictionary["counts_metrics"]["regions"] = {}
	metrics_dictionary["counts_metrics"]["regions"]["spearmanr"] = spearman_cor
	metrics_dictionary["counts_metrics"]["regions"]["pearsonr"] = pearson_cor
	metrics_dictionary["counts_metrics"]["regions"]["mse"] = mse
	
	metrics_dictionary["profile_metrics"]["regions"] = {}
	metrics_dictionary["profile_metrics"]["regions"]["median_jsd"] = np.nanmedian(jsd_pw)
	metrics_dictionary["profile_metrics"]["regions"]["median_norm_jsd"] = np.nanmedian(jsd_norm)
	profile_cor = metrics.profile_pearsonr(true_counts, profile_probs_predictions)
	valid_cor = profile_cor[np.isfinite(profile_cor)]
	metrics_dictionary["profile_metrics"]["regions"]["median_pearsonr"] = (
		float(np.median(valid_cor)) if valid_cor.size else None)
	
	metrics.plot_histogram(jsd_pw, jsd_rnd, output_prefix, "All regions provided")
	
	with open(output_prefix+'_metrics.json', 'w') as fp:
		json.dump(metrics_dictionary, fp,  indent=4)


# need full paths!
def parse_args():
    parser = argparse.ArgumentParser(description="Make model predictions on given regions and output to bigwig file.Please read all parameter argument requirements. PROVIDE ABSOLUTE PATHS!")
    parser.add_argument("-bm", "--bias-model", type=str, default=None, required=False, help="Path to bias model h5")
    parser.add_argument("-cm", "--chrombpnet-model", type=str, default=None, required=False, help="Path to chrombpnet model h5")
    parser.add_argument("-cmb", "--chrombpnet-model-nb", type=str, default=None, required=False, help="Path to chrombpnet model h5")
    parser.add_argument("-r", "--regions", type=str, required=True, help="10 column BED file of length = N which matches f['projected_shap']['seq'].shape[0]. The ith region in the BED file corresponds to ith entry in importance matrix. If start=2nd col, summit=10th col, then the input regions are assumed to be for [start+summit-(inputlen/2):start+summit+(inputlen/2)]. Should not be piped since it is read twice!")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-c", "--chrom-sizes", type=str, required=True, help="Chromosome sizes 2 column tab-separated file")
    parser.add_argument("-op", "--output-prefix", type=str, required=True, help="Output bigwig file")
    parser.add_argument("-os", "--output-prefix-stats", type=str, default=None, required=False, help="Output stats on bigwig")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-t", "--tqdm", type=int,default=0, help="Use tqdm. If yes then you need to have it installed.")
    parser.add_argument("-d", "--debug-chr", nargs="+", type=str, default=None, help="Run for specific chromosomes only (e.g. chr1 chr2) for debugging")
    parser.add_argument("-bw", "--bigwig", type=str, default=None, help="If provided .h5 with predictions are output along with calculated metrics considering bigwig as groundtruth.")
    parser.add_argument("--device", default=None, help="PyTorch device (default: CUDA when available)")
    args = parser.parse_args()
    if not any((args.bias_model, args.chrombpnet_model, args.chrombpnet_model_nb)):
        parser.error("at least one input model must be provided")
    print(args)
    return args


def _stats_path(output_prefix_stats, model_name):
    if output_prefix_stats is None:
        return None
    return f"{output_prefix_stats}_{model_name}.txt"

class _SharedInputs:
    """Everything the three model passes would otherwise recompute per model.

    ``main`` runs ``_write_model_bigwig`` up to three times. Only the model
    changes between passes -- the region file, the chromosome sizes, the
    sequences and the observed counts are functions of the arguments and of
    the model *geometry*, and for the standard bias/ChromBPNet pair the
    geometry is identical too. Each entry is therefore keyed by exactly what
    it depends on, so a bias model with a different input length still gets
    its own sequences rather than silently borrowing the accessibility
    model's.

    The observed counts are handed out read-only: three passes now share one
    array, and an in-place write by any of them would corrupt the other two.
    """

    def __init__(self, args, genome):
        self._args = args
        self._genome = genome
        self._regions_df = None
        self._chrom_sizes = None
        self._sequences = {}
        self._regions = {}
        self._observed = {}

    def regions_df(self):
        if self._regions_df is None:
            frame = pd.read_csv(self._args.regions, sep='\t', names=NARROWPEAK_SCHEMA)
            if self._args.debug_chr is not None:
                # Filter before extracting sequences, not after: comparing the
                # list-valued --debug-chr to a single chromosome string is
                # defect 2 in chrombpnet_bugs.md.
                frame = frame[frame['chr'].isin(self._args.debug_chr)].copy()
            if frame.empty:
                raise ValueError("No prediction regions remain after chromosome filtering")
            self._regions_df = frame
        return self._regions_df

    def chrom_sizes(self):
        if self._chrom_sizes is None:
            self._chrom_sizes = bigwig_helper.read_chrom_sizes(self._args.chrom_sizes)
        return self._chrom_sizes

    def sequences(self, inputlen):
        if inputlen not in self._sequences:
            self._sequences[inputlen] = bigwig_helper.get_seq(
                self.regions_df(), self._genome, inputlen)
        return self._sequences[inputlen]

    def regions(self, inputlen, outputlen):
        # regions_used comes from get_seq at inputlen, so both lengths key this.
        key = (inputlen, outputlen)
        if key not in self._regions:
            _, regions_used = self.sequences(inputlen)
            self._regions[key] = bigwig_helper.get_regions(
                self.regions_df(), outputlen, regions_used)
        return self._regions[key]

    def observed_counts(self, used_regions_df, inputlen, outputlen):
        key = (inputlen, outputlen)
        if key not in self._observed:
            counts = read_observed_counts(self._args.bigwig, used_regions_df, outputlen)
            counts.flags.writeable = False
            self._observed[key] = counts
        return self._observed[key]


def _write_model_bigwig(args, model_path, name, shared, finish):
    """Predict over the regions BED with one model and write its bigwig.

    This was three ~40-line copies -- roughly 120 of the module's 256 lines --
    one per model. They differed only incidentally: the bias copy called
    ``predict(seqs)`` where the others passed ``[seqs]`` (PredictionAdapter
    unwraps a single-element list, so both are the same call), and it ordered
    the BED write before reading chrom sizes rather than after. Those three
    statements are independent, so the merged order is observationally the
    same.

    ``name`` supplies the suffix for every output: ``<prefix>_<name>_preds.bed``,
    ``<prefix>_<name>.bw``, and the model-specific stats filename that keeps the
    three models from overwriting each other's statistics (defect 5 in
    chrombpnet_bugs.md).
    """
    model = load_model_wrapper(model_path, getattr(args, "device", None))
    inputlen = int(model.input_shape[1])
    outputlen = int(model.output_shape[0][1])

    regions_df = shared.regions_df()
    print(regions_df.head())

    seqs, regions_used = shared.sequences(inputlen)
    gs = shared.chrom_sizes()
    regions = shared.regions(inputlen, outputlen)
    # Everything downstream is indexed by the in-bounds regions only: get_seq
    # drops any region whose window runs off the chromosome, get_regions
    # applies the same mask, and the predictions follow from those sequences.
    used_regions_df = regions_df.loc[regions_used]
    used_regions_df.to_csv(
        f"{args.output_prefix}_{name}_preds.bed", sep="\t",
        header=False, index=False)

    pred_logits, pred_logcts = model.predict(
        [seqs], batch_size=args.batch_size, verbose=True)
    # Reshape with an explicit batch dimension; an unrestricted squeeze
    # collapses a single region's (1, outputlen) profile to one dimension
    # (defect 3 in chrombpnet_bugs.md).
    pred_logits = np.asarray(pred_logits).reshape(len(seqs), -1)

    # Hand the bigwig write and the metrics to the finish stage, which runs
    # them alongside the *next* model's forward. Both were pure CPU work
    # sitting behind an idle GPU.
    #
    # The filtered frame, not regions_df: compare_with_observed reads observed
    # counts per row and lines them up against the predictions, which cover
    # only the in-bounds regions. Passing the unfiltered frame made the two
    # disagree in length whenever a region sat near a chromosome edge.
    finish.submit({
        # expm1, because the training target is log(1 + counts) (defect 4).
        "data": softmax(pred_logits) * np.expm1(pred_logcts),
        "regions": regions,
        "chrom_sizes": gs,
        "bigwig_out": f"{args.output_prefix}_{name}.bw",
        "outstats_file": _stats_path(args.output_prefix_stats, name),
        "debug_chr": args.debug_chr,
        "use_tqdm": args.tqdm,
        "observed_bigwig": args.bigwig,
        "used_regions_df": used_regions_df,
        "outputlen": outputlen,
        "pred_logits": pred_logits,
        "pred_logcts": pred_logcts,
        "output_prefix": f"{args.output_prefix}_{name}",
        "obs_data": (shared.observed_counts(used_regions_df, inputlen, outputlen)
                     if args.bigwig else None),
    })


def _expected_outputs(args, models):
    """Every file this invocation promised to write.

    `exit_fast` refuses to skip teardown unless all of these are present,
    non-empty and synced, so the list has to be complete rather than
    indicative -- a missed entry is a file that could be silently truncated.
    """
    paths = []
    for _, name in models:
        prefix = f"{args.output_prefix}_{name}"
        paths.append(f"{prefix}.bw")
        paths.append(f"{prefix}_preds.bed")
        stats = _stats_path(args.output_prefix_stats, name)
        if stats:
            paths.append(stats)
        if args.bigwig:
            paths.append(f"{prefix}_predictions.h5")
            paths.append(f"{prefix}_metrics.json")
    return paths


def main(args):
    # Order preserved from the original three inlined blocks.
    models = [(path, name) for path, name in
              ((args.chrombpnet_model_nb, "chrombpnet_nobias"),
               (args.chrombpnet_model, "chrombpnet"),
               (args.bias_model, "bias")) if path]

    # Fork the finish worker FIRST, before any model is loaded: forking a
    # process that has already initialized CUDA yields a child with unusable
    # CUDA state. The worker never touches the GPU, but it inherits whatever
    # the parent had at fork time.
    with finish_stage.start(len(models)) as finish:
        if args.bigwig:
            # The metrics stage imports scipy and matplotlib, ~6s that would
            # otherwise be paid with the GPU idle. Start them now so they
            # land under the forward.
            preimport.start()
        # The FASTA is opened once for the whole invocation, not per model.
        with pyfaidx.Fasta(args.genome) as genome:
            # One cache across all models: the regions, the featurization
            # and the observed-count read are identical for each, and doing
            # them per model was two thirds of that work wasted.
            shared = _SharedInputs(args, genome)
            for model_path, name in models:
                _write_model_bigwig(args, model_path, name, shared, finish)

    # Return, do not exit. `main` is called as a library function -- by the
    # tests, and by anything embedding the package -- so terminating the
    # process here would kill the caller. The CLI entry point owns that
    # decision; see chrombpnet.CHROMBPNET.main.
    return _expected_outputs(args, models)


if __name__=="__main__":
    args = parse_args()
    main(args)
