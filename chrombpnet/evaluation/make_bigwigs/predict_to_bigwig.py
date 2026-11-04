import argparse
import pyBigWig
import numpy as np
import pandas as pd
import pyfaidx
import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
import chrombpnet.training.utils.data_utils as data_utils 
from chrombpnet.engines import load_model_wrapper
from chrombpnet.predictions import softmax, write_predictions_h5py
from chrombpnet import finish_stage
from chrombpnet.regions import NARROWPEAK_SCHEMA
from chrombpnet.seed import DEFAULT_SEED
import contextlib
import h5py
import json
from concurrent.futures import ThreadPoolExecutor

#: Most regions per forward call in pred_bw, rounded down to whole batches:
#: each call's predictions go to the finish stage as soon as they exist. The
#: optimization kit's pipelined predict uses chunks of 4,096.
STREAM_ROWS = 4096


def read_observed_counts(bigwig, regions_df, outputlen):
	"""Per-base observed counts for each region, from the observed bigwig."""
	with contextlib.closing(pyBigWig.open(bigwig)) as obs_bw:
		return data_utils.get_cts(regions_df, obs_bw, outputlen)


def compare_with_observed(bigwig, regions_df, regions, outputlen, pred_logits, pred_logcts, output_prefix, obs_data=None,
                          seed=DEFAULT_SEED):
	"""Write ``<output_prefix>_predictions.h5`` and ``_metrics.json`` (and the
	two plots) for predictions over ``regions`` against the observed bigwig."""
	comparison = ObservedComparison(bigwig, regions_df, regions, outputlen, output_prefix,
	                                obs_data=obs_data, seed=seed)
	comparison.add(0, pred_logits, pred_logcts)
	comparison.close()


class ObservedComparison:
	"""``compare_with_observed``, fed predictions as they are made.

	Adapted from the optimization kit's ``metrics_overlap``
	(github.com/anthropics/uplifting-biomolecular-modeling, Apache-2.0): what
	needs only the observed signal -- reading it, and the shuffled baseline
	with its seeded draws (``metrics.ProfileBaseline``) -- is done on
	construction, before any prediction exists, and each arriving chunk gets
	its softmax, its per-region JSD (``metrics.ProfileMetrics``) and its
	per-region profile Pearson R. ``close`` then does what needs every
	prediction: the HDF5 (on a thread, beside the plots), the counts metrics,
	the two plots and the JSON. The kit also deflates the HDF5 chunks as the
	rows arrive; here the HDF5 is written whole, as before.

	Every value is the one ``compare_with_observed`` computed before: the
	softmax, the JSD and the Pearson R are each computed row by row, so a
	chunk's rows come out as they would from the whole array, and the
	baseline is used only when the predictions leave it valid (see
	``ProfileBaseline``).
	"""

	def __init__(self, bigwig, regions_df, regions, outputlen, output_prefix, obs_data=None,
	             seed=DEFAULT_SEED, baseline=None):
		import chrombpnet.training.metrics as metrics

		if obs_data is None:
			obs_data = read_observed_counts(bigwig, regions_df, outputlen)
		self._true_counts = obs_data
		self._regions = regions
		self._output_prefix = output_prefix
		self._seed = seed
		self._profile = metrics.ProfileMetrics(obs_data, seed=seed, baseline=baseline)
		self._profile_cor = np.full(len(regions), np.nan)
		self._probs = None
		self._logcts = None
		self._arrived = np.zeros(len(regions), dtype=bool)

	def add(self, start, pred_logits, pred_logcts):
		"""Predictions for regions ``start`` to ``start + len(pred_logits)``."""
		import chrombpnet.training.metrics as metrics

		probs = softmax(pred_logits) ##
		logcts = np.asarray(pred_logcts)
		stop = start + len(probs)
		if start < 0 or stop > len(self._regions):
			raise ValueError(f"rows {start}:{stop} are outside the {len(self._regions)} regions")
		if self._probs is None and start == 0 and stop == len(self._regions):
			self._probs, self._logcts = probs, logcts
		else:
			if self._probs is None:
				self._probs = np.empty((len(self._regions),) + probs.shape[1:], dtype=probs.dtype)
				self._logcts = np.empty((len(self._regions),) + logcts.shape[1:], dtype=logcts.dtype)
			self._probs[start:stop] = probs
			self._logcts[start:stop] = logcts
		self._profile.add(start, probs)
		self._profile_cor[start:stop] = metrics.profile_pearsonr(self._true_counts[start:stop], probs)
		self._arrived[start:stop] = True

	def close(self):
		if not self._arrived.all():
			raise ValueError(f"{int((~self._arrived).sum())} of {len(self._arrived)} regions "
			                 f"have no prediction")
		output_prefix = self._output_prefix
		true_counts = self._true_counts
		true_counts_sum = np.log(np.sum(true_counts, axis=-1)+1)
		profile_probs_predictions = self._probs
		counts_sum_predictions = self._logcts.reshape(-1)
		coordinates =  [[r[0], r[-1]] for r in self._regions]

		# The HDF5 on a thread, beside everything below: its compression runs
		# in zlib, which releases the GIL, and its file is written by nothing
		# else.
		with ThreadPoolExecutor(1) as pool:
			h5 = pool.submit(write_predictions_h5py, output_prefix, profile_probs_predictions,
				counts_sum_predictions, coordinates, include_peak_flag=False)
			self._metrics_files(true_counts, true_counts_sum, counts_sum_predictions)
			h5.result()

	def _metrics_files(self, true_counts, true_counts_sum, counts_sum_predictions):
		import chrombpnet.training.metrics as metrics

		output_prefix = self._output_prefix
		# store regions, their predictions and corresponding pointwise metrics
		mnll_pw, mnll_norm, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, mnll_rnd, mnll_rnd_norm = self._profile.result()

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
		profile_cor = self._profile_cor
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
    sequences are functions of the arguments and of the model *geometry*, and for the standard bias/ChromBPNet pair the
    geometry is identical too. Each entry is therefore keyed by exactly what
    it depends on, so a bias model with a different input length still gets
    its own sequences rather than silently borrowing the accessibility
    model's. (The observed counts, which used to be cached here too, are read
    and cached by the finish stage's metrics worker; see finish_stage.)
    """

    def __init__(self, args, genome):
        self._args = args
        self._genome = genome
        self._regions_df = None
        self._chrom_sizes = None
        self._sequences = {}
        self._regions = {}

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

    # Open the model's finish work before the forward, so the metrics worker
    # reads the observed signal and draws the shuffled baseline while the GPU
    # works. The filtered frame, not regions_df: the observed counts are read
    # per row and lined up against the predictions, which cover only the
    # in-bounds regions. Passing the unfiltered frame made the two disagree in
    # length whenever a region sat near a chromosome edge.
    stream = finish.open({
        "name": name,
        "regions": regions,
        "chrom_sizes": gs,
        "bigwig_out": f"{args.output_prefix}_{name}.bw",
        "outstats_file": _stats_path(args.output_prefix_stats, name),
        "debug_chr": args.debug_chr,
        "use_tqdm": args.tqdm,
        "observed_bigwig": args.bigwig,
        "used_regions_df": used_regions_df,
        "geometry": (inputlen, outputlen),
        "outputlen": outputlen,
        "output_prefix": f"{args.output_prefix}_{name}",
        "seed": getattr(args, "seed", DEFAULT_SEED),
    })

    # The forward in runs of whole batches, each handed to the finish stage
    # as it lands, so the bigwig and the metrics are written while this same
    # model is still predicting -- not only while the next one is, which left
    # a single-model run with all of it after the forward. The batches are the
    # ones a single predict call makes, so every prediction is unchanged; they
    # run in the order the bigwig is written in (bigwig_helper.stream_plan).
    for start, stop in bigwig_helper.stream_plan(regions, gs, args.batch_size, STREAM_ROWS):
        # verbose off: one progress bar per call would be dozens of bars.
        pred_logits, pred_logcts = model.predict(
            [seqs[start:stop]], batch_size=args.batch_size, verbose=False)
        if pred_logcts is None:
            raise ValueError(
                f"{model_path} is a profile-only bias model: it predicts no counts, "
                f"so there is no counts-scaled prediction bigwig to write")
        # Reshape with an explicit batch dimension; an unrestricted squeeze
        # collapses a single region's (1, outputlen) profile to one dimension
        # (defect 3 in chrombpnet_bugs.md).
        pred_logits = np.asarray(pred_logits).reshape(len(pred_logits), -1)
        stream.add(start, pred_logits, np.asarray(pred_logcts))
    stream.close()


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

    # Fork the finish workers FIRST, before any model is loaded: forking a
    # process that has already initialized CUDA yields a child with unusable
    # CUDA state. The workers never touch the GPU, but they inherit whatever
    # the parent had at fork time.
    with finish_stage.start() as finish:
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
