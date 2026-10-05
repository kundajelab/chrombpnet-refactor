import chrombpnet.parsers as parsers
from chrombpnet import engines, exit_fast, fused_kernels
import os
from chrombpnet.data import DefaultDataFile, get_default_data_path
import chrombpnet.pipelines as pipelines
from chrombpnet.layout import RunLayout
import copy
import pandas as pd
import logging
import shutil
import subprocess
logging.getLogger('matplotlib.font_manager').disabled = True


#: Commands that never build or load a model, and so run in any engine's
#: environment.
MODEL_FREE_COMMANDS = ("prep", "modisco_motifs")


def runs_motif_report(args):
	"""Whether this command ends with a TF-MoDISco motif report."""
	if args.cmd == "bias":
		return getattr(args, "cmd_bias", None) in ("pipeline", "qc")
	return args.cmd in ("pipeline", "qc", "modisco_motifs")


def check_tomtom(args):
	"""Fail before any work if the motif report will need MEME's tomtom and
	it is not installed.

	The report runs at the very end, hours into a pipeline, and modiscolite
	looks tomtom up on PATH only then. shutil.which("tomtom") is exactly its
	lookup.
	"""
	if not runs_motif_report(args):
		return
	if getattr(args, "tomtom_lite", None):
		from chrombpnet.evaluation.modisco.api import supports_tomtom_lite
		if not supports_tomtom_lite():
			raise SystemExit(
				"chrombpnet: error: --tomtom-lite needs TF-MoDISco 2.5 or later, "
				"which this environment does not have (the TensorFlow engine's "
				"modisco-lite 2.3.2 is the newest release for Python 3.8). Leave out "
				"--tomtom-lite to match motifs with MEME's tomtom.")
		return
	if shutil.which("tomtom") is None:
		raise SystemExit(
			"chrombpnet: error: the motif report needs MEME's `tomtom`, which is "
			"not on PATH. Install MEME (it is in every envs/chrombpnet-*.yml, as "
			"`meme`), activate the environment that has it, or pass --tomtom-lite.")


def check_conventions(args):
	"""Fail before any work on a --data-aug / --dnase-convention combination
	the assay does not have (--dnase-convention on ATAC). Preprocessing and
	the generators resolve the same thing later, hours in for the latter."""
	if not hasattr(args, "data_type"):
		return
	if not (hasattr(args, "data_aug") or hasattr(args, "dnase_convention")):
		return
	from chrombpnet import conventions
	try:
		conventions.from_args(args)
	except ValueError as error:
		raise SystemExit(f"chrombpnet: error: {error}")


def check_modisco_window(args):
	"""Fail before training if --modisco-window is wider than the scored
	sequences, which are --inputlen long. qc takes the input length from the
	model, so there run_motifs reports it, before discovery starts."""
	window = getattr(args, "modisco_window", None)
	inputlen = getattr(args, "inputlen", None)
	if window is not None and inputlen is not None and window > inputlen:
		raise SystemExit(
			f"chrombpnet: error: --modisco-window {window} is wider than the "
			f"sequences it is applied to (--inputlen {inputlen}).")


# invoke pipeline modules based on command

def main():
	args = parsers.read_parser()
	# Before any work starts: PredictionAdapter reads this several frames
	# below, in code shared with training.
	fused_kernels.configure_from_args(args)
	# Likewise the deep-learning engine, and before any output directory is
	# created, so an engine this environment cannot import fails cleanly.
	try:
		engines.configure_from_args(
			args, require=args.cmd not in MODEL_FREE_COMMANDS)
	except (ValueError, engines.EngineUnavailable,
	        engines.EngineNotImplemented) as error:
		raise SystemExit(f"chrombpnet: error: {error}")
	check_tomtom(args)
	check_conventions(args)
	check_modisco_window(args)
	parsers.drop_absent(args)
	
	if args.cmd == "pipeline" or args.cmd == "train":
		RunLayout.from_args(args).create_training_dirs()
		pipelines.chrombpnet_train_pipeline(args)

	elif args.cmd == "qc":
		RunLayout.from_args(args).create_qc_dirs()
		pipelines.chrombpnet_qc(args)

	elif args.cmd == "bias":
		if args.cmd_bias == "pipeline" or args.cmd_bias == "train":
			RunLayout.from_args(args).create_training_dirs()
			pipelines.train_bias_pipeline(args)

		elif args.cmd_bias == "qc":
			RunLayout.from_args(args).create_qc_dirs()
			pipelines.bias_model_qc(args)

		else:
			print("Command not found")

	elif args.cmd == "predict":

		import chrombpnet.training.predict as predict
		from chrombpnet.engines import load_model_wrapper

		# Geometry off the checkpoint, as qc does; predict.main reads
		# model_h5 rather than model.
		model = load_model_wrapper(args.model, getattr(args, "device", None),
		                           announce=False)
		args.inputlen = int(model.input_shape[1])
		args.outputlen = int(model.output_shape[0][1])
		args.model_h5 = args.model
		predict.main(args)

	elif args.cmd == "export":

		from chrombpnet.engines.jax.export_legacy_h5 import export_legacy_h5
		out = export_legacy_h5(args.model_h5, args.output, count_head=args.count_head)
		print("wrote {} ({}, count head: {})".format(out, args.format, args.count_head))

	elif args.cmd == "pred_bw":
	
		if not any((args.bias_model, args.chrombpnet_model, args.chrombpnet_model_nb)):
			raise ValueError("At least one input model must be provided")
		import chrombpnet.evaluation.make_bigwigs.predict_to_bigwig as predict_to_bigwig

		outputs = predict_to_bigwig.main(args)
		# Only the CLI may end the process early: main() is also called as a
		# library function, where os._exit would kill the caller.
		if outputs:
			exit_fast.exit_now(outputs)

	elif args.cmd == "contribs_bw":
	
		import chrombpnet.evaluation.interpret.interpret as interpret
		interpret.main(args)
		import chrombpnet.evaluation.make_bigwigs.importance_hdf5_to_bigwig as importance_hdf5_to_bigwig
		if "counts" in  args.profile_or_counts:
			args_copy = copy.deepcopy(args)
			args_copy.hdf5 = args_copy.output_prefix + ".counts_scores.h5"
			args_copy.output_prefix = args.output_prefix + ".counts_scores"
			args_copy.regions =  args.output_prefix + ".interpreted_regions.bed"

			importance_hdf5_to_bigwig.main(args_copy)
		if "profile" in  args.profile_or_counts:
			args_copy = copy.deepcopy(args)
			args_copy.hdf5 = args_copy.output_prefix + ".profile_scores.h5"
			args_copy.output_prefix = args.output_prefix + ".profile_scores"
			args_copy.regions =  args.output_prefix + ".interpreted_regions.bed"
	
			importance_hdf5_to_bigwig.main(args_copy)
			
	elif args.cmd == "footprints":
	
		import chrombpnet.evaluation.marginal_footprints.marginal_footprinting as marginal_footprinting
		marginal_footprinting.main(args)

	elif args.cmd == "modisco_motifs":
		from chrombpnet.evaluation.modisco.api import generate_report, run_motifs
		meme_file=get_default_data_path(DefaultDataFile.motifs_meme)

		run_motifs(args.h5py, args.output_prefix+"_modisco.h5",
			max_seqlets=args.max_seqlets, window=args.window,
			n_leiden_runs=args.n_leiden, verbose=args.verbose)
		generate_report(args.output_prefix+"_modisco.h5",
			args.output_prefix+"_reports", meme_file,
			tomtom_lite=bool(getattr(args, "tomtom_lite", None)))
		
		import chrombpnet.evaluation.modisco.convert_html_to_pdf as convert_html_to_pdf
		convert_html_to_pdf.main(args.output_prefix+"_reports/motifs.html",args.output_prefix+"_reports/motifs.pdf")

	elif args.cmd == "prep":
	
		if args.cmd_prep == "nonpeaks":

			if args.inputlen % 2:
				raise ValueError("input length must be a multiple of 2")
	
			os.makedirs(args.output_prefix+"_auxiliary/", exist_ok=False)
	
			from chrombpnet.helpers.make_gc_matched_negatives.get_genomewide_gc_buckets.get_genomewide_gc_bins import get_genomewide_gc
			get_genomewide_gc(args.genome,args.output_prefix+"_auxiliary/genomewide_gc.bed",args.inputlen, args.stride)
	
			# get gc content in peaks
			import chrombpnet.helpers.make_gc_matched_negatives.get_gc_content as get_gc_content
			args_copy = copy.deepcopy(args)
			args_copy.input_bed = args_copy.peaks
			args_copy.output_prefix = args.output_prefix+"_auxiliary/foreground.gc"
			get_gc_content.main(args_copy)
	
			# prepare candidate negatives
	
			peaks_slop = args.output_prefix+"_auxiliary/peaks_slop.bed"
			with open(peaks_slop, "w") as output:
				subprocess.run(["bedtools", "slop", "-i", args.peaks, "-g", args.chrom_sizes,
					"-b", str(args.inputlen//2)], stdout=output, check=True)
			exclude_bed = pd.read_csv(args.output_prefix+"_auxiliary/peaks_slop.bed", sep="\t", header=None, usecols=[0,1,2])
	
			if args.blacklist_regions:
				with open(args.output_prefix+"_auxiliary/blacklist_slop.bed", "w") as output:
					subprocess.run(["bedtools", "slop", "-i", args.blacklist_regions,
						"-g", args.chrom_sizes, "-b", str(args.inputlen//2)], stdout=output, check=True)
										
				exclude_bed = pd.concat([exclude_bed,pd.read_csv(args.output_prefix+"_auxiliary/blacklist_slop.bed",sep="\t",header=None, usecols=[0,1,2])])

			exclude_bed.to_csv(args.output_prefix+"_auxiliary/exclude_unmerged.bed", sep="\t", header=False, index=False)
			sorted_exclude = args.output_prefix+"_auxiliary/exclude.sorted.bed"
			with open(sorted_exclude, "w") as output:
				subprocess.run(["bedtools", "sort", "-i", args.output_prefix+"_auxiliary/exclude_unmerged.bed"], stdout=output, check=True)
			with open(args.output_prefix+"_auxiliary/exclude.bed", "w") as output:
				subprocess.run(["bedtools", "merge", "-i", sorted_exclude], stdout=output, check=True)
	
	

			with open(args.output_prefix+"_auxiliary/candidates.bed", "w") as output:
				subprocess.run(["bedtools", "intersect", "-v",
					"-a", args.output_prefix+"_auxiliary/genomewide_gc.bed",
					"-b", args.output_prefix+"_auxiliary/exclude.bed"], stdout=output, check=True)
													
			# get final negatives
			import chrombpnet.helpers.make_gc_matched_negatives.get_gc_matched_negatives as get_gc_matched_negatives
			args_copy = copy.deepcopy(args)
			args_copy.candidate_negatives = args.output_prefix+"_auxiliary/candidates.bed"
			args_copy.foreground_gc_bed = args.output_prefix+"_auxiliary/foreground.gc.bed"
			args_copy.output_prefix = 	args.output_prefix+"_auxiliary/negatives"
	
			get_gc_matched_negatives.main(args_copy)
	
			negatives = pd.read_csv(args.output_prefix+"_auxiliary/negatives.bed", sep="\t", header=None)
			negatives[3]="."
			negatives[4]="."
			negatives[5]="."
			negatives[6]="."
			negatives[7]="."
			negatives[8]="."
			negatives[9]=args.inputlen//2
			negatives.to_csv(args.output_prefix+"_negatives.bed", sep="\t", header=False, index=False)

		elif args.cmd_prep == "splits":
			import chrombpnet.helpers.make_chr_splits.splits as splits
			splits.main(args)
			
		else:
			print("Command not found")
		
	else:
		print("Command not found")


if __name__=="__main__":
	main()

    
        
