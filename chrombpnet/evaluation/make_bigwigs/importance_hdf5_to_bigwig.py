import argparse
import h5py
import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper


def read_projected_shap(path):
	"""The ``/projected_shap/seq`` dataset of a contribution-score file.

	Read with h5py rather than ``deepdish.io.load``, which returns the same
	array but no longer imports on NumPy 2 (it references the removed
	``np.ComplexWarning``). ``hdf5plugin``, when installed, registers the Blosc
	filter that score files written by the original package's deepdish use.
	"""
	try:
		import hdf5plugin  # noqa: F401
	except ImportError:
		pass
	with h5py.File(path, "r") as handle:
		return handle["projected_shap"]["seq"][()]

def import_parser():

 	# need full paths!
	parser = argparse.ArgumentParser(description="Convert importance scores in hdf5 format to bigwig. The output can be visualised using WashU Epigenome Browser as a dynseq track. Please read all parameter argument requirements. PROVIDE ABSOLUTE PATHS!")
	parser.add_argument("-h5", "--hdf5", type=str, required=True, help="HDF5 file f such that f['projected_shap']['seq'] has (N x 4 x seqlen) shape with importance score * sequence so that at each f['projected_shap']['seq'][i, :, j] has 3 zeros and 1 non-zero value")
	parser.add_argument("-r", "--regions", type=str, required=True, help="10 column BED file of length = N which matches f['projected_shap']['seq'].shape[0]. The ith region in the BED file corresponds to ith entry in importance matrix. If start=2nd col, summit=10th col, then the importance scores are assumed to be for [start+summit-(seqlen/2):start+summit+(seqlen/2)]")
	parser.add_argument("-c", "--chrom-sizes", type=str, required=True, help="Chromosome sizes 2 column tab-separated file")
	parser.add_argument("-op", "--output-prefix", type=str, required=True, help="Output prefix for bigwig file (appended with .bw)")
	parser.add_argument("-os", "--output-prefix-stats", type=str, required=False, help="Output file with stats of low and high quantiles")
	parser.add_argument("-t", "--tqdm", type=int,default=0, help="Use tqdm. If yes then you need to have it installed.")
	parser.add_argument("-d", "--debug-chr", type=str, default=None, help="Run for one chromosome only (e.g. chr12) for debugging")
	args = parser.parse_args()
	print(args)
	return args

def main(args):

	d = read_projected_shap(args.hdf5)

	SEQLEN = d.shape[2]
	if SEQLEN % 2:
		raise ValueError("Importance sequence length must be even")

	gs = bigwig_helper.read_chrom_sizes(args.chrom_sizes)
	regions = bigwig_helper.get_regions(args.regions, SEQLEN)
	if d.shape[0] != len(regions):
		raise ValueError("Importance rows and BED regions must have equal length")

	bigwig_helper.write_bigwig(d.sum(1), 
                           regions, 
                           gs, 
                           args.output_prefix+".bw", 
                           outstats_file=args.output_prefix_stats, 
                           debug_chr=args.debug_chr, 
                           use_tqdm=args.tqdm)


if __name__=="__main__":
	args=import_parser()
	main(args)
