import argparse

from chrombpnet import conventions, engines
from chrombpnet.training.utils.benchmark_utils import add_benchmark_argument
from argparse import RawTextHelpFormatter


desc = """==================================================================================================
		Bias factorized, base-resolution deep learning models of chromatin accessibility reveal 
		cis-regulatory sequence syntax, transcription factor footprints and regulatory variants
		=================================================================================================="""

def ylim_pair(text):
    """Parse ``--ylim`` as a numeric (lower, upper) pair.

    The argument was declared ``type=tuple``, which makes argparse call
    ``tuple()`` on the raw string: ``"(0,0.8)"`` became
    ``('(', '0', ',', '0', '.', '8', ')')``, a 7-element tuple of characters
    that matplotlib's set_ylim rejects. The flag could not be used in the form
    its own help text documents.

    Accepts the documented ``(0,0.8)`` as well as a bare ``0,0.8``.
    """
    cleaned = text.strip().strip("()[] ")
    parts = [part for part in cleaned.replace(",", " ").split() if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected two comma-separated numbers such as (0,0.8), got {text!r}")
    try:
        lower, upper = (float(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"y-limits must be numbers, got {text!r}") from None
    return (lower, upper)

# ---------------------------------------------------------------------------
# Reusable argument groups.
#
# Thirty-one argument declarations were repeated across the twelve subparsers:
# -ibs/--interpret-batch-size seven times with identical help text, --device
# and -bs six times each, and the seven-line genomics input block three times.
# The declarations below are the single source for each. Order within a group
# is preserved exactly, because it is the order --help prints.
# ---------------------------------------------------------------------------


def _required_optional(parser):
    """Swap argparse's default groups for explicit required/optional ones."""
    parser._action_groups.pop()
    return (parser.add_argument_group('required arguments'),
            parser.add_argument_group('optional arguments'))


def _add_genomics_inputs(required):
    """Reference, reads and region files shared by the training commands."""
    required.add_argument('-g', '--genome', required=True, type=str, help="reference genome fasta file")
    required.add_argument('-c', '--chrom-sizes', type=str, required=True, help="Chrom sizes file")
    reads = required.add_mutually_exclusive_group(required=True)
    reads.add_argument('-ibam', '--input-bam-file', type=str, help="Input BAM file")
    reads.add_argument('-ifrag', '--input-fragment-file', type=str, help="Input fragment file")
    reads.add_argument('-itag', '--input-tagalign-file', type=str, help="Input tagAlign file")
    required.add_argument('-o', '--output-dir', type=str, required=True, help="Output dir (path/to/output/dir)")
    required.add_argument('-d', '--data-type', required=True, type=str, choices=['ATAC', 'DNASE'], help="assay type")
    required.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    required.add_argument("-n", "--nonpeaks", type=str, required=True, help="10 column bed file of non-peak regions, centered at summit (10th column)")
    required.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")


def _add_training_options(optional):
    """Preprocessing, geometry and optimizer options for the training commands.

    -ps/--plus-shift and -ms/--minus-shift are deliberately absent: the shift is
    always estimated, and pipelines.py sets both attributes to None for
    reads_to_bigwig.
    """
    optional.add_argument("-oth", "--outlier-threshold", type=float, default=0.9999, help="threshold to use to filter outlies")
    optional.add_argument('--ATAC-ref-path', type=str, default=None, help="Path to ATAC reference motifs (ATAC.ref.motifs.txt used by default)")
    optional.add_argument('--DNASE-ref-path', type=str, default=None, help="Path to DNASE reference motifs (DNASE.ref.motifs.txt used by default)")
    optional.add_argument('--num-samples', type=int, default=10000, help="Number of reads to sample from BAM/fragment/tagAlign file for shift estimation")
    optional.add_argument("-il", "--inputlen", type=int, default=2114, required=False, help="Sequence input length")
    optional.add_argument("-ol", "--outputlen", type=int, default=1000, required=False, help="Prediction output length")
    optional.add_argument("-s", "--seed", type=int, default=1234, help="seed to use for model training")
    optional.add_argument("-e", "--epochs", type=int, default=50, help="Maximum epochs to train")
    optional.add_argument("-es", "--early-stop", type=int, default=5, help="Early stop limit, corresponds to 'patience' in callback")
    optional.add_argument("-l", "--learning-rate", type=float, default=0.001, help="Learning rate for model training")
    optional.add_argument("-track", "--trackables", nargs="*", default=['logcount_predictions_loss', 'loss', 'logits_profile_predictions_loss', 'val_logcount_predictions_loss', 'val_loss', 'val_logits_profile_predictions_loss'], help="list of things to track per batch, such as logcount_predictions_loss,loss,profile_predictions_loss,val_logcount_predictions_loss,val_loss,val_profile_predictions_loss")
    optional.add_argument("-a", "--architecture-from-file", type=str, required=False, default=None, help="Model to use for training")
    optional.add_argument("-fp", "--file-prefix", type=str, required=False, default=None, help="File prefix for output to use. All the files will be prefixed with this string if provided.")
    optional.add_argument('-hp', '--html-prefix', required=False, default="./", help="The html prefix to use for the html file output.")
    optional.add_argument('--bsort', required=False, default=False, action='store_true', help="In prpeprocess, by deafult we sort bam using unix sort but sometimes LC collate can cause issues, so this can be set to use betools sort which works well but is memory intensive..")
    optional.add_argument('--tmpdir', required=False, default=None, type=str, help="temp dir for unix sort")
    optional.add_argument('--no-st', required=False, default=False, action='store_true', help="Dont do streaming  and filtering in preprocessing (short chromosome contrigs not in reference fasta are not removed)")
    _add_data_aug(optional)
    _add_device(optional)


def _add_data_aug(optional):
    """Register ``--data-aug`` on a command that preprocesses or trains.

    DNase records a cut as the *bond* between two bases and ATAC records it
    as a base, so reverse-complement augmentation owes them different things.
    See :mod:`chrombpnet.conventions` for the geometry.

    The default is ``None``, meaning *absent*, which
    :func:`conventions.resolve` resolves per assay --- ``both-half`` for
    DNASE, ``standard`` for ATAC. It has to be resolved there rather than
    here, because argparse cannot see ``--data-type`` and the DNase default
    is an arm that raises for ATAC. Recording ``None`` rather than the
    resolved name also keeps ``<prefix>.args.json`` a faithful record of what
    was typed, as ``--benchmark`` already does.
    """
    optional.add_argument(
        "--data-aug", type=str, default=None,
        choices=list(conventions.DATA_AUG_CHOICES),
        help="How cuts are recorded and how reverse complementation handles "
             "them (default: both-half for DNASE, standard for ATAC). "
             "standard: the behaviour this package shipped before --data-aug "
             "-- DNase 0/+1, reverse complement half of training, no "
             "re-registration. "
             "fr-only / rc-only: never / always reverse complement, in every "
             "split; the only two arms that also apply to ATAC. "
             "right: DNase 0/+1 with the +1 post-reversal shift that puts the "
             "bond back on its mirrored base. "
             "left: the -1/0 mirror of it. "
             "both-half / both-full: deposit 0.5 / 1.0 on both bases flanking "
             "the cut, which is symmetric about the bond and so needs no "
             "shift; both-half is the DNASE default and preserves the track "
             "total exactly. right/left/both-* are DNASE-only and raise for "
             "ATAC.")


def _add_ignore_bias_flag(optional):
    """Register ``--ignore-bias-flag`` on the commands that check it.

    Only ``pipeline`` and ``train`` reach
    ``stages.check_bias_model_quality``; ``bias pipeline`` trains a bias
    model but never gates on its peak-count correlation, so offering the
    flag there would imply a check that does not happen.
    """
    optional.add_argument(
        "--ignore-bias-flag", action="store_true",
        help="Downgrade the bias-model quality check from an error to a "
             "warning. By default a bias model whose predicted counts "
             "anti-correlate with observed peak counts (Pearson r <= -0.5) "
             "stops the run, because subtracting it makes the accessibility "
             "model worse. Use this only when the check is known to be "
             "measuring the data rather than the model -- e.g. a sample "
             "whose peak and background counts overlap so much that no "
             "threshold can select a representative nonpeak set. The bias "
             "correction in the resulting model is not trustworthy.")


def _add_device(optional):
    """``--device``, and ``--engine`` with it: every command that places a
    model on a device is a command that needs an engine, and no other is."""
    optional.add_argument('--device', default=None, help="Device to run the model on: cpu, cuda or cuda:N (default: a GPU when available)")
    _add_engine(optional)


def _add_engine(optional):
    """Register ``--engine`` on a command that builds or loads a model.

    The default is ``None``, meaning *absent*: the engine is then taken from
    the input models' checkpoint suffixes, else ``pytorch``
    (:func:`chrombpnet.engines.resolve_name`). An absent flag is also removed
    from the namespace before any stage sees it, so ``<prefix>.args.json``
    records ``engine`` only when it was typed.
    """
    optional.add_argument(
        "--engine", default=None, choices=list(engines.ENGINE_NAMES),
        help="Deep-learning engine. pytorch (the default): this package's "
             "PyTorch implementation. tensorflow: the original Keras "
             "implementation, byte-identical to it given the same inputs. "
             "jax: a placeholder, not implemented yet. When omitted, the "
             "input models' suffixes decide (.pt PyTorch, .h5 TensorFlow), "
             "else pytorch. Each engine runs in its own conda environment, "
             "envs/chrombpnet-<engine>.yml.")


def _add_predict_fk(optional):
    """Register ``--predict-fk`` on a command that predicts.

    Not on ``contribs_bw``: DeepLIFT needs autograd and hooks on the
    ``nn.ReLU`` modules, which a fused kernel cannot provide.
    """
    optional.add_argument(
        "--predict-fk", action="store_true",
        help="Prediction using fused kernels: run the forward on the Triton "
             "chrombpnet_k1 kernels instead of the eager graph. They are "
             "fp32-exact where the default runs TF32 convolutions, so they "
             "are closer to an fp64 reference (1.0e-05 against 1.1e-03) but "
             "they do move the output -- the profile head shifts by ~1e-5. "
             "Needs an H100 or H200; refuses by name on anything else.")


def _add_architecture_options(optional, *, filters, dilations, jitter):
    optional.add_argument("-fil", "--filters", type=int, default=filters, help="Number of filters to use in chrombpnet mode")
    optional.add_argument("-dil", "--n-dilation-layers", type=int, default=dilations, help="Number of dilation layers to use in chrombpnet model")
    optional.add_argument("-j", "--max-jitter", type=int, default=jitter, help="Maximum jitter applied on either side of region (default 500 for chrombpnet model)")


def _add_batch_sizes(optional, batch_help="batch size to use for model training"):
    optional.add_argument("-bs", "--batch-size", type=int, default=64, help=batch_help)
    _add_interpret_batch_size(optional)


def _add_interpret_batch_size(optional):
    optional.add_argument("-ibs", "--interpret-batch-size", type=int, default=None, help="Batch size for contribution scoring. Attribution needs about four times the GPU memory of a training step at the same batch size, so the default is batch-size/4.")


def _add_qc_arguments(parser, model_arguments):
    """The qc and bias-qc surfaces, which differ only in their model flags."""
    required, optional = _required_optional(parser)
    required.add_argument("-bw", "--bigwig", type=str, required=True, help="Input bigwig file of observed data")
    for flags, help_text in model_arguments:
        required.add_argument(*flags, type=str, required=True, help=help_text)
    required.add_argument('-g', '--genome', required=True, type=str, help="reference genome fasta file")
    required.add_argument('-c', '--chrom-sizes', type=str, required=True, help="Chrom sizes file")
    required.add_argument('-o', '--output-dir', type=str, required=True, help="Output dir (path/to/output/dir)")
    required.add_argument('-d', '--data-type', required=True, type=str, choices=['ATAC', 'DNASE'], help="assay type")
    required.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    required.add_argument("-n", "--nonpeaks", type=str, required=True, help="10 column bed file of non-peak regions, centered at summit (10th column)")
    required.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    optional.add_argument("-fp", "--file-prefix", type=str, required=False, default=None, help="File prefix for output to use. All the files will be prefixed with this string if provided.")
    _add_batch_sizes(optional)
    optional.add_argument('-hp', '--html-prefix', required=False, default="./", help="The html prefix to use for the html file output.")
    _add_data_aug(optional)
    _add_device(optional)
    _add_predict_fk(optional)


def _add_chrombpnet_training_arguments(parser):
    """`train` and `pipeline`, whose surfaces were byte-identical."""
    required, optional = _required_optional(parser)
    _add_genomics_inputs(required)
    _add_training_options(optional)
    add_benchmark_argument(optional)
    _add_ignore_bias_flag(optional)
    required.add_argument("-b", "--bias-model-path", type=str, required=True, help="Path for a pretrained bias model checkpoint (.pt PyTorch, .h5 TensorFlow)")
    optional.add_argument("-sr", "--negative-sampling-ratio", type=float, default=0.1, help="Ratio of negatives to positive samples per epoch")
    _add_architecture_options(optional, filters=512, dilations=8, jitter=500)
    _add_batch_sizes(optional)


def _add_bias_training_arguments(parser):
    """`bias pipeline` and `bias train`.

    These differed only in a truncated help string -- "0.8 for DNase." against
    "0.8 for DNas" -- which is why they could not simply be shared before. The
    complete wording is used for both.
    """
    required, optional = _required_optional(parser)
    _add_genomics_inputs(required)
    _add_training_options(optional)
    add_benchmark_argument(optional)
    required.add_argument("-b", "--bias-threshold-factor", type=float, required=True, help="A threshold is applied on maximum count of non-peak region for training bias model, which is set as this threshold x min(count over peak regions). Recommended start value 0.5 for ATAC and 0.8 for DNase.")
    _add_architecture_options(optional, filters=128, dilations=4, jitter=0)
    _add_batch_sizes(optional)


def build_parser():
    """Build the whole command tree without consuming argv."""
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    subparsers = parser.add_subparsers(help="Must be eithier 'pipeline', 'train', 'qc', 'bias', 'prep', 'pred_bw', 'contribs_bw', 'modisco_motifs', or 'footprints'.", required=True, dest='cmd')

    # main parsers
    pipeline_parser = subparsers.add_parser("pipeline", help="End-to-end pipline with train, quality check and test for bias factorized ChromBPNet model")
    train_parser = subparsers.add_parser("train", help="Train bias factorized ChromBPNet model")
    qc_parser = subparsers.add_parser("qc", help="Do quality checks and get test metrics for bias factorized ChromBPNet model")

    # bias parsers
    bias_parser_full = subparsers.add_parser("bias", help="Tools to train, quality check and test bias model")
    bias_parser_sub = bias_parser_full.add_subparsers(help="Must be eithier 'pipeline'  'train', 'qc'.", required=True, dest='cmd_bias')
    bias_parser = bias_parser_sub.add_parser("pipeline", help="End-to-end pipline with train, quality check and test for bias model")
    bias_parser_train = bias_parser_sub.add_parser("train", help="Train bias model")
    bias_parser_qc = bias_parser_sub.add_parser("qc", help="Do quality checks and get test metrics for the bias model")

    # helper parsers
    prep_parser_full = subparsers.add_parser("prep", help="Tools to generate preprocessing data for chrombpnet")
    prep_parser_sub = prep_parser_full.add_subparsers(help="Must be eithier 'nonpeaks' or 'splits'.", required=True, dest='cmd_prep')
    nonpeaks_parser = prep_parser_sub.add_parser("nonpeaks", help="Generate non-peak background regions given peaks")
    splits_parser = prep_parser_sub.add_parser("splits", help="Generate chromosome splits")

    # downstream tool parsers
    preds_parser = subparsers.add_parser("pred_bw", help="Get model prediction bigwigs (Metrics calculated if observed bigwig provided)")
    contribs_parser = subparsers.add_parser("contribs_bw", help="Get contribution score bigwigs")
    motifs_parser = subparsers.add_parser("modisco_motifs", help="Discover motifs from contribution scores with TF-MoDISco 2.6")
    footprints_parser = subparsers.add_parser("footprints", help="Get marginal footprinting for given model and given motifs")
    predict_parser = subparsers.add_parser("predict", help="Score one model on a fold's test split and write metrics")

    # Generate non-peak regions from peak-regions
    required_nonpeaks_parser, optional_nonpeaks_parser = _required_optional(nonpeaks_parser)
    required_nonpeaks_parser.add_argument("-g", "--genome", required=True, help="reference genome file")
    required_nonpeaks_parser.add_argument("-o", "--output-prefix", required=True, help="output BED file prefix to store the gc content of binned genome. suffix .bed will be appended by the code. If the prefix contains a directory path make sure it exists.")
    required_nonpeaks_parser.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    required_nonpeaks_parser.add_argument('-c', '--chrom-sizes', type=str, required=True, help="Chrom sizes file")
    required_nonpeaks_parser.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    optional_nonpeaks_parser.add_argument("-il", "--inputlen", type=int, default=2114, help="inputlen to use to make bins and find gc content")
    optional_nonpeaks_parser.add_argument("-st", "--stride", type=int, default=1000, help="stride to use for shifting the bins")
    optional_nonpeaks_parser.add_argument("-npr", "--neg-to-pos-ratio-train", type=int, default=2, help="Ratio of negatives to positives to sample in training set (test set always has 1:1 positive to negatives ratio")
    optional_nonpeaks_parser.add_argument("-br", "--blacklist-regions", type=str, required=False, default=None, help="TSV file with 3 columns - chr, start, end")
    optional_nonpeaks_parser.add_argument("-s", "--seed", type=int, default=1234, help="seed to use for generating nonpeaks")

    # Generate splits. Note -op/--output_prefix keeps its underscore, unlike
    # every other command's --output-prefix; renaming it would break callers.
    splits_parser._action_groups.pop()
    required_splits_parser = splits_parser.add_argument_group('required arguments')
    required_splits_parser.add_argument("-op", "--output_prefix", type=str, required=True, help="Path prefix to store the fold information (appended with .json)")
    required_splits_parser.add_argument("-c", "--chrom-sizes", type=str, required=True, help="TSV file with chromosome sizes. All chromosomes from the first column of chrom sizes file are used")
    required_splits_parser.add_argument("-tcr", "--test-chroms", nargs="*", type=str, required=True, help="Chromosomes to use for test")
    required_splits_parser.add_argument("-vcr", "--valid-chroms", nargs="*", type=str, required=True, help="Chromosomes to use for validation")

    _add_chrombpnet_training_arguments(train_parser)
    _add_chrombpnet_training_arguments(pipeline_parser)

    _add_qc_arguments(qc_parser, [
        (("-cm", "--chrombpnet-model"), "Path to combined ChromBPNet checkpoint (.pt PyTorch, .h5 TensorFlow)"),
        (("-cmb", "--chrombpnet-model-nb"), "Path to no-bias ChromBPNet checkpoint (.pt PyTorch, .h5 TensorFlow)"),
    ])

    _add_bias_training_arguments(bias_parser)
    _add_bias_training_arguments(bias_parser_train)

    _add_qc_arguments(bias_parser_qc, [
        (("-bm", "--bias-model"), "Path to bias model checkpoint (.pt PyTorch, .h5 TensorFlow)"),
    ])

    _add_predict_arguments(predict_parser)

    # Make prediction bigwigs
    required_preds, optional_preds = _required_optional(preds_parser)
    required_preds.add_argument("-bm", "--bias-model", type=str, required=False, help="Path to bias model checkpoint, .pt or .h5 (at least one model is required)")
    required_preds.add_argument("-cm", "--chrombpnet-model", type=str, required=False, help="Path to combined ChromBPNet checkpoint (.pt PyTorch, .h5 TensorFlow)")
    required_preds.add_argument("-cmb", "--chrombpnet-model-nb", type=str, required=False, help="Path to no-bias ChromBPNet checkpoint (.pt PyTorch, .h5 TensorFlow)")
    required_preds.add_argument("-r", "--regions", type=str, required=True, help="10 column bed file of regions for prediction")
    required_preds.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    required_preds.add_argument("-c", "--chrom-sizes", type=str, required=True, help="Chromosome sizes 2 column tab-separated file")
    required_preds.add_argument("-op", "--output-prefix", type=str, required=True, help="Output prefix for bigwig files")
    optional_preds.add_argument("-os", "--output-prefix-stats", type=str, default=None, required=False, help="Output stats on bigwig")
    optional_preds.add_argument("-bs", "--batch-size", type=int, default=64, help="batch size to use for prediction")
    optional_preds.add_argument("-t", "--tqdm", type=int, default=1, help="Use tqdm. If yes then you need to have it installed.")
    optional_preds.add_argument("-d", "--debug-chr", nargs="+", type=str, default=None, help="Run for specific chromosomes only (e.g. chr1 chr2) for debugging")
    optional_preds.add_argument("-bw", "--bigwig", type=str, default=None, help="If provided .h5 with predictions are output along with calculated metrics considering bigwig as groundtruth.")
    _add_device(optional_preds)
    _add_predict_fk(optional_preds)

    # Make contribution score bigwigs
    required_contribs, optional_contribs = _required_optional(contribs_parser)
    required_contribs.add_argument("-m", "--model-h5", type=str, required=True, help="Path to model checkpoint, .pt PyTorch or .h5 TensorFlow (legacy option name)")
    required_contribs.add_argument("-r", "--regions", type=str, required=True, help="10 column bed file of regions for contribution score predictions")
    required_contribs.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    required_contribs.add_argument("-c", "--chrom-sizes", type=str, required=True, help="Chromosome sizes 2 column tab-separated file")
    required_contribs.add_argument("-op", "--output-prefix", type=str, required=True, help="Output prefix for bigwig files")
    optional_contribs.add_argument("-pc", "--profile-or-counts", nargs="+", type=str, default=["counts", "profile"], choices=["counts", "profile"],
                    help="use either counts or profile or both for running shap")
    optional_contribs.add_argument("-os", "--output-prefix-stats", type=str, default=None, required=False, help="Output stats on bigwig")
    optional_contribs.add_argument("-t", "--tqdm", type=int, default=1, help="Use tqdm. If yes then you need to have it installed.")
    optional_contribs.add_argument("-d", "--debug-chr", nargs="+", type=str, default=None, help="Run for specific chromosomes only (e.g. chr1 chr2) for debugging")
    _add_device(optional_contribs)
    optional_contribs.add_argument('--batch-size', type=int, default=64)
    _add_interpret_batch_size(optional_contribs)
    optional_contribs.add_argument('--num-shuffles', type=int, default=20)
    optional_contribs.add_argument('--seed', type=int, default=1234)

    # Get marginal footprints
    required_ftps, optional_ftps = _required_optional(footprints_parser)
    required_ftps.add_argument("-m", "--model-h5", type=str, required=True, help="Path to model checkpoint, .pt PyTorch or .h5 TensorFlow (legacy option name)")
    required_ftps.add_argument("-r", "--regions", type=str, required=True, help="10 column bed file of non-peak regions")
    required_ftps.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    required_ftps.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    required_ftps.add_argument("-op", "--output-prefix", type=str, required=True, help="Output prefix for bigwig files")
    required_ftps.add_argument("-pwm_f", "--motifs-to-pwm", type=str, required=True, help="Path to a TSV file containing motifs in first column and motif string to use for footprinting in second column")
    optional_ftps.add_argument("-bs", "--batch-size", type=int, default=64, help="batch size to use for prediction")
    _add_device(optional_ftps)
    _add_predict_fk(optional_ftps)
    optional_ftps.add_argument("--ylim", default=None, type=ylim_pair, required=False, help="lower and upper y-limits for plotting the motif footprint, in the form of a tuple i.e. \
        (0,0.8). If this is set to None, ylim will be autodetermined.")

    # Run TF-Modisco
    required_tfm, optional_tfm = _required_optional(motifs_parser)
    required_tfm.add_argument("-i", "--h5py", type=str, required=True, help="ChromBPNet contribution-score HDF5 file.")
    required_tfm.add_argument("-op", "--output-prefix", type=str, required=True, help="Prefix for the results HDF5 and report directory.")
    optional_tfm.add_argument("-n", "--max-seqlets", type=int, default=50000, help="Maximum seqlets per positive/negative pattern group.")
    optional_tfm.add_argument("-l", "--n-leiden", type=int, default=2, help="Number of Leiden clustering runs.")
    optional_tfm.add_argument("-w", "--window", type=int, default=500, help="Centered contribution-score window used for discovery (must be even).")
    optional_tfm.add_argument("-v", "--verbose", action="store_true", default=False)

    return parser


def _add_predict_arguments(parser):
    """Just the prediction-and-metrics stage that `qc` runs first.

    `qc` also runs attribution and MoDISco, hours of work for numbers that
    are already written once predictions exist. Scoring a model against a
    given test split -- which is what comparing two augmentation arms needs
    -- wants only this part, and `training/predict.py` had no CLI of its own.

    Geometry is read off the checkpoint, exactly as `qc` does, so there are
    no --inputlen/--outputlen to get wrong.
    """
    required, optional = _required_optional(parser)
    required.add_argument("-m", "--model", type=str, required=True, help="Path to the checkpoint to score (.pt PyTorch, .h5 TensorFlow)")
    required.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    required.add_argument("-c", "--chrom-sizes", type=str, required=True, help="Chromosome sizes 2 column tab-separated file")
    required.add_argument("-bw", "--bigwig", type=str, required=True, help="Observed bigwig to score against")
    required.add_argument("-r", "--peaks", type=str, required=True, help="10 column bed file of peaks, or 'None' to score nonpeaks only")
    required.add_argument("-n", "--nonpeaks", type=str, required=True, help="10 column bed file of non-peak regions, or 'None' to score peaks only")
    required.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    required.add_argument("-op", "--output-prefix", type=str, required=True, help="Output prefix; _predictions.h5 and _metrics.json are appended")
    required.add_argument("-d", "--data-type", required=True, type=str, choices=['ATAC', 'DNASE'], help="assay type")
    optional.add_argument("-bs", "--batch-size", type=int, default=64, help="batch size to use for prediction")
    optional.add_argument("-s", "--seed", type=int, default=1234, help="seed for the profile-metrics shuffled draw")
    add_benchmark_argument(optional)
    _add_data_aug(optional)
    _add_device(optional)
    _add_predict_fk(optional)


def read_parser(argv=None):
    """Parse the command line.

    ``argv`` exists so the parser can be exercised without monkeypatching
    sys.argv; the CLI passes nothing and argparse reads sys.argv as before.
    """
    return build_parser().parse_args(argv)
