"""The four end-to-end workflows behind the CLI's pipeline/train/qc commands.

Each is an explicit ordered sequence of steps from :mod:`chrombpnet.stages`,
with paths resolved by :class:`chrombpnet.layout.RunLayout`. The order is the
contract -- these functions exist to run things in a particular sequence -- so
the mapping to the previous inline implementation is recorded per function.

``train`` and ``bias train`` share their pipeline with ``pipeline`` and
``bias pipeline`` respectively, returning early after the report rather than
going on to evaluate.

Checkpoint filenames take the configured engine's suffix (``.pt`` for
PyTorch), and the built-in architecture files are the engine's.
"""

from __future__ import annotations

from chrombpnet import engines, stages
from chrombpnet.engines import load_model_wrapper
from chrombpnet.layout import RunLayout


def chrombpnet_train_pipeline(args):
    """Train a bias-factorized ChromBPNet model, then evaluate it.

    Steps: preprocess -> hyperparameters -> QC the input bias model ->
    move params into logs/ -> train -> move logs -> (stop here for `train`) ->
    predict -> footprints -> interpret -> modisco -> report.
    """
    layout = RunLayout.from_args(args)
    engine = engines.current()
    sfx = engine.checkpoint_suffix

    stages.preprocess_and_qc_bigwig(args, layout)

    import chrombpnet.helpers.hyperparameters.find_chrombpnet_hyperparams as find_chrombpnet_hyperparams
    stages.find_hyperparameters(args, layout, find_chrombpnet_hyperparams)

    # Check the supplied bias model against observed peak counts before
    # spending a training run on it. Note the unprefixed "evaluation/bias"
    # output prefix, which check_bias_model_quality reads back.
    filtered_peaks = layout.aux("filtered.peaks.bed")
    stages.run_predictions(
        args, layout,
        output_prefix=layout.unprefixed("evaluation", "bias"),
        model_path=args.bias_model_path,
        peaks=filtered_peaks, nonpeaks="None")
    stages.check_bias_model_quality(
        layout, ignore=getattr(args, "ignore_bias_flag", False))

    # Separate models and params from the auxiliary scratch directory.
    layout.move("auxiliary", "models", f"bias_model_scaled{sfx}")
    layout.move("auxiliary", "logs", "chrombpnet_model_params.tsv")
    layout.move("auxiliary", "logs", "chrombpnet_data_params.tsv")

    # The params table still points at the pre-move bias model, and train.py
    # loads the model from it, so rewrite that one path.
    model_params = layout.logs("chrombpnet_model_params.tsv")
    params = open(model_params).read()
    params = params.replace(layout.aux(f"bias_model_scaled{sfx}"),
                            layout.models(f"bias_model_scaled{sfx}"))
    with open(model_params, "w") as handle:
        handle.write(params)

    import chrombpnet.training.train as train
    import copy

    args_copy = copy.deepcopy(args)
    if args_copy.architecture_from_file is None:
        args_copy.architecture_from_file = engine.default_architecture("chrombpnet")
    args_copy.peaks = filtered_peaks
    args_copy.nonpeaks = layout.aux("filtered.nonpeaks.bed")
    args_copy.output_prefix = layout.models("chrombpnet")
    args_copy.params = model_params
    train.main(args_copy)

    layout.move("models", "logs", "chrombpnet.log")
    layout.move("models", "logs", "chrombpnet.log.batch")
    layout.move("models", "logs", "chrombpnet.args.json")

    import chrombpnet.helpers.generate_reports.make_html as make_html
    if args.cmd == "train":
        stages.render_report(args, layout, make_html, args.cmd)
        print("Finished training! Exiting!")
        return

    nobias_model = layout.models(f"chrombpnet_nobias{sfx}")

    stages.run_predictions(
        args, layout,
        output_prefix=layout.evaluation("chrombpnet"),
        model_path=layout.models(f"chrombpnet{sfx}"),
        peaks=filtered_peaks, nonpeaks="None")

    motifs_to_pwm = stages.write_bias_motif_table(args, layout)
    stages.run_marginal_footprints(
        args, layout, model_path=nobias_model, regions=args.nonpeaks,
        output_prefix=layout.evaluation("chrombpnet_nobias"),
        motifs_to_pwm=motifs_to_pwm)
    layout.move("evaluation", "auxiliary", "chrombpnet_nobias_footprints.h5")

    subsample = stages.subsample_peaks_for_interpretation(
        layout, filtered_peaks, getattr(args, "interpret_subsample", None))
    # Profile only, deliberately: make_html renders just the profile MoDISco
    # table for ChromBPNet models. The bias pipelines score both.
    stages.run_interpret(
        args, layout, model_path=nobias_model, regions=subsample,
        output_prefix=layout.interpret_subsample("chrombpnet_nobias"),
        profile_or_counts=["profile"])
    stages.run_modisco(args, layout, "chrombpnet_nobias", "profile",
                       "chrombpnet_nobias_profile.pdf")

    stages.render_report(args, layout, make_html, args.cmd)


def chrombpnet_qc(args):
    """Evaluate an already-trained ChromBPNet model.

    The same tail as chrombpnet_train_pipeline from the prediction step on, but
    against user-supplied models and peak files rather than ones this run
    produced.
    """
    layout = RunLayout.from_args(args)

    # Read the geometry off the checkpoint; qc has no --inputlen/--outputlen.
    chrombpnet_md = load_model_wrapper(
        args.chrombpnet_model, getattr(args, "device", None), announce=False)
    args.inputlen = int(chrombpnet_md.input_shape[1])
    args.outputlen = int(chrombpnet_md.output_shape[0][1])

    stages.run_predictions(
        args, layout,
        output_prefix=layout.evaluation("chrombpnet"),
        model_path=args.chrombpnet_model,
        peaks=args.peaks, nonpeaks="None")

    motifs_to_pwm = stages.write_bias_motif_table(args, layout)
    stages.run_marginal_footprints(
        args, layout, model_path=args.chrombpnet_model_nb,
        regions=args.nonpeaks,
        output_prefix=layout.evaluation("chrombpnet_nobias"),
        motifs_to_pwm=motifs_to_pwm)
    layout.move("evaluation", "auxiliary", "chrombpnet_nobias_footprints.h5")

    subsample = stages.subsample_peaks_for_interpretation(
        layout, args.peaks, getattr(args, "interpret_subsample", None))
    stages.run_interpret(
        args, layout, model_path=args.chrombpnet_model_nb, regions=subsample,
        output_prefix=layout.interpret_subsample("chrombpnet_nobias"),
        profile_or_counts=["profile"])
    stages.run_modisco(args, layout, "chrombpnet_nobias", "profile",
                       "chrombpnet_nobias_profile.pdf")

    import chrombpnet.helpers.generate_reports.make_html as make_html
    stages.render_report(args, layout, make_html, args.cmd)


def train_bias_pipeline(args):
    """Train a bias model on low-count nonpeaks, then evaluate it.

    Unlike the ChromBPNet pipeline this trains on nonpeaks only (peaks are
    passed as the "None" sentinel) and scores both counts and profile
    contributions.
    """
    layout = RunLayout.from_args(args)
    engine = engines.current()

    stages.preprocess_and_qc_bigwig(args, layout)

    import chrombpnet.helpers.hyperparameters.find_bias_hyperparams as find_bias_hyperparams
    stages.find_hyperparameters(args, layout, find_bias_hyperparams)

    layout.move("auxiliary", "logs", "bias_model_params.tsv")
    layout.move("auxiliary", "logs", "bias_data_params.tsv")

    import chrombpnet.training.train as train
    import copy

    filtered_nonpeaks = layout.aux("filtered.bias_nonpeaks.bed")
    args_copy = copy.deepcopy(args)
    if args_copy.architecture_from_file is None:
        args_copy.architecture_from_file = engine.default_architecture("bias")
    args_copy.peaks = "None"
    args_copy.nonpeaks = filtered_nonpeaks
    args_copy.output_prefix = layout.models("bias")
    args_copy.params = layout.logs("bias_model_params.tsv")
    train.main(args_copy)

    layout.move("models", "logs", "bias.args.json")
    layout.move("models", "logs", "bias.log")
    layout.move("models", "logs", "bias.log.batch")

    import chrombpnet.helpers.generate_reports.make_html_bias as make_html_bias
    if args.cmd_bias == "train":
        stages.render_report(args, layout, make_html_bias, args.cmd_bias)
        print("Finished training! Exiting!")
        return

    bias_model = layout.models(f"bias{engine.checkpoint_suffix}")
    filtered_peaks = layout.aux("filtered.bias_peaks.bed")

    stages.run_predictions(
        args, layout,
        output_prefix=layout.evaluation("bias"),
        model_path=bias_model,
        peaks=filtered_peaks, nonpeaks=filtered_nonpeaks)

    subsample = stages.subsample_peaks_for_interpretation(
        layout, filtered_peaks, getattr(args, "interpret_subsample", None))
    stages.run_interpret(
        args, layout, model_path=bias_model, regions=subsample,
        output_prefix=layout.interpret_subsample("bias"),
        profile_or_counts=["counts", "profile"])
    stages.run_modisco(args, layout, "bias", "profile", "bias_profile.pdf")
    stages.run_modisco(args, layout, "bias", "counts", "bias_counts.pdf")

    stages.render_report(args, layout, make_html_bias, args.cmd_bias)


def bias_model_qc(args):
    """Evaluate an already-trained bias model."""
    layout = RunLayout.from_args(args)

    bias_md = load_model_wrapper(
        args.bias_model, getattr(args, "device", None), announce=False)
    args.inputlen = int(bias_md.input_shape[1])
    args.outputlen = int(bias_md.output_shape[0][1])

    stages.run_predictions(
        args, layout,
        output_prefix=layout.evaluation("bias"),
        model_path=args.bias_model,
        peaks=args.peaks, nonpeaks=args.nonpeaks)

    subsample = stages.subsample_peaks_for_interpretation(
        layout, args.peaks, getattr(args, "interpret_subsample", None))
    stages.run_interpret(
        args, layout, model_path=args.bias_model, regions=subsample,
        output_prefix=layout.interpret_subsample("bias"),
        profile_or_counts=["counts", "profile"])
    stages.run_modisco(args, layout, "bias", "profile", "bias_profile.pdf")
    stages.run_modisco(args, layout, "bias", "counts", "bias_counts.pdf")

    import chrombpnet.helpers.generate_reports.make_html_bias as make_html_bias
    stages.render_report(args, layout, make_html_bias, args.cmd_bias)
