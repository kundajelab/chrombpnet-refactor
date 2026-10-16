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

Each takes the run's read-only :class:`~chrombpnet.options.Options`. What a
pipeline decides -- the bigwig preprocessing generates, the filtered peaks a
model trains on, the geometry qc reads off a checkpoint -- it passes on
explicitly, through :meth:`Options.step`, so it reaches only the steps that
come after it.
"""

from __future__ import annotations

from chrombpnet import engines, stages
from chrombpnet.engines import load_model_wrapper
from chrombpnet.layout import RunLayout
from chrombpnet.options import as_options


def chrombpnet_train_pipeline(options):
    """Train a bias-factorized ChromBPNet model, then evaluate it.

    Steps: preprocess -> hyperparameters -> QC the input bias model ->
    move params into logs/ -> train -> move logs -> (stop here for `train`) ->
    predict -> footprints -> interpret -> modisco -> report.
    """
    options = as_options(options)
    layout = RunLayout.from_args(options)
    engine = engines.current()
    sfx = engine.checkpoint_suffix

    # Every later step reads the observed signal from this bigwig.
    options = options.step(bigwig=stages.preprocess_and_qc_bigwig(options, layout))

    import chrombpnet.helpers.hyperparameters.find_chrombpnet_hyperparams as find_chrombpnet_hyperparams
    stages.find_hyperparameters(options, layout, find_chrombpnet_hyperparams)

    # Check the supplied bias model against observed peak counts before
    # spending a training run on it. Note the unprefixed "evaluation/bias"
    # output prefix, which check_bias_model_quality reads back.
    filtered_peaks = layout.aux("filtered.peaks.bed")
    stages.run_predictions(
        options, layout,
        output_prefix=layout.unprefixed("evaluation", "bias"),
        model_path=options.bias_model_path,
        peaks=filtered_peaks, nonpeaks="None")
    stages.check_bias_model_quality(
        layout, ignore=getattr(options, "ignore_bias_flag", False))

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

    train.main(options.step(
        replacing={"peaks": filtered_peaks,
                   "nonpeaks": layout.aux("filtered.nonpeaks.bed")},
        architecture_from_file=default_architecture(options, engine, "chrombpnet"),
        output_prefix=layout.models("chrombpnet"),
        params=model_params))

    layout.move("models", "logs", "chrombpnet.log")
    layout.move("models", "logs", "chrombpnet.log.batch")
    layout.move("models", "logs", "chrombpnet.args.json")

    import chrombpnet.helpers.generate_reports.make_html as make_html
    if options.cmd == "train":
        stages.render_report(options, layout, make_html, options.cmd)
        print("Finished training! Exiting!")
        return

    nobias_model = layout.models(f"chrombpnet_nobias{sfx}")

    stages.run_predictions(
        options, layout,
        output_prefix=layout.evaluation("chrombpnet"),
        model_path=layout.models(f"chrombpnet{sfx}"),
        peaks=filtered_peaks, nonpeaks="None")

    motifs_to_pwm = stages.write_bias_motif_table(options, layout)
    stages.run_marginal_footprints(
        options, layout, model_path=nobias_model, regions=options.nonpeaks,
        output_prefix=layout.evaluation("chrombpnet_nobias"),
        motifs_to_pwm=motifs_to_pwm)
    layout.move("evaluation", "auxiliary", "chrombpnet_nobias_footprints.h5")

    subsample = stages.subsample_peaks_for_interpretation(
        layout, filtered_peaks, getattr(options, "interpret_subsample", None),
        seed=getattr(options, "seed", None))
    # Both heads are scored and kept, as the bias pipelines do, so the
    # subsample's contribution scores can be compared across runs. MoDISco
    # stays profile-only: make_html renders just the profile table for
    # ChromBPNet models.
    stages.run_interpret(
        options, layout, model_path=nobias_model, regions=subsample,
        output_prefix=layout.interpret_subsample("chrombpnet_nobias"),
        profile_or_counts=["counts", "profile"])
    stages.run_modisco(options, layout, "chrombpnet_nobias", "profile",
                       "chrombpnet_nobias_profile.pdf")

    stages.render_report(options, layout, make_html, options.cmd)


def default_architecture(options, engine, kind):
    """The -a architecture file, else the engine's built-in one for ``kind``."""
    if options.architecture_from_file is None:
        return engine.default_architecture(kind)
    return options.architecture_from_file


def model_geometry(options, model_path):
    """``options`` plus the input and output lengths read off a checkpoint;
    qc has no --inputlen/--outputlen."""
    model = load_model_wrapper(
        model_path, getattr(options, "device", None), announce=False)
    return options.step(inputlen=int(model.input_shape[1]),
                        outputlen=int(model.output_shape[0][1]))


def chrombpnet_qc(options):
    """Evaluate an already-trained ChromBPNet model.

    The same tail as chrombpnet_train_pipeline from the prediction step on, but
    against user-supplied models and peak files rather than ones this run
    produced.
    """
    options = as_options(options)
    layout = RunLayout.from_args(options)
    options = model_geometry(options, options.chrombpnet_model)

    stages.run_predictions(
        options, layout,
        output_prefix=layout.evaluation("chrombpnet"),
        model_path=options.chrombpnet_model,
        peaks=options.peaks, nonpeaks="None")

    motifs_to_pwm = stages.write_bias_motif_table(options, layout)
    stages.run_marginal_footprints(
        options, layout, model_path=options.chrombpnet_model_nb,
        regions=options.nonpeaks,
        output_prefix=layout.evaluation("chrombpnet_nobias"),
        motifs_to_pwm=motifs_to_pwm)
    layout.move("evaluation", "auxiliary", "chrombpnet_nobias_footprints.h5")

    subsample = stages.subsample_peaks_for_interpretation(
        layout, options.peaks, getattr(options, "interpret_subsample", None),
        seed=getattr(options, "seed", None))
    stages.run_interpret(
        options, layout, model_path=options.chrombpnet_model_nb, regions=subsample,
        output_prefix=layout.interpret_subsample("chrombpnet_nobias"),
        profile_or_counts=["counts", "profile"])
    stages.run_modisco(options, layout, "chrombpnet_nobias", "profile",
                       "chrombpnet_nobias_profile.pdf")

    import chrombpnet.helpers.generate_reports.make_html as make_html
    stages.render_report(options, layout, make_html, options.cmd)


def train_bias_pipeline(options):
    """Train a bias model on low-count nonpeaks, then evaluate it.

    Unlike the ChromBPNet pipeline this trains on nonpeaks only (peaks are
    passed as the "None" sentinel) and scores both counts and profile
    contributions.
    """
    options = as_options(options)
    layout = RunLayout.from_args(options)
    engine = engines.current()

    # Every later step reads the observed signal from this bigwig.
    options = options.step(bigwig=stages.preprocess_and_qc_bigwig(options, layout))

    import chrombpnet.helpers.hyperparameters.find_bias_hyperparams as find_bias_hyperparams
    stages.find_hyperparameters(options, layout, find_bias_hyperparams)

    layout.move("auxiliary", "logs", "bias_model_params.tsv")
    layout.move("auxiliary", "logs", "bias_data_params.tsv")

    import chrombpnet.training.train as train

    filtered_nonpeaks = layout.aux("filtered.bias_nonpeaks.bed")
    train.main(options.step(
        replacing={"peaks": "None", "nonpeaks": filtered_nonpeaks},
        architecture_from_file=default_architecture(options, engine, "bias"),
        output_prefix=layout.models("bias"),
        params=layout.logs("bias_model_params.tsv")))

    layout.move("models", "logs", "bias.args.json")
    layout.move("models", "logs", "bias.log")
    layout.move("models", "logs", "bias.log.batch")

    import chrombpnet.helpers.generate_reports.make_html_bias as make_html_bias
    if options.cmd_bias == "train":
        stages.render_report(options, layout, make_html_bias, options.cmd_bias)
        print("Finished training! Exiting!")
        return

    bias_model = layout.models(f"bias{engine.checkpoint_suffix}")
    filtered_peaks = layout.aux("filtered.bias_peaks.bed")

    stages.run_predictions(
        options, layout,
        output_prefix=layout.evaluation("bias"),
        model_path=bias_model,
        peaks=filtered_peaks, nonpeaks=filtered_nonpeaks)

    subsample = stages.subsample_peaks_for_interpretation(
        layout, filtered_peaks, getattr(options, "interpret_subsample", None),
        seed=getattr(options, "seed", None))
    stages.run_interpret(
        options, layout, model_path=bias_model, regions=subsample,
        output_prefix=layout.interpret_subsample("bias"),
        profile_or_counts=["counts", "profile"])
    stages.run_modisco(options, layout, "bias", "profile", "bias_profile.pdf")
    stages.run_modisco(options, layout, "bias", "counts", "bias_counts.pdf")

    stages.render_report(options, layout, make_html_bias, options.cmd_bias)


def bias_model_qc(options):
    """Evaluate an already-trained bias model."""
    options = as_options(options)
    layout = RunLayout.from_args(options)
    options = model_geometry(options, options.bias_model)

    stages.run_predictions(
        options, layout,
        output_prefix=layout.evaluation("bias"),
        model_path=options.bias_model,
        peaks=options.peaks, nonpeaks=options.nonpeaks)

    subsample = stages.subsample_peaks_for_interpretation(
        layout, options.peaks, getattr(options, "interpret_subsample", None),
        seed=getattr(options, "seed", None))
    stages.run_interpret(
        options, layout, model_path=options.bias_model, regions=subsample,
        output_prefix=layout.interpret_subsample("bias"),
        profile_or_counts=["counts", "profile"])
    stages.run_modisco(options, layout, "bias", "profile", "bias_profile.pdf")
    stages.run_modisco(options, layout, "bias", "counts", "bias_counts.pdf")

    import chrombpnet.helpers.generate_reports.make_html_bias as make_html_bias
    stages.render_report(options, layout, make_html_bias, options.cmd_bias)
