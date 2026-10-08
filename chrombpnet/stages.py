"""Reusable steps shared by the four pipelines in :mod:`chrombpnet.pipelines`.

``chrombpnet_train_pipeline``, ``chrombpnet_qc``, ``train_bias_pipeline`` and
``bias_model_qc`` ran to 417 lines between them, most of it the same tail
repeated with different model paths: the 30k-peak subsample appeared four
times, the interpret setup four times, the report call six times, the bias-motif
table twice, the preprocessing preamble twice.

Each stage takes the run's read-only :class:`~chrombpnet.options.Options`
and its own inputs as explicit parameters, and hands the step module a new
namespace built with :meth:`Options.step` -- never a modified copy of the
user's. The step modules' ``<prefix>.args.json`` records are dumps of that
namespace, so they hold the options plus that step's inputs and nothing
another step set.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from chrombpnet.data import DefaultDataFile, get_default_data_path
from chrombpnet.options import as_options

#: Contribution scores are computed on at most this many peaks, sampled with a
#: fixed seed so a rerun scores the same regions (--interpret-subsample).
INTERPRET_SUBSAMPLE_SIZE = 30_000
INTERPRET_SUBSAMPLE_SEED = 1234

#: Width of the QC PWM built from the shifted bigwig.
PWM_WIDTH = 24

#: A bias model whose peak count correlation is at or below this is rejected:
#: it is likely fitting AT content rather than enzyme bias.
MIN_BIAS_PEAK_PEARSONR = -0.5

#: Enzyme-bias motifs inserted for marginal footprinting, by assay.
#:
#: Kept in step with chrombpnet/data/motif_to_pwm.{ATAC,DNASE}.tsv by
#: tests/test_stages.py. The table is written out rather than copied from those
#: files because the packaged copies use CRLF line endings, and the generated
#: auxiliary TSV has always been LF.
BIAS_MOTIFS = {
    "ATAC": [["tn5_1", "GCACAGTACAGAGCTG"],
             ["tn5_2", "GTGCACAGTTCTAGAGTGTGCAG"],
             ["tn5_3", "CCTCTACACTGTGCAGAA"],
             ["tn5_4", "GCACAGTTCTAGACTGTGCAG"],
             ["tn5_5", "CTGCACAGTGTAGAGTTGTGC"]],
    "DNASE": [["dnase_1", "TTTACAAGTCCA"],
              ["dnase_2", "TGTACTTACGAA"]],
}


def preprocess_and_qc_bigwig(options, layout):
    """Shift reads into a bigwig, build the PWM that QCs the shift, and return
    the bigwig's path, which the caller passes on to the later steps.

    With -ibw the given bigwig is used as is: only the PWM is built, which is
    the evidence that its shift is right.
    """
    import chrombpnet.helpers.preprocessing.reads_to_bigwig as reads_to_bigwig
    import chrombpnet.helpers.preprocessing.analysis.build_pwm_from_bigwig as build_pwm_from_bigwig

    options = as_options(options)
    bigwig = getattr(options, "input_bigwig", None)
    if bigwig is None:
        # --plus-shift/--minus-shift are not exposed by the pipeline parsers,
        # so the shift is always estimated; reads_to_bigwig requires the
        # attributes.
        reads_to_bigwig.main(options.step(
            output_prefix=layout.aux("data"), plus_shift=None, minus_shift=None))
        bigwig = layout.aux("data_unstranded.bw")
    else:
        warn_if_dnase_convention_assumed(options)

    folds = json.load(open(options.chr_fold_path))
    if not folds["valid"]:
        raise ValueError("Validation chromosome list is empty")
    build_pwm_from_bigwig.main(options.step(
        bigwig=bigwig, output_prefix=layout.evaluation("bw_shift_qc"),
        chr=folds["valid"][0], pwm_width=PWM_WIDTH))
    return bigwig


def warn_if_dnase_convention_assumed(options):
    """A DNASE bigwig given with -ibw but no --dnase-convention is taken to
    use the default, which reverse-complement augmentation relies on and
    nothing can check -- and which a chrombpnet 1.x bigwig does not use."""
    from chrombpnet import conventions

    if (getattr(options, "data_type", None) != "DNASE"
            or getattr(options, "dnase_convention", None) is not None
            or getattr(options, "data_aug", None) == "forward-only"):
        return
    warnings.warn(
        f"-ibw: the DNASE bigwig is taken to record cuts as "
        f"--dnase-convention {conventions.DNASE_CONVENTION_DEFAULT} (the "
        f"default), which reverse-complement augmentation relies on. A bigwig "
        f"from chrombpnet 1.x records them as `right`: pass --dnase-convention "
        f"right, or right-deprecated to reproduce 1.x training.",
        RuntimeWarning, stacklevel=3)


def find_hyperparameters(options, layout, finder):
    """Run one of the hyperparameter finders, writing into ``auxiliary/``."""
    finder.main(as_options(options).step(output_prefix=layout.aux()))


def check_bias_model_quality(layout, ignore=False):
    """Reject a bias model that anti-correlates with observed peak counts.

    A bias model whose predicted counts run *opposite* to observed counts in
    peaks is not describing sequence bias, and subtracting it makes the
    accessibility model worse rather than better -- so this refuses by
    default and the refusal is worth heeding.

    ``ignore`` downgrades it to a warning, for the case where the check is
    known to be measuring the data rather than the model: where peak and
    background counts overlap so heavily that no threshold can select a
    representative low-count nonpeak set, the bias model is fitted to a
    truncated tail and anti-correlates for reasons that no amount of
    retraining will fix. Proceeding then is a deliberate choice to study a
    sample the pipeline was not designed for, and the resulting model's
    bias correction should not be trusted quantitatively.
    """
    metrics_path = layout.unprefixed("evaluation", "bias_metrics.json")
    bias_metrics = json.load(open(metrics_path))
    pearsonr = bias_metrics["counts_metrics"]["peaks"]["pearsonr"]
    print("Bias model pearsonr performance in peaks is: {}".format(
        str(np.round(pearsonr, 2))))
    if pearsonr > MIN_BIAS_PEAK_PEARSONR:
        return
    message = ("Bias model has strongly negative peak count correlation "
               f"({pearsonr:.2f} <= {MIN_BIAS_PEAK_PEARSONR})")
    if not ignore:
        raise ValueError(message)
    # Loud, and repeated at the end of the run, because a warning in the
    # middle of a two-day job's log is a warning nobody reads.
    banner = "!" * 78
    print(f"\n{banner}\n!! WARNING: {message}\n"
          "!! --ignore-bias-flag was given, so training continues anyway.\n"
          "!! The bias correction in this model is not trustworthy; treat its\n"
          "!! scores as descriptive, not comparable with runs that passed.\n"
          f"{banner}\n", flush=True)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def run_predictions(options, layout, output_prefix, model_path, peaks, nonpeaks):
    """Predict over the test split and write metrics.

    ``peaks``/``nonpeaks`` are the regions to score, in place of ``-p``/``-n``;
    the string ``"None"`` disables a set, the sentinel ``initializers`` checks
    for.
    """
    import chrombpnet.training.predict as predict

    predict.main(as_options(options).step(
        replacing={"peaks": peaks, "nonpeaks": nonpeaks},
        output_prefix=output_prefix, model_h5=model_path))


def write_bias_motif_table(options, layout):
    """Write the motif table marginal footprinting inserts, and return its path.

    Not file-prefixed: this filename never carried the prefix.
    """
    if options.data_type not in BIAS_MOTIFS:
        raise ValueError(
            f"unknown data type {options.data_type!r}; expected one of "
            f"{sorted(BIAS_MOTIFS)}")
    path = layout.unprefixed("auxiliary", "motif_to_pwm.tsv")
    pd.DataFrame(BIAS_MOTIFS[options.data_type]).to_csv(
        path, sep="\t", header=False, index=False)
    return path


def run_marginal_footprints(options, layout, model_path, regions, output_prefix,
                            motifs_to_pwm):
    """Insert each bias motif and measure the model's footprint response."""
    import chrombpnet.evaluation.marginal_footprints.marginal_footprinting as marginal_footprinting

    marginal_footprinting.main(as_options(options).step(
        model_h5=model_path, regions=regions, output_prefix=output_prefix,
        motifs_to_pwm=motifs_to_pwm, ylim=None))


def subsample_peaks_for_interpretation(layout, peaks_path, size=None):
    """Sample at most ``size`` (default INTERPRET_SUBSAMPLE_SIZE) peaks and
    return the BED path, which keeps its ``30K_`` name whatever the size.

    Also creates ``auxiliary/interpret_subsample/``, with ``exist_ok=False`` as
    before, so a rerun into a populated directory fails rather than mixing
    contribution scores from two runs.
    """
    if size is None:
        size = INTERPRET_SUBSAMPLE_SIZE
    peaks = pd.read_csv(peaks_path, sep="\t", header=None)
    if peaks.shape[0] > size:
        sub_peaks = peaks.sample(size, random_state=INTERPRET_SUBSAMPLE_SEED)
    else:
        sub_peaks = peaks
    subsample_path = layout.aux("30K_subsample_peaks.bed")
    sub_peaks.to_csv(subsample_path, sep="\t", header=False, index=False)
    layout.create_interpret_subsample_dir()
    return subsample_path


def run_interpret(options, layout, model_path, regions, output_prefix,
                  profile_or_counts):
    """Compute DeepLIFT-SHAP contribution scores."""
    import chrombpnet.evaluation.interpret.interpret as interpret

    interpret.main(as_options(options).step(
        profile_or_counts=profile_or_counts, regions=regions,
        model_h5=model_path, output_prefix=output_prefix, debug_chr=None))


def run_modisco(options, layout, scores_name, kind, pdf_name):
    """Discover motifs from one contribution track and render its report.

    ``kind`` is "profile" or "counts" and selects both the results filename and
    the ``evaluation/modisco_<kind>/`` report directory. ``options`` supplies the
    report options (``--tomtom-lite``, ``--modisco-max-seqlets``,
    ``--modisco-window``), each absent unless given; absent ones keep
    run_motifs' defaults.
    """
    from chrombpnet.evaluation.modisco.api import generate_report, run_motifs
    import chrombpnet.evaluation.modisco.convert_html_to_pdf as convert_html_to_pdf

    meme_file = get_default_data_path(DefaultDataFile.motifs_meme)
    results = layout.interpret_subsample(
        f"modisco_results_{kind}_scores.h5")
    report_dir = layout.modisco_dir(kind)

    settings = {name: getattr(options, option)
                for name, option in (("max_seqlets", "modisco_max_seqlets"),
                                     ("window", "modisco_window"))
                if getattr(options, option, None) is not None}
    run_motifs(layout.interpret_subsample(f"{scores_name}.{kind}_scores.h5"),
               results, **settings)
    generate_report(results, report_dir, meme_file,
                    tomtom_lite=bool(getattr(options, "tomtom_lite", None)))
    convert_html_to_pdf.main(report_dir + "motifs.html",
                             layout.evaluation(pdf_name))


def render_report(options, layout, module, command):
    """Render the HTML and PDF summary report for a finished run."""
    options = as_options(options)
    module.main(options.step(input_dir=options.output_dir, command=command))
