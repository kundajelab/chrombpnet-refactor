#!/usr/bin/env python
"""Run one step of the original chrombpnet, imported from its own checkout.

In the original package's own environment (TensorFlow 2.8 on Python 3.8, an
install of that checkout with its pinned requirements), on a GPU, the steps in
order (README.md gives them in full):

    run_original_retrain.py --original-repo R chrombpnet bias train ...
    run_original_retrain.py --original-repo R --ignore-bias-flag chrombpnet train ...
    run_original_retrain.py --original-repo R predict -m ... -o ...

`chrombpnet` runs the original's command line (chrombpnet.CHROMBPNET.main) on
the arguments after it; `predict` runs its chrombpnet/training/predict.py,
whose own command line takes the arguments after it. Both come from
kundajelab/chrombpnet (--original-repo, at 09938fd): any other chrombpnet on
sys.path (an editable install of chrombpnet-refactor) is taken off first, and
the import is checked.

Why `predict` on its own: the original's `train` stops once the model is
trained, before the test-set predictions its `pipeline` makes next (the rest
of `pipeline` -- footprints, DeepSHAP on its own subsample, MoDISco, reports
-- is not needed here). The `predict` step makes that one call as
`pipeline` does: the trained chrombpnet.h5 on auxiliary/filtered.peaks.bed,
nonpeaks "None", into evaluation/chrombpnet (_metrics.json, _predictions.h5).

One change, with --ignore-bias-flag and only for `train` and `pipeline`: the
original asserts that its bias model's counts Pearson r in peaks is above -0.5
and stops otherwise. The engines check the same thing (stages.py,
MIN_BIAS_PEAK_PEARSONR = -0.5), and the sweep ran them with
--ignore-bias-flag, which makes it a warning -- the stomach samples' bias
models fail it with every engine. This does the same to the original: that one
assert becomes the same warning; nothing else in it changes.
"""

import argparse
import inspect
import os
import sys
import warnings

#: The original's check, chrombpnet/pipelines.py chrombpnet_train_pipeline.
BIAS_CHECK = 'assert(bias_metrics["counts_metrics"]["peaks"]["pearsonr"] > -0.5)'
MIN_BIAS_PEAK_PEARSONR = -0.5


def import_original(repo):
    """Import chrombpnet from `repo` and nowhere else."""
    original = os.path.realpath(repo)
    sys.path[:] = [original] + [
        p for p in sys.path
        if p and os.path.realpath(p) != original
        and not os.path.isfile(os.path.join(p, "chrombpnet", "__init__.py"))]
    import chrombpnet
    where = os.path.dirname(os.path.dirname(os.path.realpath(chrombpnet.__file__)))
    if where != original:
        sys.exit(f"Error: imported chrombpnet from {where}, not the original at {repo}")
    return where


def warn_bias_check(bias_metrics):
    """The engines' --ignore-bias-flag warning, in place of the original's assert."""
    pearsonr = bias_metrics["counts_metrics"]["peaks"]["pearsonr"]
    if pearsonr > MIN_BIAS_PEAK_PEARSONR:
        return
    message = ("Bias model has strongly negative peak count correlation "
               f"({pearsonr:.2f} <= {MIN_BIAS_PEAK_PEARSONR})")
    banner = "!" * 78
    print(f"\n{banner}\n!! WARNING: {message}\n"
          "!! --ignore-bias-flag was given, so training continues anyway (the original\n"
          "!! asserts here; run_original_retrain.py made it this warning).\n"
          f"{banner}\n", flush=True)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def relax_bias_check():
    """Recompile the original's chrombpnet_train_pipeline with its bias-model
    assert replaced by warn_bias_check, in its own module and at its own line
    numbers; refuse if the assert is not there exactly once."""
    import chrombpnet.pipelines as pipelines
    function = pipelines.chrombpnet_train_pipeline
    source = inspect.getsource(function)
    if source.count(BIAS_CHECK) != 1:
        sys.exit(f"Error: {inspect.getsourcefile(function)} does not hold the bias-model check "
                 f"exactly once ({BIAS_CHECK}); is --original-repo the 09938fd original?")
    source = source.replace(BIAS_CHECK, "_warn_bias_check(bias_metrics)")
    code = compile("\n" * (function.__code__.co_firstlineno - 1) + source,
                   inspect.getsourcefile(function), "exec")
    pipelines._warn_bias_check = warn_bias_check
    exec(code, pipelines.__dict__)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original-repo", required=True,
                    help="a checkout of kundajelab/chrombpnet (the original package)")
    ap.add_argument("--ignore-bias-flag", action="store_true",
                    help="the bias-model check warns instead of stopping (train, pipeline)")
    ap.add_argument("step", choices=("chrombpnet", "predict"))
    ap.add_argument("argv", nargs=argparse.REMAINDER,
                    help="the original's own arguments for that step")
    args = ap.parse_args(argv)

    where = import_original(args.original_repo)
    print(f"original chrombpnet: {where}", flush=True)
    if args.step == "chrombpnet":
        if args.ignore_bias_flag:
            relax_bias_check()
        from chrombpnet import CHROMBPNET
        sys.argv = ["chrombpnet"] + args.argv
        CHROMBPNET.main()
    else:
        from chrombpnet.training import predict
        from chrombpnet.training.utils import argmanager
        sys.argv = ["predict"] + args.argv
        predict.main(argmanager.fetch_predict_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
