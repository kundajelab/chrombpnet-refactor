#!/usr/bin/env python
"""Score a finished run's 30K peak subsample again with DeepSHAP, and keep it.

    python run_contribution.py --run-dir <root>/<engine>/<ASSAY>/<ENCID>/fold_N \
        [--heads counts profile] [--profile-weighting midpoint|stopgrad] [--device cuda]

In the environment of the run's engine (envs/chrombpnet-<engine>.yml), on a
GPU like the run's.

`chrombpnet pipeline` scores the ChromBPNet (no-bias) model on a fixed 30K
subsample of the peaks, for TF-MoDISco. Before the change that keeps both
heads (CHANGELOG: "`pipeline` and `qc` keep both heads' contribution scores")
it scored only the profile head, so the sweep's finished runs have
`chrombpnet_nobias.profile_scores.h5` and no counts scores. This scores the
heads that are missing, exactly as the pipeline would have:

  - the same regions, model, genome, seed, number of shuffled references and
    batch size, read from the pipeline's own record of its interpret step
    (model/auxiliary/interpret_subsample/chrombpnet_nobias.interpret.args.json);
  - through `chrombpnet.evaluation.interpret.interpret.main`, the function
    the pipeline calls;
  - into the same directory and file names the pipeline now writes:
    chrombpnet_nobias.counts_scores.h5 (and .profile_scores.h5 with
    --heads profile --force). Each holds raw (one-hot), shap (hypothetical)
    and projected_shap, one row per line of
    chrombpnet_nobias.interpreted_regions.bed.

--profile-weighting midpoint|stopgrad rescores the profile head with that
weighting of the profile target (chrombpnet's --contribs-profile-stopgrad;
midpoint is its default, and what chrombpnet 1.x computes), into files of
their own, chrombpnet_nobias.<weighting>.profile_scores.h5, beside the
pipeline's, which are left alone. run_summarize.py --profile-stopgrad off
compares the midpoint ones (on: the pipeline's). The installed chrombpnet must have the option; an older one
would score with whatever it did by default.

Scores are written under a .partial prefix and renamed when complete, so an
interrupted job leaves nothing that looks finished. The regions actually
scored must equal the pipeline's interpreted_regions.bed line for line, or
nothing is renamed: scores of two region sets never share a directory.
A record of what was run is written beside them,
chrombpnet_nobias.contribution.args.json.
"""

import argparse
import datetime
import filecmp
import inspect
import json
import os
import sys

HEADS = ("counts", "profile")
PREFIX = "chrombpnet_nobias"
PROFILE_WEIGHTINGS = ("midpoint", "stopgrad")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="one run: <root>/<engine>/<ASSAY>/<ENCID>/fold_N")
    ap.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS),
                    help="heads to score (default: both); ones already scored are skipped")
    ap.add_argument("--force", action="store_true",
                    help="score the requested heads even if their files exist")
    ap.add_argument("--device", default=None,
                    help="device for the engine (default: what the pipeline used)")
    ap.add_argument("--interpret-batch-size", type=int, default=None,
                    help="override the pipeline's contribution-scoring batch size")
    ap.add_argument("--profile-weighting", choices=PROFILE_WEIGHTINGS, default=None,
                    help="rescore the profile head with this weighting of the profile target "
                         "(stopgrad: --contribs-profile-stopgrad), into "
                         f"{PREFIX}.<weighting>.profile_scores.h5")
    args = ap.parse_args(argv)
    if args.profile_weighting and args.heads != ["profile"]:
        ap.error("--profile-weighting rescores the profile head only (--heads profile); "
                 "the counts head does not depend on it")

    interp_dir = os.path.join(args.run_dir, "model", "auxiliary", "interpret_subsample")
    pipeline_prefix = os.path.join(interp_dir, PREFIX)
    record = f"{pipeline_prefix}.interpret.args.json"
    # Where the scores go: the pipeline's own names, or the weighting's.
    final = (f"{pipeline_prefix}.{args.profile_weighting}" if args.profile_weighting
             else pipeline_prefix)
    if not os.path.isfile(record):
        sys.exit(f"Error: no {record}. Is {args.run_dir} a finished `chrombpnet pipeline` run?")

    todo = [h for h in args.heads
            if args.force or not os.path.isfile(f"{final}.{h}_scores.h5")]
    for head in args.heads:
        if head not in todo:
            print(f"have: {final}.{head}_scores.h5 -- skipped (--force rescoring)")
    if not todo:
        print("Nothing to score.")
        return 0

    with open(record) as fh:
        recorded = json.load(fh)
    for key in ("regions", "model_h5", "genome"):
        if not os.path.isfile(recorded.get(key) or ""):
            sys.exit(f"Error: {record} names {key}={recorded.get(key)!r}, which does not exist")

    # The namespace the pipeline gave interpret.main, but for the missing
    # heads and a scratch prefix.
    partial = f"{final}.partial"
    ns = argparse.Namespace(**recorded)
    ns.profile_or_counts = todo
    ns.output_prefix = partial
    ns.debug_chr = None
    if args.profile_weighting:
        ns.contribs_profile_stopgrad = args.profile_weighting == "stopgrad"
    if args.device:
        ns.device = args.device
    if args.interpret_batch_size:
        ns.interpret_batch_size = args.interpret_batch_size

    from chrombpnet import engines
    from chrombpnet.evaluation.interpret import interpret

    # The engine the pipeline ran with (its --engine is in the record), set up
    # as the CLI does. Not left to the checkpoint's suffix: the PyTorch
    # engine can read a TensorFlow .h5 too.
    if not getattr(ns, "engine", None):
        sys.exit(f"Error: {record} does not say which engine the pipeline used")
    try:
        engine = engines.configure_from_args(ns, require=True)
    except (ValueError, engines.EngineUnavailable, engines.EngineNotImplemented) as error:
        sys.exit(f"Error: {error}")
    if args.profile_weighting and \
            "profile_stop_gradient" not in inspect.signature(engine.attribute).parameters:
        sys.exit("Error: the installed chrombpnet "
                 f"({os.path.dirname(os.path.dirname(os.path.realpath(engines.__file__)))}) "
                 "predates --contribs-profile-stopgrad, so it cannot choose the profile weighting")
    print(f"run:      {args.run_dir}")
    print(f"engine:   {engine.name}")
    print(f"model:    {ns.model_h5}")
    print(f"regions:  {ns.regions}")
    print(f"heads:    {' '.join(todo)}")
    if args.profile_weighting:
        print(f"profile weighting: {args.profile_weighting}")
    print(f"seed {getattr(ns, 'seed', None)}, shuffles {getattr(ns, 'num_shuffles', 20)}, "
          f"batch size {interpret.resolve_interpret_batch_size(ns)}, device {getattr(ns, 'device', None)}")
    started = datetime.datetime.now()
    interpret.main(ns)

    regions_new = f"{partial}.interpreted_regions.bed"
    regions_old = f"{pipeline_prefix}.interpreted_regions.bed"
    if os.path.isfile(regions_old):
        if not filecmp.cmp(regions_new, regions_old, shallow=False):
            sys.exit(f"Error: the regions scored ({regions_new}) differ from the pipeline's "
                     f"({regions_old}); the new scores are left under {partial}.*")
        os.remove(regions_new)
    else:
        os.replace(regions_new, regions_old)
    for head in todo:
        os.replace(f"{partial}.{head}_scores.h5", f"{final}.{head}_scores.h5")
        print(f"wrote:    {final}.{head}_scores.h5")
    os.remove(f"{partial}.interpret.args.json")

    with open(f"{final}.contribution.args.json", "w") as fh:
        json.dump({
            "script": os.path.abspath(__file__),
            "started": started.isoformat(timespec="seconds"),
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
            "heads": todo,
            "profile_weighting": args.profile_weighting,
            "engine": engine.name,
            "chrombpnet": os.path.dirname(os.path.dirname(os.path.realpath(engines.__file__))),
            "interpret_args": {k: v for k, v in vars(ns).items() if k != "output_prefix"},
        }, fh, indent=4, default=str)
    print(f"Done in {(datetime.datetime.now() - started).total_seconds() / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
