#!/usr/bin/env python3
"""A chrombpnet CLI command (the original package's or chrombpnet-refactor's,
whichever is imported) with its output writers replaced by no-ops, so that it
does only the work cherimoya's matching command does: the blog's head-to-head
rows (run_headtohead.sh).

    python headtohead.py SLICE <chrombpnet CLI arguments>

    predict    pred_bw with one model: the predictions over the regions and
               their metrics against the observed bigWig (cherimoya
               evaluate); the predicted bigWig and the predictions HDF5 are
               not written
    attribute  contribs_bw: the DeepLIFT/SHAP attributions (cherimoya
               attribute); the scores HDF5 and its bigWig are not written
    none       the command as is (its full time, for the total rows)

Everything else is the package's own code path, unchanged. At the end it
prints how many times each writer was skipped, and refuses (exit 3) if a
writer it was meant to skip never ran, so a slice cannot silently measure
something else.

chrombpnet-refactor's pred_bw writes in forked worker processes since da8536e
(chrombpnet/finish_stage.py) and its bigWig goes through
bigwig_helper.BigWigWriter, not write_bigwig. So the skips are counted in a
file every process appends to, and the writer stubbed there is BigWigWriter:
the worker still receives each chunk and computes the counts-scaled profile,
and writes nothing.
"""

import os
import sys
import tempfile
from collections import Counter

#: Every process (the CLI's and its forked workers) appends one line per skip.
SKIP_LOG = os.environ.setdefault(
    "HEADTOHEAD_SKIP_LOG", os.path.join(tempfile.mkdtemp(prefix="headtohead-"), "skipped.log"))


def _note(name):
    with open(SKIP_LOG, "a") as handle:
        handle.write(name + "\n")


def _skipped():
    try:
        with open(SKIP_LOG) as handle:
            return Counter(line.strip() for line in handle if line.strip())
    except FileNotFoundError:
        return Counter()


def _skip(name):
    def stub(*args, **kwargs):
        _note(name)
    return stub


class _NoBigWig:
    """bigwig_helper.BigWigWriter that writes nothing (chrombpnet-refactor's
    streaming bigWig writer)."""

    def __init__(self, *args, **kwargs):
        pass

    def add(self, start, rows):
        pass

    def close(self):
        _note("bigwig")


def _refactor():
    try:
        import chrombpnet.predictions  # noqa: F401  -- only chrombpnet-refactor has it
        return True
    except ImportError:
        return False


def stub_predict(refactor):
    from chrombpnet.evaluation.make_bigwigs import bigwig_helper, predict_to_bigwig
    bigwig_helper.write_bigwig = _skip("bigwig")
    if hasattr(bigwig_helper, "BigWigWriter"):
        bigwig_helper.BigWigWriter = _NoBigWig
    predict_to_bigwig.write_predictions_h5py = _skip("predictions_h5")
    if refactor:
        # The CLI checks every promised output before skipping teardown;
        # promise only what this slice still writes, so it exits as usual.
        expected = predict_to_bigwig._expected_outputs
        predict_to_bigwig._expected_outputs = lambda args, models: [
            path for path in expected(args, models)
            if not path.endswith((".bw", "_predictions.h5"))]
    return ("bigwig", "predictions_h5")


def stub_attribute(refactor):
    from chrombpnet.evaluation.make_bigwigs import importance_hdf5_to_bigwig
    importance_hdf5_to_bigwig.main = _skip("scores_bigwig")
    if refactor:
        from chrombpnet.evaluation.interpret import interpret
        interpret._save_scores = _skip("scores_h5")
    else:
        import deepdish
        deepdish.io.save = _skip("scores_h5")
    return ("scores_h5", "scores_bigwig")


def main():
    slice_, argv = sys.argv[1], sys.argv[2:]
    refactor = _refactor()
    expected = {"predict": stub_predict, "attribute": stub_attribute}.get(slice_, lambda _: ())(refactor)

    reported = []

    def report():
        if reported:
            return
        reported.append(True)
        skipped = _skipped()
        print("headtohead: %s slice of %s; writers skipped: %s"
              % (slice_, "chrombpnet-refactor" if refactor else "chrombpnet", dict(skipped)), flush=True)
        missing = [name for name in expected if not skipped[name]]
        if missing:
            sys.exit("headtohead: %s never ran, so this was not the %s slice" % (", ".join(missing), slice_))

    if refactor:
        # pred_bw ends in exit_fast's os._exit, which skips everything after it.
        from chrombpnet import exit_fast
        exit_now = exit_fast.exit_now
        exit_fast.exit_now = lambda outputs: (report(), exit_now(outputs))
    from chrombpnet.CHROMBPNET import main as cli_main
    sys.argv = ["chrombpnet"] + argv
    try:
        cli_main()
    except SystemExit as exit_:
        if exit_.code not in (None, 0):
            raise
    report()


if __name__ == "__main__":
    main()
