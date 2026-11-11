#!/usr/bin/env python3
"""Blog 2026-015's Sections 3 and 4 cells from run_sections34.sh: one GPU per
run tag (*_sections34*), every number from the same job, dataset, fold, model
(the full model, -cm), regions and batch size.

    python summarize_sections34.py ANALYSIS_DIR      # the jobs' OUT_ROOT (site.env's OPTIMIZATION_OUT)

Section 3 (chrombpnet 1.x vs 1.x with the optimization kit, `fast` where the
  run has it, else `exact`): per lever, the stage it replaces: stock (the
  `stages` arm, stock_stage_timer.py) -> the kit (its run record,
  opt_manifest.json; summarize_kit_levers.py's pairing); overall, stock
  pred_bw -> the kit's pred_bw (wall clock); malloc_env, stock -> stockmalloc.
Section 4 (chrombpnet 1.x vs chrombpnet-refactor): stock -> refactor4 (the
  original's HDF5 compression, gzip 4), with each forward route; the inference
  row, stock's forward stage -> chrombpnet-refactor's forward alone on each
  route (forward_timer.py) beside the kit's own forward (its run record).
Medians over the rounds; peak memory is peak_rss.py's (the largest process).
"""

import csv
import glob
import json
import importlib.util
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("usage: python summarize_sections34.py ANALYSIS_DIR  (the jobs' OUT_ROOT: site.env's OPTIMIZATION_OUT)")
ROOT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("stock", "stages", "stockmalloc", "fast_k1", "fast", "exact", "refactor4", "refactor4_fast", "refactor4_exact",
        "refactor4_nomalloc", "refactor4_prev", "refactor", "refactor_nomalloc", "forward_eager", "forward_fast",
        "forward_exact", "original")


def _levers():
    spec = importlib.util.spec_from_file_location("summarize_kit_levers", os.path.join(HERE, "summarize_kit_levers.py"))
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0], ROOT]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def _gb(peaks, arm):
    return "%.2f" % median(peaks[arm]) if peaks.get(arm) else "-"


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def speedup(before, after):
    if before is None or after is None:
        return "n/a"
    return "%.1fx (%.1f s -> %.1f s)" % (before / after, before, after)


def main():
    levers = _levers()
    runs = sorted(glob.glob(os.path.join(ROOT, "*_sections34*", "fold_*")))
    if not runs:
        sys.exit("no *_sections34*/fold_* under " + ROOT)
    for run in runs:
        seconds, peaks, notes = defaultdict(list), defaultdict(list), []
        for row in csv.DictReader(open(os.path.join(run, "timings.tsv")), delimiter="\t"):
            if row["status"] != "ok" or not row["seconds"]:
                notes.append("%s %s" % (row["arm"], row["status"]))
                continue
            seconds[row["arm"]].append(float(row["seconds"]))
            peak = os.path.join(run, "r%s_%s" % (row["round"], row["arm"]), "peak_rss_mb")
            if os.path.exists(peak):
                peaks[row["arm"]].append(float(open(peak).read()) / 1024)
        med = {arm: median(values) for arm, values in seconds.items()}
        print("\n## %s  (%s)" % (os.path.basename(os.path.dirname(run)), os.path.basename(run)))
        for arm in ARMS:
            if arm in seconds:
                print("  %-9s n=%-2d median %6.1f s   peak %s GB" % (
                    arm, len(seconds[arm]), med[arm],
                    "%.2f" % median(peaks[arm]) if peaks.get(arm) else "-"))
        if notes:
            print("  not timed: " + ", ".join(sorted(set(notes))))

        # Section 3, per lever: stock's stage (stages arm) -> the kit's (its run record)
        stock = [levers.stock_stages(d) for d in sorted(glob.glob(os.path.join(run, "r*_stages")))
                 if os.path.exists(os.path.join(d, "stages.json"))]
        kit_mode = next((m for m in ("fast_k1", "fast", "exact")
                         if glob.glob(os.path.join(run, "r*_%s" % m, "opt_manifest.json"))), "exact")
        kit = [levers.kit_stages(p) for p in sorted(glob.glob(os.path.join(run, "r*_%s" % kit_mode, "opt_manifest.json")))]
        if stock and kit:
            s = {k: median([r.get(k) for r in stock]) for k in ("featurise", "metrics", "bigwig_write", "forward",
                                                                 "h5_write", "finish_imports", "teardown")}
            k = {k: median([r.get(k) for r in kit]) for k in ("featurise", "metrics", "bigwig_write", "forward",
                                                               "h5_write", "finish_imports_overlapped")}
            print("  Section 3, per lever (stock stage -> the kit's, `%s`):" % kit_mode)
            for lever, before, after in (("prefetch", s["featurise"], k["featurise"]),
                                         ("metrics_stage", s["metrics"], k["metrics"]),
                                         ("bigwig_writer", s["bigwig_write"], k["bigwig_write"]),
                                         ("forward_route", s["forward"], k["forward"]),
                                         ("h5_writer", s["h5_write"], k["h5_write"])):
                print("    %-14s %s" % (lever, speedup(before, after)))
            print("    %-14s %.1f s of imports moved under the forward (stock's finish-stage imports %.1f s)" % (
                "preimport", k["finish_imports_overlapped"] or 0, s["finish_imports"] or 0))
            print("    %-14s %.1f s of teardown skipped" % ("exit_fast", s["teardown"] or 0))
        for mode in ("fast_k1", "fast", "exact"):
            if mode in med:
                print("  Section 3, overall (1.x -> 1.x with the kit): stock -> %-7s  %s" % (mode, speedup(med.get("stock"), med.get(mode))))
        if "stockmalloc" in med:
            print("  malloc_env in the original: stock -> stockmalloc  %s; peak %s -> %s GB" % (
                speedup(med.get("stock"), med.get("stockmalloc")), _gb(peaks, "stock"), _gb(peaks, "stockmalloc")))
        for arm, label in (("refactor4", "gzip 4, PyTorch TF32 forward"), ("refactor4_fast", "gzip 4, Triton TF32 (--predict-fused-kernel-fast)"),
                           ("refactor4_exact", "gzip 4, Triton fp32 (--predict-fused-kernel-exact)"), ("refactor", "gzip 9")):
            if arm in med:
                print("  Section 4, overall (1.x -> chrombpnet-refactor, %s): stock -> %s  %s" % (label, arm, speedup(med.get("stock"), med.get(arm))))
        if "refactor4_prev" in med:
            print("  Streaming, incremental bigWig writer, faster observed read (fa45de5 -> HEAD): refactor4_prev -> refactor4  %s" % speedup(med.get("refactor4_prev"), med.get("refactor4")))
        for nomalloc, tuned in (("refactor4_nomalloc", "refactor4"), ("refactor_nomalloc", "refactor")):
            if nomalloc in med:
                print("  malloc_env in chrombpnet-refactor: %s -> %s  %s; peak %s -> %s GB" % (
                    nomalloc, tuned, speedup(med.get(nomalloc), med.get(tuned)), _gb(peaks, nomalloc), _gb(peaks, tuned)))

        # Section 4, inference: stock's forward stage -> chrombpnet-refactor's forward alone, per route,
        # beside the kit's own forward (its run record)
        stock_forward = median([r.get("forward") for r in stock]) if stock else None
        forwards = {}
        for arm in ("forward_eager", "forward_fast", "forward_exact"):
            values = []
            for path in sorted(glob.glob(os.path.join(run, "r*_%s" % arm, "forward.json"))):
                values += json.load(open(path))["seconds"]
            if values:
                forwards[arm] = median(values)
        kit_forward = {mode: median([levers.kit_stages(p).get("forward") for p in
                                     sorted(glob.glob(os.path.join(run, "r*_%s" % mode, "opt_manifest.json")))])
                       for mode in ("fast_k1", "fast", "exact")}
        if forwards or any(kit_forward.values()):
            print("  Section 4, inference (forward over the regions; stock's forward stage %s):" % (
                "%.1f s" % stock_forward if stock_forward else "n/a"))
            for arm, label in (("forward_eager", "chrombpnet-refactor, PyTorch (cuDNN, TF32)"),
                               ("forward_fast", "chrombpnet-refactor, Triton TF32 (--predict-fused-kernel-fast)"),
                               ("forward_exact", "chrombpnet-refactor, Triton fp32 (--predict-fused-kernel-exact)")):
                if arm in forwards:
                    print("    %-62s %s" % (label, speedup(stock_forward, forwards[arm])))
            for mode, label in (("fast_k1", "the kit's own forward, `fast` with Triton TF32 (L40S hook)"),
                                ("fast", "the kit's own forward, `fast`"), ("exact", "the kit's own forward, `exact`")):
                if kit_forward.get(mode):
                    print("    %-62s %s" % (label, speedup(stock_forward, kit_forward[mode])))
        if "original" in med:
            print("  check: original (09938fd) %.1f s vs stock (1.0.1, the kit's --mode off) %.1f s" % (med["original"], med.get("stock") or float("nan")))


if __name__ == "__main__":
    main()
