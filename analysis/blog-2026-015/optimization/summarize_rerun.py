#!/usr/bin/env python3
"""Collate run_pred_bw_rerun.sh, run_pred_bw_kit_fused.sh and run_kit_modes.sh: per run (GPU and
job), the median pred_bw time of each arm over every fold and round, and the
speedups the blog tables quote. Runs are kept apart: arms are only compared
within one job.

    python summarize_rerun.py ANALYSIS_DIR      # the jobs' OUT_ROOT (site.env's OPTIMIZATION_OUT)

Also prints each arm's median by its position in the round, so an order
effect (the cold-cache problem this re-run exists to remove) would show, and
its median peak memory where the runs recorded one (peak_rss.py: the peak
resident memory of the largest process).
"""

import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("usage: python summarize_rerun.py ANALYSIS_DIR  (the jobs' OUT_ROOT: site.env's OPTIMIZATION_OUT)")
ROOT = sys.argv[1]
ARMS = ["original", "none", "kit", "ours", "both", "main", "kitfk", "fkonly", "fk", "mallocenv", "nomalloc",
        "stock", "stockdet", "stockmalloc", "exact", "fast",   # run_kit_modes.sh
        "evaluate", "attribute", "marginalize",         # run_cherimoya.sh
        "orig_predict", "ref_predict", "ref_predict_fast", "orig_attr", "ref_attr", "orig_contribs", "ref_contribs"]  # run_headtohead.sh
RATIOS = [("kit", "none"), ("ours", "none"), ("both", "none"), ("main", "none"),
          ("kitfk", "none"), ("fkonly", "none"), ("fk", "none"),
          ("kitfk", "kit"), ("main", "original"), ("fk", "main"), ("mallocenv", "main"), ("main", "nomalloc"), ("main", "both"),
          ("fast", "stock"), ("exact", "stock"), ("exact", "fast"), ("stockdet", "stock"),
          ("stockmalloc", "stock"),
          ("ref_predict", "orig_predict"), ("ref_predict_fast", "orig_predict"), ("ref_predict_fast", "ref_predict"),
          ("ref_attr", "orig_attr"), ("ref_contribs", "orig_contribs")]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "fold_*", "timings.tsv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["peak"] = os.path.join(os.path.dirname(path), "r{}_{}".format(row["round"], row["arm"]),
                                           "peak_rss_mb")
                rows.append(row)
    if not rows:
        sys.exit("no timings.tsv under " + ROOT)
    by_gpu = defaultdict(lambda: defaultdict(list))
    by_position = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    peaks = defaultdict(lambda: defaultdict(list))   # GB
    notes = defaultdict(set)
    for row in rows:
        gpu = row["run"]
        if row["status"] == "ok" and row["seconds"]:
            seconds = float(row["seconds"])
            by_gpu[gpu][row["arm"]].append(seconds)
            by_position[gpu][row["arm"]][int(row["position"])].append(seconds)
            if os.path.exists(row["peak"]):
                peaks[gpu][row["arm"]].append(float(open(row["peak"]).read()) / 1024)
        else:
            notes[gpu].add("{} {}".format(row["arm"], row["status"]))
    for gpu in sorted(by_gpu):
        arms = by_gpu[gpu]
        folds = sorted({row["fold"] for row in rows if row["run"] == gpu})
        print("\n## {}  (folds {})".format(gpu, ",".join(folds)))
        print("{:<9} {:>4} {:>9} {:>9} {:>9}{}".format("arm", "n", "median s", "min s", "max s",
                                                     "  peak GB (min-max)" if peaks[gpu] else ""))
        for arm in ARMS:
            if arm in arms:
                values = arms[arm]
                peak = peaks[gpu].get(arm)
                print("{:<9} {:>4} {:>9.1f} {:>9.1f} {:>9.1f}{}".format(
                    arm, len(values), statistics.median(values), min(values), max(values),
                    "  {:6.2f} ({:.2f}-{:.2f})".format(statistics.median(peak), min(peak), max(peak))
                    if peak else ""))
        for faster, slower in RATIOS:
            if faster in arms and slower in arms:
                a, b = statistics.median(arms[slower]), statistics.median(arms[faster])
                print("  {:>8} vs {:<8} {:6.1f} s -> {:6.1f} s  ({:.2f}x)".format(
                    faster, slower, a, b, a / b))
        print("  median by position in the round (0 = first):")
        for arm in ARMS:
            if arm in by_position[gpu]:
                cells = ["{}:{:.1f}".format(pos, statistics.median(v))
                         for pos, v in sorted(by_position[gpu][arm].items())]
                print("    {:<9} {}".format(arm, "  ".join(cells)))
        if notes[gpu]:
            print("  not timed: " + ", ".join(sorted(notes[gpu])))


if __name__ == "__main__":
    main()
