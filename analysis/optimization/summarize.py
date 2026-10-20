#!/usr/bin/env python3
"""Collate benchmark_pred_bw.py runs: per GPU, the median pred_bw time of
each arm over every run directory and round, and the speedups between arms.

    python summarize.py RESULTS_DIR [RESULTS_DIR ...]

Each RESULTS_DIR is searched recursively for timings.tsv, so it can be one
--out-dir or a directory holding several (one per fold or GPU). Also prints
each arm's median by its position in the round, so an order effect (such as
a cold file cache on whichever arm runs first) would show.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = ["original", "none", "kit", "ours", "both", "main", "kitfk", "fkonly", "fk"]
#: (faster arm, baseline arm) pairs whose median times are compared.
RATIOS = [("kit", "none"), ("ours", "none"), ("both", "none"), ("main", "none"),
          ("kitfk", "none"), ("fkonly", "none"), ("fk", "none"),
          ("kitfk", "kit"), ("main", "original"), ("fk", "main")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", help="directories to search for timings.tsv")
    args = parser.parse_args()
    rows = []
    for root in args.results:
        for path in sorted(Path(root).rglob("timings.tsv")):
            with open(path) as handle:
                rows += list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit("no timings.tsv under " + ", ".join(args.results))
    by_gpu = defaultdict(lambda: defaultdict(list))
    by_position = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    notes = defaultdict(set)
    for row in rows:
        gpu = row["run"]
        if row["status"] == "ok" and row["seconds"]:
            seconds = float(row["seconds"])
            by_gpu[gpu][row["arm"]].append(seconds)
            by_position[gpu][row["arm"]][int(row["position"])].append(seconds)
        else:
            notes[gpu].add("{} {}".format(row["arm"], row["status"]))
    for gpu in sorted(set(by_gpu) | set(notes)):
        arms = by_gpu[gpu]
        labels = sorted({row["fold"] for row in rows if row["run"] == gpu})
        print("\n## {}  ({})".format(gpu, ", ".join(label or "-" for label in labels)))
        print("{:<9} {:>4} {:>9} {:>9} {:>9}".format("arm", "n", "median s", "min s", "max s"))
        for arm in ARMS:
            if arm in arms:
                values = arms[arm]
                print("{:<9} {:>4} {:>9.1f} {:>9.1f} {:>9.1f}".format(
                    arm, len(values), statistics.median(values), min(values), max(values)))
        for faster, baseline in RATIOS:
            if faster in arms and baseline in arms:
                before, after = statistics.median(arms[baseline]), statistics.median(arms[faster])
                print("  {:>8} vs {:<8} {:6.1f} s -> {:6.1f} s  ({:.2f}x)".format(
                    faster, baseline, before, after, before / after))
        if by_position[gpu]:
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
