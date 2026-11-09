#!/usr/bin/env python3
"""The blog's Section 4 "Measured effect" cells, from run_section4.sh (each
change alone, old form against new) and the kit-fused jobs (pred_bw totals),
in the form

    {memory or size advantage, if any}; {x}x faster (L40S: {a} s -> {b} s; H200: {c} s -> {d} s)

    python summarize_section4.py ANALYSIS_DIR      # the jobs' OUT_ROOT (site.env's OPTIMIZATION_OUT)

Medians over every round of a run tag; GPUs from different jobs are never
combined into one number, only placed side by side.
"""

import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("usage: python summarize_section4.py ANALYSIS_DIR  (the jobs' OUT_ROOT: site.env's OPTIMIZATION_OUT)")
ROOT = sys.argv[1]

# (row, stage or "pred_bw", old form, new form, what the memory/size note reports)
ROWS = [
    ("GC-matched non-peaks (prep nonpeaks)", "gc", "old", "new", "peak"),
    ("Shared inputs across models (pred_bw)", "shared", "unshared", "shared", None),
    ("Fixed-step bigWigs", "bigwig", "list", "fixed", "size"),
    ("NumPy bigWig writer arguments", "bigwig", "list", "numpy", None),
    ("Batched cast into a preallocated output", "cast", "legacy", "batched", "added"),
    ("All of the above (without the kit's changes)", "pred_bw", "none", "ours", None),
    ("Together with the kit's changes", "pred_bw", "none", "both", None),
    ("Maximum HDF5 compression: gzip 9 + shuffle vs gzip 4", "h5", "gzip4", "gzip9", "size"),
    ("  ... on threads (the kit's h5_writer)", "h5", "gzip9", "gzip9threads", None),
    ("Maximum HDF5 compression, in pred_bw (the default vs the same without it)", "pred_bw", "both", "main", None),
    ("Fully optimized chrombpnet-refactor (the default) vs original chrombpnet", "pred_bw", "original", "main", None),
]


# The HDF5 stage alone at the original's filters (submit_h5_gzip4.sh: run_section4.sh
# STAGES=h5 RUN_KIND=h5-gzip4), its own job: its variants are compared only with each
# other, and it is left out of the rows above.
H5_GZIP4_ROWS = [
    ("HDF5 on threads at the original's filters (--h5-gzip-level 4 --no-h5-shuffle) vs gzip 4",
     "h5", "gzip4", "gzip4threads", "size"),
    ("HDF5 on threads at gzip 9 + shuffle (the default) vs gzip 4", "h5", "gzip4", "gzip9threads", "size"),
]


def gpu_of(tag):
    return "H200" if "H200" in tag else "L40S" if "L40S" in tag else None


def load(runs="*_section4*", kit_fused=True):
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # gpu -> stage -> variant -> rows
    for path in glob.glob(os.path.join(ROOT, runs, "fold_*", "results.jsonl")):
        gpu = gpu_of(path.split(os.sep)[-3])
        for line in open(path):
            row = json.loads(line)
            stats[gpu][row["stage"]][row["variant"]].append(row)
    for path in glob.glob(os.path.join(ROOT, "*_kit-fused", "fold_*", "timings.tsv")) if kit_fused else ():
        gpu = gpu_of(path.split(os.sep)[-3])
        for row in csv.DictReader(open(path), delimiter="\t"):
            if row["status"] == "ok" and row["seconds"]:
                stats[gpu]["pred_bw"][row["arm"]].append({"seconds": float(row["seconds"])})
    return stats


def med(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(values) if values else None


def cell(stats, stage, old, new, note):
    parts, ratios, memo = [], [], ""
    for gpu in ("L40S", "H200"):
        a, b = med(stats[gpu][stage][old], "seconds"), med(stats[gpu][stage][new], "seconds")
        if a is None or b is None:
            parts.append("%s: pending" % gpu)
            continue
        parts.append("%s: %.0f s -> %.0f s" % (gpu, a, b) if min(a, b) >= 10 else "%s: %.1f s -> %.1f s" % (gpu, a, b))
        ratios.append(a / b)
        if note and not memo:
            o, n = stats[gpu][stage][old], stats[gpu][stage][new]
            if note == "size":
                sa, sb = med(o, "out_bytes"), med(n, "out_bytes")
                memo = "files %.1fx smaller (%.0f MB -> %.0f MB)" % (sa / sb, sa / 1e6, sb / 1e6) if sa and sb else ""
            elif note == "peak":
                pa, pb = med(o, "rss_peak_mb"), med(n, "rss_peak_mb")
                memo = "peak memory %.1f GB -> %.1f GB" % (pa / 1024, pb / 1024) if pa and pb else ""
            elif note == "added":
                pa, pb = med(o, "rss_added_mb"), med(n, "rss_added_mb")
                memo = "%.0f MB less peak memory" % (pa - pb) if pa is not None and pb is not None else ""
    if ratios:
        lo, hi = min(ratios), max(ratios)
        word = "faster" if lo >= 1 else "slower"
        low, high = sorted(r if r >= 1 else 1 / r for r in (lo, hi))
        speed = "%.1fx %s" % (low, word) if high - low < 0.05 else "%.1f-%.1fx %s" % (low, high, word)
    else:
        speed = "pending"
    return "%s%s (%s)" % (memo + "; " if memo else "", speed, "; ".join(parts))


def main():
    stats = load()
    if not stats:
        sys.exit("no *_section4*/fold_*/results.jsonl or *_kit-fused/fold_*/timings.tsv under " + ROOT)
    for row, stage, old, new, note in ROWS:
        print("%-72s %s" % (row, cell(stats, stage, old, new, note)))
    gzip4 = load("*_h5-gzip4*", kit_fused=False)
    if gzip4:
        print("\n## the HDF5 stage at the original's filters (*_h5-gzip4, one job)")
        for row, stage, old, new, note in H5_GZIP4_ROWS:
            print("%-72s %s" % (row, cell(gzip4, stage, old, new, note)))


if __name__ == "__main__":
    main()
