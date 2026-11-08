#!/usr/bin/env python3
"""Totals over the positions both prediction bigWigs cover, for the models
whose header totals cannot be compared directly.

    python matched_totals.py prediction_totals.tsv OUT.tsv SCRATCH_DIR

Some models' with-bias bigWig covers fewer positions than its no-bias one
(their header totals then cover different regions), and a few contain NaN
values (their header totals are NaN). For each such model both files are
downloaded, summed per chromosome over the positions where both hold a finite
value, and deleted. Models already in OUT.tsv are skipped, so an interrupted
run resumes.
"""

import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyBigWig

URL = "https://www.encodeproject.org/files/{0}/@@download/{1}"
FIELDS = ["experiment", "with_bias_sum", "without_bias_sum", "bases", "with_bias_finite_bases",
          "without_bias_finite_bases"]


def needs_matching(row):
    sums = (float(row["with_bias_sum"]), float(row["without_bias_sum"]))
    return not all(np.isfinite(sums)) or row["with_bias_bases"] != row["without_bias_bases"]


def download(name, scratch):
    """Downloads to a temporary name and renames, so a file under its own name is complete."""
    path = os.path.join(scratch, name)
    if not os.path.exists(path):
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "10", "-o", path + ".part",
                        URL.format(name.split(".")[0], name)], check=True)
        os.rename(path + ".part", path)
    return path


def matched(with_path, without_path):
    with_bw, without_bw = pyBigWig.open(with_path), pyBigWig.open(without_path)
    totals = np.zeros(2)
    bases = finite_with = finite_without = 0
    for chrom, length in with_bw.chroms().items():
        if chrom not in without_bw.chroms():
            continue
        with_values = with_bw.values(chrom, 0, length, numpy=True)
        without_values = without_bw.values(chrom, 0, length, numpy=True)
        both = np.isfinite(with_values) & np.isfinite(without_values)
        totals += with_values[both].sum(dtype=np.float64), without_values[both].sum(dtype=np.float64)
        bases += int(both.sum())
        finite_with += int(np.isfinite(with_values).sum())
        finite_without += int(np.isfinite(without_values).sum())
    return float(totals[0]), float(totals[1]), bases, finite_with, finite_without


def main():
    totals_path, out, scratch = sys.argv[1:4]
    os.makedirs(scratch, exist_ok=True)
    with open(totals_path) as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if needs_matching(row)]
    done = set()
    if os.path.exists(out):
        with open(out) as handle:
            done = {row["experiment"] for row in csv.DictReader(handle, delimiter="\t")}
    else:
        with open(out, "w") as handle:
            handle.write("\t".join(FIELDS) + "\n")
    todo = [row for row in rows if row["experiment"] not in done]
    print("%d models need matched totals, %d left" % (len(rows), len(todo)), flush=True)

    with ProcessPoolExecutor(16) as pool:  # processes: pyBigWig decodes under the GIL
        futures = {pool.submit(run, row, scratch): row["experiment"] for row in todo}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                print("failed %s: %s" % (futures[future], error), flush=True)
                continue
            with open(out, "a") as handle:
                handle.write("\t".join([futures[future]] + [repr(value) for value in result]) + "\n")
            print("done %s" % futures[future], flush=True)


def run(row, scratch):
    paths = [os.path.join(scratch, row[key]) for key in ("with_bias_file", "without_bias_file")]
    try:
        return matched(*[download(os.path.basename(path), scratch) for path in paths])
    finally:
        for path in paths:
            for leftover in (path, path + ".part"):
                if os.path.exists(leftover):
                    os.remove(leftover)


if __name__ == "__main__":
    main()
