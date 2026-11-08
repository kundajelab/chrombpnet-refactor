#!/usr/bin/env python3
"""Checks bigwig_totals.py's header totals against sums over every base.

    python check_exact.py FILE.bigWig [URL]

For a downloaded bigWig: the total summary read from its header (and, given
the file's URL, read remotely), against the sum of value x bases over every
interval of every chromosome, and against pyBigWig's exact per-chromosome sum.
"""

import os
import sys

import numpy as np
import pyBigWig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bigwig_totals import total_summary  # noqa: E402


def main():
    path = sys.argv[1]
    header = total_summary(path)
    print("header (local): sum %.6f, bases %d" % (header["sum"], header["bases_covered"]))
    if len(sys.argv) > 2:
        remote = total_summary(sys.argv[2])
        print("header (remote): sum %.6f, bases %d, identical %s" % (
            remote["sum"], remote["bases_covered"], remote == header))
    bw = pyBigWig.open(path)
    interval_sum, bases, stats_sum = 0.0, 0, 0.0
    for chrom, length in bw.chroms().items():
        intervals = bw.intervals(chrom)
        if not intervals:
            continue
        array = np.array(intervals, dtype=np.float64)
        spans = array[:, 1] - array[:, 0]
        interval_sum += float(np.sum(spans * array[:, 2]))
        bases += int(spans.sum())
        stats_sum += bw.stats(chrom, 0, length, type="sum", exact=True)[0] or 0.0
    print("every interval: sum %.6f, bases %d" % (interval_sum, bases))
    print("pyBigWig exact stats: sum %.6f" % stats_sum)
    print("relative difference, header vs intervals: %.3e" % (abs(header["sum"] - interval_sum) / interval_sum))


if __name__ == "__main__":
    main()
