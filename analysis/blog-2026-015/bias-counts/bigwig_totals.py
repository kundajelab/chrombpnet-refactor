#!/usr/bin/env python3
"""Total predicted counts of every ENCODE ChromBPNet model's two prediction
bigWigs (with bias: "predicted signal profile"; without: "bias-corrected
predicted signal profile"), read from each file's total summary.

    python bigwig_totals.py OUT.tsv MODEL_DOWNLOADS.tsv

MODEL_DOWNLOADS.tsv: each model's two prediction bigWigs on the ENCODE portal (their download URLs).

A bigWig's header points at a 40-byte total summary (bases covered, min, max,
sum, sum of squares) computed over the full-resolution data when the file was
written, so each file's total takes two small HTTP range requests instead of
streaming its 1.3 GB. check_exact.py compares this with a sum over every base
of downloaded files.
"""

import csv
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

COLUMNS = {"with_bias": "predicted signal profile bigWig",
           "without_bias": "bias-corrected predicted signal profile bigWig"}
MAGIC = 0x888FFC26
FIELDS = ("bases_covered", "min", "max", "sum", "sum_squares")


def _read(source, start, length):
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(
            source, headers={"Range": "bytes=%d-%d" % (start, start + length - 1)})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise IOError("%s: HTTP %d, not a partial response" % (source, response.status))
                    return response.read()
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
    with open(source, "rb") as handle:
        handle.seek(start)
        return handle.read(length)


def total_summary(source):
    """The bigWig's total summary, as a dict of FIELDS."""
    head = _read(source, 0, 512)
    endian = "<" if struct.unpack("<I", head[:4])[0] == MAGIC else ">"
    if struct.unpack(endian + "I", head[:4])[0] != MAGIC:
        raise ValueError("%s is not a bigWig" % source)
    offset = struct.unpack(endian + "Q", head[44:52])[0]
    if offset == 0:
        raise ValueError("%s has no total summary" % source)
    block = head[offset:offset + 40] if offset + 40 <= len(head) else _read(source, offset, 40)
    return dict(zip(FIELDS, struct.unpack(endian + "Qdddd", block)))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out, downloads = sys.argv[1], sys.argv[2]
    with open(downloads) as handle:
        models = [row for row in csv.DictReader(handle, delimiter="\t") if row["type"] == "ChromBPNet"]
    jobs = [(row, kind, row[column]) for row in models for kind, column in COLUMNS.items()]

    def summary_or_error(job):
        try:
            return total_summary(job[2])
        except Exception as error:  # recorded per file, so one refusal does not lose the rest
            return error

    with ThreadPoolExecutor(6) as pool:
        summaries = list(pool.map(summary_or_error, jobs))
    failed = [(url, error) for (_, _, url), error in zip(jobs, summaries) if isinstance(error, Exception)]
    for url, error in failed:
        print("failed: %s: %s" % (url, error), file=sys.stderr)
    totals = {}
    for (row, kind, url), summary in zip(jobs, summaries):
        if not isinstance(summary, Exception):
            totals.setdefault(row["experiment"], {"row": row})[kind] = (url, summary)
    totals = {experiment: entry for experiment, entry in totals.items() if len(entry) == 3}
    with open(out, "w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["experiment", "assay", "biosample", "with_bias_file", "without_bias_file",
                         "with_bias_sum", "without_bias_sum", "with_bias_bases", "without_bias_bases"])
        for experiment, entry in totals.items():
            row = entry["row"]
            (with_url, with_bias), (without_url, without_bias) = entry["with_bias"], entry["without_bias"]
            writer.writerow([experiment, row["assay"], row["biosample"],
                             with_url.rsplit("/", 1)[1], without_url.rsplit("/", 1)[1],
                             repr(with_bias["sum"]), repr(without_bias["sum"]),
                             with_bias["bases_covered"], without_bias["bases_covered"]])
    print("%d models -> %s" % (len(totals), out))


if __name__ == "__main__":
    main()
