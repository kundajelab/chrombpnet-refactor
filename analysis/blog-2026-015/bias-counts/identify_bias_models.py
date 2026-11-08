#!/usr/bin/env python3
"""Which pretrained bias model each ENCODE ChromBPNet model (and fold) uses.

    python identify_bias_models.py OUT.tsv [encode_paths.tsv]

The training logs name the bias model's directory inconsistently across
batches, so models are grouped by weights instead: a scaled bias model differs
from the bias model it was made from only in its counts head, so the
convolutions and the profile head identify the source. Writes one row per
model and fold: the hash of those weights, and of the counts head.
"""

import csv
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor

import h5py

PATHS = "/path/to/workdir/sherlock/ids/encode/path/to/lab/encode_paths.tsv"
COUNTS_HEAD = "logcount_predictions"


def weight_hashes(path):
    body, counts = hashlib.sha1(), hashlib.sha1()
    with h5py.File(path, "r") as handle:
        weights = handle["model_weights"]
        names = []
        weights.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
        for name in sorted(names):
            (counts if name.startswith(COUNTS_HEAD + "/") else body).update(name.encode() + weights[name][()].tobytes())
        counts_bias = float(weights[COUNTS_HEAD][COUNTS_HEAD]["bias:0"][0])
    return body.hexdigest()[:12], counts.hexdigest()[:12], counts_bias


def main():
    out = sys.argv[1]
    paths = sys.argv[2] if len(sys.argv) > 2 else PATHS
    with open(paths) as handle:
        models = [row for row in csv.DictReader(handle, delimiter="\t") if row["model"] == "chrombpnet"]
    jobs = [(row["experiment"], row["assay"], fold, row["bias_fold%d" % fold]) for row in models for fold in range(5)]

    def run(job):
        try:
            return weight_hashes(job[3])
        except (OSError, KeyError) as error:
            return ("missing", "missing", float("nan")) if isinstance(error, FileNotFoundError) else ("error", str(error)[:60], float("nan"))

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(run, jobs))
    with open(out, "w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["experiment", "assay", "fold", "body_hash", "counts_hash", "counts_bias", "path"])
        for (experiment, assay, fold, path), result in zip(jobs, results):
            writer.writerow([experiment, assay, fold, *result, path])
    print("%d bias models -> %s" % (len(jobs), out))


if __name__ == "__main__":
    main()
