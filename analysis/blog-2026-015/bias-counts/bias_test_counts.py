#!/usr/bin/env python3
"""The three bias models most ENCODE ChromBPNet models use: their counts head's
predictions on held-out test chromosomes (fold 1) against the observed counts.

    python bias_test_counts.py OUT_DIR ATLAS_DIR

ATLAS_DIR is the ENCODE chromatin atlas project directory (ATAC/ and DNASE/ with each
experiment's preprocessing/ and model runs, as the atlas lays them out).

Each bias model's predictions on its training experiment's fold-1 test peaks
and non-peaks were saved when that experiment's ChromBPNet model was trained
(bias_predictions.h5, from the scaled bias model, which differs from the
pretrained one only in the counts head's constant, so correlations are the
same). The observed counts are the experiment's bigWig summed over the same
1,000 bp, as log(1 + counts), the target the counts head is trained on. Writes
one TSV per bias model and checks Pearson r against the run's bias_metrics.json.
"""

import json
import os
import sys

import h5py
import numpy as np
import pyBigWig
from scipy.stats import pearsonr

OUTPUT_LENGTH = 1000
# (label, assay, fold-1 run directory, observed bigWig), relative to the atlas directory
BIAS_MODELS = [
    ("ENCSR880CUB", "DNase-seq", "DNASE/ENCSR880CUB/chrombppnet_model_encsr880cub_bias_fold_1",
     "DNASE/ENCSR880CUB/preprocessing/bigWigs/ENCSR880CUB.bigWig"),
    ("ENCSR283TME", "DNase-seq", "DNASE/ENCSR283TME/chrombppnet_model_encsr283tme_bias_fold_1",
     "DNASE/ENCSR283TME/preprocessing/bigWigs/ENCSR283TME.bigWig"),
    ("ENCSR868FGK", "ATAC-seq", "ATAC/ENCSR868FGK/chrombpnet_model_feb15_fold_1",
     "ATAC/ENCSR868FGK/preprocessing/bigWigs/ENCSR868FGK.bigWig"),
]


def observed_log_counts(bigwig, chroms, centers):
    bw = pyBigWig.open(bigwig)
    half = OUTPUT_LENGTH // 2
    totals = np.empty(len(centers))
    for i, (chrom, center) in enumerate(zip(chroms, centers)):
        values = np.nan_to_num(bw.values(chrom, int(center) - half, int(center) + half, numpy=True))
        totals[i] = values.sum()
    return np.log1p(totals)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out_dir, atlas = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for label, assay, run, bigwig in BIAS_MODELS:
        run, bigwig = os.path.join(atlas, run), os.path.join(atlas, bigwig)
        with h5py.File(os.path.join(run, "bias_predictions.h5"), "r") as handle:
            chroms = handle["coords/coords_chrom"][()].astype(str)
            centers = handle["coords/coords_center"][()]
            is_peak = handle["coords/coords_peak"][()].astype(int)
            predicted = handle["predictions/logcounts"][()]
        observed = observed_log_counts(bigwig, chroms, centers)
        with open(os.path.join(run, "bias_metrics.json")) as handle:
            recorded = json.load(handle)["counts_metrics"]
        for name, mask in (("peaks", is_peak == 1), ("nonpeaks", is_peak == 0)):
            print("%s %s: n %d, Pearson r %.4f (recorded %.4f)" % (
                label, name, mask.sum(), pearsonr(observed[mask], predicted[mask])[0], recorded[name]["pearsonr"]))
        np.savetxt(os.path.join(out_dir, "%s_fold1_test_counts.tsv" % label),
                   np.column_stack([is_peak, predicted, observed]), fmt=["%d", "%.6f", "%.6f"], delimiter="\t",
                   header="is_peak\tpredicted_log_counts\tobserved_log_counts", comments="")


if __name__ == "__main__":
    main()
