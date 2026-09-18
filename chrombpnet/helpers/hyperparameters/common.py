"""Shared steps of the two hyperparameter finders.

``find_chrombpnet_hyperparams.py`` and ``find_bias_hyperparams.py`` open the
same inputs, split them the same way and write the same shape of TSV. What they
do *between* those steps differs -- the bias finder thresholds nonpeaks against
a quantile of the peak counts, the ChromBPNet finder rescales a bias model --
so only the shared ends live here.
"""

from __future__ import annotations

import json

import pandas as pd
import pyBigWig
import pyfaidx

from chrombpnet.regions import NARROWPEAK_SCHEMA


class FoldInputs:
    """Peaks, nonpeaks and open handles for one fold, split train+valid / test.

    Attribute names match the local variables the two finders used, so the
    surrounding code reads as it did before.
    """

    def __init__(self, args):
        splits_dict = json.load(open(args.chr_fold_path))
        self.chroms_to_keep = splits_dict["train"] + splits_dict["valid"]
        self.test_chroms_to_keep = splits_dict["test"]
        print("evaluating hyperparameters on the following chromosomes",
              self.chroms_to_keep)

        self.bw = pyBigWig.open(args.bigwig)
        self.genome = pyfaidx.Fasta(args.genome)

        in_peaks = pd.read_csv(args.peaks, sep='\t', header=None,
                               names=NARROWPEAK_SCHEMA)
        in_nonpeaks = pd.read_csv(args.nonpeaks, sep='\t', header=None,
                                  names=NARROWPEAK_SCHEMA)

        if in_peaks.empty or in_nonpeaks.empty:
            raise ValueError("Peak and nonpeak files must both be non-empty")
        if args.inputlen < args.outputlen:
            raise ValueError("inputlen must be at least outputlen")

        # inputlen and outputlen are chosen based on the filters and dilation
        # layers used.
        self.peaks = in_peaks[in_peaks["chr"].isin(self.chroms_to_keep)]
        self.test_peaks = in_peaks[in_peaks["chr"].isin(self.test_chroms_to_keep)]
        self.nonpeaks = in_nonpeaks[in_nonpeaks["chr"].isin(self.chroms_to_keep)]
        self.test_nonpeaks = in_nonpeaks[
            in_nonpeaks["chr"].isin(self.test_chroms_to_keep)]


def counts_loss_weight(final_cnts, lower_thresh, upper_thresh, *, inclusive):
    """Median of the in-range counts over ten, with a floor of 1.0.

    ``inclusive`` selects the boundary comparison, and the two finders differ
    here: find_chrombpnet_hyperparams uses ``<=`` / ``>=`` while
    find_bias_hyperparams uses ``<`` / ``>``. They agree on every ordinary
    input and diverge only when the outlier quantiles collapse onto a single
    data value, at which point the strict form selects nothing and the median
    is nan. The asymmetry is inherited from the TensorFlow original, so it is a
    parameter rather than something to settle -- see
    tests/test_hyperparameters.py, which pins both behaviours.
    """
    import numpy as np

    if inclusive:
        selected = final_cnts[(final_cnts <= upper_thresh) & (final_cnts >= lower_thresh)]
    else:
        selected = final_cnts[(final_cnts < upper_thresh) & (final_cnts > lower_thresh)]
    weight = np.median(selected) / 10
    if weight == 0:
        raise ValueError("Computed counts loss weight is zero")
    if weight < 1.0:
        weight = 1.0
        print("WARNING: you are training on low-read depth data")
    return weight


def write_params_tsv(path, rows, *, trailing_newline):
    """Write a parameter table as ``key<TAB>value`` lines.

    The originals built these with fourteen bare ``file.write`` calls and no
    context manager, and the two tables ended differently: the ``*_data_params``
    files emit a newline after every row and so end with one, while the
    ``*_model_params`` files omit it after the last row. train.py's
    get_model_param_dict skips blank lines and would tolerate either, but the
    files are compared byte-for-byte by the regression harness, so
    ``trailing_newline`` is explicit and has no default.
    """
    body = "\n".join(f"{key}\t{value}" for key, value in rows)
    with open(path, "w") as handle:
        handle.write(body + "\n" if trailing_newline else body)
