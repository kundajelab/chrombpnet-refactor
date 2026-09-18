"""Characterization tests for the two hyperparameter finders.

``find_bias_hyperparams`` and ``find_chrombpnet_hyperparams`` share about 30
byte-identical lines of fold reading, edge filtering and quantile thresholding,
and an almost identical TSV-writing tail. Those TSVs are a contract: ``train.py``
reads them back through ``get_model_param_dict`` and validates specific keys.
Nothing pinned their contents before this module, which makes consolidating the
two files unsafe.

The point of these tests is to pin the *current* numbers so a refactor that
merges the shared code has to reproduce them exactly -- including the one place
the two finders deliberately disagree, which is easy to "clean up" by accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chrombpnet.helpers.hyperparameters import find_bias_hyperparams

CHROM_SIZES = {"chr1": 4000, "chr2": 4000, "chr3": 4000}
INPUTLEN = 100
OUTPUTLEN = 50


def _build_inputs(directory: Path, peak_counts, nonpeak_counts) -> Path:
    """Write a genome, bigwig and BEDs giving each region an exact count.

    ``param_utils.get_seqs_cts`` sums the bigwig over
    ``[centre - outputlen // 2, centre + outputlen // 2)``, so placing a single
    value at the start of that window makes each region's count exact and the
    expected results hand-checkable.
    """
    import pyBigWig

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "genome.fa").write_text("".join(
        f">{chrom}\n" + "ACGT" * (size // 4) + "\n"
        for chrom, size in CHROM_SIZES.items()))
    (directory / "folds.json").write_text(json.dumps(
        {"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}))

    centres = [300 + 300 * index
               for index in range(max(len(peak_counts), len(nonpeak_counts)))]
    handle = pyBigWig.open(str(directory / "data.bw"), "w")
    handle.addHeader(list(CHROM_SIZES.items()))
    for chrom, size in CHROM_SIZES.items():
        values = np.zeros(size, dtype=np.float64)
        for index, count in enumerate(peak_counts):
            values[centres[index] - OUTPUTLEN // 2] = float(count)
        for index, count in enumerate(nonpeak_counts):
            values[centres[index] - OUTPUTLEN // 2 + 150] = float(count)
        handle.addEntries(chrom, 0, values=values, span=1, step=1)
    handle.close()

    def write_bed(path, how_many, shift):
        with open(path, "w") as bed:
            for chrom in CHROM_SIZES:
                for index in range(how_many):
                    start = centres[index] + shift - INPUTLEN // 2
                    bed.write("\t".join(str(field) for field in (
                        chrom, start, start + INPUTLEN,
                        ".", ".", ".", ".", ".", ".", INPUTLEN // 2)) + "\n")

    write_bed(directory / "peaks.bed", len(peak_counts), 0)
    write_bed(directory / "nonpeaks.bed", len(nonpeak_counts), 150)
    return directory


def _args(directory: Path, prefix: str) -> SimpleNamespace:
    return SimpleNamespace(
        genome=str(directory / "genome.fa"),
        bigwig=str(directory / "data.bw"),
        peaks=str(directory / "peaks.bed"),
        nonpeaks=str(directory / "nonpeaks.bed"),
        bias_threshold_factor=0.5,
        outlier_threshold=0.9999,
        max_jitter=0,
        chr_fold_path=str(directory / "folds.json"),
        inputlen=INPUTLEN,
        outputlen=OUTPUTLEN,
        filters=4,
        n_dilation_layers=1,
        output_prefix=str(directory / prefix),
        negative_sampling_ratio=0.5,
        seed=1234,
        benchmark=None,
    )


def _read_tsv(path: Path) -> dict[str, str]:
    return dict(line.split("\t", 1)
                for line in path.read_text().splitlines() if line)


def test_bias_finder_writes_the_expected_parameter_tables(tmp_path):
    """Pin both TSVs for a well-separated peak/nonpeak split.

    Peaks are [100, 110, 120, 130, 140] per chromosome and nonpeaks
    [10 .. 100]. Train+valid is chr1+chr2, so every value appears twice.

      counts_threshold = quantile(peaks, 0.01) * 0.5 = 100.0 * 0.5 = 50.0
      final_cnts       = nonpeaks < 50.0            = [10, 20, 30, 40] x2
      upper / lower    = quantile(final, 0.9999 / 0.0001) = 40.0 / 10.0
      kept             = 10 < x < 40                = [20, 20, 30, 30]
      counts_loss_weight = median(kept) / 10        = 25.0 / 10 = 2.5
    """
    directory = _build_inputs(tmp_path / "spread",
                              [100, 110, 120, 130, 140],
                              [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    find_bias_hyperparams.main(_args(directory, "out_"))

    data = _read_tsv(directory / "out_bias_data_params.tsv")
    assert data == {
        "counts_sum_min_thresh": "10.0",
        "counts_sum_max_thresh": "40.0",
        "trainings_pts_post_thresh": "4",
    }

    model = _read_tsv(directory / "out_bias_model_params.tsv")
    assert model == {
        "counts_loss_weight": "2.5",
        "filters": "4",
        "n_dil_layers": "1",
        "inputlen": str(INPUTLEN),
        "outputlen": str(OUTPUTLEN),
        "max_jitter": "0",
        "chr_fold_path": str(directory / "folds.json"),
        # Hardcoded: bias training uses every negative, but train.py requires
        # the key to be present.
        "negative_sampling_ratio": "1.0",
    }

    # train.py:get_model_param_dict rejects a table missing any of these.
    required = {"counts_loss_weight", "filters", "n_dil_layers", "inputlen",
                "outputlen", "negative_sampling_ratio", "max_jitter",
                "chr_fold_path"}
    assert required <= set(model)


def test_bias_finder_emits_a_nan_loss_weight_when_counts_are_degenerate(tmp_path):
    """Documents a defect, so a refactor does not quietly change the symptom.

    With every nonpeak count equal, the 0.9999/0.0001 outlier quantiles collapse
    onto that single value and the strict ``final_cnts < upper & > lower`` mask
    selects nothing. ``np.median([])`` is nan, and neither guard catches it:
    ``counts_loss_weight == 0`` is False for nan, and so is
    ``counts_loss_weight < 1.0``. The finder therefore writes
    ``counts_loss_weight nan`` and reports zero training points instead of
    failing.

    The run does die later -- with the unrelated-looking "No BED regions are
    available for the train chromosome split" from initializers.py -- because
    the same collapsed thresholds empty the nonpeak BED. The nan is what makes
    the failure hard to read.

    Note the chrombpnet finder uses ``<=`` / ``>=`` for this same mask
    (find_chrombpnet_hyperparams.py:140 vs find_bias_hyperparams.py:110), so it
    returns the value rather than nan here. That asymmetry is inherited from
    the TensorFlow original and must survive any consolidation of the two files.
    """
    directory = _build_inputs(tmp_path / "degenerate",
                              [100, 100, 100, 100, 100],
                              [30, 30, 30, 30, 30])
    with pytest.warns(RuntimeWarning, match="empty slice"):
        find_bias_hyperparams.main(_args(directory, "out_"))

    model = _read_tsv(directory / "out_bias_model_params.tsv")
    assert model["counts_loss_weight"] == "nan"
    data = _read_tsv(directory / "out_bias_data_params.tsv")
    assert data["trainings_pts_post_thresh"] == "0"
    assert data["counts_sum_min_thresh"] == data["counts_sum_max_thresh"] == "30.0"


def test_the_two_finders_disagree_on_the_boundary_mask(tmp_path):
    """Pin the ``<`` vs ``<=`` divergence directly, with no file plumbing.

    This is the single behavioural difference hiding inside ~30 lines of
    otherwise identical code, and it only shows up when the outlier bounds
    collapse onto a data value.
    """
    counts = np.array([30.0, 30.0, 30.0])
    upper = np.quantile(counts, 0.9999)
    lower = np.quantile(counts, 1 - 0.9999)

    bias_mask = (counts < upper) & (counts > lower)
    chrombpnet_mask = (counts <= upper) & (counts >= lower)

    assert not bias_mask.any(), "find_bias_hyperparams.py:110 uses < and >"
    assert chrombpnet_mask.all(), "find_chrombpnet_hyperparams.py:140 uses <= and >="
    assert np.isnan(np.median(counts[bias_mask]))
    assert np.median(counts[chrombpnet_mask]) == 30.0
