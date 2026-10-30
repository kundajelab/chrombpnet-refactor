import numpy as np
import pandas as pd
import pytest

from chrombpnet.helpers.make_gc_matched_negatives.get_gc_matched_negatives import (
    match_gc_negatives,
)


def _write_bed(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", header=False, index=False)


def test_gc_matching_uses_nearest_numeric_bins_and_is_deterministic(tmp_path):
    candidates = tmp_path / "candidates.bed"
    foreground = tmp_path / "foreground.bed"
    _write_bed(candidates, [
        ["chr1", 10, 20, 0.30], ["chr1", 30, 40, 0.31],
        ["chr2", 50, 60, 0.70], ["chr2", 60, 70, 0.69],
        ["chr3", 70, 80, 0.50],
    ])
    _write_bed(foreground, [
        ["chr1", 100, 110, 0.29], ["chr2", 120, 130, 0.71],
        ["chr3", 140, 150, 0.49], ["ignored", 0, 10, 0.5],
    ])
    splits = {"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}
    first = match_gc_negatives(candidates, foreground, splits, ratios=(2, 2, 1), seed=7)
    second = match_gc_negatives(candidates, foreground, splits, ratios=(2, 2, 1), seed=7)
    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[0].shape[0] == 5
    assert set(first[0]["gc"]) == {0.30, 0.31, 0.50, 0.69, 0.70}
    assert first[3].tolist() == ["ignored"]


def test_gc_matching_fails_immediately_when_fold_pool_is_exhausted(tmp_path):
    candidates = tmp_path / "candidates.bed"
    foreground = tmp_path / "foreground.bed"
    _write_bed(candidates, [["chr1", 10, 20, 0.4]])
    _write_bed(foreground, [["chr1", 100, 110, 0.4]])
    splits = {"train": ["chr1"], "valid": [], "test": []}
    with pytest.raises(ValueError, match="Requested 2 negatives but only 1"):
        match_gc_negatives(candidates, foreground, splits, ratios=(2, 2, 1), seed=1)


def _three_folds(tmp_path):
    candidates = tmp_path / "candidates.bed"
    foreground = tmp_path / "foreground.bed"
    _write_bed(candidates, [
        [chrom, start, start + 10, round(0.40 + start / 2000, 2)]
        for chrom in ("chr1", "chr2", "chr3")
        for start in range(0, 400, 10)
    ])
    _write_bed(foreground, [
        ["chr1", 1000, 1010, 0.41], ["chr1", 1020, 1030, 0.45], ["chr1", 1040, 1050, 0.50],
        ["chr2", 1000, 1010, 0.43], ["chr2", 1020, 1030, 0.48],
        ["chr3", 1000, 1010, 0.42], ["chr3", 1020, 1030, 0.47],
    ])
    return candidates, foreground, {"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}


@pytest.mark.parametrize("ratios", [(2, 2, 1), (1, 1, 4), (3, 0, 2)])
def test_gc_matching_draws_each_folds_own_ratio(tmp_path, ratios):
    candidates, foreground, splits = _three_folds(tmp_path)
    negatives = match_gc_negatives(candidates, foreground, splits, ratios=ratios, seed=3)[0]
    counts = negatives["chrom"].value_counts().reindex(["chr1", "chr2", "chr3"], fill_value=0)
    assert counts.tolist() == [3 * ratios[0], 2 * ratios[1], 2 * ratios[2]]


def test_gc_matching_test_ratio_leaves_train_and_valid_negatives_unchanged(tmp_path):
    # The folds draw from one generator in train, valid, test order, so a
    # larger test fold (e.g. 1:4 for evaluation) changes only the test fold.
    candidates, foreground, splits = _three_folds(tmp_path)
    default = match_gc_negatives(candidates, foreground, splits, ratios=(2, 2, 1), seed=3)[0]
    wider = match_gc_negatives(candidates, foreground, splits, ratios=(2, 2, 4), seed=3)[0]
    for chrom in ("chr1", "chr2"):
        pd.testing.assert_frame_equal(
            default[default.chrom == chrom].reset_index(drop=True),
            wider[wider.chrom == chrom].reset_index(drop=True))


@pytest.mark.parametrize("ratios", [(2, 2), (2, -1, 1), (2, 1.5, 1)])
def test_gc_matching_rejects_bad_ratios(tmp_path, ratios):
    candidates, foreground, splits = _three_folds(tmp_path)
    with pytest.raises(ValueError, match="three non-negative integers"):
        match_gc_negatives(candidates, foreground, splits, ratios=ratios, seed=3)
