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
    first = match_gc_negatives(candidates, foreground, splits, ratio=2, seed=7)
    second = match_gc_negatives(candidates, foreground, splits, ratio=2, seed=7)
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
        match_gc_negatives(candidates, foreground, splits, ratio=2, seed=1)
