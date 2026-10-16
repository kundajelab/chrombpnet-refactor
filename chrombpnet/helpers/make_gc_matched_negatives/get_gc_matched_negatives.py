"""Generate GC-matched negatives using compact numeric arrays.

Candidates are represented by integer GC bins and fold-group codes. Within
each fold, candidates are shuffled once, sorted by GC bin, and consumed without
replacement. A binary search selects the nearest non-empty bin when an exact
match is unavailable, so exhausted inputs fail instead of looping forever.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chrombpnet.seed import DEFAULT_SEED


GROUP_NAMES = np.array(["chrom_train", "chrom_valid", "chrom_test"])


def parse_args():
    parser = argparse.ArgumentParser(description="generate GC-matched non-peak regions")
    parser.add_argument("-c", "--candidate-negatives", required=True)
    parser.add_argument("-f", "--foreground-gc-bed", required=True)
    parser.add_argument("-o", "--output-prefix", required=True)
    parser.add_argument("-fl", "--chr-fold-path", required=True)
    parser.add_argument("-npr", "--neg-to-pos-ratio-train", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _chromosome_group_map(splits):
    group_map = {}
    for code, key in enumerate(("train", "valid", "test")):
        if key not in splits:
            raise ValueError(f"Fold file is missing required '{key}' chromosome list")
        for chromosome in splits[key]:
            if chromosome in group_map:
                raise ValueError(f"Chromosome {chromosome!r} appears in multiple folds")
            group_map[chromosome] = code
    return group_map


def _read_gc_bed(path, group_map, label):
    frame = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3])
    if frame.empty:
        raise ValueError(f"{label} BED is empty: {path}")
    frame.columns = ["chrom", "start", "end", "gc"]
    frame["start"] = pd.to_numeric(frame["start"], errors="raise").astype(np.int64)
    frame["end"] = pd.to_numeric(frame["end"], errors="raise").astype(np.int64)
    gc = pd.to_numeric(frame["gc"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(gc).all() or np.any((gc < 0.0) | (gc > 1.0)):
        raise ValueError(f"{label} GC fractions must be finite values in [0, 1]")
    frame["gc_bin"] = np.rint(gc * 100.0).astype(np.int16)
    groups = frame["chrom"].map(group_map)
    ignored = np.unique(frame.loc[groups.isna(), "chrom"].astype(str))
    frame = frame.loc[groups.notna()].copy()
    frame["group"] = groups.loc[groups.notna()].astype(np.int8).to_numpy()
    return frame.reset_index(drop=True), ignored


def _nearest_available_bin(target, remaining, rng):
    available = np.flatnonzero(remaining > 0)
    if available.size == 0:
        raise ValueError("GC-matched candidate pool is exhausted")
    insertion = np.searchsorted(available, target)
    if insertion == 0:
        return int(available[0])
    if insertion == available.size:
        return int(available[-1])
    lower, upper = int(available[insertion - 1]), int(available[insertion])
    lower_distance, upper_distance = target - lower, upper - target
    if lower_distance == upper_distance:
        return lower if rng.integers(2) == 0 else upper
    return lower if lower_distance < upper_distance else upper


def _sample_group(candidate_frame, target_bins, rng):
    """Return candidate row indices matched to target bins without replacement."""
    candidate_bins = candidate_frame["gc_bin"].to_numpy(dtype=np.int16)
    if target_bins.size > candidate_bins.size:
        raise ValueError(
            f"Requested {target_bins.size} negatives but only {candidate_bins.size} "
            "candidates are available in this chromosome fold"
        )
    random_keys = rng.random(candidate_bins.size)
    order = np.lexsort((random_keys, candidate_bins))
    sorted_bins = candidate_bins[order]
    starts = np.searchsorted(sorted_bins, np.arange(101), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(101), side="right")
    remaining = ends - starts
    consumed = np.zeros(101, dtype=np.int64)
    selected = np.empty(target_bins.size, dtype=np.int64)
    selected_bins = np.empty(target_bins.size, dtype=np.int16)
    frame_indices = candidate_frame.index.to_numpy()

    for index, target in enumerate(target_bins):
        chosen_bin = _nearest_available_bin(int(target), remaining, rng)
        sorted_position = starts[chosen_bin] + consumed[chosen_bin]
        selected[index] = frame_indices[order[sorted_position]]
        selected_bins[index] = chosen_bin
        consumed[chosen_bin] += 1
        remaining[chosen_bin] -= 1
    return selected, selected_bins


def match_gc_negatives(candidate_negatives, foreground_gc_bed, splits, ratio, seed):
    if ratio < 0:
        raise ValueError("neg_to_pos_ratio_train must be non-negative")
    group_map = _chromosome_group_map(splits)
    candidates, ignored_candidates = _read_gc_bed(candidate_negatives, group_map, "Candidate")
    foreground, ignored_foreground = _read_gc_bed(foreground_gc_bed, group_map, "Foreground")
    if foreground.empty:
        raise ValueError("No foreground regions belong to the requested folds")

    group_codes = foreground["group"].to_numpy(dtype=np.int8)
    repetitions = np.where(group_codes < 2, ratio, 1).astype(np.int64)
    expanded_rows = np.repeat(np.arange(len(foreground)), repetitions)
    expanded_groups = group_codes[expanded_rows]
    target_bins = foreground["gc_bin"].to_numpy(dtype=np.int16)[expanded_rows]
    rng = np.random.default_rng(seed)

    output_parts = []
    target_parts = []
    for group in range(3):
        request_positions = np.flatnonzero(expanded_groups == group)
        if request_positions.size == 0:
            continue
        group_candidates = candidates.loc[candidates["group"] == group]
        if group_candidates.empty:
            raise ValueError(f"No candidate negatives are available for {GROUP_NAMES[group]}")
        selected, selected_bins = _sample_group(group_candidates, target_bins[request_positions], rng)
        part = candidates.loc[selected, ["chrom", "start", "end"]].copy()
        part["gc"] = selected_bins.astype(np.float64) / 100.0
        part["request_position"] = request_positions
        output_parts.append(part)
        target_parts.append(pd.DataFrame({
            "request_position": request_positions,
            "foreground_gc": target_bins[request_positions] / 100.0,
        }))

    if not output_parts:
        output = pd.DataFrame(columns=["chrom", "start", "end", "gc"])
        foreground_gc = np.empty(0, dtype=np.float64)
    else:
        output = pd.concat(output_parts).sort_values("request_position")
        foreground_gc = (
            pd.concat(target_parts).sort_values("request_position")["foreground_gc"].to_numpy()
        )
        output = output.drop(columns="request_position").reset_index(drop=True)
    return output, foreground_gc, ignored_candidates, ignored_foreground


def main(args):
    with open(args.chr_fold_path) as handle:
        splits = json.load(handle)
    output, foreground_gc, ignored_candidates, ignored_foreground = match_gc_negatives(
        args.candidate_negatives, args.foreground_gc_bed, splits,
        args.neg_to_pos_ratio_train, args.seed,
    )
    print("Ignored background chromosomes: " + ",".join(ignored_candidates))
    print("Ignored foreground chromosomes: " + ",".join(ignored_foreground))
    output_path = Path(args.output_prefix + ".bed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep="\t", index=False, header=False, quoting=csv.QUOTE_NONE)

    bins = np.linspace(0, 1, 101)
    plt.figure()
    plt.hist([output["gc"].to_numpy(), foreground_gc], bins, density=True,
             label=["negatives gc distribution", "foreground gc distribution"])
    plt.xlabel("GC content")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.savefig(args.output_prefix + "_compared_with_foreground.png")
    plt.close()


if __name__ == "__main__":
    main(parse_args())
