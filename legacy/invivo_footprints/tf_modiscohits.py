"""Call attribution-supported motif hits from TF-MoDISco 2.6 patterns.

TF-MoDISco 2.6 removed the legacy ``densityadapted_hitscoring`` module.  This
replacement scans each contribution-weight matrix in both orientations and
reports the highest cosine-similarity hit per pattern and input example.
"""

import os

import click
import numpy as np
import pandas as pd

from legacy.invivo_footprints import run_tfmodisco
from chrombpnet.evaluation.modisco.api import iter_patterns


HIT_COLUMNS = [
    "chrom", "start", "end", "key", "strand", "peak_index",
    "imp_total_signed_score", "imp_total_score", "imp_frac_score",
    "imp_ic_avg_score", "agg_sim", "mod_delta", "mod_precision",
    "mod_percentile", "fann_perclasssum_perc", "fann_perclassavg_perc",
]


def import_tfmodisco_hits(hits_bed):
    return pd.read_csv(hits_bed, sep="\t", header=None, names=HIT_COLUMNS)


def _information_content(ppm, background):
    ppm = np.asarray(ppm, dtype=float)
    ppm = (ppm + 0.001) / (ppm.sum(axis=1, keepdims=True) + 0.004)
    return np.sum(ppm * np.log2(ppm / np.asarray(background)[None, :]), axis=1)


def _trim_pattern(pattern, background, min_ic):
    ic = _information_content(pattern.sequence, background)
    passing = np.flatnonzero(ic >= min_ic)
    if not passing.size:
        raise ValueError(f"IC threshold {min_ic} removes all of {pattern.tag}")
    start, end = int(passing[0]), int(passing[-1]) + 1
    return pattern.contrib_scores[start:end], ic[start:end]


def _best_hit(track, motif):
    """Return (start, end, reverse-complement, cosine similarity)."""
    length = motif.shape[0]
    if length > track.shape[0]:
        raise ValueError("Motif is longer than the contribution-score track")
    orientations = (motif, motif[::-1, ::-1])
    best = None
    for reverse, oriented in enumerate(orientations):
        motif_flat = oriented.reshape(-1)
        motif_norm = np.linalg.norm(motif_flat)
        for start in range(track.shape[0] - length + 1):
            window = track[start:start + length].reshape(-1)
            denominator = np.linalg.norm(window) * motif_norm
            similarity = float(np.dot(window, motif_flat) / denominator) if denominator else 0.0
            candidate = (similarity, start, bool(reverse))
            if best is None or candidate[0] > best[0]:
                best = candidate
    similarity, start, reverse = best
    return start, start + length, reverse, similarity


@click.command()
@click.option("-o", "--outdir", required=True)
@click.option("-i", "--input-length", default=2114, type=int)
@click.option("-c", "--center-cut-size", default=1000, type=int)
@click.option("--keep-non-acgt", is_flag=True)
@click.option("-m", "--min-ic", default=0.2, type=float)
@click.option("-mc", "--metacluster-ind", default=0, type=int)
@click.option("-p", "--pattern-inds", default=None, type=str)
@click.argument("shap_scores_path")
@click.argument("tfm_results_path")
@click.argument("peak_bed_path")
def main(shap_scores_path, tfm_results_path, peak_bed_path, outdir,
         input_length, center_cut_size, keep_non_acgt, min_ic,
         metacluster_ind, pattern_inds):
    if metacluster_ind not in (0, 1):
        raise ValueError("metacluster_ind must be 0 (positive) or 1 (negative)")
    selected = None if pattern_inds is None else {int(value) for value in pattern_inds.split(",")}
    group = "pos_patterns" if metacluster_ind == 0 else "neg_patterns"
    patterns = [pattern for pattern in iter_patterns(tfm_results_path) if pattern.group == group]
    if selected is not None:
        patterns = [pattern for index, pattern in enumerate(patterns) if index in selected]
    if not patterns:
        raise ValueError(f"No selected patterns found in {group}")

    peaks = pd.read_csv(peak_bed_path, sep="\t", header=None, usecols=[0, 1, 2, 9],
                        names=["chrom", "start", "end", "summit"])
    peak_starts = peaks["start"].to_numpy() + peaks["summit"].to_numpy() - input_length // 2
    hyp, actual, sequences, _ = run_tfmodisco.import_shap_scores_part2(
        shap_scores_path, peaks.rename(columns={"chrom": "peak_chrom", "start": "peak_start",
                                                "end": "peak_end"}),
        center_cut_size=center_cut_size)
    if not keep_non_acgt:
        valid = sequences.sum(axis=(1, 2)) == center_cut_size
        hyp, actual, sequences = hyp[valid], actual[valid], sequences[valid]
        peaks, peak_starts = peaks.loc[valid].reset_index(drop=True), peak_starts[valid]

    background = sequences.mean(axis=(0, 1))
    background = background / background.sum()
    trimmed = [(_trim_pattern(pattern, background, min_ic), pattern) for pattern in patterns]
    offset = (input_length - center_cut_size) // 2
    rows, raw_rows = [], []
    for example, track in enumerate(actual):
        total_track = np.abs(track).sum()
        for pattern_index, ((motif, ic), pattern) in enumerate(trimmed):
            start, end, reverse, similarity = _best_hit(track, motif)
            values = track[start:end].sum(axis=1)
            oriented_ic = ic[::-1] if reverse else ic
            key = f"{metacluster_ind}_{pattern.name.rsplit('_', 1)[-1]}"
            genomic_start = int(peak_starts[example] + offset + start)
            metrics = [float(values.sum()), float(np.abs(values).sum()),
                       float(np.abs(values).sum() / total_track) if total_track else 0.0,
                       float(np.mean(values * oriented_ic)), similarity]
            rows.append([peaks.loc[example, "chrom"], genomic_start,
                         genomic_start + (end - start), key, "-" if reverse else "+",
                         example, *metrics, np.nan, np.nan, np.nan, np.nan, np.nan])
            raw_rows.append([example, pattern_index, start, end, reverse, similarity])

    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(raw_rows, columns=["example_index", "pattern_index", "start", "end",
                                    "revcomp", "agg_sim"]).to_csv(
        os.path.join(outdir, "tfm_matches_raw.tsv"), sep="\t", index=False)
    pd.DataFrame(rows, columns=HIT_COLUMNS).to_csv(
        os.path.join(outdir, "tfm_matches.bed"), sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
