#!/usr/bin/env python
"""The ensemble-vs-distillation table, in Markdown, from compare_fold_means.py's output.

    python fold_mean_table.py --dir <compare_fold_means.py --out-dir> [--label "chrombpnet-refactor (updated)"]

One row per comparison against the five fold models' mean: a fold model
(against the mean that includes it, and against the other four's), the LOFO
students and the final student. Columns: their predictions' agreement with
the mean's (log-count Pearson r, profile 1 - median JSD) and their DeepSHAP
scores' (median per-region Pearson r of the projected scores, counts and
profile heads). Each value is the mean over datasets and folds (the final
student: over datasets), with the range over datasets in parentheses with
--ranges.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROWS = (("fold model (vs 5-fold mean)", "One fold model vs. the 5-fold mean (includes itself)"),
        ("fold model (vs other 4)", "One fold model vs. the other 4 folds' mean"),
        ("LOFO student", "LOFO student vs. the 5-fold mean"),
        ("final student", "Final student vs. the 5-fold mean"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", default="chrombpnet-refactor (updated)")
    ap.add_argument("--ranges", action="store_true", help="add the range over datasets")
    args = ap.parse_args(argv)
    pred = pd.read_csv(os.path.join(args.dir, "fold_mean_predictions.tsv"), sep="\t")
    contrib = pd.read_csv(os.path.join(args.dir, "fold_mean_contributions.tsv"), sep="\t")
    contrib = contrib[contrib["kind"] == "projected"]

    def cell(frame, column):
        per_dataset = frame.groupby("encid")[column].mean()
        text = f"{frame[column].mean():.3f}"
        if args.ranges and len(per_dataset) > 1:
            text += f" ({per_dataset.min():.3f}–{per_dataset.max():.3f})"
        return text

    lines = ["| Comparison | Counts prediction (Pearson r) | Profile prediction (1 − JSD) "
             "| Counts contributions (Pearson r) | Profile contributions (Pearson r) |",
             "|---|---|---|---|---|"]
    for model, name in ROWS:
        p = pred[pred["model"] == model]
        c = contrib[contrib["model"] == model]
        if p.empty or c.empty:
            continue
        lines.append(f"| {name} | {cell(p, 'counts_pearsonr')} | {cell(p, 'one_minus_jsd')} "
                     f"| {cell(c[c['head'] == 'counts'], 'pearsonr_median')} "
                     f"| {cell(c[c['head'] == 'profile'], 'pearsonr_median')} |")
    n_datasets = pred["encid"].nunique()
    datasets = f"{n_datasets} dataset" + ("s" if n_datasets != 1 else "")
    n_regions = int(pred["n_regions"].median())
    lines.append("")
    lines.append(f"_Agreement with the 5-fold mean (the ensemble) of {args.label}, on {n_regions:,} peaks shared by "
                 f"all models, per dataset; means over {datasets} × 5 folds (the final student: over the "
                 f"{datasets}). Predictions of the bias-free models: Pearson r of the log counts, and "
                 f"1 − the median Jensen-Shannon distance (base 2) of the profile probabilities, against the mean "
                 f"log counts and the mean probabilities. Contributions: median per-peak Pearson r of the projected "
                 f"DeepSHAP scores against the mean scores, with the same 20 shuffled references for every model. "
                 f"A fold model is one fifth of the 5-fold mean, which flatters it; against the other four folds' "
                 f"mean it is not. LOFO student k never saw fold k's test chromosomes, though four of its five "
                 f"teachers did; the final student trained on every chromosome._")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
