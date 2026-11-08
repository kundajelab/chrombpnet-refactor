#!/usr/bin/env python3
"""Redraw compare_fold_means.py's three plots from its saved TSVs (the plotting tail of its main())."""
import os
import sys

import pandas as pd

sys.path.insert(0, "/path/to/workdir/software/chrombpnet-refactor/analysis/expand")
import compare_fold_means as cfm  # noqa: E402

out_dir, label = sys.argv[1], sys.argv[2]
contrib = pd.read_csv(os.path.join(out_dir, "fold_mean_contributions.tsv"), sep="\t")
pred = pd.read_csv(os.path.join(out_dir, "fold_mean_predictions.tsv"), sep="\t")
teachers = f"the five {label} fold models"
note = (f"Bars: the mean over folds (dots: each fold). Fold model k: against the five-fold mean, of which it "
        f"is a fifth.\nLOFO student k: trained on "
        f"fold k's training chromosomes (four of its five teachers saw its test chromosomes).\nFinal student: "
        f"trained on every chromosome. Both distilled from {teachers}.")
n_regions = f"{int(pred['n_regions'].median()):,} shared peaks per dataset"
for metric, (column, metric_label) in (("pearson", ("pearsonr_median", "Pearson r")),
                                       ("cosine", ("cosine_median", "cosine similarity"))):
    panels = [((lambda f, h=h, k=k: (f["head"] == h) & (f["kind"] == k)), column, f"{h}\n{k}")
              for h in cfm.HEADS for k in cfm.KINDS]
    cfm.plot(out_dir, contrib, panels, f"fold_mean_contributions_{metric}",
             f"DeepLIFT/DeepSHAP scores against the five-fold mean of {label}",
             f"{n_regions}  |  bar = median over regions of the per-region {metric_label}  |  "
             f"same DeepLIFT/DeepSHAP references (seed 1234) for every model", note, [(0, 1)] * len(panels))
panels = [((lambda f: f["model"].notna()), "counts_pearsonr", "log counts\nPearson r"),
          ((lambda f: f["model"].notna()), "one_minus_jsd", "profile\n1 - median JSD")]
cfm.plot(out_dir, pred, panels, "fold_mean_predictions",
         f"Predictions against the five-fold mean of {label}",
         f"{n_regions}  |  bias-free models  |  reference: the mean log counts and the mean "
         "profile probabilities of the fold models", note, [(0, 1), (0, 1)])
print("redrawn in", out_dir)
