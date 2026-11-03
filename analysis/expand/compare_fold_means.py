#!/usr/bin/env python
"""Students against their teachers' five-fold mean, on shared regions.

    python compare_fold_means.py --scores <root>/shared/<teacher set> --label "<teacher label>" \
        --out-dir <dir> [--biosamples biosamples.tsv]

--scores holds, per dataset (<ASSAY>/<ENCID>/), score_shared_regions.py's
output for eleven bias-free models scored on the same regions.bed: the five
fold models (fold_0 .. fold_4, the teachers), the five LOFO students
(lofo_0 .. lofo_4) and the final student (final). The fold models' mean is
their ensemble: the mean of their hypothetical DeepSHAP scores, and of their
predictions -- the mean of their profile probabilities and of their log
counts, the targets the students were distilled on.

Compared against it, per dataset:

  fold k (vs 5-fold mean)    fold model k against the five-fold mean, of which
                             it is a fifth -- which flatters it
  fold k (vs other 4)        fold model k against the mean of the other four
                             (in the TSVs only: the only row with another
                             reference, so the plots leave it out)
  LOFO student k             against the five-fold mean (every student
                             learned from all five)
  final student              against the five-fold mean

DeepSHAP, as ../engine/run_summarize.py measures it: per region, over the
whole input window, the Pearson r and cosine similarity of the hypothetical
scores (4 x 2114, flattened) and of the projected scores (the observed
base's); the median over regions. Regions are kept where every model has
them and the one-hot sequences agree.

Predictions: the Pearson r and the MSE, over regions, of the model's log
counts against the reference's; and the median over regions of the
Jensen-Shannon distance (base 2) between their profile probabilities.

Writes to --out-dir:
  fold_mean_contributions.tsv, fold_mean_predictions.tsv
  fold_mean_contributions_{pearson,cosine}.{png,pdf,tsv}, fold_mean_predictions.{png,pdf,tsv}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))
import run_summarize as rs  # noqa: E402

HEADS = ("counts", "profile")
KINDS = ("hypothetical", "projected")
FOLDS = range(5)
VS_ALL = "fold model (vs 5-fold mean)"
VS_OTHERS = "fold model (vs other 4)"
LOFO = "LOFO student"
FINAL = "final student"
MODELS = (VS_ALL, VS_OTHERS, LOFO, FINAL)
PLOTTED = (VS_ALL, LOFO, FINAL)


def load(prefix, head, keys):
    raw, hyp, rows = rs.load_rows(f"{prefix}.{head}_scores.h5", f"{prefix}.interpreted_regions.bed", keys)
    return raw, hyp, rows


def contributions(dataset_dir, keys):
    """Rows (model, k, head, kind, n, r_median, r_mean, cos_median, cos_mean)."""
    sys.path.insert(0, str(HERE))
    from compare_original_mean import compare, summarize
    out = []
    labels = [f"fold_{k}" for k in FOLDS] + [f"lofo_{k}" for k in FOLDS] + ["final"]
    for head in HEADS:
        loaded = {label: load(os.path.join(dataset_dir, label), head, keys) for label in labels}
        have = np.ones(len(keys), dtype=bool)
        for raw, hyp, rows in loaded.values():
            have &= np.asarray([r is not None for r in rows])
        aligned = {}
        for label, (raw, hyp, rows) in loaded.items():
            idx = np.asarray([r for r, ok in zip(rows, have) if ok], dtype=np.int64)
            aligned[label] = (raw[idx], hyp[idx].astype(np.float32))
        raw0 = aligned["fold_0"][0]
        total = sum(aligned[f"fold_{k}"][1] for k in FOLDS)
        mean5 = total / len(FOLDS)
        for k in FOLDS:
            raw_k, hyp_k = aligned[f"fold_{k}"]
            for model, ref in ((VS_ALL, mean5), (VS_OTHERS, (total - hyp_k) / (len(FOLDS) - 1))):
                metrics, n, _ = compare(hyp_k, raw_k, ref, raw0)
                out += [(model, k, head, kind) + (n,) + v for kind, v in summarize(metrics).items()]
            raw_s, hyp_s = aligned[f"lofo_{k}"]
            metrics, n, _ = compare(hyp_s, raw_s, mean5, raw0)
            out += [(LOFO, k, head, kind) + (n,) + v for kind, v in summarize(metrics).items()]
        raw_s, hyp_s = aligned["final"]
        metrics, n, _ = compare(hyp_s, raw_s, mean5, raw0)
        out += [(FINAL, "", head, kind) + (n,) + v for kind, v in summarize(metrics).items()]
        del loaded, aligned, total, mean5
    return out


def predictions(dataset_dir, keys):
    """Rows (model, k, n, counts_pearsonr, counts_mse, profile_median_jsd)."""
    import h5py
    from scipy.spatial.distance import jensenshannon
    labels = [f"fold_{k}" for k in FOLDS] + [f"lofo_{k}" for k in FOLDS] + ["final"]
    sets = {}
    for label in labels:
        prefix = os.path.join(dataset_dir, label)
        index = {key: i for i, key in enumerate(rs.region_keys(f"{prefix}.interpreted_regions.bed"))}
        with h5py.File(f"{prefix}.predictions.h5", "r") as f:
            sets[label] = (index, f["logcounts"][:], f["probs"][:])
    common = [key for key in keys if all(key in s[0] for s in sets.values())]
    aligned = {}
    for label, (index, logcounts, probs) in sets.items():
        rows = np.asarray([index[key] for key in common])
        aligned[label] = (logcounts[rows].astype(np.float64), probs[rows].astype(np.float64))

    def measure(model, k, a, b):
        (la, pa), (lb, pb) = a, b
        r = float(np.corrcoef(la, lb)[0, 1])
        mse = float(np.mean((la - lb) ** 2))
        jsd = float(np.nanmedian(jensenshannon(pa, pb, axis=1, base=2)))
        return (model, k, len(la), r, mse, jsd)

    folds = [aligned[f"fold_{k}"] for k in FOLDS]
    mean5 = (np.mean([f[0] for f in folds], axis=0), np.mean([f[1] for f in folds], axis=0))
    out = []
    for k in FOLDS:
        others = [folds[j] for j in FOLDS if j != k]
        loo = (np.mean([f[0] for f in others], axis=0), np.mean([f[1] for f in others], axis=0))
        out.append(measure(VS_ALL, k, folds[k], mean5))
        out.append(measure(VS_OTHERS, k, folds[k], loo))
        out.append(measure(LOFO, k, aligned[f"lofo_{k}"], mean5))
    out.append(measure(FINAL, "", aligned["final"], mean5))
    return out


def plot(out_dir, frame, panels, out_name, title, subtitle, footnote, ylims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_summarize as ps

    # The blog's colours: the fold models white (the engine figure's hollow reference bar),
    # the LOFO student blue and the final student green (plot_summarize.SLOTS).
    styles = {VS_ALL: ps.RETRAINED_BAR, LOFO: dict(color=ps.SLOTS[0], linewidth=0),
              FINAL: dict(color=ps.SLOTS[2], linewidth=0)}
    datasets = sorted({(r.biosample, r.encid, r.assay) for r in frame.itertuples()},
                      key=lambda d: ps._encid_rank(d[1]))
    fig, axes = plt.subplots(len(panels), 1, squeeze=False, figsize=(9.5, 1.7 * len(panels) + 2.4))
    axes = [a[0] for a in axes]
    fig.patch.set_facecolor(ps.SURFACE)
    step = 0.2
    offsets = {m: (i - (len(PLOTTED) - 1) / 2) * step for i, m in enumerate(PLOTTED)}
    for index, (ax, (selector, column, label)) in enumerate(zip(axes, panels)):
        sub = frame[selector(frame)]
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=ps.GRID, linewidth=0.8)
        ax.set_facecolor(ps.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ps.GRID)
        for x, (biosample, encid, assay) in enumerate(datasets):
            for model in PLOTTED:
                values = sub[(sub["encid"] == encid) & (sub["model"] == model)][column].astype(float)
                if values.empty:
                    continue
                ax.bar(x + offsets[model], values.mean(), width=step - 0.03, zorder=3, **styles[model])
                if len(values) > 1:   # one dot per fold
                    ax.scatter([x + offsets[model]] * len(values), values, s=7, color=ps.INK, zorder=4)
        ax.set_ylim(*ylims[index])
        ax.set_ylabel(label, fontsize=9, color=ps.INK, rotation=0, ha="right", va="center", labelpad=14)
        ax.tick_params(axis="y", labelsize=7, colors=ps.INK_MUTED, length=0)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([f"{e} ({a})\n{b}" for b, e, a in datasets] if index == len(panels) - 1 else [],
                           fontsize=7.5, color=ps.INK)
        ax.tick_params(axis="x", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, **styles[m]) for m in PLOTTED]
    ps.finish(plt, fig, os.path.join(out_dir, out_name), title, subtitle, footnote, handles, list(PLOTTED))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="score_shared_regions.py's tree for one teacher set")
    ap.add_argument("--label", required=True, help='the teachers\' name, e.g. "chrombpnet-refactor (updated)"')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--biosamples", default=None)
    args = ap.parse_args(argv)
    import pandas as pd
    biosamples = rs.load_biosamples(args.biosamples) if args.biosamples else {}
    contrib_rows, pred_rows = [], []
    for assay in sorted(os.listdir(args.scores)):
        for encid in sorted(os.listdir(os.path.join(args.scores, assay))):
            d = os.path.join(args.scores, assay, encid)
            if not os.path.isfile(os.path.join(d, "regions.bed")):
                continue
            missing = [m for m in [f"fold_{k}" for k in FOLDS] + [f"lofo_{k}" for k in FOLDS] + ["final"]
                       if not os.path.isfile(os.path.join(d, f"{m}.COMPLETED"))]
            if missing:
                print(f"# {encid}: not scored yet: {', '.join(missing)} -- skipped", file=sys.stderr)
                continue
            keys = rs.region_keys(os.path.join(d, "regions.bed"))
            base = dict(assay=assay, biosample=biosamples.get(encid, encid), encid=encid)
            for model, k, head, kind, n, r_med, r_mean, c_med, c_mean in contributions(d, keys):
                contrib_rows.append(dict(base, model=model, fold=k, head=head, kind=kind, n_regions=n,
                                         pearsonr_median=r_med, pearsonr_mean=r_mean,
                                         cosine_median=c_med, cosine_mean=c_mean))
            for model, k, n, r, mse, jsd in predictions(d, keys):
                pred_rows.append(dict(base, model=model, fold=k, n_regions=n, counts_pearsonr=r,
                                      counts_mse=mse, profile_median_jsd=jsd, one_minus_jsd=1 - jsd))
            print(f"# {encid}: compared", file=sys.stderr, flush=True)
    if not contrib_rows:
        sys.exit("Error: no dataset has all eleven models scored")
    os.makedirs(args.out_dir, exist_ok=True)
    contrib = pd.DataFrame(contrib_rows)
    pred = pd.DataFrame(pred_rows)
    contrib.to_csv(os.path.join(args.out_dir, "fold_mean_contributions.tsv"), sep="\t", index=False,
                   float_format="%.6f")
    pred.to_csv(os.path.join(args.out_dir, "fold_mean_predictions.tsv"), sep="\t", index=False,
                float_format="%.6f")
    print(f"wrote {args.out_dir}/fold_mean_contributions.tsv, fold_mean_predictions.tsv")

    teachers = f"the five {args.label} fold models"
    note = (f"Bars: the mean over folds (dots: each fold). Fold model k: against the five-fold mean, of which it "
            f"is a fifth.\nLOFO student k: trained on "
            f"fold k's training chromosomes (four of its five teachers saw its test chromosomes).\nFinal student: "
            f"trained on every chromosome. Both distilled from {teachers}.")
    n_regions = f"{int(pred['n_regions'].median()):,} shared peaks per dataset"
    for metric, (column, metric_label) in (("pearson", ("pearsonr_median", "Pearson r")),
                                           ("cosine", ("cosine_median", "cosine similarity"))):
        panels = [((lambda f, h=h, k=k: (f["head"] == h) & (f["kind"] == k)), column, f"{h}\n{k}")
                  for h in HEADS for k in KINDS]
        plot(args.out_dir, contrib, panels, f"fold_mean_contributions_{metric}",
             f"DeepLIFT/DeepSHAP scores against the five-fold mean of {args.label}",
             f"{n_regions}  |  bar = median over regions of the per-region {metric_label}  |  "
             f"same DeepLIFT/DeepSHAP references (seed 1234) for every model", note, [(0, 1)] * len(panels))
        contrib[["model", "fold", "biosample", "encid", "assay", "head", "kind", "n_regions", column]].to_csv(
            os.path.join(args.out_dir, f"fold_mean_contributions_{metric}.tsv"), sep="\t", index=False,
            float_format="%.6f")
    panels = [((lambda f: f["model"].notna()), "counts_pearsonr", "log counts\nPearson r"),
              ((lambda f: f["model"].notna()), "one_minus_jsd", "profile\n1 - median JSD")]
    plot(args.out_dir, pred, panels, "fold_mean_predictions",
         f"Predictions against the five-fold mean of {args.label}",
         f"{n_regions}  |  bias-free models  |  reference: the mean log counts and the mean "
         "profile probabilities of the fold models", note, [(0, 1), (0, 1)])
    pred.to_csv(os.path.join(args.out_dir, "fold_mean_predictions.tsv"), sep="\t", index=False, float_format="%.6f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
