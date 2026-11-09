#!/usr/bin/env python3
"""The blog's bias-model figure: (1) the main bias models' counts predictions
on held-out chromosomes against the observed counts; (2) each ENCODE ChromBPNet
model's total predicted counts with and without its bias model.

    python make_figure.py DATA_DIR OUT.png

DATA_DIR holds bias_test_counts.py's fig1/*.tsv, bigwig_totals.py's
prediction_totals.tsv and matched_totals.py's matched_totals.tsv. Prints the
numbers the figure shows.
"""

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Peaks red, non-peaks blue, as the blog's test-set figure colours counts and profile; the
# assays green and orange, the blog's two-group colours (plot_summarize.SLOT_NAMES, as the
# default/updated and LOFO/final-student figures use them).
RED, BLUE, GREEN, ORANGE = "#e34948", "#2a78d6", "#1baf7a", "#eb6834"
# (bias model, assay, biosample, ENCODE models whose fold-1 bias it is)
BIAS_MODELS = [("ENCSR880CUB", "DNase-seq", "prefrontal cortex", 389),
               ("ENCSR283TME", "DNase-seq", "CD8+ T cells", 348),
               ("ENCSR868FGK", "ATAC-seq", "K562", 315)]
ASSAY_COLOURS = {"DNase-seq": GREEN, "ATAC-seq": ORANGE}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def totals(data_dir):
    frame = pd.read_csv(os.path.join(data_dir, "prediction_totals.tsv"), sep="\t")
    # Files whose header total is NaN also hold overflowed values (their finite totals reach 1e41,
    # against 1e6 to 1e9 for the rest): broken predictions, with no meaningful total.
    broken = ~(np.isfinite(frame.with_bias_sum) & np.isfinite(frame.without_bias_sum))
    frame.attrs["broken"] = sorted(frame.experiment[broken])
    frame = frame[~broken].copy()
    frame["matched"] = False
    matched_path = os.path.join(data_dir, "matched_totals.tsv")
    if os.path.exists(matched_path):
        matched = pd.read_csv(matched_path, sep="\t").set_index("experiment")
        rows = frame.experiment.isin(matched.index)
        for column in ("with_bias_sum", "without_bias_sum"):
            frame.loc[rows, column] = frame.loc[rows, "experiment"].map(matched[column])
        frame.loc[rows, "matched"] = True
    comparable = np.isfinite(frame.with_bias_sum) & np.isfinite(frame.without_bias_sum) & (
        frame.matched | (frame.with_bias_bases == frame.without_bias_bases))
    broken = frame.attrs["broken"]
    frame = frame[comparable].copy()
    frame.attrs["broken"] = broken
    frame["difference"] = frame.with_bias_sum - frame.without_bias_sum
    frame["share"] = 100 * frame.difference / frame.with_bias_sum
    return frame


def main():
    data_dir, out = sys.argv[1:3]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK_2,
                         "axes.labelsize": 10})
    figure = plt.figure(figsize=(10, 9.7), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 1, height_ratios=[1, 1.02], hspace=0.62, left=0.075, right=0.975,
                               top=0.868, bottom=0.135)

    # 1. Bias models' counts head on held-out chromosomes.
    top = grid[0].subgridspec(1, 3, wspace=0.28)
    figure.text(0.02, 0.975, "1. The bias models' counts predictions barely change across regions", fontsize=13,
                weight="bold")
    figure.text(0.02, 0.951, "fold-1 held-out chromosomes of each bias model's own experiment  |  log(1 + counts) over "
                "1,000 bp  |  dashed: predicted = observed", fontsize=8.6,
                color=INK_2)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=RED, label="peaks"),
               plt.Line2D([], [], ls="", marker="o", ms=6, color=BLUE, label="non-peaks (what bias models are trained on)")]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.947), ncol=2, frameon=False,
                  fontsize=9, handletextpad=0.3, columnspacing=1.4)
    for column, (label, assay, biosample, users) in enumerate(BIAS_MODELS):
        data = pd.read_csv(os.path.join(data_dir, "fig1", "%s_fold1_test_counts.tsv" % label), sep="\t")
        ax = figure.add_subplot(top[column])
        style(ax)
        low = min(data.observed_log_counts.quantile(0.0005), data.predicted_log_counts.quantile(0.0005))
        high = max(data.observed_log_counts.quantile(0.9995), data.predicted_log_counts.quantile(0.9995))
        low, high = np.floor(low), np.ceil(high)
        ax.plot([low, high], [low, high], color=INK_2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        means = {}
        for is_peak, colour, name in ((0, BLUE, "non-peaks"), (1, RED, "peaks")):
            subset = data[data.is_peak == is_peak]
            ax.scatter(subset.observed_log_counts, subset.predicted_log_counts, s=1.2, color=colour, alpha=0.12,
                       linewidths=0, rasterized=True, zorder=2)
            r = pearsonr(subset.observed_log_counts, subset.predicted_log_counts)[0]
            means[name] = (np.expm1(subset.predicted_log_counts).mean(), np.expm1(subset.observed_log_counts).mean(), r)
            print("%s %s: n %d, r %.4f, mean predicted %.0f, mean observed %.0f" % (label, name, len(subset), r,
                                                                                 means[name][0], means[name][1]))
        # Observed counts are not the counts the enzyme's bias accounts for, so a correlation with
        # them in peaks says nothing; what the panel shows is how little the prediction moves.
        rows = [("mean counts", "peaks", "non-peaks", INK_2),
                ("predicted", "%.0f" % means["peaks"][0], "%.0f" % means["non-peaks"][0], INK),
                ("observed", "%.0f" % means["peaks"][1], "%.0f" % means["non-peaks"][1], INK)]
        for row, (name, peaks, nonpeaks, colour) in enumerate(rows):
            y = 0.95 - 0.07 * row
            ax.text(0.04, y, name, transform=ax.transAxes, va="center", fontsize=8.6, color=colour, zorder=4)
            ax.text(0.57, y, peaks, transform=ax.transAxes, va="center", ha="right", fontsize=8.6,
                    color=RED if row else INK_2, zorder=4)
            ax.text(0.88, y, nonpeaks, transform=ax.transAxes, va="center", ha="right", fontsize=8.6,
                    color=BLUE if row else INK_2, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))  # over the dashed diagonal
        ax.text(0.04, 0.95 - 0.07 * 3, "Pearson r (non-peaks) = %.2f" % means["non-peaks"][2], transform=ax.transAxes,
                va="center", fontsize=8.6, color=INK, zorder=4)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        # Three lines: one line is wider than a panel.
        ax.set_title("Bias model of %d %s\nChromBPNet models:\nFrom %s (%s)." % (users, assay, label, biosample),
                     fontsize=9.4, color=INK, loc="left")
        ax.set_xlabel("observed log counts")
        if column == 0:
            ax.set_ylabel("bias model's predicted log counts")

    # 2. Total predicted counts with and without the bias model.
    frame = totals(data_dir)
    figure.text(0.02, 0.475, "2. Total predicted counts with and without the bias model, per ChromBPNet model",
                fontsize=13, weight="bold")
    figure.text(0.02, 0.451, "Predicted signal with bias vs. Predicted signal without bias, across all 1,512 ENCODE "
                "GRAMMAR ChromBPNet models", fontsize=8.6, color=INK_2)

    # One square panel at the row's left, under the first panel above; the table beside it.
    ax = figure.add_subplot(grid[1])
    ax.set_anchor("W")
    style(ax)
    for assay, colour in ASSAY_COLOURS.items():
        subset = frame[frame.assay == assay]
        ax.scatter(subset.without_bias_sum, subset.with_bias_sum, s=9, color=colour, alpha=0.75, linewidths=0,
                   label="%s (%d)" % (assay, len(subset)), zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    low = 10 ** np.floor(np.log10(min(frame.without_bias_sum.min(), frame.with_bias_sum.min())))
    high = 10 ** np.ceil(np.log10(max(frame.without_bias_sum.max(), frame.with_bias_sum.max())))
    ax.plot([low, high], [low, high], color=INK_2, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.set_xlabel("total predicted counts without bias")
    ax.set_ylabel("total predicted counts with bias")
    ax.legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.2, markerscale=1.6)
    ax.text(0.97, 0.04, "dashed: equal totals", transform=ax.transAxes, ha="right", fontsize=8.4, color=INK_2)

    valid = frame[frame.difference >= 0]
    mean, sd = valid.difference.mean(), valid.difference.std()
    share_mean, share_sd, share_median = valid.share.mean(), valid.share.std(), valid.share.median()
    # Beside the square panel: each total's median ± SD across models, in counts and as a share of
    # the with-bias total (the with-bias column's share is 100% by definition).
    ax.apply_aspect()
    box = ax.get_position()
    columns = [("Predict\nwith bias", valid.with_bias_sum, None),
               ("Predict\nwithout bias", valid.without_bias_sum, 100 - valid.share),
               ("Predicted\nbias", valid.difference, valid.share)]
    left, centres = box.x1 + 0.065, [box.x1 + 0.265, box.x1 + 0.39, box.x1 + 0.515]
    middle = box.y0 + 0.5 * box.height
    figure.text(left, middle + 0.085, "Per model, median ± SD (%s models)" % format(len(valid), ","), fontsize=9.4,
                color=INK, va="center")
    figure.add_artist(plt.Line2D([left, centres[-1] + 0.06], [middle + 0.006] * 2, transform=figure.transFigure,
                                 color=GRID, lw=0.8))
    for centre, (header, counts, share) in zip(centres, columns):
        figure.text(centre, middle + 0.035, header, fontsize=9, color=INK_2, ha="center", va="center",
                    linespacing=1.3)
        figure.text(centre, middle - 0.02, "%.1f ± %.1f" % (counts.median() / 1e6, counts.std() / 1e6), fontsize=9,
                    color=INK, ha="center", va="center")
        figure.text(centre, middle - 0.055, "100" if share is None else "%.1f ± %.1f" % (share.median(), share.std()),
                    fontsize=9, color=INK, ha="center", va="center")
    figure.text(left, middle - 0.02, "million counts", fontsize=9, color=INK_2, va="center")
    figure.text(left, middle - 0.055, "% of with-bias total", fontsize=9, color=INK_2, va="center")

    excluded = 1512 - len(valid)
    note = ("Predicted bias = with - without, per model. Each total is over the positions both bigWigs cover: from the file "
            "header, or position by position where the two files differ in coverage or hold NaN values.")
    lower = sorted(frame[frame.difference < 0].experiment)
    if lower:
        note += (" Left out: %s, whose with-bias total is below its total without bias, which cannot happen "
                 "(its two files may be swapped)" % ", ".join(lower))
    if frame.attrs["broken"]:
        note += "%s %d models whose bigWigs hold NaN and overflowed values" % (
            ";" if lower else " Left out:", len(frame.attrs["broken"]))
    note += "." if lower or frame.attrs["broken"] else ""
    missing = sorted(set(pd.read_csv(os.path.join(data_dir, "prediction_totals.tsv"), sep="\t").experiment)
                     - set(frame.experiment) - set(frame.attrs["broken"]))
    if missing:
        note += " Not yet summed: %d models." % len(missing)
    figure.text(0.02, 0.018, "\n".join(textwrap.wrap(note, 158)), fontsize=7.8, color=INK_2, va="bottom")
    figure.savefig(out, dpi=200, facecolor=SURFACE)

    print("models %d (%d DNase, %d ATAC); with - without: mean %.4g, SD %.4g; share mean %.2f%%, SD %.2f%%, "
          "median %.2f%%; left out %d" % (len(valid), (valid.assay == "DNase-seq").sum(),
                                          (valid.assay == "ATAC-seq").sum(), mean, sd, share_mean, share_sd,
                                          share_median, excluded))
    for assay, subset in valid.groupby("assay"):
        print("  %s: share mean %.2f%%, SD %.2f%%, median %.2f%%; difference mean %.4g, SD %.4g" % (
            assay, subset.share.mean(), subset.share.std(), subset.share.median(), subset.difference.mean(),
            subset.difference.std()))


if __name__ == "__main__":
    main()
