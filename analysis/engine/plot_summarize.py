#!/usr/bin/env python
"""Plot the engines sweep collated by run_summarize.py.

    python plot_summarize.py --dir <run_summarize.py --out-dir> [<dir> ...]

Five figures, each written as .png, .pdf and a .tsv table view:

    summary_scores     each engine's model on its own test set, one panel per
                       engine (the old TensorFlow ChromBPNet, `original`,
                       first when present, then the original retrained on
                       the engines' inputs, `original (newly trained)`)
    summary_agreement  how closely each engine's model agrees with the
                       original TensorFlow model, one panel per engine (and
                       one for the retrained original)
    summary_contributions_pearson, summary_contributions_cosine
                       how closely each engine's DeepSHAP scores on its 30K
                       peak subsample agree with the original model's, one
                       panel per head (counts, profile) and kind
                       (hypothetical, projected), one bar per engine; with
                       run_original_retrain.py's and run_deepshap_seeds.py's
                       runs, two neutral bars per fold before the engines':
                       the original retrained on the engines' inputs against
                       the stored scores (hollow, "original (newly trained)"),
                       then the original's DeepSHAP rerun on the model whose
                       scores are stored, with a new seed, against them
                       (grey, "original (same model; diff seed)": the
                       ceiling); with two
                       or more seeds, a solid line per fold for that model's
                       runs against one another
    summary_runtime    wall time of each run, one panel per pipeline step

    x   biosample  >  ENCID (assay)  >  fold, datasets in ENCID_ORDER (as in
        ../dnase-cutsite's plots)
    y   scores and agreement: 0 to 1, higher is better for both measures
          red  counts Pearson r       (agreement: between the engine's and
                                       the original's predicted log counts)
          blue 1 - median profile JSD (agreement: between their predicted
                                       profiles)
          horizontal rule per measure = mean over that dataset's folds
        contributions: the median over regions of the per-region Pearson r
          (or cosine similarity), one bar per engine
        runtime: hours, one bar per engine

The runs drawn are run_summarize.py's (its runs.tsv: the engines, or its
--runs labels), in that order, each in the next categorical colour; panels
for the original and the retrained original come first, and pairs of runs
from its --versus follow the runs vs the original in summary_agreement.
--noun names them in the titles (default "engine").

In summary_scores every panel other than the reference run (--reference,
default the first run) carries "(+0.02)" after each mean: that engine minus the
reference, averaged over the folds both have. Higher is better for both
measures, so + means the engine scores higher. The engines share nonpeaks
and preprocessing, so they are scored on the same regions (run_summarize.py
checks this, in inputs.tsv); `original` used its own peak set, so its
difference is only indicative.

Both measures share one axis on purpose: plotting 1 - JSD puts the
divergence on the same higher-is-better 0-1 scale as a correlation.

The original's bars are neutral, not colour: a fourth and fifth categorical
slot beside the engines' three would not clear the colour-vision checks. The
retrained original's bar is hollow (an ink outline), the new seed's a solid
grey, and the seed-vs-seed line ink.

Colours: the measures are the validated red/blue pair #e34948 / #2a78d6
(OKLab Delta E 21.6 protan, 32.3 normal vision); the runs are the
reference palette's categorical slots in order, blue / orange / aqua /
yellow. The first three pass every check all-pairs (worst CVD Delta E 9.2,
normal vision 24.0); with a fourth the bars stay side by side in that order,
where every adjacent pair passes (worst CVD Delta E 9.1, normal vision
22.9). Aqua and yellow are below 3:1 against the surface, so every legend
names each run and the table view exists. Identity is never colour alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RED, BLUE = "#e34948", "#2a78d6"
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c58", "#dededa"
SURFACE = "#fcfcfb"
ENGINES = ("pytorch", "tensorflow", "jax")
#: Categorical slots, in order (the reference palette's first four).
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
ENGINE_COLOURS = dict(zip(ENGINES, SLOTS))
ORIGINAL = "original"
RETRAINED = "original_retrained"   # run_original_retrain.py
#: Panel names as drawn.
DISPLAY = {RETRAINED: "original\n(newly trained)",
           f"{RETRAINED} vs {ORIGINAL}": f"original (newly trained)\nvs {ORIGINAL}"}
RETRAINED_LABEL = "original (newly trained)"


def display(name):
    """A panel's name as drawn: DISPLAY's, else a run name with a
    parenthesized part, e.g. "chrombpnet-refactor (default)", on two lines."""
    return DISPLAY.get(name) or name.replace(" (", "\n(", 1)
RETRAINED_BAR = dict(facecolor=SURFACE, edgecolor=INK, linewidth=0.8)
#: run_deepshap_seeds.py's runs on the model whose scores are stored, against them.
NEW_SEED = "new_seed"
NEW_SEED_LABEL = "original (same model; diff seed)"
NEW_SEED_BAR = dict(color=INK_MUTED, linewidth=0)

SCORE_MEASURES = (
    ("counts Pearson r", RED, ("counts", "pearsonr"), lambda v: v),
    ("1 - profile JSD", BLUE, ("profile", "median_jsd"), lambda v: 1.0 - v),
)
AGREEMENT_MEASURES = (
    ("counts r between models", RED, "counts_pearsonr", lambda v: v),
    ("1 - profile JSD between models", BLUE, "profile_median_jsd", lambda v: 1.0 - v),
)
CONTRIBUTION_PANELS = (("counts", "hypothetical"), ("counts", "projected"),
                       ("profile", "hypothetical"), ("profile", "projected"))
CONTRIBUTION_METRICS = {"pearson": ("pearsonr_median", "Pearson r"),
                        "cosine": ("cosine_median", "cosine similarity")}
#: The original's DeepSHAP on its own model across seeds (run_deepshap_seeds.py).
SEED_LINES = (
    ("itself", "original vs itself (seeds)", dict(color=INK, linewidth=1.6, linestyle="-")),
)
STEP_LABELS = {"bias_pipeline": "bias pipeline", "chrombpnet_pipeline": "chrombpnet pipeline"}

# Datasets left to right, as in ../dnase-cutsite's plots. A biosample sits
# where its first listed ENCID does, so the biosample > ENCID nesting is
# kept; unlisted ENCIDs go last, alphabetically.
ENCID_ORDER = ["ENCSR000EOT", "ENCSR868FGK", "ENCSR163PKT", "ENCSR337UIU"]


def read_runs(directory):
    """The runs' labels in run_summarize.py's order (runs.tsv), or the
    engines for a directory written before it existed."""
    path = Path(directory) / "runs.tsv"
    if not path.exists():
        return ENGINES
    return tuple(pd.read_csv(path, sep="\t").sort_values("order")["engine"].astype(str))


def set_runs(labels):
    """Make `labels` the runs every figure draws, each in its slot's colour."""
    global ENGINES, ENGINE_COLOURS
    if len(labels) > len(SLOTS):
        sys.exit(f"{len(labels)} runs, but only {len(SLOTS)} validated colours: plot fewer at a time")
    ENGINES = tuple(labels)
    ENGINE_COLOURS = dict(zip(ENGINES, SLOTS))


def _encid_rank(encid):
    if encid in ENCID_ORDER:
        return (ENCID_ORDER.index(encid), "")
    return (len(ENCID_ORDER), encid)


# ---------------------------------------------------------------- data

def score_cells(results, source, subset):
    """engine -> {(biosample, encid, assay, fold): {measure: value}}."""
    frame = results[(results["source"] == source) & (results["subset"] == subset)]
    out = {}
    for label, _colour, (group, metric), transform in SCORE_MEASURES:
        rows = frame[(frame["group"] == group) & (frame["metric"] == metric)]
        for r in rows.itertuples():
            key = (r.biosample, r.encid, r.assay, int(r.fold))
            out.setdefault(r.engine, {}).setdefault(key, {})[label] = transform(float(r.value))
    return out


def agreement_cells(agreement):
    """'<engine> vs original' -> {(biosample, encid, assay, fold): {measure: value}}."""
    out = {}
    for r in agreement.itertuples():
        key = (r.biosample, r.encid, r.assay, int(r.fold))
        cell = out.setdefault(f"{r.engine} vs {r.reference}", {}).setdefault(key, {})
        for label, _colour, column, transform in AGREEMENT_MEASURES:
            value = float(getattr(r, column))
            if np.isfinite(value):
                cell[label] = transform(value)
    return out


# ---------------------------------------------------------------- layout

def layout(keys):
    """Bar positions and the spans each grouping level labels.

    Nesting is biosample > encid > fold, with a wider gap at each coarser
    boundary. Computed once over every panel, so a column is the same
    dataset and fold in every panel and a missing run is a visible gap.
    """
    positions, fold_ticks, encid_spans, biosample_spans = {}, [], [], []
    x = 0.0
    biosamples = sorted({k[0] for k in keys},
                        key=lambda b: (min(_encid_rank(k[1]) for k in keys if k[0] == b), b))
    for biosample in biosamples:
        bio_start = x
        for encid, assay in sorted({(k[1], k[2]) for k in keys if k[0] == biosample},
                                   key=lambda ea: _encid_rank(ea[0])):
            enc_start = x
            for fold in sorted({k[3] for k in keys if k[0] == biosample and k[1] == encid}):
                positions[(biosample, encid, fold)] = x
                fold_ticks.append((x, str(fold)))
                x += 1.0
            encid_spans.append((enc_start - 0.5, x - 0.5, f"{encid}\n({assay})"))
            x += 0.45
        biosample_spans.append((bio_start - 0.5, x - 0.45 - 0.5, biosample))
        x += 0.9
    return positions, fold_ticks, encid_spans, biosample_spans


def _below(ax, points):
    """x in data, y at the bottom of the axes moved down by `points`: the
    grouping labels sit a fixed distance under the axis whatever the panel
    height."""
    from matplotlib.transforms import blended_transform_factory, offset_copy
    return offset_copy(blended_transform_factory(ax.transData, ax.transAxes),
                       fig=ax.figure, y=-points, units="points")


def style_axes(ax, label, geometry, ylim, yticks, show_xlabels):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    _pos, fold_ticks, encid_spans, bio_spans = geometry
    ax.set_xticks([p for p, _ in fold_ticks])
    ax.set_xticklabels([t for _, t in fold_ticks], fontsize=7, color=INK_MUTED)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=7, colors=INK_MUTED, length=0)
    ax.set_ylim(*ylim)
    if yticks is None:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    else:
        ax.set_yticks(yticks)
    if fold_ticks:
        ax.set_xlim(min(p for p, _ in fold_ticks) - 0.7, max(p for p, _ in fold_ticks) + 0.7)
    ax.set_ylabel(label, fontsize=9, color=INK, rotation=0, ha="right", va="center", labelpad=14)
    if not show_xlabels:
        return
    for start, end, text in encid_spans:
        ax.text((start + end) / 2, 0, text, transform=_below(ax, 20),
                ha="center", va="top", fontsize=7.5, color=INK)
        ax.plot([start + 0.05, end - 0.05], [0, 0], transform=_below(ax, 16),
                color=GRID, linewidth=1, clip_on=False)
    for start, end, text in bio_spans:
        ax.text((start + end) / 2, 0, text, transform=_below(ax, 50),
                ha="center", va="top", fontsize=8.5, color=INK, fontweight="bold")
        ax.plot([start + 0.05, end - 0.05], [0, 0], transform=_below(ax, 46),
                color=INK_MUTED, linewidth=1, clip_on=False)
    ax.annotate("fold", xy=(0, 0), xycoords="axes fraction", xytext=(-6, -3),
                textcoords="offset points", ha="right", va="top", fontsize=7.5,
                color=INK_MUTED, annotation_clip=False)


def paired_difference(inside, reference, label):
    """Mean of cell - reference over the folds both have (paired, because
    folds differ by more than engines do)."""
    diffs = [value - reference[k][label] for k, value in inside.items() if label in reference.get(k, {})]
    return float(np.mean(diffs)) if diffs else None


def draw_measures(ax, cells, geometry, measures, reference=None, decimals=2):
    """Two bars per fold, a mean rule per dataset and measure, and its label."""
    positions, _ticks, encid_spans, _bio = geometry
    at_x = {k: positions[(k[0], k[1], k[3])] for k in cells if (k[0], k[1], k[3]) in positions}
    for (label, colour, _src, _fn), dx in zip(measures, (-0.17, 0.17)):
        pts = [(at_x[k] + dx, got[label]) for k, got in sorted(cells.items()) if label in got and k in at_x]
        if pts:
            ax.bar([p[0] for p in pts], [p[1] for p in pts], width=0.30, color=colour,
                   linewidth=0, zorder=3)
    for start, end, _text in encid_spans:
        for index, (label, colour, _src, _fn) in enumerate(measures):
            inside = {k: got[label] for k, got in cells.items()
                      if label in got and k in at_x and start < at_x[k] < end}
            if not inside:
                continue
            mean = float(np.mean(list(inside.values())))
            ax.hlines(mean, start + 0.1, end - 0.1, color=colour, linewidth=2.0, zorder=4)
            text = f"{mean:.{decimals}f}"
            if reference is not None:
                diff = paired_difference(inside, reference, label)
                if diff is not None and round(diff, decimals) != 0:
                    text += f" ({diff:+.{decimals}f})"
            at, align = (start + 0.15, "left") if index == 0 else (end - 0.15, "right")
            ax.text(at, mean + 0.025, text, color=INK, fontsize=7, va="bottom", ha=align, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.75))


def finish(plt, fig, out, title, subtitle, footnote, handles, names):
    """Title block, legend and footnote at fixed distances (inches) from the
    figure edges, so they keep their spacing at any number of panels."""
    height = fig.get_figheight()

    def up(inches):
        return 1.0 - inches / height

    fig.suptitle(title, fontsize=11, color=INK, x=0.012, ha="left", va="top", y=up(0.10))
    fig.text(0.012, up(0.36), subtitle, fontsize=8, color=INK_MUTED, ha="left", va="top")
    def legend(columns):
        return fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.012, up(0.52)), frameon=False,
                          fontsize=8, ncol=max(columns, 1), labelcolor=INK, handlelength=1.1,
                          handleheight=1.1, columnspacing=1.4)

    # One row, unless it is wider than the figure: then two, and the panels
    # start lower by a row.
    drawn, rows = legend(len(names)), 1
    if names and drawn.get_window_extent(fig.canvas.get_renderer()).width > 0.97 * fig.bbox.width:
        drawn.remove()
        drawn, rows = legend(-(-len(names) // 2)), 2
    top = up(0.85 + 0.2 * (rows - 1))
    if footnote:
        fig.text(0.012, 0.08 / height, footnote, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
    # Below the last axis: fold ticks and the two grouping rows (~0.9 in),
    # then the footnote when there is one, 0.16 in a line.
    bottom = (1.05 + (0.16 * (footnote.count("\n") + 1) if footnote else 0.0)) / height
    fig.tight_layout(rect=(0.0, bottom, 1.0, top))
    fig.subplots_adjust(bottom=bottom, top=top)
    for suffix in (".png", ".pdf"):
        fig.savefig(out + suffix, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}.png, {out}.pdf")


def figure(plt, n_panels, n_columns):
    width = max(7.5, 0.60 * n_columns + 3.2)
    fig, axes = plt.subplots(n_panels, 1, squeeze=False, figsize=(width, 1.5 * n_panels + 2.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    return fig, [a[0] for a in axes]


def write_table(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figures

def plot_scores(plt, results, args):
    cells = score_cells(results, "own", args.subset)
    panels = [e for e in (ORIGINAL, RETRAINED) + ENGINES if e in cells]
    if not args.with_original and ORIGINAL in panels:
        panels.remove(ORIGINAL)
    if not panels:
        print("skipped summary_scores: no own scores yet", file=sys.stderr)
        return
    reference_run = args.reference or ENGINES[0]
    reference = cells.get(reference_run)
    geometry = layout({k for p in panels for k in cells[p]})
    fig, axes = figure(plt, len(panels), len(geometry[1]))
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        style_axes(ax, display(panel), geometry, (0, 1), np.arange(0, 1.01, 0.25),
                   index == len(panels) - 1)
        draw_measures(ax, cells[panel], geometry, SCORE_MEASURES,
                      reference=reference if panel != reference_run else None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _l, c, _s, _f in SCORE_MEASURES]
    footnote = (f"(+0.02) after a mean = that panel minus {reference_run}, averaged over the folds both "
                f"have; + means it scores higher; not shown when it rounds to 0.00.")
    if ORIGINAL in panels:
        footnote += (f"\n{ORIGINAL}: the old TensorFlow ChromBPNet from the atlas, scored by its own "
                     f"pipeline on its own peak set, so not on exactly the engines' test regions.")
    if RETRAINED in panels:
        footnote += (f"\n{RETRAINED_LABEL}: the original chrombpnet retrained on the engines' inputs "
                     f"(run_original_retrain.py), on the engines' test regions.")
    finish(plt, fig, args.out + "summary_scores", f"ChromBPNet test scores by {args.noun}",
           f"each {args.noun}'s model on its own test set  |  {args.subset} subset  |  higher is better for "
           f"both measures  |  horizontal rule = mean over that dataset's folds",
           footnote, handles, [l for l, *_ in SCORE_MEASURES])
    rows = [(p, *k, label, round(v, 6)) for p in panels for k, got in sorted(cells[p].items())
            for label, v in got.items()]
    write_table(args.out + "summary_scores.tsv", rows,
                ["engine", "biosample", "encid", "assay", "fold", "measure", "value"])


def plot_agreement(plt, agreement, args):
    if "engine" not in agreement.columns:
        print("skipped summary_agreement: agreement.tsv has no engine column "
              "(an older format); run run_summarize.py again", file=sys.stderr)
        return
    cells = agreement_cells(agreement)
    panels = [f"{e} vs {ORIGINAL}" for e in ENGINES + (RETRAINED,) if f"{e} vs {ORIGINAL}" in cells]
    # run_summarize.py --versus pairs, by the order of their runs.
    rank = {e: i for i, e in enumerate(ENGINES)}
    pairs = sorted((p for p in cells if p not in panels and not p.endswith(f" vs {ORIGINAL}")),
                   key=lambda p: tuple(rank.get(e, len(rank)) for e in p.split(" vs ", 1)))
    panels += pairs
    if not panels:
        print("skipped summary_agreement: no engine has predictions to compare yet", file=sys.stderr)
        return
    geometry = layout({k for p in panels for k in cells[p]})
    fig, axes = figure(plt, len(panels), len(geometry[1]))
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        style_axes(ax, display(panel), geometry, (0, 1), np.arange(0, 1.01, 0.25),
                   index == len(panels) - 1)
        draw_measures(ax, cells[panel], geometry, AGREEMENT_MEASURES, decimals=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _l, c, _s, _f in AGREEMENT_MEASURES]
    finish(plt, fig, args.out + "summary_agreement",
           f"Agreement of each {args.noun}'s ChromBPNet model with the {ORIGINAL} TensorFlow model"
           + (" and with one another" if pairs else ""),
           "the two models' predictions on the test regions both scored, compared with each other rather "
           "than with the data  |  1 = identical  |  horizontal rule = mean over that dataset's folds",
           "Counts: Pearson r between the two models' predicted log counts. Profile: median "
           "Jensen-Shannon distance between their predicted profiles (scipy's, as in median_jsd).\n"
           f"{ORIGINAL}: the atlas's TensorFlow ChromBPNet, a separately trained model (DNASE: the "
           "original 0/+1 cut convention, now --dnase-convention right-deprecated; the engines: the default, both-half)."
           + (f"\n{RETRAINED_LABEL}: the original chrombpnet retrained on the engines' inputs -- how closely "
              f"two models of the same code agree." if f"{RETRAINED} vs {ORIGINAL}" in panels else ""),
           handles, [l for l, *_ in AGREEMENT_MEASURES])
    rows = [(p, *k, label, round(v, 6)) for p in panels for k, got in sorted(cells[p].items())
            for label, v in got.items()]
    write_table(args.out + "summary_agreement.tsv", rows,
                ["pair", "biosample", "encid", "assay", "fold", "measure", "value"])


def seed_footnote(lines, retrained=False, new_seed=False):
    """The footnote's last lines: what the original's bars and lines drawn are."""
    parts = []
    if retrained:
        parts.append(f"hollow bar: {RETRAINED_LABEL}, the original chrombpnet retrained on the engines' inputs "
                     "(its code, its own bias model), its DeepLIFT/DeepSHAP vs the stored scores -- the same code as the "
                     "stored scores, another training run.")
    if new_seed:
        parts.append(f"grey bar: {NEW_SEED_LABEL}, the original's DeepLIFT/DeepSHAP rerun on the model whose scores are "
                     "stored, with a new seed, vs those scores -- the same model and code, only the shuffled "
                     "references differ: the ceiling.")
    if "itself" in lines:
        parts.append("solid line: the original's DeepLIFT/DeepSHAP rerun on that model, seed vs seed -- only the shuffled "
                     "references differ.")
    if not parts:
        return f"{ORIGINAL}: a separately trained model, so 1 is not expected; compare the engines with one another."
    return "\n".join(["coloured bars: each engine's own model vs the stored scores."] + parts)


def plot_contributions(plt, contributions, args, metric):
    column, metric_label = CONTRIBUTION_METRICS[metric]
    out = args.out + f"summary_contributions_{metric}"
    if contributions is None or contributions.empty:
        print(f"skipped {out}: no contribution comparisons yet", file=sys.stderr)
        return
    frame = contributions[np.isfinite(contributions[column].astype(float))]
    # Runs against another run (run_summarize.py --versus) are not bars here.
    frame = frame[~(frame["engine"].isin(ENGINES) & (frame["reference"].astype(str) != ORIGINAL))]
    # The original's seeded runs: "itself" pairs two seeds, "stored" pairs a
    # seed with the stored scores. One value per fold: pairs, seeds averaged.
    seeded = frame[frame["engine"].astype(str).str.startswith("original:")]
    seeded = seeded.assign(line=np.where(seeded["reference"].astype(str).str.startswith("original:"),
                                         "itself", "stored"))
    seed_cells = (seeded.groupby(["biosample", "encid", "assay", "fold", "head", "kind", "line"])[column]
                  .apply(lambda v: float(np.mean(v.astype(float)))).reset_index())
    # The retrained original against the stored scores, seeds averaged.
    retrained = frame[frame["engine"].astype(str).str.startswith(RETRAINED + ":")
                      & (frame["reference"].astype(str) == ORIGINAL)]
    retrained_cells = (retrained.groupby(["biosample", "encid", "assay", "fold", "head", "kind"])[column]
                       .apply(lambda v: float(np.mean(v.astype(float)))).reset_index())
    panels = [(h, k) for h, k in CONTRIBUTION_PANELS
              if ((frame["head"] == h) & (frame["kind"] == k)).any()]
    if not panels:
        print(f"skipped {out}: no finite comparisons yet", file=sys.stderr)
        return
    keys = {(r.biosample, r.encid, r.assay, int(r.fold)) for r in frame.itertuples()}
    geometry = layout(keys)
    positions = geometry[0]
    low = min(0.0, float(frame[column].min()) - 0.05)
    fig, axes = figure(plt, len(panels), len(geometry[1]))
    new_seed_cells = seed_cells[seed_cells["line"] == "stored"]
    bars = (([RETRAINED] if len(retrained_cells) else []) + ([NEW_SEED] if len(new_seed_cells) else [])
            + list(ENGINES))
    step = {1: 0.5, 2: 0.36, 3: 0.28, 4: 0.22, 5: 0.18}.get(len(bars), 0.9 / len(bars))
    width = step - 0.02
    offsets = {bar: (i - (len(bars) - 1) / 2) * step for i, bar in enumerate(bars)}
    for index, (ax, (head, kind)) in enumerate(zip(axes, panels)):
        style_axes(ax, f"{head}\n{kind}", geometry, (low, 1),
                   np.arange(0, 1.01, 0.25) if low == 0 else None, index == len(panels) - 1)
        sub = frame[(frame["head"] == head) & (frame["kind"] == kind)]
        for engine in ENGINES:
            rows = sub[sub["engine"] == engine]
            xs = [positions[(r.biosample, r.encid, int(r.fold))] + offsets[engine] for r in rows.itertuples()]
            # No mean rules: three engines' rules, a few hundredths apart,
            # would sit on top of one another. The legend gives each mean.
            ax.bar(xs, rows[column].astype(float), width=width, color=ENGINE_COLOURS[engine],
                   linewidth=0, zorder=3)
        for bar, bar_cells, style in ((RETRAINED, retrained_cells, RETRAINED_BAR),
                                      (NEW_SEED, new_seed_cells, NEW_SEED_BAR)):
            if bar not in offsets:
                continue
            cells = bar_cells[(bar_cells["head"] == head) & (bar_cells["kind"] == kind)]
            xs = [positions[(r.biosample, r.encid, int(r.fold))] + offsets[bar] for r in cells.itertuples()]
            ax.bar(xs, cells[column].astype(float), width=width, zorder=3, **style)
        for line, _label, style in reversed(SEED_LINES):
            cells = seed_cells[(seed_cells["head"] == head) & (seed_cells["kind"] == kind)
                               & (seed_cells["line"] == line)]
            for r in cells.itertuples():
                x = positions[(r.biosample, r.encid, int(r.fold))]
                ax.hlines(float(getattr(r, column)), x - 0.45, x + 0.45, zorder=5, **style)
    means = frame.groupby("engine")[column].apply(lambda v: float(np.mean(v.astype(float))))
    line_means = seed_cells.groupby("line")[column].apply(lambda v: float(np.mean(v.astype(float))))
    handles = [plt.Line2D([0], [0], **style) for line, _l, style in SEED_LINES if line in line_means]
    names = [f"{label} (mean {line_means[line]:.3f})" for line, label, _s in SEED_LINES if line in line_means]
    for bar_cells, style, label in ((retrained_cells, RETRAINED_BAR, RETRAINED_LABEL),
                                    (new_seed_cells, NEW_SEED_BAR, NEW_SEED_LABEL)):
        if len(bar_cells):
            handles.append(plt.Rectangle((0, 0), 1, 1, **style))
            names.append(f"{label} (mean {float(np.mean(bar_cells[column])):.3f})")
    handles += [plt.Rectangle((0, 0), 1, 1, color=ENGINE_COLOURS[e]) for e in ENGINES if e in means]
    names += [f"{e} (mean {means[e]:.3f})" for e in ENGINES if e in means]
    profile_source = ""
    if "profile_scores" in frame.columns:
        engines_profile = (frame["head"] == "profile") & frame["engine"].isin(ENGINES)
        # run_summarize.py's labels; "midpoint" is what it wrote before --profile-stopgrad.
        labels = {"stopgrad-off": "stop-gradient off", "midpoint": "stop-gradient off",
                  "stopgrad-on": "stop-gradient on"}
        tags = sorted({labels.get(t, t) for t in frame.loc[engines_profile, "profile_scores"].astype(str)}
                      - {"pipeline", "nan"})
        if tags:
            profile_source = f"  |  profile scores: {', '.join(tags)}"
    finish(plt, fig, out,
           f"DeepLIFT/DeepSHAP contribution scores: each {args.noun} vs the {ORIGINAL} TensorFlow model",
           f"30K peak subsample, regions matched to the {ORIGINAL}'s genome-wide scores  |  bar = median "
           f"over regions of the per-region {metric_label}  |  legend: mean over datasets and folds"
           f"{profile_source}",
           "hypothetical: the 4 x 2114 hypothetical scores, flattened. projected: the 2114 per-base scores of "
           "the observed base. Each over the whole input window.\n"
           + seed_footnote(set(seed_cells["line"]), retrained=bool(len(retrained_cells)),
                           new_seed=bool(len(new_seed_cells))),
           handles, names)
    write_table(out + ".tsv",
                frame[["engine", "reference", "biosample", "encid", "assay", "fold", "head", "kind", "n_matched",
                       "pearsonr_median", "pearsonr_mean", "cosine_median", "cosine_mean"]].values.tolist(),
                ["engine", "reference", "biosample", "encid", "assay", "fold", "head", "kind", "n_matched",
                 "pearsonr_median", "pearsonr_mean", "cosine_median", "cosine_mean"])


def short_gpu(name):
    for prefix in ("NVIDIA ", "Tesla "):
        name = name.replace(prefix, "")
    return name.replace("-SXM4", "").replace("-PCIE", "").replace(" ", "")


def plot_runtime(plt, timings, args):
    if timings.empty:
        print("skipped summary_runtime: no timings yet", file=sys.stderr)
        return
    timings = timings.assign(hours=timings["seconds"] / 3600.0, gpu=timings["gpu"].fillna("?").map(short_gpu))
    found = list(dict.fromkeys(timings["step"]))
    steps = [s for s in STEP_LABELS if s in found] + [s for s in found if s not in STEP_LABELS]
    keys = {(r.biosample, r.encid, r.assay, int(r.fold)) for r in timings.itertuples()}
    geometry = layout(keys)
    positions = geometry[0]
    fig, axes = figure(plt, len(steps), len(geometry[1]))
    several_gpus = timings["gpu"].nunique() > 1
    step_x = {1: 0.0, 2: 0.36, 3: 0.28}.get(len(ENGINES), 0.84 / len(ENGINES))
    offsets = {e: (i - (len(ENGINES) - 1) / 2) * step_x for i, e in enumerate(ENGINES)}
    for index, (ax, step) in enumerate(zip(axes, steps)):
        frame = timings[timings["step"] == step]
        top = max(1.0, float(frame["hours"].max()) * (1.35 if several_gpus else 1.1))
        style_axes(ax, STEP_LABELS.get(step, step.replace("_", " ")) + "\nhours", geometry, (0, top),
                   None, index == len(steps) - 1)
        for engine in ENGINES:
            sub = frame[frame["engine"] == engine]
            xs = [positions[(r.biosample, r.encid, int(r.fold))] + offsets[engine] for r in sub.itertuples()]
            ax.bar(xs, sub["hours"], width=max(step_x - 0.02, 0.2), color=ENGINE_COLOURS[engine],
                   linewidth=0, zorder=3)
            if several_gpus:
                for x, r in zip(xs, sub.itertuples()):
                    ax.text(x, r.hours + top * 0.015, r.gpu, rotation=90, fontsize=5, color=INK_MUTED,
                            ha="center", va="bottom", zorder=5)
    medians = timings.groupby(["engine", "encid", "fold"])["hours"].sum().groupby("engine").median()
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENGINE_COLOURS[e]) for e in ENGINES if e in medians]
    names = [f"{e} (median {medians[e]:.1f} h per run)" for e in ENGINES if e in medians]
    gpus = sorted(timings["gpu"].unique())
    subtitle = ("wall time of each step  |  " +
                ("GPU written above each bar: runtimes compare only on the same GPU type"
                 if several_gpus else f"all runs on {gpus[0]}"))
    finish(plt, fig, args.out + "summary_runtime", f"Runtime by {args.noun}", subtitle, None,
           handles, names)
    write_table(args.out + "summary_runtime.tsv",
                timings[["engine", "biosample", "encid", "assay", "fold", "step", "hours", "gpu"]]
                .round({"hours": 4}).values.tolist(),
                ["engine", "biosample", "encid", "assay", "fold", "step", "hours", "gpu"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="+", required=True,
                    help="run_summarize.py output directories (its --out-dir), each plotted in place")
    ap.add_argument("--out", default=None, help="output prefix (default: <--dir>/; only with one --dir)")
    ap.add_argument("--subset", default="peaks")
    ap.add_argument("--reference", default=None,
                    help="the run every other score panel is compared against (default: the first run; "
                         f"or {ORIGINAL} / {RETRAINED})")
    ap.add_argument("--noun", default="engine",
                    help='what the titles call a run (default "engine")')
    ap.add_argument("--no-original", dest="with_original", action="store_false",
                    help="leave the old TensorFlow ChromBPNet out of summary_scores")
    args = ap.parse_args(argv)
    directories = [Path(d) for d in args.dir]
    if args.out is not None and len(directories) > 1:
        sys.exit("--out needs a single --dir")
    out = args.out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for directory in directories:
        print(f"== {directory}", file=sys.stderr)
        args.out = out if out is not None else str(directory) + "/"
        plot_directory(plt, directory, args)
    return 0


def plot_directory(plt, directory, args):
    """Every figure of one run_summarize.py output directory."""
    set_runs(read_runs(directory))

    def read(name, required=True):
        path = directory / name
        if not path.exists():
            if required:
                sys.exit(f"{path} not found: run run_summarize.py first")
            print(f"{path} not found (run_summarize.py --no-contributions?)", file=sys.stderr)
            return None
        return pd.read_csv(path, sep="\t", dtype={"fold": int})

    plot_scores(plt, read("results.tsv"), args)
    plot_agreement(plt, read("agreement.tsv"), args)
    contributions = read("contributions.tsv", required=False)
    for metric in CONTRIBUTION_METRICS:
        plot_contributions(plt, contributions, args, metric)
    plot_runtime(plt, read("timings.tsv"), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
