#!/usr/bin/env python
"""Plot the scores in results.tsv as grouped bars, one panel per --data-aug arm.

    python plot_summarize.py --results <results.tsv> --out <prefix> \
        [--fill-original ENCID:FOLD ...] [--arm ARM ...] [--source own|headtohead]

--results is summarize.py's output; each figure is written as <prefix>[suffix]
.png, .pdf and .tsv.

Seven figures, all with the same x layout:

    summary_scores                         each model on its own test set
    summary_scores_aug_on_standard_tests   each arm's model on the standard
                                           models' test sets
        ..._original, ..._refactor         its left (standard_tensorflow's test
                                           sets) and right (standard's) halves
                                           as figures of their own
    summary_scores_standard_on_aug_tests   the standard models on each arm's
                                           test set
        ..._original, ..._refactor         its left (the standard_tensorflow
                                           model) and right (standard) halves
                                           as figures of their own

In every figure the two standard models' rows come first, in the reference
greys, and a rule separates them from the arms' rows.

    x   biosample  >  ENCID (assay)  >  fold  >  {counts r, 1 - profile JSD}
    y   0 to 1, higher is better for both measures
        red  counts Pearson r
        blue 1 - median profile JSD
        horizontal line per measure = mean across that dataset's folds

On the figures standard_tensorflow is labelled "original" and standard
"original-refactor" (ARM_LABELS); the data keep the arm names.

In summary_scores the top panel, standard_tensorflow, is the old TensorFlow
ChromBPNet scored by its own pipeline. While it is plotted, every other
panel's means also carry "(+0.02)": that arm minus standard_tensorflow,
averaged over the folds both have. Higher is better for both measures, so +
always means the arm beats the old model. A difference that rounds to 0.00
is not printed.

The two cross-test figures (from the cross-evaluation step) are twice as wide: the left
half belongs to standard_tensorflow, the right half to standard. In
_aug_on_standard_tests a half is that model's TEST SET, and each arm row is
the arm's model scored on it; in _standard_on_aug_tests a half is that
MODEL, and each arm row is it scored on the arm's test set. The grey rows
are standard_tensorflow and standard on their own test sets -- the same
numbers as in summary_scores -- each in its own half, and every arm row's
"(+0.02)" is measured against the grey row of the same half.

Both measures share one axis on purpose. Profile JSD is a divergence, so
lower is better; plotting 1 - JSD flips it to "higher is better" and puts it
on the same 0-1 scale as a correlation. That is what makes a single y-axis
honest here -- two measures of different scale would need two charts, never
two y-axes.

Colours are the validated red/blue pair from the project palette
(#e34948 / #2a78d6), not pure red/blue: OKLab Delta E 21.6 under protanopia
and 32.3 under normal vision on a light surface, both clear of the >= 8 and
>= 15 floors. The reference greys are #5c5c58 and #a3a29c (6.5:1 and 2.5:1
on the surface, 2.6:1 apart), and position within the pair encodes the
measure as well. Identity is never colour alone -- there is always a legend.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

# Validated pair; see module docstring. Text never wears these.
RED, BLUE = "#e34948", "#2a78d6"
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c58", "#dededa"
SURFACE = "#fcfcfb"

MEASURES = (
    ("counts Pearson r", RED, ("counts", "pearsonr"), lambda v: v),
    ("1 - profile JSD", BLUE, ("profile", "median_jsd"), lambda v: 1.0 - v),
)
# Reference rows, in MEASURES order.
GREYS = (INK_MUTED, "#a3a29c")

# The old TensorFlow ChromBPNet, collated by summarize.py from the ids
# table's TENSORFLOW_DIR column. It exists only under source "own".
TF_ARM = "standard_tensorflow"
STANDARD = "standard"
REFERENCES = (TF_ARM, STANDARD)        # left half, right half

ARM_ORDER = [TF_ARM, STANDARD, "fr-only", "rc-only", "right", "left",
             "both-half", "both-full"]

# How the two reference models are named on the figures. results.tsv, the
# .tsv tables and --arm keep the names above.
ARM_LABELS = {TF_ARM: "original", STANDARD: "original-refactor"}

# Arms defined only for DNase: in summary_scores their ATAC columns are
# shaded rather than left as an unexplained gap.
DNASE_ONLY_ARMS = ("right", "left", "both-half", "both-full")
NOT_APPLICABLE = "#efeeea"             # one step off the surface, under the grid

# (encid, fold) with no original-model run, from --fill-original. (In the
# published figures, ENCSR868FGK fold 2's original run directory is empty, so
# it has neither a model nor a test set.) Every set of bars that misses such a
# fold because of it (the original model's own row, and in the cross-test
# figures each arm row's original-model half) is drawn there at the mean of
# its other folds of that dataset. Drawing only: those mean rules are
# unchanged by construction, the "(+0.02)" differences still pair measured
# folds, and the .tsv tables keep only measured values.
TF_GAPS = ()


def arm_label(arm):
    return ARM_LABELS.get(arm, arm)


def biosample_label(name):
    """Donor codes in parentheses: 'stomach M54' -> 'stomach (M54)'."""
    return re.sub(r"^(.+) ([MF]\d+)$", r"\1 (\2)", name)


def fill_with_fold_mean(cells, encid, fold):
    """`cells` plus `fold` of `encid` at the mean of its other folds.

    Returns (cells, the added key or None). Nothing is added when the fold
    is already there or the dataset has no folds to average.
    """
    same = {k: got for k, got in cells.items() if k[1] == encid}
    if not same or any(k[3] == fold for k in same):
        return cells, None
    biosample, _encid, assay, _fold = next(iter(same))
    key = (biosample, encid, assay, fold)
    filled = dict(cells)
    filled[key] = {label: float(np.mean([got[label] for got in same.values() if label in got]))
                   for label, *_ in MEASURES}
    return filled, key


def fill_tf_gaps(cells):
    """`cells` with every TF_GAPS fold filled; returns (cells, added keys)."""
    added = []
    for encid, fold in TF_GAPS:
        cells, key = fill_with_fold_mean(cells, encid, fold)
        if key is not None:
            added.append(key)
    return cells, added

HALF_GAP = 1.6                         # x units between the two halves

# Datasets left to right (--encid-order; the default is the published
# figures' four). A biosample sits where its first listed ENCID does, so the
# biosample > ENCID nesting is kept; unlisted ENCIDs go last, alphabetically.
ENCID_ORDER = ["ENCSR000EOT", "ENCSR868FGK", "ENCSR163PKT", "ENCSR337UIU"]


def _encid_rank(encid):
    if encid in ENCID_ORDER:
        return (ENCID_ORDER.index(encid), "")
    return (len(ENCID_ORDER), encid)


def load(results, subset):
    frame = pd.read_csv(results, sep="\t", dtype=str)
    frame["value"] = frame["value"].astype(float)
    frame["fold"] = frame["fold"].astype(int)
    if "test" not in frame:
        # results.tsv from before summarize.py recorded the test set: every
        # row then was scored on its arm's own test data.
        frame["test"] = frame["arm"]
    frame = frame[frame["subset"] == subset]
    if frame.empty:
        sys.exit(f"no rows with subset={subset!r} in {results}")
    return frame


def measure_values(frame):
    """(source, arm, model, test, biosample, encid, assay, fold) -> {label: value}."""
    out = {}
    for label, _colour, (group, metric), transform in MEASURES:
        rows = frame[(frame["group"] == group) & (frame["metric"] == metric)]
        for r in rows.itertuples():
            key = (r.source, r.arm, r.model, r.test,
                   r.biosample, r.encid, r.assay, r.fold)
            out.setdefault(key, {})[label] = transform(r.value)
    return out


def select(values, source, arm, model, test):
    """One (source, arm, model, test): (biosample, encid, assay, fold) -> {label: value}."""
    want = (source, arm, model, test)
    return {k[4:]: got for k, got in values.items() if k[:4] == want}


def layout(keys, x0=0.0):
    """Bar positions, plus the spans each grouping level needs to label.

    Nesting is biosample > encid > fold, so the x position is built by
    walking those in order and inserting a wider gap at each coarser
    boundary. Groups are drawn adjacent, never interleaved. `x0` shifts the
    whole layout, which is how the second half of a cross-test figure is
    placed beside the first.
    """
    positions, fold_ticks, encid_spans, biosample_spans = {}, [], [], []
    x = x0
    biosamples = sorted({k[0] for k in keys},
                        key=lambda b: (min(_encid_rank(k[1]) for k in keys if k[0] == b), b))
    for biosample in biosamples:
        bio_start = x
        encids = sorted({(k[1], k[2]) for k in keys if k[0] == biosample},
                        key=lambda ea: _encid_rank(ea[0]))
        for encid, assay in encids:
            enc_start = x
            # set(): the same fold appears once per arm, and the geometry
            # is shared across arms, so one column per fold.
            folds = sorted({k[3] for k in keys
                            if k[0] == biosample and k[1] == encid})
            for fold in folds:
                positions[(biosample, encid, fold)] = x
                fold_ticks.append((x, str(fold)))
                x += 1.0
            encid_spans.append((enc_start - 0.5, x - 0.5, f"{encid}\n({assay})"))
            x += 0.45                      # gap between datasets
        biosample_spans.append((bio_start - 0.5, x - 0.45 - 0.5, biosample))
        x += 0.9                           # wider gap between biosamples
    return positions, fold_ticks, encid_spans, biosample_spans


def paired_difference(inside, reference, label):
    """Mean of cell - reference over the folds both have.

    Paired rather than a difference of the two rows' means, because the
    folds differ by more than the arms do: a fold present in one and missing
    from the other would move a difference of means on its own.
    """
    diffs = [value - reference[k][label] for k, value in inside.items()
             if label in reference.get(k, {})]
    return float(np.mean(diffs)) if diffs else None


def draw_half(ax, cells, geometry, reference=None, grey=False):
    """One row's bars, mean rules and labels over one geometry.

    `cells` maps (biosample, encid, assay, fold) to {label: value};
    `reference`, when given, is the cells each mean is compared against.
    """
    positions, _ticks, encid_spans, _bio = geometry
    at_x = {k: positions[(k[0], k[1], k[3])] for k in cells
            if (k[0], k[1], k[3]) in positions}

    # Two bars per fold, with a visible gap between them and at group edges.
    width = 0.30
    offsets = (-0.17, 0.17)          # leaves a clear gap between the pair
    for index, ((label, colour, _src, _fn), dx) in enumerate(zip(MEASURES, offsets)):
        xs, ys = [], []
        for k, got in sorted(cells.items()):
            if label in got and k in at_x:
                xs.append(at_x[k] + dx)
                ys.append(got[label])
        ax.bar(xs, ys, width=width, color=GREYS[index] if grey else colour,
               linewidth=0, zorder=3)

    # Mean across folds, per dataset, per measure: a short horizontal rule
    # spanning only the dataset it summarizes, so it cannot be misread as a
    # global threshold.
    for start, end, _text in encid_spans:
        for index, (label, colour, _src, _fn) in enumerate(MEASURES):
            inside = {k: got[label] for k, got in cells.items()
                      if label in got and k in at_x and start < at_x[k] < end}
            if not inside:
                continue
            mean = float(np.mean(list(inside.values())))
            ax.hlines(mean, start + 0.1, end - 0.1,
                      color=GREYS[index] if grey else colour,
                      linewidth=2.0, zorder=4)
            text = f"{mean:.2f}"
            if reference is not None:
                diff = paired_difference(inside, reference, label)
                # A difference that rounds to zero is left unsaid rather
                # than printed as a signed "(-0.00)".
                if diff is not None and round(diff, 2) != 0:
                    text += f" ({diff:+.2f})"
            # One label at each end, so the two rules can sit at almost the
            # same height without their numbers overlapping. The surface
            # backing matters for the lower rule's label, which sits where
            # the taller measure's bars rise through it; translucent, so the
            # bar behind still reads as one bar rather than two.
            at, align = ((start + 0.15, "left") if index == 0
                         else (end - 0.15, "right"))
            ax.text(at, mean + 0.025, text, color=INK_MUTED if grey else colour,
                    fontsize=7, va="bottom", ha=align, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE,
                              edgecolor="none", alpha=0.75))


def frame_panel(ax, label, geometries, show_xlabels, half_labels=None):
    """Axes styling, ticks, the row label and (bottom panel only) the
    grouping labels, for one or more side-by-side geometries.

    The layout is computed once over ALL rows, not per panel: rows cover
    different datasets and folds, so a per-panel layout would put different
    things in the same column and leave the single label row at the bottom
    describing only the last panel. With one geometry a column means the
    same dataset and fold in every row, and a missing run is a visible gap.
    """
    # Recessive grid, behind the marks.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    fold_ticks = [t for g in geometries for t in g[1]]
    ax.set_xticks([p for p, _ in fold_ticks])
    ax.set_xticklabels([t for _, t in fold_ticks], fontsize=7, color=INK_MUTED)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=7, colors=INK_MUTED, length=0)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    if fold_ticks:
        ax.set_xlim(min(p for p, _ in fold_ticks) - 0.7,
                    max(p for p, _ in fold_ticks) + 0.7)
    ax.set_ylabel(label, fontsize=9, color=INK, rotation=0,
                  ha="right", va="center", labelpad=14)

    # Halves are told apart by a hairline in the gap between them.
    for left, right in zip(geometries, geometries[1:]):
        divide = (max(p for p, _ in left[1]) + min(p for p, _ in right[1])) / 2
        ax.axvline(divide, color=GRID, linewidth=1.0, zorder=1)

    if not show_xlabels:
        return
    # The coarser grouping levels, written below the axis rather than folded
    # into the tick labels, so the nesting is visible as nesting.
    for geometry in geometries:
        _pos, _ticks, encid_spans, bio_spans = geometry
        for start, end, text in encid_spans:
            ax.text((start + end) / 2, -0.30, text, transform=_blend(ax),
                    ha="center", va="top", fontsize=7.5, color=INK)
            ax.plot([start + 0.05, end - 0.05], [-0.22, -0.22],
                    transform=_blend(ax), color=GRID, linewidth=1,
                    clip_on=False)
        for start, end, text in bio_spans:
            ax.text((start + end) / 2, -0.66, biosample_label(text), transform=_blend(ax),
                    ha="center", va="top", fontsize=8.5, color=INK,
                    fontweight="bold")
            ax.plot([start + 0.05, end - 0.05], [-0.60, -0.60],
                    transform=_blend(ax), color=INK_MUTED, linewidth=1,
                    clip_on=False)
    # The outermost level, when there are halves: what each half holds.
    for geometry, text in zip(geometries, half_labels or ()):
        spans = geometry[3]
        start, end = spans[0][0], spans[-1][1]
        ax.text((start + end) / 2, -0.98, text, transform=_blend(ax),
                ha="center", va="top", fontsize=9.5, color=INK,
                fontweight="bold")
        ax.plot([start + 0.05, end - 0.05], [-0.92, -0.92],
                transform=_blend(ax), color=INK, linewidth=1.2,
                clip_on=False)
    # "fold" names the tick row, so it sits beside it rather than under
    # the grouping rows.
    ax.annotate("fold", xy=(0, -0.055), xycoords="axes fraction",
                xytext=(-6, 0), textcoords="offset points",
                ha="right", va="top", fontsize=7.5, color=INK_MUTED,
                annotation_clip=False)


def _blend(ax):
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transData, ax.transAxes)


def render(plt, out, panels, geometries, title, subtitle, footnote,
           half_labels=None, grey_legend=None, table_skip=()):
    """Draw one figure and write its .png, .pdf and .tsv.

    `panels` is a list of (arm, [(cells, reference, grey) or None per half],
    [(model, test) per half]), optionally followed by the assays whose
    columns that row shades as not applicable. Rows are labelled with
    arm_label(arm). (arm, model, test, cell key) entries in `table_skip` are
    drawn but left out of the .tsv. `grey_legend`, when given, adds the two
    reference greys to the legend as "<measure>, <grey_legend>". Where the
    reference rows end and the arm rows begin, a rule runs across the figure.
    """
    columns = len(geometries[0][1])
    width = max(7.5, 0.60 * columns + 3.2) * len(geometries)
    fig, axes = plt.subplots(len(panels), 1, squeeze=False,
                             figsize=(width, 1.5 * len(panels) + 2.6),
                             sharex=True)
    axes = [a[0] for a in axes]
    fig.patch.set_facecolor(SURFACE)

    for index, (ax, panel) in enumerate(zip(axes, panels)):
        label, halves, _who = panel[:3]
        shaded = panel[3] if len(panel) > 3 else ()
        last = index == len(panels) - 1
        frame_panel(ax, arm_label(label), geometries, show_xlabels=last,
                    half_labels=half_labels if last else None)
        for geometry in geometries:
            for start, end, text in geometry[2]:
                if any(text.endswith(f"({assay})") for assay in shaded):
                    ax.axvspan(start, end, color=NOT_APPLICABLE, linewidth=0,
                               zorder=0.4)
        for geometry, half in zip(geometries, halves):
            if half is not None:
                cells, reference, grey = half
                draw_half(ax, cells, geometry, reference=reference, grey=grey)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour)
               for _l, colour, _s, _f in MEASURES]
    names = [label for label, *_ in MEASURES]
    if grey_legend:
        handles += [plt.Rectangle((0, 0), 1, 1, color=g) for g in GREYS]
        names += [f"{label}, {grey_legend}" for label, *_ in MEASURES]
    fig.legend(handles, names,
               loc="upper left", bbox_to_anchor=(0.012, 0.958),
               frameon=False, fontsize=8, ncol=len(names), labelcolor=INK,
               handlelength=1.1, handleheight=1.1, columnspacing=1.4)
    fig.suptitle(title, fontsize=11, color=INK, x=0.012, ha="left", y=0.992)
    fig.text(0.012, 0.972, subtitle,
             fontsize=8, color=INK_MUTED, ha="left", va="top")
    if footnote:
        fig.text(0.012, 0.010, footnote,
                 fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    if half_labels:
        fig.subplots_adjust(bottom=0.20 if len(panels) > 2 else 0.36)
    else:
        fig.subplots_adjust(bottom=0.17 if len(panels) > 2 else 0.30)
    # The reference rows (the original models) are set off from the arms by
    # a rule midway between the last of them and the first arm, drawn after
    # the layout is final so it lands in the gap.
    for index in range(1, len(panels)):
        if panels[index - 1][0] in REFERENCES and panels[index][0] not in REFERENCES:
            above, below = axes[index - 1].get_position(), axes[index].get_position()
            y = (above.y0 + below.y1) / 2
            from matplotlib.lines import Line2D
            fig.add_artist(Line2D([0.012, below.x1], [y, y], transform=fig.transFigure,
                                  color=INK_MUTED, linewidth=1.0))
            break
    written = []
    for suffix in (".png", ".pdf"):
        path = out + suffix
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        written.append(path)
    plt.close(fig)
    print("wrote " + ", ".join(written))

    # A table view always exists alongside the figure, so the data is
    # reachable without reading colour.
    table = out + ".tsv"
    rows = []
    for label, halves, who, *_ in panels:
        for half, (model, test) in zip(halves, who):
            if half is None:
                continue
            for (b, e, asy, f), got in sorted(half[0].items()):
                if (label, model, test, (b, e, asy, f)) in table_skip:
                    continue
                for measure, v in got.items():
                    rows.append((label, model, test, b, e, asy, f, measure,
                                 round(v, 6)))
    columns = ["arm", "model", "test", "biosample", "encid", "assay", "fold",
               "measure", "value"]
    if all(model == test == label for label, _h, who, *_ in panels for model, test in who):
        # Every row its own model on its own test set (summary_scores): the
        # table keeps its original columns.
        rows = [(r[0],) + r[3:] for r in rows]
        columns = ["arm"] + columns[3:]
    pd.DataFrame(rows, columns=columns).to_csv(table, sep="\t", index=False)
    print(f"wrote {table}")


def main(argv=None):
    global TF_GAPS, ENCID_ORDER
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results TSV from summarize.py")
    ap.add_argument("--out", required=True,
                    help="output prefix, e.g. <dir>/summary_scores; .png, .pdf "
                         "and .tsv are written, and the cross-test figures get "
                         "suffixes")
    ap.add_argument("--source", default="own", choices=["own", "headtohead"],
                    help="for summary_scores: 'own' scores each arm's model on "
                         "its own test data from training; 'headtohead' uses "
                         "the same scores recomputed by an older cross-evaluation")
    ap.add_argument("--subset", default="peaks")
    ap.add_argument("--arm", action="append",
                    help="restrict to this arm (repeatable); the cross-test "
                         "figures always keep their two grey reference rows")
    ap.add_argument("--fill-original", action="append", default=[], metavar="ENCID:FOLD",
                    help="a fold with no original-model run, drawn at the mean of that "
                         "dataset's other folds (repeatable; the published figures "
                         "use ENCSR868FGK:2)")
    ap.add_argument("--encid-order", default=",".join(ENCID_ORDER),
                    help="comma-separated ENCIDs, left to right; unlisted ones go last")
    args = ap.parse_args(argv)

    try:
        TF_GAPS = tuple((encid, int(fold)) for encid, fold in
                        (item.rsplit(":", 1) for item in args.fill_original))
    except ValueError:
        sys.exit(f"--fill-original takes ENCID:FOLD, got {args.fill_original}")
    ENCID_ORDER = [e for e in args.encid_order.split(",") if e]

    values = measure_values(load(args.results, args.subset))
    if not values:
        sys.exit("no counts/profile measures found")

    # Each arm's model on its own test set, from the chosen source.
    own = {}
    for arm in {k[1] for k in values}:
        cells = select(values, args.source, arm, arm, arm)
        if cells:
            own[arm] = cells
    if not own:
        sys.exit(f"no rows with source={args.source!r} subset={args.subset!r} "
                 f"in {args.results}")

    arms = [a for a in ARM_ORDER if a in own]
    arms += sorted(set(own) - set(arms))
    if args.arm:
        arms = [a for a in arms if a in set(args.arm)]
        if not arms:
            sys.exit(f"none of {args.arm} present in {args.results}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- summary_scores: each model on its own test set.
    # Compare against the old model only when its panel is on the page, so
    # every "(+0.02)" has its reference in view.
    reference = own[TF_ARM] if TF_ARM in arms else None
    drawn, table_skip = dict(own), set()
    if TF_ARM in drawn:
        drawn[TF_ARM], added = fill_tf_gaps(own[TF_ARM])
        table_skip.update((TF_ARM, TF_ARM, TF_ARM, k) for k in added)
    # The two original models in the reference greys, as in the cross-test
    # figures; the arms in the measure colours.
    panels = [(arm, [(drawn[arm], reference if arm != TF_ARM else None, arm in REFERENCES)],
               [(arm, arm)], ("ATAC",) if arm in DNASE_ONLY_ARMS else ())
              for arm in arms]
    geometry = layout({k for arm in arms for k in drawn[arm]})
    footnote = None
    if reference is not None:
        tf = arm_label(TF_ARM)
        footnote = (f"{tf}: the original TensorFlow ChromBPNet, scored by its "
                    f"own pipeline on its own peak set.\n"
                    f"(+0.02) after a mean = that arm minus {tf}, averaged "
                    f"over the folds both have; + means the arm is better; "
                    f"not shown when it rounds to 0.00.")
    render(plt, args.out, panels, [geometry],
           "ChromBPNet test set performance, by assay cut site convention",
           f"{args.source} models, {args.subset} subset  |  higher is better "
           f"for both measures  |  horizontal rule = mean over that "
           f"dataset's folds",
           footnote, table_skip=table_skip,
           grey_legend="original models" if any(a in REFERENCES for a in arms) else None)

    # ---- The two cross-test figures, from the cross-evaluation. Their reference
    # rows are always the standard models on their own test sets (source
    # own, whatever --source says), so each half's grey row and arm rows
    # describe the same models and test sets.
    refs = {r: select(values, "own", r, r, r) for r in REFERENCES}
    if not all(refs.values()):
        missing = [r for r, cells in refs.items() if not cells]
        print(f"skipped the cross-test figures: no own scores for {missing}",
              file=sys.stderr)
        return 0
    tested = [a for a in arms if a not in REFERENCES]
    tf, std = arm_label(TF_ARM), arm_label(STANDARD)
    figures = (
        ("_aug_on_standard_tests",
         lambda arm, ref: (arm, ref),          # arm's model, ref's test set
         "ChromBPNet test set performance: cut site convention models on the "
         "original models' test sets",
         f"left: {tf}'s test sets  |  right: {std}'s test sets",
         [f"on {tf}'s test sets", f"on {std}'s test sets"],
         "same test set, a different model; + means the arm's model is better",
         "ChromBPNet test set performance: cut site convention models on {ref}'s "
         "test sets"),
        ("_standard_on_aug_tests",
         lambda arm, ref: (ref, arm),          # ref's model, arm's test set
         "ChromBPNet test set performance: original models on each cut site "
         "convention's test set",
         f"left: the {tf} model  |  right: the {std} model",
         [f"{tf} model", f"{std} model"],
         "the same model on a different test set; + means it scores higher "
         "on that arm's test set than on its own",
         "ChromBPNet test set performance: the {ref} model on each cut site "
         "convention's test set"),
    )
    # The grey rows as drawn: the original model's own row gets its TF_GAPS
    # folds filled, as in summary_scores. The "(+0.02)" references stay `refs`.
    refs_drawn, ref_added = dict(refs), {}
    refs_drawn[TF_ARM], ref_added[TF_ARM] = fill_tf_gaps(refs[TF_ARM])
    for suffix, pick, title, sides, half_labels, meaning, split_title in figures:
        panels = [(r, [(refs_drawn[r], None, True) if r == side else None
                       for side in REFERENCES],
                   [(side, side) for side in REFERENCES])
                  for r in REFERENCES]
        table_skip = {(TF_ARM, TF_ARM, TF_ARM, k) for k in ref_added[TF_ARM]}
        measured = 0
        for arm in tested:
            halves, who = [], []
            for side in REFERENCES:
                model, test = pick(arm, side)
                cells = select(values, "headtohead", arm, model, test)
                measured += len(cells)
                if side == TF_ARM:
                    # The original-model half misses TF_GAPS folds for the
                    # same reason the grey row does.
                    cells, added = fill_tf_gaps(cells)
                    table_skip.update((arm, model, test, k) for k in added)
                halves.append((cells, refs[side], False) if cells else None)
                who.append((model, test))
            panels.append((arm, halves, who,
                           ("ATAC",) if arm in DNASE_ONLY_ARMS else ()))
        if not measured:
            print(f"skipped {args.out}{suffix}: no cross-test results yet "
                  f"(run the cross-evaluation step)", file=sys.stderr)
            continue
        keys = {k for _l, halves, *_ in panels for h in halves if h
                for k in h[0]}
        left = layout(keys)
        right = layout(keys, x0=max(p for p, _ in left[1]) + 1.0 + HALF_GAP)
        render(plt, args.out + suffix, panels, [left, right], title,
               f"{sides}  |  grey rows = each original model on its own test "
               f"set  |  {args.subset} subset  |  higher is better for both "
               f"measures  |  horizontal rule = mean over that dataset's folds",
               f"Grey rows: {tf} and {std} on their own test sets, "
               f"the same numbers as in summary_scores, each in its own half.\n"
               f"(+0.02) after a mean = that row minus the grey row of its "
               f"half, averaged over the folds both have: {meaning}; not "
               f"shown when it rounds to 0.00.",
               half_labels=half_labels, grey_legend="grey rows: the model on its own test set",
               table_skip=table_skip)
        # The same figure one half at a time: that half's grey row, then every
        # arm row's cells in that half, on the same columns as the combined one.
        for index, (side, name) in enumerate(((TF_ARM, "_original"), (STANDARD, "_refactor"))):
            ref = arm_label(side)
            split = [(side, [(refs_drawn[side], None, True)], [(side, side)])]
            split += [(arm, [halves[index]], [who[index]], shaded)
                      for arm, halves, who, shaded in panels[len(REFERENCES):]]
            render(plt, args.out + suffix + name, split, [left],
                   split_title.format(ref=ref),
                   f"grey row = {ref} on its own test set  |  {args.subset} subset  |  "
                   f"higher is better for both measures  |  horizontal rule = mean over "
                   f"that dataset's folds",
                   f"Grey row: {ref} on its own test set, the same numbers as in "
                   f"summary_scores.\n(+0.02) after a mean = that row minus the grey row, "
                   f"averaged over the folds both have: {meaning}; not shown when it "
                   f"rounds to 0.00.",
                   half_labels=[half_labels[index]],
                   grey_legend="grey row: the model on its own test set",
                   table_skip=table_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
