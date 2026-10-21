#!/usr/bin/env python
"""The figures this comparison adds to ../engine/plot_summarize.py's.

    python plot_expand.py --dir <run_summarize.py --out-dir>

Reads run_summarize.py's tables and writes, as .png, .pdf and a .tsv table
view each:

    expand_epochs          epochs each run's ChromBPNet (or student) trained,
                           and the epoch whose weights were kept (1-based),
                           from epochs.tsv
    expand_bias            each run's bias model on its test regions
                           (results.tsv, source bias): profile 1 - median JSD
                           on nonpeaks and on peaks, and counts Pearson r on
                           nonpeaks where the bias model has a counts head
    expand_bias_response   the bias-free model's largest marginal-footprint
                           response to the assay's bias motifs
                           (bias_response.tsv); the pipeline calls a model
                           corrected when every motif's is under 0.003, the
                           rule drawn
    expand_contributions_versus_{pearson,cosine}
                           run_summarize.py --versus pairs' DeepSHAP scores
                           against each other (contributions.tsv rows whose
                           reference is a run, not the original): one panel
                           per head and kind, one bar per pair

x is biosample > ENCID (assay) > fold, as in plot_summarize.py; bars are one
per run, side by side, in the colour plot_summarize.py gives that run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import plot_summarize as ps  # noqa: E402

CORRECTED_BELOW = 0.003   # marginal_footprinting's corrected/uncorrected cut


def offsets_for(names):
    step = {1: 0.0, 2: 0.36, 3: 0.28}.get(len(names), 0.84 / max(len(names), 1))
    return {n: (i - (len(names) - 1) / 2) * step for i, n in enumerate(names)}, max(step - 0.02, 0.2)


def run_bars(plt, frame, panels, colours, out, title, subtitle, footnote, ylim=None, yticks=None,
             rule=None, names=None):
    """One panel per (key, label) in `panels`; in each, one bar per series
    (frame's `series` column) per fold. `frame` has biosample, encid, assay,
    fold, panel, series and value."""
    frame = frame[np.isfinite(frame["value"].astype(float))]
    panels = [(key, label) for key, label in panels if (frame["panel"] == key).any()]
    if not panels:
        print(f"skipped {out}: nothing to draw", file=sys.stderr)
        return False
    series = [s for s in colours if (frame["series"] == s).any()]
    keys = {(r.biosample, r.encid, r.assay, int(r.fold)) for r in frame.itertuples()}
    geometry = ps.layout(keys)
    positions = geometry[0]
    offsets, width = offsets_for(series)
    fig, axes = ps.figure(plt, len(panels), len(geometry[1]))
    for index, (ax, (key, label)) in enumerate(zip(axes, panels)):
        sub = frame[frame["panel"] == key]
        top = ylim[1] if ylim else max(float(sub["value"].max()) * 1.15, rule * 1.3 if rule else 0.0, 1e-9)
        ps.style_axes(ax, label, geometry, (ylim[0] if ylim else 0, top), yticks, index == len(panels) - 1)
        for name in series:
            rows = sub[sub["series"] == name]
            xs = [positions[(r.biosample, r.encid, int(r.fold))] + offsets[name] for r in rows.itertuples()]
            ax.bar(xs, rows["value"].astype(float), width=width, color=colours[name], linewidth=0, zorder=3)
        if rule is not None:
            ax.axhline(rule, color=ps.INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[s]) for s in series]
    legend = [(names or {}).get(s, s) for s in series]
    if len(panels) == 1:
        # A mean is only meaningful over one measure.
        means = frame.groupby("series")["value"].apply(lambda v: float(np.mean(v.astype(float))))
        legend = [f"{name} (mean {means[s]:.3g})" for name, s in zip(legend, series)]
    ps.finish(plt, fig, out, title, subtitle, footnote, handles, legend)
    ps.write_table(out + ".tsv", frame[["series", "panel", "biosample", "encid", "assay", "fold", "value"]]
                   .values.tolist(), ["run", "panel", "biosample", "encid", "assay", "fold", "value"])
    return True


def plot_epochs(plt, epochs, colours, args):
    if epochs is None or epochs.empty:
        print("skipped expand_epochs: no epochs.tsv rows", file=sys.stderr)
        return
    chrombpnet = epochs[epochs["model"] == "chrombpnet"]
    frame = pd.concat([
        chrombpnet.assign(panel="trained", value=chrombpnet["epochs"].astype(float)),
        chrombpnet.assign(panel="kept", value=pd.to_numeric(chrombpnet["best_epoch"], errors="coerce") + 1),
    ]).rename(columns={"engine": "series"})
    selections = sorted(set(chrombpnet["selection"].astype(str)))
    run_bars(plt, frame, [("trained", "epochs\ntrained"), ("kept", "epoch\nkept")], colours,
             args.out + "expand_epochs", "Epochs trained and the epoch kept",
             "ChromBPNet model (or distilled student) of each run  |  epoch kept counted from 1  |  "
             f"selection: {', '.join(selections)}",
             "The epoch kept: lowest validation loss (loss), highest validation counts r + 1 - median JSD "
             "(pearsonr-jsd), or the last epoch when nothing was validated (last: the all-chromosome student).\n"
             "Training stops after --early-stop (5) epochs without improvement, so trained = kept + 5 unless "
             "--epochs (50) ran out first.")


def plot_bias(plt, results, colours, args):
    bias = results[results["source"] == "bias"]
    if bias.empty:
        print("skipped expand_bias: no bias-model metrics", file=sys.stderr)
        return
    parts = []
    for key, group, subset, metric, transform in (
            ("jsd_nonpeaks", "profile", "nonpeaks", "median_jsd", lambda v: 1 - v),
            ("jsd_peaks", "profile", "peaks", "median_jsd", lambda v: 1 - v),
            ("r_nonpeaks", "counts", "nonpeaks", "pearsonr", lambda v: v)):
        rows = bias[(bias["group"] == group) & (bias["subset"] == subset) & (bias["metric"] == metric)]
        parts.append(rows.assign(panel=key, value=transform(rows["value"].astype(float))))
    frame = pd.concat(parts).rename(columns={"engine": "series"})
    run_bars(plt, frame, [("jsd_nonpeaks", "1 - profile JSD\nnonpeaks"),
                          ("jsd_peaks", "1 - profile JSD\npeaks"),
                          ("r_nonpeaks", "counts r\nnonpeaks")], colours,
             args.out + "expand_bias", "Bias model test scores",
             "each run's bias model on its test chromosomes  |  higher is better",
             "A profile-only bias model (--bias-profile-only) has no counts head, so no counts r.\n"
             "The bias model trains on low-count nonpeaks; on peaks it is expected to do worse.",
             ylim=(0, 1), yticks=np.arange(0, 1.01, 0.25))


def plot_bias_response(plt, responses, colours, args):
    if responses is None or responses.empty:
        print("skipped expand_bias_response: no bias_response.tsv rows", file=sys.stderr)
        return
    frame = responses.assign(panel="max", value=responses["max"].astype(float)).rename(columns={"engine": "series"})
    run_bars(plt, frame, [("max", "largest\nresponse")], colours,
             args.out + "expand_bias_response", "Bias left in the bias-free model",
             "marginal footprint of the bias-free model at the assay's bias motifs, inserted into nonpeaks  |  "
             "lower is better  |  dashed rule: 0.003",
             "Largest over the bias motifs of the peak predicted probability at the insertion "
             "(chrombpnet_nobias_max_bias_response.txt). The pipeline calls a model corrected when every\n"
             "motif's response is under 0.003. Values are rounded to 0.001 by the pipeline.",
             rule=CORRECTED_BELOW)


def plot_versus(plt, contributions, colours, args, metric):
    column, metric_label = ps.CONTRIBUTION_METRICS[metric]
    out = args.out + f"expand_contributions_versus_{metric}"
    if contributions is None or contributions.empty:
        print(f"skipped {out}: no contributions.tsv", file=sys.stderr)
        return
    runs = set(colours)
    pairs = contributions[contributions["engine"].isin(runs)
                          & contributions["reference"].astype(str).isin(runs)]
    if pairs.empty:
        print(f"skipped {out}: no --versus comparisons", file=sys.stderr)
        return
    pairs = pairs.assign(series=pairs["engine"].astype(str) + " vs " + pairs["reference"].astype(str),
                         panel=pairs["head"] + ":" + pairs["kind"], value=pairs[column].astype(float))
    order = [f"{a} vs {b}" for a in colours for b in colours if (pairs["series"] == f"{a} vs {b}").any()]
    pair_colours = {name: colours[name.split(" vs ", 1)[0]] for name in order}
    run_bars(plt, pairs, [(f"{h}:{k}", f"{h}\n{k}") for h, k in ps.CONTRIBUTION_PANELS], pair_colours, out,
             f"DeepSHAP contribution scores: run vs run ({metric_label})",
             f"the same regions and DeepSHAP references in both  |  bar = median over regions of the per-region "
             f"{metric_label}  |  legend: mean over datasets and folds",
             "hypothetical: the 4 x 2114 hypothetical scores, flattened. projected: the 2114 per-base scores of "
             "the observed base. Each over the whole input window.",
             ylim=(min(0.0, float(pairs["value"].min()) - 0.05), 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="run_summarize.py's --out-dir")
    ap.add_argument("--out", default=None, help="output prefix (default: <--dir>/)")
    args = ap.parse_args(argv)
    directory = Path(args.dir)
    args.out = args.out if args.out is not None else str(directory) + "/"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps.set_runs(ps.read_runs(directory))
    colours = dict(ps.ENGINE_COLOURS)

    def read(name):
        path = directory / name
        if not path.exists():
            print(f"{path} not found", file=sys.stderr)
            return None
        return pd.read_csv(path, sep="\t", dtype={"fold": int})

    results = read("results.tsv")
    if results is None:
        sys.exit("run run_summarize.py first")
    plot_epochs(plt, read("epochs.tsv"), colours, args)
    plot_bias(plt, results, colours, args)
    plot_bias_response(plt, read("bias_response.tsv"), colours, args)
    contributions = read("contributions.tsv")
    for metric in ps.CONTRIBUTION_METRICS:
        plot_versus(plt, contributions, colours, args, metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
