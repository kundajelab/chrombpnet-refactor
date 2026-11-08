#!/usr/bin/env python3
"""A copy of a run_summarize.py output directory, edited for the blog's figures.

    python prepare_inputs.py SRC DST [--drop RUN ...] [--fill-original ENCID:FOLD ...] [--jsd-base-e RUN ...]

--drop leaves a run out of every table (and runs.tsv), so plot_summarize.py
does not draw it. --fill-original adds the original model's scores for a fold
it has none for (results.tsv), as the mean over its other folds, per subset,
group and metric. --jsd-base-e puts those runs' median_jsd from base e into
base 2 (x 1/sqrt(ln 2), as run_summarize.py's option of that name does), for a
summary collated without it. SRC is not changed; DST gets results, agreement,
contributions, timings, inputs and runs .tsv, and NOTE.txt saying what was
done.
"""

import argparse
import math
import os

import pandas as pd

TABLES = ("results", "agreement", "contributions", "timings", "inputs", "runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--fill-original", nargs="*", default=[], metavar="ENCID:FOLD")
    ap.add_argument("--jsd-base-e", nargs="*", default=[], metavar="RUN")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    notes = []
    for name in TABLES:
        path = os.path.join(args.src, name + ".tsv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if args.drop and "engine" in frame.columns and name != "inputs":
            frame = frame[~frame["engine"].isin(args.drop)]
            if name == "runs":
                frame["order"] = [str(i) for i in range(len(frame))]
        if name == "results" and args.jsd_base_e:
            rows = frame["engine"].isin(args.jsd_base_e) & (frame["metric"] == "median_jsd")
            frame.loc[rows, "value"] = frame.loc[rows, "value"].map(
                lambda v: "%.6f" % (float(v) / math.sqrt(math.log(2))))
            notes.append(f"results.tsv: median_jsd of {', '.join(args.jsd_base_e)} from base e to base 2 "
                         f"({int(rows.sum())} rows)")
        if name == "results":
            for item in args.fill_original:
                encid, fold = item.split(":")
                rows = frame[(frame["engine"] == "original") & (frame["encid"] == encid)]
                if (rows["fold"] == fold).any():
                    raise SystemExit(f"original already has {encid} fold {fold}")
                keys = [c for c in frame.columns if c not in ("fold", "value")]
                filled = (rows.assign(value=rows["value"].astype(float)).groupby(keys, sort=False)["value"].mean()
                          .reset_index())
                filled["fold"] = fold
                filled["value"] = filled["value"].map(lambda v: "%.6f" % v)
                frame = pd.concat([frame, filled[frame.columns]], ignore_index=True)
                notes.append(f"results.tsv: original {encid} fold {fold} = mean of folds "
                             f"{','.join(sorted(rows['fold'].unique()))} ({len(filled)} rows)")
        frame.to_csv(os.path.join(args.dst, name + ".tsv"), sep="\t", index=False)
    if args.drop:
        notes.append("left out: " + ", ".join(args.drop))
    with open(os.path.join(args.dst, "NOTE.txt"), "w") as handle:
        handle.write(f"from {args.src}\n" + "\n".join(notes) + "\n")
    print("\n".join(notes) or "copied")


if __name__ == "__main__":
    main()
