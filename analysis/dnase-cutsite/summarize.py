#!/usr/bin/env python
"""Collate every arm's test metrics into one TSV, for all datasets and folds.

    python summarize.py --root <runs> --out <results.tsv> \
        [--ids <ids.tsv>] [--biosamples biosamples.tsv]

The sources read under --root, laid out as README.md describes, answer
different questions:

    <arm>/<ASSAY>/<ENCID>/fold_N/model/evaluation/chrombpnet_metrics.json
        each arm's own model on its own test data -- "how good is this arm?"

    <TENSORFLOW_DIR>/[evaluation/]chrombpnet_metrics.json, per ids-table row
        the old TensorFlow ChromBPNet on its own test data, recorded as arm
        `standard_tensorflow` -- "is the refactor at least as good as what
        it replaces?" (only with --ids)

    comparisons/<arm>/<ASSAY>/<ENCID>/fold_N/<who>_metrics.json
        model <who> (standard, standard_tensorflow; `arm` in older runs) on
        the arm's test data -- the cross-evaluation step in README.md.

    comparisons/<arm>/<ASSAY>/<ENCID>/fold_N/on_<test>/arm_metrics.json
        the arm's model on <test>'s test data (standard or
        standard_tensorflow), the other direction of the same comparison.

Every row names both the model that predicted (`model`) and whose test set
it was scored on (`test`); for source `own` the two are the same.

standard_tensorflow is filed under source `own`, not `headtohead`: its
pipeline chose its own peaks and nonpeaks, so it was not scored on the same
regions as the refactor.

The number to watch is profile_metrics/peaks/median_jsd (lower is better)
and counts_metrics/peaks/pearsonr (higher is better).
"""

import argparse
import collections
import glob
import json
import os
import sys

TF_ARM = "standard_tensorflow"

# Where a TensorFlow pipeline run keeps its metrics, relative to the run
# directory. The atlas runs differ by vintage: the newer `chrombpnet
# pipeline` writes an evaluation/ subdirectory, the older ones wrote
# everything flat. First match wins.
TF_METRICS = ("evaluation/chrombpnet_metrics.json", "chrombpnet_metrics.json")


def load_biosamples(path):
    """ENCID -> short biosample label, from a TSV with ENCID and BIOSAMPLE
    columns (biosamples.tsv here lists this analysis's four datasets).

    A table rather than an ENCODE lookup, so summarizing works offline and
    on a compute node with no outbound network. An ENCID with no entry --
    or no table at all -- falls back to its own accession, which degrades
    the label rather than dropping the row.
    """
    if path is None:
        return {}
    table = {}
    try:
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            try:
                i_id, i_bs = header.index("ENCID"), header.index("BIOSAMPLE")
            except ValueError:
                return table
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) > max(i_id, i_bs) and f[i_bs]:
                    table[f[i_id]] = f[i_bs]
    except OSError:
        pass
    return table


def tensorflow_runs(ids_path):
    """(assay, encid, fold, run dir, metrics path or None), one per ids row.

    Columns are found by header name, not position, so the table can grow
    columns without this silently reading the wrong one.
    """
    try:
        with open(ids_path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            lines = fh.read().splitlines()
    except OSError as err:
        print(f"# {TF_ARM}: cannot read ids table: {err}", file=sys.stderr)
        return []
    try:
        cols = [header.index(c)
                for c in ("ASSAY", "ENCID", "FOLD_NUM", "TENSORFLOW_DIR")]
    except ValueError:
        print(f"# {TF_ARM}: no TENSORFLOW_DIR column in {ids_path}",
              file=sys.stderr)
        return []
    runs = []
    for line in lines:
        f = line.rstrip("\r").split("\t")
        if len(f) <= max(cols) or not f[cols[3]]:
            continue
        assay, encid, fold, run_dir = (f[i] for i in cols)
        found = next((p for p in (os.path.join(run_dir, rel) for rel in TF_METRICS)
                      if os.path.isfile(p)), None)
        runs.append((assay, encid, fold, run_dir, found))
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="the runs: <root>/<arm>/<ASSAY>/<ENCID>/fold_N/model and "
                         "<root>/comparisons/ (README.md)")
    ap.add_argument("--out", required=True, help="results TSV to write")
    ap.add_argument("--ids", default=None,
                    help="TSV with ASSAY, ENCID, FOLD_NUM and TENSORFLOW_DIR columns; "
                         "TENSORFLOW_DIR is each fold's original (TensorFlow) "
                         "ChromBPNet run, collated as standard_tensorflow. "
                         "Omit to leave that arm out")
    ap.add_argument("--biosamples", default=None,
                    help="TSV with ENCID and BIOSAMPLE columns, for the biosample "
                         "labels (e.g. biosamples.tsv); without it each dataset "
                         "is labelled by its ENCID")
    args = ap.parse_args(argv)
    root = args.root

    BIOSAMPLE = load_biosamples(args.biosamples)
    rows = []

    def grab(path, source, arm, assay, encid, fold, model, test):
        try:
            with open(path) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            return
        biosample = BIOSAMPLE.get(encid, encid)
        for group in ("counts_metrics", "profile_metrics"):
            for subset, vals in (m.get(group) or {}).items():
                for key, value in (vals or {}).items():
                    if isinstance(value, (int, float)):
                        rows.append((source, arm, assay, biosample, encid, fold,
                                     model, test, group.split("_")[0], subset,
                                     key, f"{value:.6f}"))

    # Each arm's own model on its own data.
    for path in glob.glob(f"{root}/*/*/*/fold_*/model/evaluation/chrombpnet_metrics.json"):
        parts = path.split(os.sep)
        arm, assay, encid, fold = parts[-7], parts[-6], parts[-5], parts[-4].replace("fold_", "")
        if arm == "comparisons":
            continue
        grab(path, "own", arm, assay, encid, fold, arm, arm)

    # The old TensorFlow model on its own data. A row with no metrics file is
    # named rather than skipped silently: a missing baseline would otherwise
    # look like a dataset the comparison never covered.
    for assay, encid, fold, run_dir, path in (tensorflow_runs(args.ids) if args.ids else []):
        if path is None:
            print(f"# {TF_ARM}: no chrombpnet_metrics.json for {assay} {encid} "
                  f"fold {fold} under {run_dir}", file=sys.stderr)
            continue
        grab(path, "own", TF_ARM, assay, encid, fold, TF_ARM, TF_ARM)

    # Head-to-head: other models on the arm's test data ...
    for path in glob.glob(f"{root}/comparisons/*/*/*/fold_*/*_metrics.json"):
        parts = path.split(os.sep)
        arm, assay, encid, fold = parts[-5], parts[-4], parts[-3], parts[-2].replace("fold_", "")
        who = os.path.basename(path).replace("_metrics.json", "")
        grab(path, "headtohead", arm, assay, encid, fold,
             arm if who == "arm" else who, arm)

    # ... and the arm's model on theirs.
    for path in glob.glob(f"{root}/comparisons/*/*/*/fold_*/on_*/arm_metrics.json"):
        parts = path.split(os.sep)
        arm, assay, encid, fold = parts[-6], parts[-5], parts[-4], parts[-3].replace("fold_", "")
        grab(path, "headtohead", arm, assay, encid, fold, arm,
             parts[-2].replace("on_", "", 1))

    header = ("source", "arm", "assay", "biosample", "encid", "fold", "model",
              "test", "group", "subset", "metric", "value")
    rows.sort()
    with open(args.out, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")

    print("\t".join(header))
    for r in rows:
        print("\t".join(r))
    print(f"\n# {len(rows)} rows -> {args.out}", file=sys.stderr)

    # A compact headline: the two metrics that matter, peaks only, per arm/assay.
    print("\n# headline (peaks subset, mean over folds)", file=sys.stderr)
    print(f"#   {'source':<10} {'arm':<19} {'assay':<6} {'biosample':<13} "
          f"{'model':<19} {'test set':<19} {'metric':<16} mean", file=sys.stderr)
    agg = collections.defaultdict(list)
    for (source, arm, assay, biosample, encid, fold, model, test,
         group, subset, key, value) in rows:
        if subset == "peaks" and key in ("pearsonr", "median_jsd", "median_pearsonr"):
            agg[(source, arm, assay, biosample, model, test, key)].append(float(value))
    for k in sorted(agg):
        source, arm, assay, biosample, model, test, key = k
        vals = agg[k]
        print(f"#   {source:<10} {arm:<19} {assay:<6} {biosample:<13} {model:<19} "
              f"{test:<19} {key:<16} {sum(vals)/len(vals):.4f}  (n={len(vals)})",
              file=sys.stderr)

    # Every own model against the old TensorFlow one, paired on dataset and
    # fold. Paired, not a difference of the two means above, because the
    # folds differ by more than the arms do, so a fold one side lacks would
    # move a difference of means on its own.
    own = {}
    for (source, arm, assay, biosample, encid, fold, model, test,
         group, subset, key, value) in rows:
        if source == "own" and subset == "peaks" and key in ("pearsonr", "median_jsd"):
            own[(arm, assay, biosample, encid, fold, key)] = float(value)
    delta = collections.defaultdict(list)
    for (arm, assay, biosample, encid, fold, key), value in own.items():
        tf = own.get((TF_ARM, assay, biosample, encid, fold, key))
        if arm != TF_ARM and tf is not None:
            delta[(arm, assay, biosample, key)].append(value - tf)
    print(f"\n# vs {TF_ARM} (own, peaks subset, arm minus {TF_ARM}, "
          f"mean over the folds both have)", file=sys.stderr)
    print("#   pearsonr: + means the arm is better;  "
          "median_jsd: - means the arm is better", file=sys.stderr)
    if not delta:
        print(f"#   (no dataset and fold has both an arm and {TF_ARM})", file=sys.stderr)
    for k in sorted(delta):
        arm, assay, biosample, key = k
        vals = delta[k]
        print(f"#   {arm:<10} {assay:<6} {biosample:<13} {key:<16} "
              f"{sum(vals)/len(vals):+.4f}  (n={len(vals)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
