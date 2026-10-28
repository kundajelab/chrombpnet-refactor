#!/usr/bin/env python
"""Each model's DeepSHAP scores against the original's five-fold mean.

    python compare_original_mean.py --ids <ids.tsv> --engines-root <engines tree> \
        --root <expand root> --out-dir <dir> [--biosamples biosamples.tsv] [--workers 4]

The original ChromBPNet's released DeepSHAP scores are per fold and
genome-wide: each fold model scored every peak of the dataset
(<ENCID>.<head>_scores.h5 in the ids table's DEEPLIFT_COUNTS_DIR /
DEEPLIFT_PROFILE_DIR). Their mean over the five folds is the original's
ensemble attribution, defined on every peak; the atlas stores it as
<head>_mean_shap_scores.h5 in DEEPLIFT_MEAN_COUNTS_DIR /
DEEPLIFT_MEAN_PROFILE_DIR, row for row in the per-fold files' (identical)
region order.

For dataset D and fold k the regions are fold k's 30K peak subsample, R_k
(the PyTorch runs' interpret_subsample/chrombpnet_nobias.interpreted_regions.bed):
the old and new fold-k models, LOFO student k and the final student were
all scored on it, with seed 1234. The reference on R_k is the stored
five-fold mean's hypothetical scores (shap/seq), read by (chromosome,
summit); without the mean columns, the mean of the five per-fold files.
Compared against it:

  student-final       the dataset's final student (trained on every chromosome)
  student-lofo        LOFO student k
  old, new            fold k's PyTorch models
  original (1 fold)   the original fold-k model's stored scores, against the
                      mean of the other four folds: it is a fifth of the
                      five-fold mean, which would flatter it
  original (1 fold, new seed)
                      the same model rescored by run_deepshap_seeds.py with
                      seed 1, against the other four: other shuffled
                      references than the stored scores, as every PyTorch
                      model's have
  original (5-fold mean, from the folds)
                      the five per-fold files averaged here, against the
                      stored mean: a check that the two are the same (~1)

The one-fold baselines need the five per-fold files, ~80 GB a dataset;
--no-baselines reads only the stored means and the models' scores.

Regions are kept only where every original fold has them and the one-hot
sequences are identical. Per region, over the whole input window, the
Pearson r and cosine similarity of the hypothetical scores (4 x 2114,
flattened) and of the projected scores (the observed base's), as in
../engine/run_summarize.py.

Writes to --out-dir:
  original_mean.tsv     one row per dataset, fold, head, kind and model
  summary_contributions_original_mean_{pearson,cosine}.{png,pdf,tsv}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))
import run_summarize as rs  # noqa: E402

HEADS = ("counts", "profile")
KINDS = ("hypothetical", "projected")
SUBSAMPLE = "model/auxiliary/interpret_subsample/chrombpnet_nobias"
ONE_FOLD = "original (1 fold)"
NEW_SEED = "original (1 fold, new seed)"
REFERENCE = "original (5-fold mean)"
REFERENCE_LOO = "original (other 4 folds' mean)"
FROM_FOLDS = "original (5-fold mean, from the folds)"


def run_scores(run, head):
    """A run's subsample scores for one head (the midpoint-weighted profile
    scores, as run_summarize.py --profile-stopgrad off reads) and regions."""
    name = "midpoint.profile" if head == "profile" else "counts"
    return f"{run}/{SUBSAMPLE}.{name}_scores.h5", f"{run}/{SUBSAMPLE}.interpreted_regions.bed"


def models(engines_root, root, assay, encid, fold):
    return {
        "old": f"{engines_root}/pytorch/{assay}/{encid}/fold_{fold}",
        "new": f"{root}/new/{assay}/{encid}/fold_{fold}",
        "student-lofo": f"{root}/distill/lofo/{assay}/{encid}/fold_{fold}",
        "student-final": f"{root}/distill/final/{assay}/{encid}/fold_{fold}",
    }


def compare(hyp_a, raw_a, hyp_b, raw_b, chunk=2048):
    """Per-region metrics of aligned (n, 4, L) score arrays, over the regions
    whose one-hot sequences agree: ({kind: (r, cos)}, n_compared, n_mismatch)."""
    out = {kind: ([], []) for kind in KINDS}
    mismatched = 0
    for lo in range(0, len(hyp_a), chunk):
        ha, hb = hyp_a[lo:lo + chunk], hyp_b[lo:lo + chunk]
        ra, rb = raw_a[lo:lo + chunk], raw_b[lo:lo + chunk]
        same = (ra == rb).all(axis=(1, 2))
        mismatched += int((~same).sum())
        ha, hb, ra = ha[same].astype(np.float32), hb[same].astype(np.float32), ra[same]
        n = len(ha)
        if not n:
            continue
        for kind, (x, y) in {"hypothetical": (ha.reshape(n, -1), hb.reshape(n, -1)),
                             "projected": ((ha * ra).sum(1), (hb * ra).sum(1))}.items():
            r, cos = rs.row_metrics(x, y)
            out[kind][0].append(r)
            out[kind][1].append(cos)
    return out, len(hyp_a) - mismatched, mismatched


def summarize(metrics):
    rows = {}
    for kind in KINDS:
        r = np.concatenate(metrics[kind][0]) if metrics[kind][0] else np.full(1, np.nan)
        cos = np.concatenate(metrics[kind][1]) if metrics[kind][1] else np.full(1, np.nan)
        rows[kind] = (float(np.nanmedian(r)), float(np.nanmean(r)),
                      float(np.nanmedian(cos)), float(np.nanmean(cos)))
    return rows


def read_union(scores, regions_file, union, upos):
    """Rows of a genome-wide scores file for the regions in ``union``:
    (hypothetical scores, one-hot, mask), each over ``union``; the mask is
    False where the file lacks the region."""
    import h5py
    try:
        import hdf5plugin  # noqa: F401  (the originals are Blosc-compressed)
    except ImportError:
        pass
    listed = rs.region_keys(regions_file)
    index = {}
    for i, key in enumerate(listed):
        index.setdefault(key, i)
    pairs = sorted((index[key], upos[key]) for key in union if key in index)
    file_rows = np.asarray([p[0] for p in pairs], dtype=np.int64)
    union_rows = np.asarray([p[1] for p in pairs], dtype=np.int64)
    with h5py.File(scores, "r") as f:
        if f["shap/seq"].shape[0] != len(listed):
            raise ValueError(f"{scores} has {f['shap/seq'].shape[0]} rows, but {regions_file} "
                             f"lists {len(listed)} regions")
        hyp = np.zeros((len(union),) + f["shap/seq"].shape[1:], dtype=np.float16)
        raw = np.zeros((len(union),) + f["raw/seq"].shape[1:], dtype=np.int8)
        hyp[union_rows] = rs.read_rows(f["shap/seq"], file_rows)
        raw[union_rows] = rs.read_rows(f["raw/seq"], file_rows)
    mask = np.zeros(len(union), dtype=bool)
    mask[union_rows] = True
    return hyp, raw, mask


def dataset_job(job):
    """Every comparison of one dataset: rows (fold, head, kind, model,
    reference, n_regions, n_compared, n_seq_mismatch, r_median, r_mean,
    cos_median, cos_mean) and notes."""
    assay, encid = job["assay"], job["encid"]
    folds = job["folds"]                       # fold -> {head: original scores dir}
    mean_dirs = job.get("mean_dirs") or {}     # head -> stored five-fold mean dir
    notes, rows = [], []
    regions = {k: rs.region_keys(run_scores(models(job["engines_root"], job["root"], assay, encid, k)["old"],
                                            "counts")[1])[:job.get("max_regions")] for k in folds}
    union = sorted(set().union(*map(set, regions.values())))
    upos = {key: i for i, key in enumerate(union)}
    for head in HEADS:
        # The reference: the stored five-fold mean, rows in the per-fold files'
        # region order (the averaging script averaged row by row, and the
        # five region lists are identical), else the mean of the folds.
        stored = None
        if mean_dirs.get(head):
            any_fold = folds[min(folds)][head]
            stored = read_union(os.path.join(mean_dirs[head], f"{head}_mean_shap_scores.h5"),
                                os.path.join(any_fold, f"{encid}.interpreted_regions.bed.gz"), union, upos)
            print(f"#   {encid} {head}: stored five-fold mean read", file=sys.stderr, flush=True)
        total, count, raw_u, own = None, np.zeros(len(union), dtype=np.int16), None, {}
        if job.get("baselines", True) or stored is None:
            for k, dirs in sorted(folds.items()):
                hyp, raw, mask = read_union(os.path.join(dirs[head], f"{encid}.{head}_scores.h5"),
                                            os.path.join(dirs[head], f"{encid}.interpreted_regions.bed.gz"),
                                            union, upos)
                if total is None:
                    total = np.zeros(hyp.shape, dtype=np.float32)
                    raw_u = raw
                total += hyp
                count += mask
                at = np.asarray([upos[key] for key in regions[k]])
                own[k] = (hyp[at], mask[at])
                del hyp, raw
                print(f"#   {encid} {head}: original fold {k} read", file=sys.stderr, flush=True)
        complete = count == len(folds)
        for k in sorted(folds):
            at = np.asarray([upos[key] for key in regions[k]])
            if stored is not None:
                reference, raw_ref, keep_ref = stored[0][at].astype(np.float32), stored[1][at], stored[2][at]
            else:
                reference, raw_ref, keep_ref = total[at] / len(folds), raw_u[at], complete[at]
            if own:
                own_hyp, own_ok = own[k]
                keep_loo = complete[at] & own_ok
                loo = (total[at] - own_hyp) / (len(folds) - 1)
                metrics, n_cmp, n_mis = compare(own_hyp[keep_loo], raw_u[at][keep_loo], loo[keep_loo],
                                                raw_u[at][keep_loo])
                for kind, values in summarize(metrics).items():
                    rows.append((k, head, kind, ONE_FOLD, REFERENCE_LOO, int(keep_loo.sum()), n_cmp, n_mis) + values)
                if stored is not None:
                    # The stored mean against the folds' own mean: ~1 if they agree.
                    both = keep_ref & complete[at]
                    metrics, n_cmp, n_mis = compare((total[at] / len(folds))[both], raw_u[at][both],
                                                    reference[both], raw_ref[both])
                    for kind, values in summarize(metrics).items():
                        rows.append((k, head, kind, FROM_FOLDS, REFERENCE, int(both.sum()), n_cmp, n_mis) + values)
            sets = {}
            for label, run in models(job["engines_root"], job["root"], assay, encid, k).items():
                scores, regions_file = run_scores(run, head)
                if not (os.path.isfile(os.path.join(run, "COMPLETED")) and os.path.isfile(scores)):
                    notes.append(f"{encid} fold {k} {head}: no {label} scores")
                    continue
                sets[label] = (scores, regions_file, REFERENCE)
            seed_dir = rs.seed_dirs(os.path.join(job["engines_root"], "original", assay, encid, f"fold_{k}")).get(1)
            if own and seed_dir and os.path.isfile(os.path.join(seed_dir, f"scores.{head}_scores.h5")):
                sets[NEW_SEED] = (os.path.join(seed_dir, f"scores.{head}_scores.h5"),
                                  os.path.join(seed_dir, "scores.interpreted_regions.bed"), REFERENCE_LOO)
            for label, (scores, regions_file, which) in sets.items():
                m_raw, m_hyp, m_rows = rs.load_rows(scores, regions_file, regions[k])
                have = np.asarray([r is not None for r in m_rows])
                if which == REFERENCE_LOO:
                    ok = have & keep_loo
                    target, target_raw = loo[ok], raw_u[at][ok]
                else:
                    ok = have & keep_ref
                    target, target_raw = reference[ok], raw_ref[ok]
                idx = np.asarray([r for r, use in zip(m_rows, ok) if use], dtype=np.int64)
                metrics, n_cmp, n_mis = compare(m_hyp[idx], m_raw[idx], target, target_raw)
                for kind, values in summarize(metrics).items():
                    rows.append((k, head, kind, label, which, int(ok.sum()), n_cmp, n_mis) + values)
                del m_raw, m_hyp
            print(f"#   {encid} {head}: fold {k} compared", file=sys.stderr, flush=True)
        del stored, total, raw_u, own
    return rows, notes


def plot(out_dir, table, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import plot_summarize as ps

    column, metric_label = ps.CONTRIBUTION_METRICS[metric]
    frame = pd.DataFrame(table)
    runs = ["old", "new", "student-lofo", "student-final"]
    bars = [ONE_FOLD, NEW_SEED] + runs
    styles = {ONE_FOLD: ps.RETRAINED_BAR, NEW_SEED: ps.NEW_SEED_BAR}
    styles.update({run: dict(color=colour, linewidth=0) for run, colour in zip(runs, ps.SLOTS)})
    frame = frame[frame["model"].isin(bars)]
    keys = {(r.biosample, r.encid, r.assay, int(r.fold)) for r in frame.itertuples()}
    geometry = ps.layout(keys)
    positions = geometry[0]
    step = 0.15
    offsets = {bar: (i - (len(bars) - 1) / 2) * step for i, bar in enumerate(bars)}
    low = min(0.0, float(frame[column].min()) - 0.05)
    fig, axes = ps.figure(plt, len(ps.CONTRIBUTION_PANELS), len(geometry[1]))
    for index, (ax, (head, kind)) in enumerate(zip(axes, ps.CONTRIBUTION_PANELS)):
        ps.style_axes(ax, f"{head}\n{kind}", geometry, (low, 1),
                      np.arange(0, 1.01, 0.25) if low == 0 else None, index == len(axes) - 1)
        sub = frame[(frame["head"] == head) & (frame["kind"] == kind)]
        for bar in bars:
            rows = sub[sub["model"] == bar]
            xs = [positions[(r.biosample, r.encid, int(r.fold))] + offsets[bar] for r in rows.itertuples()]
            ax.bar(xs, rows[column].astype(float), width=step - 0.02, zorder=3, **styles[bar])
    means = frame.groupby("model")[column].mean()
    handles = [plt.Rectangle((0, 0), 1, 1, **styles[b]) for b in bars if b in means]
    names = [f"{b} (mean {means[b]:.3f})" for b in bars if b in means]
    out = os.path.join(out_dir, f"summary_contributions_original_mean_{metric}")
    ps.finish(plt, fig, out,
              "DeepSHAP contribution scores: each model vs the original's five-fold mean",
              f"fold k's 30K peak subsample  |  reference: the original's stored five-fold mean DeepSHAP on those "
              f"peaks  |  bar = median over regions of the per-region {metric_label}  |  "
              f"legend: mean over datasets and folds",
              "hypothetical: the 4 x 2114 hypothetical scores, flattened. projected: the 2114 per-base scores of "
              "the observed base. Each over the whole input window.\n"
              f"hollow bar: {ONE_FOLD}, the original's fold-k model's stored scores vs the mean of the other four "
              "folds (the five-fold mean includes it). grey bar: the same model rescored with another seed, vs the "
              "other four:\nother shuffled references than the stored scores, as every coloured bar's. "
              "student-final trained on every chromosome; four of student-lofo's five teachers trained on fold k's "
              "test chromosomes.",
              handles, names)
    ps.write_table(out + ".tsv",
                   frame[["model", "reference", "biosample", "encid", "assay", "fold", "head", "kind",
                          "n_compared", "pearsonr_median", "cosine_median"]].values.tolist(),
                   ["model", "reference", "biosample", "encid", "assay", "fold", "head", "kind",
                    "n_compared", "pearsonr_median", "cosine_median"])


def read_mean_dirs(path):
    """ENCID -> {head: stored five-fold mean dir}, from the ids table's
    DEEPLIFT_MEAN_COUNTS_DIR / DEEPLIFT_MEAN_PROFILE_DIR (absent: none)."""
    with open(path) as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        out = {}
        for line in fh:
            f = line.rstrip("\r\n").split("\t")
            if len(f) <= col["ENCID"]:
                continue
            dirs = {head: f[col[name]] for head, name in (("counts", "DEEPLIFT_MEAN_COUNTS_DIR"),
                                                          ("profile", "DEEPLIFT_MEAN_PROFILE_DIR"))
                    if name in col and len(f) > col[name] and f[col[name]]}
            if dirs:
                out[f[col["ENCID"]]] = dirs
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="the engines sweep's ids table")
    ap.add_argument("--engines-root", required=True, help="the engines sweep's run tree")
    ap.add_argument("--root", required=True, help="this comparison's run tree (new/, distill/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--biosamples", default=None)
    ap.add_argument("--workers", type=int, default=4, help="datasets at once (each holds ~15 GB)")
    ap.add_argument("--datasets", nargs="+", default=None, metavar="ENCID", help="only these datasets")
    ap.add_argument("--max-regions", type=int, default=None,
                    help="only the first N regions of each fold's subsample (a quick check)")
    ap.add_argument("--no-baselines", dest="baselines", action="store_false",
                    help="skip the one-fold baselines, and with them reading the per-fold files "
                         "(~80 GB per dataset): only the stored means and the models' scores are read")
    args = ap.parse_args(argv)
    biosamples = rs.load_biosamples(args.biosamples) if args.biosamples else {}
    datasets, mean_dirs = {}, {}
    for assay, encid, fold, _tf_dir, dirs in rs.ids_rows(args.ids):
        datasets.setdefault((assay, encid), {})[int(fold)] = dirs
    for encid, dirs in read_mean_dirs(args.ids).items():
        mean_dirs[encid] = dirs
    jobs = [{"assay": a, "encid": e, "folds": f, "engines_root": args.engines_root, "root": args.root,
             "max_regions": args.max_regions, "baselines": args.baselines,
             "mean_dirs": mean_dirs.get(e)}
            for (a, e), f in datasets.items() if not args.datasets or e in args.datasets]
    table = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(dataset_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            found, notes = future.result()
            for note in notes:
                print(f"#   {note}", file=sys.stderr)
            for row in found:
                fold, head, kind, model, reference = row[:5]
                table.append(dict(assay=job["assay"], biosample=biosamples.get(job["encid"], job["encid"]),
                                  encid=job["encid"], fold=fold, head=head, kind=kind, model=model,
                                  reference=reference, n_regions=row[5], n_compared=row[6],
                                  n_seq_mismatch=row[7], pearsonr_median=row[8], pearsonr_mean=row[9],
                                  cosine_median=row[10], cosine_mean=row[11]))
            print(f"# {job['encid']}: {len(found)} comparisons", file=sys.stderr, flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    columns = ["assay", "biosample", "encid", "fold", "head", "kind", "model", "reference", "n_regions",
               "n_compared", "n_seq_mismatch", "pearsonr_median", "pearsonr_mean", "cosine_median", "cosine_mean"]
    table.sort(key=lambda r: (r["encid"], r["fold"], r["head"], r["kind"], r["model"]))
    rs.write_tsv(os.path.join(args.out_dir, "original_mean.tsv"), columns,
                 [[rs.fmt(r[c]) for c in columns] for r in table])
    for metric in ("pearson", "cosine"):
        plot(args.out_dir, table, metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
