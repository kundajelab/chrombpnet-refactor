#!/usr/bin/env python
"""Collate the engines sweep: scores, agreement with the original, inputs, runtimes.

    python run_summarize.py --root <run tree> --ids <ids.tsv> --out-dir <dir> \
        [--biosamples biosamples.tsv] [--profile-stopgrad off|on] [--workers 4] \
        [--runs LABEL=DIR ...] [--versus LABEL=REFERENCE ...]

--root is the run tree the steps in README.md write (one directory per
engine, plus original/ and original_retrained/); --ids is the dataset x fold
table (README.md, "Inputs"). In any environment with this package's
dependencies: numpy, scipy, h5py.

The runs collated are <root>/<engine> for pytorch, tensorflow and jax, or
with --runs any labelled set of trees, each laid out as an engine's
(<DIR>/<ASSAY>/<ENCID>/fold_N/{bias_model,model}/); the tables then name the
runs by label where they would name the engine, and the first label is the
headline's reference. --versus LABEL=REFERENCE also compares run LABEL with
run REFERENCE, dataset and fold for dataset and fold: their test predictions
(agreement.tsv) and their subsample DeepSHAP scores (contributions.tsv), with
REFERENCE in the reference column.

Eight tables are written to --out-dir:

results.tsv
    Test-set metrics, one row per (source, engine, dataset, fold, metric):
      own       <engine>/<ASSAY>/<ENCID>/fold_N/model/evaluation/
                chrombpnet_metrics.json -- the engine's ChromBPNet model on
                its own test set. Engine `original` is the old TensorFlow
                ChromBPNet from the ids table's TENSORFLOW_DIR, scored by
                its own pipeline on its own peak set. Engine
                `original_retrained` is the original chrombpnet retrained on
                the sweep's inputs (run_original_retrain.py, --retrained-root),
                on its test set as the engines are on theirs.
      bias      <engine>/.../bias_model/evaluation/<ENCID>_bias_metrics.json
                -- the engine's bias model.

agreement.tsv
    For every engine, dataset and fold: how closely the engine's ChromBPNet
    test predictions (model/evaluation/chrombpnet_predictions.h5) agree with
    the original TensorFlow model's (TENSORFLOW_DIR/evaluation/
    chrombpnet_predictions.h5), on the regions both scored (matched by
    chromosome and summit; some original files store the centre shifted by
    half the input window, so the offset that matches the most regions is
    used and recorded as center_offset) -- Pearson r of the predicted log counts, and the
    median Jensen-Shannon distance between the predicted profiles (scipy's,
    as in median_jsd). Independent of the observed data. The retrained
    original (engine original_retrained) is compared the same way.

contributions.tsv
    For every engine, dataset, fold, head (counts, profile) and kind
    (hypothetical, projected): how closely the engine's DeepSHAP scores on
    its 30K peak subsample (model/auxiliary/interpret_subsample/
    chrombpnet_nobias.<head>_scores.h5, rows as in
    chrombpnet_nobias.interpreted_regions.bed) agree with the original
    model's genome-wide scores (<ENCID>.<head>_scores.h5 in the ids table's
    DEEPLIFT_COUNTS_DIR / DEEPLIFT_PROFILE_DIR -- or DEEPLIFT_DIR for both --
    rows as in that directory's <ENCID>.interpreted_regions.bed.gz). Regions are matched by chromosome
    and summit, and kept only if both one-hot sequences are identical. Per
    region, over the whole 2114 bp input:
      hypothetical  the 4 x 2114 hypothetical scores (`shap`), flattened
      projected     the 2114 per-base scores of the observed base (hypothetical
                    x one-hot, summed over the 4 bases)
    and from them the Pearson r and the cosine similarity; the table gives
    their median and mean over regions. The counts scores of a run exist only
    once run_contribution.py has scored them (or for runs whose pipeline
    already kept them). --profile-stopgrad picks the engines' profile scores:
      off  (the default) chrombpnet_nobias.midpoint.profile_scores.h5, what
           run_contribution.py --profile-weighting midpoint writes: the
           original's weighting, the profile target's stop-gradient ignored
      on   the pipeline's own chrombpnet_nobias.profile_scores.h5, which in
           this sweep honour it (every run predates the midpoint default,
           fcccade)
    and the last column, profile_scores, says which (stopgrad-off or
    stopgrad-on; "pipeline" for the counts head, which does not depend on it).

    It also holds the original's DeepSHAP against itself, once
    run_deepshap_seeds.py has run the original chrombpnet's DeepSHAP on the
    original model with several seeds (<root>/original, --seeds-root):
      engine original:seed<s>, reference original
                    that seeded run against the stored original scores,
                    matched as the engines are
      engine original:seed<a>, reference original:seed<b>
                    two seeded runs against each other: the same model and
                    code, only the shuffled references differ. No comparison
                    of DeepSHAP scores is expected to clear this.
    and, once run_original_retrain.py has retrained the original on the
    sweep's inputs and scored it the same way (<root>/original_retrained,
    --retrained-root):
      engine original_retrained:seed<s>, reference original
                    the retrained model against the stored scores: what the
                    engines' rows are to be read against -- the same code as
                    the stored scores, another training run
      engine original_retrained:seed<s>, reference original:seed<s>
                    against the ENCODE model's run with the same seed, so the
                    same references: the two models alone differ
    Their profile_scores column reads "original": the original's own
    weighting, whatever --profile-stopgrad says. Cached per dataset and fold, apart
    from the engines' comparisons. Reading the originals is most of the cost (~11 GB per
    dataset and fold), so each comparison is cached in --cache-dir, keyed by
    the sizes and times of its input files.

inputs.tsv
    Whether the engines' runs of one dataset and fold (and the retrained
    original's) saw identical test regions: a sha256 per run of
    model/auxiliary/filtered.peaks.bed and
    filtered.nonpeaks.bed (and of data_unstranded.bw with --check-bigwigs,
    which reads ~0.5 GB per run). A mismatch means the runs are not
    comparable.

timings.tsv
    Wall time of each step (timings.tsv in each finished run) with the GPU it
    ran on (provenance.txt). Runtimes compare only on the same GPU type.

epochs.tsv
    For each run's ChromBPNet and bias model, from its epoch log
    (model/logs/chrombpnet.log, or a distilled student's
    model/models/chrombpnet_nobias.log; bias_model/logs/<ENCID>_bias.log):
    the epochs trained and the epoch whose weights were kept, 0-based --
    the first with the lowest val_loss, or the highest val_selection_score
    for a log that has one (--model-selection pearsonr-jsd), or the last
    when nothing was validated (distill --all-chromosomes).

bias_response.tsv
    The bias-free model's marginal-footprint response to the assay's bias
    motifs (model/evaluation/chrombpnet_nobias_max_bias_response.txt):
    the pipeline's corrected/uncorrected call (every motif under 0.003),
    the mean and the largest response, and each motif's.

runs.tsv
    The runs' labels and trees, in order (plot_summarize.py reads it).

The headline, printed to stderr, gives each engine's mean over folds of the
test scores, each engine minus pytorch and minus the original (paired on
dataset and fold), the agreement with the original, and the contribution
agreement.
"""

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import os
import sys

ENGINES = ("pytorch", "tensorflow", "jax")
ORIGINAL = "original"
RETRAINED = "original_retrained"   # run_original_retrain.py
TF_METRICS = ("evaluation/chrombpnet_metrics.json", "chrombpnet_metrics.json")
TF_PREDICTIONS = ("evaluation/chrombpnet_predictions.h5", "chrombpnet_predictions.h5")
HEADS = ("counts", "profile")
KINDS = ("hypothetical", "projected")
SCORES_PREFIX = "model/auxiliary/interpret_subsample/chrombpnet_nobias"


def scores_rel(head, profile_tag=None):
    """A run's scores file for one head: the pipeline's, or for the profile
    head with a tag, run_contribution.py --profile-weighting <tag>'s."""
    if head == "profile" and profile_tag:
        return f"{SCORES_PREFIX}.{profile_tag}.profile_scores.h5"
    return f"{SCORES_PREFIX}.{head}_scores.h5"
BLOCK_ROWS = 31 * 64          # a whole number of the originals' 31-row chunks


def seed_dirs(fold_dir):
    """seed -> run_deepshap_seeds.py's directory, for every COMPLETED seed."""
    found = {}
    for name in (os.listdir(fold_dir) if os.path.isdir(fold_dir) else ()):
        path = os.path.join(fold_dir, name)
        if name.startswith("seed_") and not name.endswith(".partial") and \
                os.path.isfile(os.path.join(path, "COMPLETED")):
            found[int(name[len("seed_"):])] = path
    return dict(sorted(found.items()))


def load_rows(scores, regions, keys):
    """(raw, hyp, rows) for `keys`, from a scores file whose rows follow
    `regions`: rows[i] indexes raw/hyp for keys[i], or is None where the file
    lacks that region."""
    import numpy as np
    import h5py
    try:
        import hdf5plugin  # noqa: F401  (the original's writer compresses with Blosc)
    except ImportError:
        pass
    index = {}
    for i, k in enumerate(region_keys(regions)):
        index.setdefault(k, i)
    wanted = [index.get(k) for k in keys]
    needed = np.unique(np.asarray([w for w in wanted if w is not None], dtype=np.int64))
    with h5py.File(scores, "r") as f:
        raw = read_rows(f["raw/seq"], needed)
        hyp = read_rows(f["shap/seq"], needed)
    where = {j: n for n, j in enumerate(needed.tolist())}
    return raw, hyp, [where.get(w) if w is not None else None for w in wanted]


def pair_metrics(a, b, chunk=1024):
    """Per-region metrics of two load_rows sets over the regions both have
    with identical sequences: ({kind: (r, cos)}, n_compared, n_seq_mismatch)."""
    import numpy as np
    raw_a, hyp_a, rows_a = a
    raw_b, hyp_b, rows_b = b
    both = [(i, j) for i, j in zip(rows_a, rows_b) if i is not None and j is not None]
    out = {kind: ([], []) for kind in KINDS}
    mismatched = 0
    for lo in range(0, len(both), chunk):
        ia = np.asarray([i for i, _ in both[lo:lo + chunk]])
        ib = np.asarray([j for _, j in both[lo:lo + chunk]])
        ra, rb, ha, hb = raw_a[ia], raw_b[ib], hyp_a[ia], hyp_b[ib]
        same = (ra == rb).all(axis=(1, 2))
        mismatched += int((~same).sum())
        ra, rb, ha, hb = ra[same], rb[same], ha[same], hb[same]
        n = len(ha)
        if not n:
            continue
        for kind, (x, y) in {
            "hypothetical": (ha.reshape(n, -1), hb.reshape(n, -1)),
            "projected": ((ha.astype(np.float32) * ra).sum(1), (hb.astype(np.float32) * rb).sum(1)),
        }.items():
            r, cos = row_metrics(x, y)
            out[kind][0].append(r)
            out[kind][1].append(cos)
    return out, len(both) - mismatched, mismatched


def seeded_runs(job):
    """'original:seed<s>' and 'original_retrained:seed<s>' -> the directory
    of that COMPLETED seed, for one dataset and fold."""
    runs = {f"{ORIGINAL}:seed{s}": p for s, p in seed_dirs(job["fold_dir"]).items()}
    runs.update({f"{RETRAINED}:seed{s}": p for s, p in seed_dirs(job.get("retrained_dir") or "").items()})
    return runs


def compare_seeds(job):
    """The original's seeded DeepSHAP runs of one dataset and fold: on the
    ENCODE model, each against the stored scores and against each other; on
    the retrained model, against the stored scores and against the ENCODE
    model's run with the same seed. Returns rows (engine, reference, head,
    kind, n_engine, n_matched, n_seq_mismatch, r_median, r_mean, cos_median,
    cos_mean) and notes."""
    import itertools
    import numpy as np
    runs = seeded_runs(job)
    notes, rows = [], []
    if not runs:
        return rows, notes
    regions = {name: os.path.join(p, "scores.interpreted_regions.bed") for name, p in runs.items()}
    keys = region_keys(next(iter(regions.values())))
    for name in list(runs):
        if region_keys(regions[name]) != keys:
            notes.append(f"{job['encid']} fold {job['fold']}: {name} scored other regions; left out")
            del runs[name]
    for head in HEADS:
        sets = {name: load_rows(os.path.join(p, f"scores.{head}_scores.h5"), regions[name], keys)
                for name, p in runs.items() if os.path.isfile(os.path.join(p, f"scores.{head}_scores.h5"))}
        stored_dir = job["deeplift_dirs"].get(head) or ""
        stored = os.path.join(stored_dir, f"{job['encid']}.{head}_scores.h5")
        pairs = list(itertools.combinations([n for n in sets if n.startswith(ORIGINAL + ":")], 2))
        # The retrained model against the ENCODE model's run of its seed: the
        # same references, so the models alone differ.
        pairs += [(n, f"{ORIGINAL}:{n.split(':', 1)[1]}") for n in sets
                  if n.startswith(RETRAINED + ":") and f"{ORIGINAL}:{n.split(':', 1)[1]}" in sets]
        if os.path.isfile(stored):
            sets[ORIGINAL] = load_rows(stored, os.path.join(
                stored_dir, f"{job['encid']}.interpreted_regions.bed.gz"), keys)
            pairs = [(s, ORIGINAL) for s in sets if s != ORIGINAL] + pairs
        else:
            notes.append(f"{job['encid']} fold {job['fold']}: no stored {head} scores ({stored})")
        for a, b in pairs:
            metrics, n_compared, n_mismatch = pair_metrics(sets[a], sets[b])
            for kind in KINDS:
                r_list, cos_list = metrics[kind]
                r = np.concatenate(r_list) if r_list else np.full(1, np.nan)
                cos = np.concatenate(cos_list) if cos_list else np.full(1, np.nan)
                rows.append((a, b, head, kind, len(keys), n_compared, n_mismatch,
                             float(np.nanmedian(r)), float(np.nanmean(r)),
                             float(np.nanmedian(cos)), float(np.nanmean(cos))))
        del sets
    return rows, notes


def seeds_job(job, cache_dir):
    """compare_seeds, cached on the sizes and times of the files it reads."""
    inputs = []
    for path in seeded_runs(job).values():
        inputs += [os.path.join(path, n) for n in sorted(os.listdir(path))]
    for head in HEADS:
        p = os.path.join(job["deeplift_dirs"].get(head) or "", f"{job['encid']}.{head}_scores.h5")
        if os.path.isfile(p):
            inputs.append(p)
    key = {"version": 1, "inputs": signature(inputs)}
    cache = os.path.join(cache_dir, f"{job['assay']}_{job['encid']}_fold{job['fold']}.seeds.json")
    try:
        with open(cache) as fh:
            cached = json.load(fh)
        if cached.get("key") == key:
            return [tuple(r) for r in cached["rows"]], cached["notes"], True
    except (OSError, ValueError):
        pass
    rows, notes = compare_seeds(job)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache + ".partial", "w") as fh:
        json.dump({"key": key, "rows": rows, "notes": notes}, fh)
    os.replace(cache + ".partial", cache)
    return rows, notes, False


def compare_versus(job):
    """One dataset and fold: for each --versus pair, run LABEL's subsample
    scores against run REFERENCE's, regions matched by chromosome and summit
    and kept only if both sequences are identical. Returns rows (label,
    reference, head, kind, n_label, n_matched, n_seq_mismatch, r_median,
    r_mean, cos_median, cos_mean) and notes."""
    import numpy as np
    rows, notes = [], []
    for label, reference in job["versus"]:
        a, b = job["runs"][label], job["runs"][reference]
        if not all(os.path.isfile(os.path.join(r, "COMPLETED")) for r in (a, b)):
            continue
        regions_a, regions_b = (os.path.join(r, f"{SCORES_PREFIX}.interpreted_regions.bed") for r in (a, b))
        if not (os.path.isfile(regions_a) and os.path.isfile(regions_b)):
            notes.append(f"{job['encid']} fold {job['fold']}: {label} or {reference} has no scored regions")
            continue
        keys = region_keys(regions_a)
        for head in HEADS:
            scores_a, scores_b = (os.path.join(r, scores_rel(head, job.get("profile_tag"))) for r in (a, b))
            if not (os.path.isfile(scores_a) and os.path.isfile(scores_b)):
                notes.append(f"{job['encid']} fold {job['fold']}: no {head} scores to compare for "
                             f"{label} vs {reference}")
                continue
            metrics, n_compared, n_mismatch = pair_metrics(load_rows(scores_a, regions_a, keys),
                                                           load_rows(scores_b, regions_b, keys))
            for kind in KINDS:
                r_list, cos_list = metrics[kind]
                r = np.concatenate(r_list) if r_list else np.full(1, np.nan)
                cos = np.concatenate(cos_list) if cos_list else np.full(1, np.nan)
                rows.append((label, reference, head, kind, len(keys), n_compared, n_mismatch,
                             float(np.nanmedian(r)), float(np.nanmean(r)),
                             float(np.nanmedian(cos)), float(np.nanmean(cos))))
    return rows, notes


def versus_job(job, cache_dir):
    """compare_versus, cached on the sizes and times of the files it reads."""
    inputs = []
    for label, reference in job["versus"]:
        for run in (job["runs"][label], job["runs"][reference]):
            for rel in [scores_rel(h, job.get("profile_tag")) for h in HEADS] + ["COMPLETED"]:
                p = os.path.join(run, rel)
                if os.path.isfile(p):
                    inputs.append(p)
    key = {"version": 1, "pairs": [list(p) for p in job["versus"]], "inputs": signature(inputs)}
    tag = f".{job['profile_tag']}" if job.get("profile_tag") else ""
    cache = os.path.join(cache_dir, f"{job['assay']}_{job['encid']}_fold{job['fold']}{tag}.versus.json")
    try:
        with open(cache) as fh:
            cached = json.load(fh)
        if cached.get("key") == key:
            return [tuple(r) for r in cached["rows"]], cached["notes"], True
    except (OSError, ValueError):
        pass
    rows, notes = compare_versus(job)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache + ".partial", "w") as fh:
        json.dump({"key": key, "rows": rows, "notes": notes}, fh)
    os.replace(cache + ".partial", cache)
    return rows, notes, False


def load_biosamples(path):
    """ENCID -> short biosample label (cached ENCODE lookup); falls back to the ENCID."""
    table = {}
    try:
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            i_id, i_bs = header.index("ENCID"), header.index("BIOSAMPLE")
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) > max(i_id, i_bs) and f[i_bs]:
                    table[f[i_id]] = f[i_bs]
    except (OSError, ValueError):
        pass
    return table


def ids_rows(path):
    """(assay, encid, fold, tensorflow_dir, {head: deeplift dir}), one per ids row, by
    header name. DEEPLIFT_COUNTS_DIR / DEEPLIFT_PROFILE_DIR name each head's
    original scores; a table with only DEEPLIFT_DIR uses it for both."""
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        lines = [l.rstrip("\r\n") for l in fh if l.strip()]
    col = {name: i for i, name in enumerate(header)}

    def get(f, name):
        return f[col[name]] if name in col and len(f) > col[name] else ""

    rows = []
    for f in (line.split("\t") for line in lines):
        both = get(f, "DEEPLIFT_DIR")
        dirs = {head: get(f, f"DEEPLIFT_{head.upper()}_DIR") or both for head in HEADS}
        rows.append((get(f, "ASSAY"), get(f, "ENCID"), get(f, "FOLD_NUM"),
                     get(f, "TENSORFLOW_DIR"), dirs))
    return rows


def metric_rows(path, source, engine, assay, biosample, encid, fold):
    try:
        with open(path) as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return []
    out = []
    for group in ("counts_metrics", "profile_metrics"):
        for subset, vals in (m.get(group) or {}).items():
            for key, value in (vals or {}).items():
                if isinstance(value, (int, float)):
                    out.append((source, engine, assay, biosample, encid, fold,
                                group.split("_")[0], subset, key, f"{value:.6f}"))
    return out


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def first_file(directory, candidates):
    if not directory:
        return None
    return next((p for p in (os.path.join(directory, rel) for rel in candidates)
                 if os.path.isfile(p)), None)


# -- predictions ---------------------------------------------------------------

def read_predictions(path, attempts=3):
    """(keys, log counts, profiles), or None, with a note, if the file cannot
    be read: a run still writing it, or Oak's NFS returning a short read
    (seen: "truncated file: eof = 8388608"), which a retry usually clears."""
    import time
    import h5py
    import numpy as np
    for attempt in range(attempts):
        try:
            with h5py.File(path, "r") as f:
                chrom = [c.decode() if isinstance(c, bytes) else str(c) for c in f["coords/coords_chrom"][:]]
                keys = list(zip(chrom, f["coords/coords_center"][:].tolist()))
                return keys, np.asarray(f["predictions/logcounts"][:]), np.asarray(f["predictions/profs"][:])
        except OSError as error:
            if attempt + 1 == attempts:
                print(f"# agreement: cannot read {path}: {error}", file=sys.stderr)
                return None
            time.sleep(5 * (attempt + 1))


#: Offsets tried between two files' coords_center. Not every version of the
#: original package wrote the summit there: some atlas runs' files hold it
#: shifted by half the input window (1057 = 2114 / 2), which no region
#: survives matching on unless corrected.
CENTER_OFFSETS = (0, -1057, 1057, -500, 500)


def agreement(pred_a, pred_b):
    """(n_a, n_b, n_matched, offset, counts r, median JSD) between two loaded
    prediction files, matched on (chrom, center + offset) with the offset in
    CENTER_OFFSETS that matches the most regions."""
    import numpy as np
    from scipy.spatial.distance import jensenshannon
    keys_a, counts_a, profs_a = pred_a
    keys_b, counts_b, profs_b = pred_b
    index_b = {}
    for i, k in enumerate(keys_b):
        index_b.setdefault(k, i)
    best_pairs, best_offset = [], 0
    for offset in CENTER_OFFSETS:
        pairs = [(i, index_b[(c, x + offset)]) for i, (c, x) in enumerate(keys_a)
                 if (c, x + offset) in index_b]
        if len(pairs) > len(best_pairs):
            best_pairs, best_offset = pairs, offset
    if len(best_pairs) < 3:
        return len(keys_a), len(keys_b), len(best_pairs), best_offset, float("nan"), float("nan")
    ia, ib = map(np.asarray, zip(*best_pairs))
    r = float(np.corrcoef(counts_a[ia], counts_b[ib])[0, 1])
    jsd = float(np.nanmedian(jensenshannon(profs_a[ia], profs_b[ib], axis=1)))
    return len(keys_a), len(keys_b), len(best_pairs), best_offset, r, jsd


# -- contribution scores -------------------------------------------------------

def region_keys(path):
    """(chrom, summit position) per line of a narrowPeak-style BED (gzipped or not)."""
    opener = gzip.open if path.endswith(".gz") else open
    keys = []
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 10:
                keys.append((f[0], int(f[1]) + int(f[9])))
    return keys


def signature(paths):
    out = []
    for p in paths:
        st = os.stat(p)
        out.append([p, st.st_size, int(st.st_mtime)])
    return out


def row_metrics(a, b):
    """Per-row Pearson r and cosine similarity of two (n, m) float arrays."""
    import numpy as np
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (a * b).sum(1) / (np.sqrt((a * a).sum(1)) * np.sqrt((b * b).sum(1)))
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        r = (a * b).sum(1) / (np.sqrt((a * a).sum(1)) * np.sqrt((b * b).sum(1)))
    return r, cos


def read_rows(dataset, rows):
    """dataset[rows] for sorted unique row indices, read a block of chunks at a time."""
    import numpy as np
    out = np.empty((len(rows),) + dataset.shape[1:], dtype=dataset.dtype)
    rows = np.asarray(rows)
    done = 0
    for start in range(0, dataset.shape[0], BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, dataset.shape[0])
        lo, hi = np.searchsorted(rows, [start, stop])
        if hi > lo:
            block = dataset[start:stop]
            out[done:done + hi - lo] = block[rows[lo:hi] - start]
            done += hi - lo
    return out


def compare_contributions(job):
    """One dataset and fold: every engine's subsample scores against the original's.

    Returns rows (engine, head, kind, n_engine, n_matched, n_seq_mismatch,
    r_median, r_mean, cos_median, cos_mean) and notes.
    """
    import numpy as np
    import h5py
    try:
        import hdf5plugin  # noqa: F401  (the originals are Blosc-compressed)
    except ImportError:
        pass

    encid, dirs, runs = job["encid"], job["deeplift_dirs"], job["runs"]
    notes = []
    indices = {}            # one per directory: the heads often share one

    rows = []
    for head in HEADS:
        orig_dir = dirs.get(head) or ""
        orig_scores = os.path.join(orig_dir, f"{encid}.{head}_scores.h5")
        orig_regions = os.path.join(orig_dir, f"{encid}.interpreted_regions.bed.gz")
        if not os.path.isfile(orig_scores):
            notes.append(f"{encid} fold {job['fold']}: no original {head} scores ({orig_scores})")
            continue
        if not os.path.isfile(orig_regions):
            notes.append(f"{encid} fold {job['fold']}: no {orig_regions}")
            continue
        if orig_dir not in indices:
            index = {}
            for i, k in enumerate(region_keys(orig_regions)):
                index.setdefault(k, i)
            indices[orig_dir] = index
        index = indices[orig_dir]
        # Which original rows each engine needs, then each original row read once.
        wanted = {}
        for engine, run in runs.items():
            if not os.path.isfile(os.path.join(run, "COMPLETED")):
                continue
            scores = os.path.join(run, scores_rel(head, job.get("profile_tag")))
            regions = os.path.join(run, f"{SCORES_PREFIX}.interpreted_regions.bed")
            if not (os.path.isfile(scores) and os.path.isfile(regions)):
                continue
            keys = region_keys(regions)
            pairs = [(i, index[k]) for i, k in enumerate(keys) if k in index]
            wanted[engine] = (scores, len(keys), pairs)
        if not wanted:
            continue
        needed = np.unique(np.concatenate([np.asarray([j for _, j in p], dtype=np.int64)
                                           for _, _, p in wanted.values()] or [np.zeros(0, np.int64)]))
        with h5py.File(orig_scores, "r") as f:
            o_raw = read_rows(f["raw/seq"], needed)
            o_hyp = read_rows(f["shap/seq"], needed)
        where = {j: n for n, j in enumerate(needed.tolist())}
        for engine, (scores, n_engine, pairs) in wanted.items():
            if not pairs:
                rows.append((engine, head, "hypothetical", n_engine, 0, 0) + (float("nan"),) * 4)
                rows.append((engine, head, "projected", n_engine, 0, 0) + (float("nan"),) * 4)
                continue
            mine = np.asarray([i for i, _ in pairs])
            theirs = np.asarray([where[j] for _, j in pairs])
            with h5py.File(scores, "r") as f:
                e_raw_all = f["raw/seq"][:]
                e_hyp_all = f["shap/seq"][:]
            metrics = {kind: ([], []) for kind in KINDS}
            mismatched = 0
            # In slices: float64 copies of 30K x 8456 scores would be ~2 GB each.
            for lo in range(0, len(mine), 2048):
                m, t = mine[lo:lo + 2048], theirs[lo:lo + 2048]
                e_raw, e_hyp = e_raw_all[m], e_hyp_all[m]
                t_raw, t_hyp = o_raw[t], o_hyp[t]
                same = (e_raw == t_raw).all(axis=(1, 2))
                mismatched += int((~same).sum())
                e_raw, e_hyp, t_raw, t_hyp = e_raw[same], e_hyp[same], t_raw[same], t_hyp[same]
                n = len(e_hyp)
                if not n:
                    continue
                pairs_by_kind = {
                    "hypothetical": (e_hyp.reshape(n, -1), t_hyp.reshape(n, -1)),
                    "projected": ((e_hyp.astype(np.float32) * e_raw).sum(1),
                                  (t_hyp.astype(np.float32) * t_raw).sum(1)),
                }
                for kind, (a, b) in pairs_by_kind.items():
                    r, cos = row_metrics(a, b)
                    metrics[kind][0].append(r)
                    metrics[kind][1].append(cos)
            del e_raw_all, e_hyp_all
            for kind in KINDS:
                rs, coss = metrics[kind]
                r = np.concatenate(rs) if rs else np.full(1, np.nan)
                cos = np.concatenate(coss) if coss else np.full(1, np.nan)
                rows.append((engine, head, kind, n_engine, len(pairs) - mismatched, mismatched,
                             float(np.nanmedian(r)), float(np.nanmean(r)),
                             float(np.nanmedian(cos)), float(np.nanmean(cos))))
    return rows, notes


def contributions_job(job, cache_dir):
    """compare_contributions with a cache keyed by every input file's size and time."""
    inputs = []
    for head in HEADS:
        p = os.path.join(job["deeplift_dirs"].get(head) or "", f"{job['encid']}.{head}_scores.h5")
        if os.path.isfile(p):
            inputs.append(p)
    for engine in sorted(job["runs"]):
        # COMPLETED too: only finished runs are compared, and a run's scores
        # exist before it finishes.
        for rel in [scores_rel(h, job.get("profile_tag")) for h in HEADS] + ["COMPLETED"]:
            p = os.path.join(job["runs"][engine], rel)
            if os.path.isfile(p):
                inputs.append(p)
    key = {"version": 2, "inputs": signature(inputs)}
    tag = f".{job['profile_tag']}" if job.get("profile_tag") else ""
    cache = os.path.join(cache_dir, f"{job['assay']}_{job['encid']}_fold{job['fold']}{tag}.json")
    try:
        with open(cache) as fh:
            cached = json.load(fh)
        if cached.get("key") == key:
            return [tuple(r) for r in cached["rows"]], cached["notes"], True
    except (OSError, ValueError):
        pass
    rows, notes = compare_contributions(job)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache + ".partial", "w") as fh:
        json.dump({"key": key, "rows": rows, "notes": notes}, fh)
    os.replace(cache + ".partial", cache)
    return rows, notes, False


# -- the rest ------------------------------------------------------------------

def read_timings(run_dir):
    gpu = ""
    try:
        with open(os.path.join(run_dir, "provenance.txt")) as fh:
            for line in fh:
                if line.startswith("gpu:"):
                    gpu = line.split(":", 1)[1].strip().split(",")[0]
    except OSError:
        pass
    steps = []
    try:
        with open(os.path.join(run_dir, "timings.tsv")) as fh:
            next(fh)
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 2:
                    steps.append((f[0], int(f[1])))
    except (OSError, StopIteration, ValueError):
        pass
    return gpu, steps


#: Each model's epoch log, relative to a run: the pipeline's, then that of
#: a distilled student (chrombpnet distill -op model/models/chrombpnet_nobias).
EPOCH_LOGS = {"chrombpnet": ("model/logs/chrombpnet.log", "model/models/chrombpnet_nobias.log"),
              "bias": ("bias_model/logs/{encid}_bias.log",)}


def read_epochs(path):
    """(epochs trained, epoch kept, how it was chosen) from an epoch log, or
    None. The epoch kept is the one the training loop keeps: the first with
    the lowest val_loss, or the first with the highest val_selection_score
    (--model-selection pearsonr-jsd), or the last when no epoch has a
    validation loss (distill --all-chromosomes)."""
    import csv
    import math
    try:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows:
        return None

    def number(row, key):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    if "val_selection_score" in rows[0]:
        finite = [(s, i) for i, s in enumerate(number(r, "val_selection_score") for r in rows)
                  if math.isfinite(s)]
        best = max(finite, key=lambda t: (t[0], -t[1]))[1] if finite else ""
        return len(rows), best, "pearsonr-jsd"
    finite = [(v, i) for i, v in enumerate(number(r, "val_loss") for r in rows) if math.isfinite(v)]
    if finite:
        return len(rows), min(finite)[1], "loss"
    return len(rows), len(rows) - 1, "last"


def read_bias_response(path):
    """(call, mean, largest, per-motif values) from marginal_footprinting's
    <prefix>_max_bias_response.txt ("corrected_0.001_0.002/0.001/..."), or None."""
    try:
        with open(path) as fh:
            call, mean, values = fh.read().strip().split("_", 2)
        values = [float(v) for v in values.split("/") if v]
        return call, float(mean), max(values), values
    except (OSError, ValueError):
        return None


def parse_runs(items, root):
    """label -> run tree: --runs LABEL=DIR ..., or <root>/<engine> per engine."""
    if not items:
        return {engine: os.path.join(root, engine) for engine in ENGINES}
    runs = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not (sep and label and path):
            sys.exit(f"--runs takes LABEL=DIR, got {item!r}")
        if label in (ORIGINAL, RETRAINED) or ":" in label:
            sys.exit(f"--runs label {label!r} is reserved")
        runs[label] = path
    return runs


def parse_versus(items, labels):
    """[(label, reference)] from --versus LABEL=REFERENCE ..., both --runs labels."""
    pairs = []
    for item in items or ():
        label, sep, reference = item.partition("=")
        if not sep or label not in labels or reference not in labels or label == reference:
            sys.exit(f"--versus takes LABEL=REFERENCE, two different --runs labels "
                     f"({', '.join(labels)}); got {item!r}")
        pairs.append((label, reference))
    return pairs


def write_tsv(path, header, rows):
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"# {len(rows):>6} rows -> {path}", file=sys.stderr)


def mean(values):
    return sum(values) / len(values)


def fmt(x):
    return f"{x:.6f}" if isinstance(x, float) else str(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the sweep's run tree")
    ap.add_argument("--ids", required=True,
                    help="ids table (ASSAY, ENCID, FOLD_NUM, TENSORFLOW_DIR, "
                         "DEEPLIFT_COUNTS_DIR and DEEPLIFT_PROFILE_DIR, or DEEPLIFT_DIR)")
    ap.add_argument("--biosamples", default=None,
                    help="ENCID -> biosample label table (ENCID, BIOSAMPLE columns; default: label by ENCID)")
    ap.add_argument("--out-dir", required=True, help="where the TSVs go")
    ap.add_argument("--cache-dir", default=None,
                    help="contribution comparisons are cached here (default: <out-dir>/.contributions_cache)")
    ap.add_argument("--check-bigwigs", action="store_true",
                    help="also hash each run's data_unstranded.bw (slow)")
    ap.add_argument("--no-agreement", action="store_true",
                    help="skip reading the prediction files")
    ap.add_argument("--no-contributions", action="store_true",
                    help="skip the contribution-score comparison")
    ap.add_argument("--profile-stopgrad", choices=("off", "on"), default="off",
                    help="the engines' profile scores to compare: off (default), the original's weighting "
                         "(chrombpnet_nobias.midpoint.profile_scores.h5, run_contribution.py "
                         "--profile-weighting midpoint); on, the pipeline's own chrombpnet_nobias."
                         "profile_scores.h5, which in this sweep honour the stop-gradient")
    # The name before --profile-stopgrad: chrombpnet_nobias.<tag>.profile_scores.h5.
    ap.add_argument("--profile-tag", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seeds-root", default=None,
                    help="run_deepshap_seeds.py's output: the original's DeepSHAP with several seeds "
                         "(default: <root>/original)")
    ap.add_argument("--retrained-root", default=None,
                    help="run_original_retrain.py's output: the original retrained on the sweep's inputs "
                         "(default: <root>/original_retrained)")
    ap.add_argument("--workers", type=int, default=4,
                    help="datasets and folds compared at once (default 4; each holds ~3 GB)")
    ap.add_argument("--runs", nargs="+", default=None, metavar="LABEL=DIR",
                    help="the runs to collate, by label, each a tree of <ASSAY>/<ENCID>/fold_N "
                         "(default: <root>/<engine> for pytorch, tensorflow and jax). The first label "
                         "is the headline's reference")
    ap.add_argument("--versus", nargs="+", default=[], metavar="LABEL=REFERENCE",
                    help="also compare run LABEL's test predictions and subsample DeepSHAP scores with "
                         "run REFERENCE's (two --runs labels), dataset and fold for dataset and fold")
    args = ap.parse_args(argv)
    # Which scores files the profile head is read from (the cache files are
    # named by it), and the label the rows carry.
    if args.profile_tag is None:
        args.profile_tag = {"off": "midpoint", "on": None}[args.profile_stopgrad]
    profile_label = {"midpoint": "stopgrad-off", None: "stopgrad-on"}.get(args.profile_tag, args.profile_tag)
    root = args.root
    seeds_root = args.seeds_root or os.path.join(root, "original")
    retrained_root = args.retrained_root or os.path.join(root, RETRAINED)
    cache_dir = args.cache_dir or os.path.join(args.out_dir, ".contributions_cache")
    biosamples = load_biosamples(args.biosamples) if args.biosamples else {}
    rows = ids_rows(args.ids)
    trees = parse_runs(args.runs, root)
    labels = tuple(trees)
    versus = parse_versus(args.versus, labels)

    results, timings, agree, inputs, contribs, jobs = [], [], [], [], [], []
    epochs, responses = [], []
    for assay, encid, fold, tf_dir, deeplift_dirs in rows:
        biosample = biosamples.get(encid, encid)
        runs = {e: os.path.join(tree, assay, encid, f"fold_{fold}") for e, tree in trees.items()}
        for engine, run in runs.items():
            results += metric_rows(os.path.join(run, "model/evaluation/chrombpnet_metrics.json"),
                                   "own", engine, assay, biosample, encid, fold)
            results += metric_rows(os.path.join(run, f"bias_model/evaluation/{encid}_bias_metrics.json"),
                                   "bias", engine, assay, biosample, encid, fold)
            for model, candidates in EPOCH_LOGS.items():
                found = next((read_epochs(p) for p in (os.path.join(run, c.format(encid=encid))
                                                       for c in candidates) if os.path.isfile(p)), None)
                if found:
                    epochs.append((engine, assay, biosample, encid, fold, model) + found)
            response = read_bias_response(os.path.join(
                run, "model/evaluation/chrombpnet_nobias_max_bias_response.txt"))
            if response:
                call, mean_response, largest, values = response
                responses.append((engine, assay, biosample, encid, fold, call, f"{mean_response:.4f}",
                                  f"{largest:.4f}", "/".join(f"{v:g}" for v in values)))
            # Finished runs only: a failed or running one has a partial
            # timings.tsv, whose total would pass for a fast run.
            if os.path.isfile(os.path.join(run, "COMPLETED")):
                gpu, steps = read_timings(run)
                timings += [(engine, assay, biosample, encid, fold, step, seconds, gpu)
                            for step, seconds in steps]
        retrained = os.path.join(retrained_root, assay, encid, f"fold_{fold}")
        results += metric_rows(os.path.join(retrained, "model/evaluation/chrombpnet_metrics.json"),
                               "own", RETRAINED, assay, biosample, encid, fold)
        if tf_dir:
            path = first_file(tf_dir, TF_METRICS)
            if path is None:
                print(f"# {ORIGINAL}: no chrombpnet_metrics.json for {assay} {encid} fold {fold} "
                      f"under {tf_dir}", file=sys.stderr)
            else:
                results += metric_rows(path, "own", ORIGINAL, assay, biosample, encid, fold)

        files = ["model/auxiliary/filtered.peaks.bed", "model/auxiliary/filtered.nonpeaks.bed"]
        if args.check_bigwigs:
            files.append("model/auxiliary/data_unstranded.bw")
        for rel in files:
            digests = {e: sha256(os.path.join(run, rel)) for e, run in dict(runs, **{RETRAINED: retrained}).items()
                       if os.path.isfile(os.path.join(run, rel))}
            if len(digests) >= 2:
                identical = "yes" if len(set(digests.values())) == 1 else "NO"
                inputs.append((assay, biosample, encid, fold, os.path.basename(rel),
                               ",".join(sorted(digests)), identical,
                               ",".join(f"{e}:{d[:12]}" for e, d in sorted(digests.items()))))

        if not args.no_agreement:
            reference = first_file(tf_dir, TF_PREDICTIONS)
            if reference is None:
                print(f"# agreement: no original predictions for {assay} {encid} fold {fold}"
                      f"{' under ' + tf_dir if tf_dir else ' (no TENSORFLOW_DIR)'}", file=sys.stderr)
            else:
                original = read_predictions(reference)
                for engine, run in dict(runs, **{RETRAINED: retrained}).items():
                    path = os.path.join(run, "model/evaluation/chrombpnet_predictions.h5")
                    # Only finished runs: a running one may be writing the file.
                    if original is None or not os.path.isfile(os.path.join(run, "COMPLETED")):
                        continue
                    mine = read_predictions(path) if os.path.isfile(path) else None
                    if mine is not None:
                        n_e, n_o, n_matched, offset, r, jsd = agreement(mine, original)
                        agree.append((engine, ORIGINAL, assay, biosample, encid, fold, n_e, n_o,
                                      n_matched, offset, f"{r:.6f}", f"{jsd:.6f}"))
                del original
            # --versus: one run against another, both finished.
            for label, reference in versus:
                paths = [os.path.join(runs[e], "model/evaluation/chrombpnet_predictions.h5")
                         for e in (label, reference)]
                if not all(os.path.isfile(os.path.join(runs[e], "COMPLETED")) for e in (label, reference)) \
                        or not all(os.path.isfile(p) for p in paths):
                    continue
                mine, theirs = read_predictions(paths[0]), read_predictions(paths[1])
                if mine is not None and theirs is not None:
                    n_e, n_o, n_matched, offset, r, jsd = agreement(mine, theirs)
                    agree.append((label, reference, assay, biosample, encid, fold, n_e, n_o,
                                  n_matched, offset, f"{r:.6f}", f"{jsd:.6f}"))
                del mine, theirs

        if not args.no_contributions and versus:
            jobs.append({"assay": assay, "biosample": biosample, "encid": encid, "fold": fold,
                         "runs": runs, "versus": versus, "profile_tag": args.profile_tag})
        if not args.no_contributions:
            if any(d and os.path.isdir(d) for d in deeplift_dirs.values()):
                jobs.append({"assay": assay, "biosample": biosample, "encid": encid, "fold": fold,
                             "deeplift_dirs": deeplift_dirs, "runs": runs,
                             "profile_tag": args.profile_tag})
                seeds_job_spec = {"assay": assay, "biosample": biosample, "encid": encid,
                                  "fold": fold, "deeplift_dirs": deeplift_dirs,
                                  "fold_dir": os.path.join(seeds_root, assay, encid, f"fold_{fold}"),
                                  "retrained_dir": retrained, "seeds": True}
                if seeded_runs(seeds_job_spec):
                    jobs.append(seeds_job_spec)
            else:
                print(f"# contributions: no original DeepLIFT directory for {assay} {encid} "
                      f"fold {fold}", file=sys.stderr)

    if jobs:
        print(f"# contributions: {len(jobs)} datasets x folds, {args.workers} at a time "
              f"(cache: {cache_dir})", file=sys.stderr)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(seeds_job if job.get("seeds") else
                                   versus_job if job.get("versus") else contributions_job, job, cache_dir): job
                       for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    found, notes, cached = future.result()
                except Exception as error:  # noqa: BLE001 -- report and go on
                    print(f"#   {job['encid']} fold {job['fold']}: FAILED: {type(error).__name__}: {error}",
                          file=sys.stderr)
                    continue
                for note in notes:
                    print(f"#   {note}", file=sys.stderr)
                kind_of = "seeded-original " if job.get("seeds") else "run-vs-run " if job.get("versus") else ""
                print(f"#   {job['encid']} fold {job['fold']}: {len(found)} {kind_of}comparisons"
                      f"{' (cached)' if cached else ''}", file=sys.stderr)
                if job.get("versus"):
                    for r in found:
                        engine, reference, head, kind = r[:4]
                        source = profile_label if head == "profile" else "pipeline"
                        contribs.append((engine, reference, job["assay"], job["biosample"],
                                         job["encid"], job["fold"], head, kind)
                                        + tuple(fmt(x) for x in r[4:]) + (source,))
                    continue
                if job.get("seeds"):
                    for r in found:
                        engine, reference, head, kind = r[:4]
                        contribs.append((engine, reference, job["assay"], job["biosample"],
                                         job["encid"], job["fold"], head, kind)
                                        + tuple(fmt(x) for x in r[4:]) + (ORIGINAL,))
                    continue
                for r in found:
                    engine, head, kind = r[:3]
                    source = profile_label if head == "profile" else "pipeline"
                    contribs.append((engine, ORIGINAL, job["assay"], job["biosample"], job["encid"],
                                     job["fold"], head, kind) + tuple(fmt(x) for x in r[3:])
                                    + (source,))
        contribs.sort()

    results.sort()
    agree.sort()
    os.makedirs(args.out_dir, exist_ok=True)
    write_tsv(os.path.join(args.out_dir, "results.tsv"),
              ("source", "engine", "assay", "biosample", "encid", "fold",
               "group", "subset", "metric", "value"), results)
    write_tsv(os.path.join(args.out_dir, "agreement.tsv"),
              ("engine", "reference", "assay", "biosample", "encid", "fold",
               "n_engine", "n_reference", "n_matched", "center_offset", "counts_pearsonr",
               "profile_median_jsd"), agree)
    if not args.no_contributions:
        write_tsv(os.path.join(args.out_dir, "contributions.tsv"),
                  ("engine", "reference", "assay", "biosample", "encid", "fold", "head", "kind",
                   "n_engine", "n_matched", "n_seq_mismatch",
                   "pearsonr_median", "pearsonr_mean", "cosine_median", "cosine_mean",
                   "profile_scores"), contribs)
    write_tsv(os.path.join(args.out_dir, "inputs.tsv"),
              ("assay", "biosample", "encid", "fold", "file", "engines", "identical", "sha256"), inputs)
    write_tsv(os.path.join(args.out_dir, "timings.tsv"),
              ("engine", "assay", "biosample", "encid", "fold", "step", "seconds", "gpu"), timings)
    write_tsv(os.path.join(args.out_dir, "epochs.tsv"),
              ("engine", "assay", "biosample", "encid", "fold", "model", "epochs", "best_epoch",
               "selection"), sorted(epochs))
    write_tsv(os.path.join(args.out_dir, "bias_response.tsv"),
              ("engine", "assay", "biosample", "encid", "fold", "call", "mean", "max", "values"),
              sorted(responses))
    write_tsv(os.path.join(args.out_dir, "runs.tsv"), ("order", "engine", "tree"),
              [(i, label, tree) for i, (label, tree) in enumerate(trees.items())])

    headline(results, agree, contribs, inputs, timings, labels, versus)
    return 0


def headline(results, agree, contribs, inputs, timings, labels=ENGINES, versus=()):
    err = sys.stderr
    own = {}
    for source, engine, assay, biosample, encid, fold, group, subset, key, value in results:
        if source == "own" and subset == "peaks" and key in ("pearsonr", "median_jsd"):
            own[(engine, assay, biosample, encid, fold, key)] = float(value)

    print("\n# test scores (own models, peaks subset, mean over folds)", file=err)
    print(f"#   {'engine':<18} {'assay':<6} {'biosample':<13} {'counts r':>9} {'prof JSD':>9}  n", file=err)
    agg = collections.defaultdict(list)
    for (engine, assay, biosample, encid, fold, key), value in own.items():
        agg[(engine, assay, biosample, key)].append(value)
    order = {e: i for i, e in enumerate((ORIGINAL, RETRAINED) + tuple(labels))}
    for engine, assay, biosample in sorted({k[:3] for k in agg}, key=lambda k: (k[2], k[1], order.get(k[0], 9))):
        r = agg.get((engine, assay, biosample, "pearsonr"), [])
        j = agg.get((engine, assay, biosample, "median_jsd"), [])
        print(f"#   {engine:<18} {assay:<6} {biosample:<13} "
              f"{mean(r) if r else float('nan'):>9.4f} {mean(j) if j else float('nan'):>9.4f}  {len(r)}",
              file=err)

    for reference in (labels[0], ORIGINAL):
        delta = collections.defaultdict(list)
        for (engine, assay, biosample, encid, fold, key), value in own.items():
            ref = own.get((reference, assay, biosample, encid, fold, key))
            if engine != reference and ref is not None:
                delta[(engine, assay, biosample, key)].append(value - ref)
        print(f"\n# minus {reference}, paired on dataset and fold (pearsonr: + is better; "
              f"median_jsd: - is better)", file=err)
        if not delta:
            print(f"#   (no dataset and fold has both an engine and {reference})", file=err)
        for k in sorted(delta):
            engine, assay, biosample, key = k
            print(f"#   {engine:<18} {assay:<6} {biosample:<13} {key:<11} "
                  f"{mean(delta[k]):+.4f}  (n={len(delta[k])})", file=err)

    print(f"\n# agreement of each engine's test predictions with the {ORIGINAL} model's "
          f"(mean over datasets and folds)", file=err)
    by_engine = collections.defaultdict(list)
    for engine, ref, assay, biosample, encid, fold, n_e, n_o, n_matched, _offset, r, jsd in agree:
        name = engine if ref == ORIGINAL else f"{engine} vs {ref}"
        by_engine[name].append((float(r), float(jsd), int(n_matched), int(n_e), int(n_o)))
    if not by_engine:
        print("#   (no engine has predictions to compare yet)", file=err)
    for engine in tuple(labels) + (RETRAINED,) + tuple(f"{a} vs {b}" for a, b in versus):
        vals = by_engine.get(engine)
        if vals:
            print(f"#   {engine:<18} counts r {mean([v[0] for v in vals]):.4f}   "
                  f"profile JSD {mean([v[1] for v in vals]):.4f}   "
                  f"(n={len(vals)}; matched {mean([v[2] / max(1, v[3]) for v in vals]):.1%} of its regions)",
                  file=err)

    if contribs:
        print(f"\n# contribution scores vs the {ORIGINAL} model, 30K subsample "
              f"(mean over datasets and folds of the per-region median)", file=err)
        print(f"#   {'engine':<31} {'head':<8} {'kind':<13} {'pearson r':>9} {'cosine':>9}  n", file=err)
        agg = collections.defaultdict(list)
        for row in contribs:
            engine, ref, assay, biosample, encid, fold, head, kind = row[:8]
            r_med, cos_med = float(row[11]), float(row[13])
            agg[(engine if ref == ORIGINAL else f"{engine} vs {ref}", head, kind)].append((r_med, cos_med))
        for engine in tuple(labels) + tuple(f"{a} vs {b}" for a, b in versus):
            for head in HEADS:
                for kind in KINDS:
                    vals = agg.get((engine, head, kind))
                    if vals:
                        print(f"#   {engine:<31} {head:<8} {kind:<13} {mean([v[0] for v in vals]):>9.4f} "
                              f"{mean([v[1] for v in vals]):>9.4f}  {len(vals)}", file=err)
        # The original's seeded runs (run_deepshap_seeds.py) and the
        # retrained original's (run_original_retrain.py), if any.
        seeded = collections.defaultdict(list)
        for row in contribs:
            engine, ref, head, kind = str(row[0]), str(row[1]), row[6], row[7]
            source = engine.split(":", 1)[0]
            if source in (ORIGINAL, RETRAINED) and ":" in engine:
                versus = "same seed" if source == RETRAINED else "itself"
                label = f"{source} vs {versus if ref.startswith(ORIGINAL + ':') else 'stored'}"
                seeded[(label, head, kind)].append((float(row[11]), float(row[13])))
        labels = (f"{ORIGINAL} vs itself", f"{ORIGINAL} vs stored",
                  f"{RETRAINED} vs stored", f"{RETRAINED} vs same seed")
        if any(label in labels[2:] for label, _h, _k in seeded):
            print(f"#   ({RETRAINED} vs same seed: against the {ORIGINAL} model's run with the same "
                  f"references)", file=err)
        for label in labels:
            for head in HEADS:
                for kind in KINDS:
                    vals = seeded.get((label, head, kind))
                    if vals:
                        print(f"#   {label:<31} {head:<8} {kind:<13} "
                              f"{mean([v[0] for v in vals]):>9.4f} {mean([v[1] for v in vals]):>9.4f}"
                              f"  {len(vals)}", file=err)

    bad = [r for r in inputs if r[6] != "yes"]
    print(f"\n# identical test regions across engines: {len(inputs) - len(bad)} of {len(inputs)} "
          f"file comparisons", file=err)
    for r in bad:
        print(f"#   DIFFERENT: {r[2]} fold {r[3]} {r[4]}  {r[7]}", file=err)

    print("\n# wall time per run, hours (median over datasets and folds)", file=err)
    totals = collections.defaultdict(float)
    gpus = collections.defaultdict(set)
    for engine, assay, biosample, encid, fold, step, seconds, gpu in timings:
        totals[(engine, encid, fold)] += seconds / 3600
        gpus[engine].add(gpu or "?")
    for engine in labels:
        hours = sorted(v for (e, _, _), v in totals.items() if e == engine)
        if hours:
            print(f"#   {engine:<11} {hours[len(hours) // 2]:6.1f} h  (n={len(hours)}; "
                  f"GPUs: {', '.join(sorted(gpus[engine]))})", file=err)


if __name__ == "__main__":
    sys.exit(main())
