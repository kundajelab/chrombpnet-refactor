#!/usr/bin/env python
"""Score one original model with the original chrombpnet's DeepSHAP, once per seed.

    python run_deepshap_seeds.py --original-repo <kundajelab/chrombpnet checkout> \
        --model <original chrombpnet_nobias.h5> --regions-source <regions.bed> \
        --genome <genome.fa> --out <root>/original/<ASSAY>/<ENCID>/fold_N [--seeds 1 2 3]

In the original package's own environment (TensorFlow 2.8 on Python 3.8, the
kundajelab shap fork, deeplift), on a GPU.

How far apart can two DeepSHAP runs on the same model be? DeepSHAP averages
over dinucleotide-shuffled references, and the original draws them unseeded,
so two runs of it on one model differ by the references alone. That spread is
the ceiling of every comparison of contribution scores, and the scale on which
the engines' disagreement with the original's scores is read (run_summarize.py
adds these runs to contributions.tsv; plot_summarize.py draws them).

What runs: kundajelab/chrombpnet itself (--original-repo, at 09938fd), its
chrombpnet.evaluation.interpret.interpret.main imported from that checkout and
not from chrombpnet-refactor, with the kundajelab shap fork -- so the profile
target's weights are the original's midpoint ones. One change: the original draws each reference with
deeplift's dinuc_shuffle and no random state; here shap_utils.dinuc_shuffle is
given rng=np.random.RandomState(seed), fresh for each head, as
tests/parity/tf_engine.py does to compare it with the TensorFlow engine bit
for bit. Different seeds give different references; nothing else changes.
Each seed runs in its own process, because the original disables eager
execution when it is imported.

Model: the original model whose genome-wide scores run_summarize.py compares
against -- for the ENCODE models, the released
model.chrombpnet_nobias.fold_N.<ENCID>.h5 of that dataset and fold.

Regions: the engines' 30K peak subsample (--regions-source: a run's
model/auxiliary/interpret_subsample/chrombpnet_nobias.interpreted_regions.bed;
the three engines' are identical), all of it unless --n-regions takes the first N lines,
written once to <out>/regions.bed so every seed scores the same ones. Seeds can
run as separate jobs at the same time.

Output, under --out (<root>/original/<ASSAY>/<ENCID>/fold_N):
  regions.bed
  seed_<s>/scores.counts_scores.h5, scores.profile_scores.h5
      the original's own writer: raw, shap (hypothetical), projected_shap,
      Blosc-compressed, one row per line of scores.interpreted_regions.bed
  seed_<s>/scores.interpret.args.json, run.json, COMPLETED
A seed is written under seed_<s>.partial and renamed when every head is done;
seeds already COMPLETED are skipped (--force rescores them).
"""

import argparse
import datetime
import functools
import json
import os
import shutil
import socket
import subprocess
import sys

HEADS = ("counts", "profile")


def worker(args):
    """One seed, in a fresh process: the original code, seeded references."""
    # The original first, and no other chrombpnet: the env's editable install
    # of chrombpnet-refactor puts that checkout on sys.path too.
    original = os.path.realpath(args.original_repo)
    sys.path[:] = [original] + [
        p for p in sys.path
        if p and os.path.realpath(p) != original
        and not os.path.isfile(os.path.join(p, "chrombpnet", "__init__.py"))]
    import chrombpnet
    where = os.path.dirname(os.path.dirname(os.path.realpath(chrombpnet.__file__)))
    if where != os.path.realpath(args.original_repo):
        sys.exit(f"Error: imported chrombpnet from {where}, not the original at {args.original_repo}")
    import numpy as np
    from deeplift.dinuc_shuffle import dinuc_shuffle
    from chrombpnet.evaluation.interpret import interpret, shap_utils

    for head in args.heads:
        # The original's only source of randomness, seeded: a fresh stream per
        # head, so both heads explain against the same references.
        shap_utils.dinuc_shuffle = functools.partial(
            dinuc_shuffle, rng=np.random.RandomState(args.seed))
        interpret.main(argparse.Namespace(
            genome=args.genome, regions=args.regions, model_h5=args.model,
            output_prefix=os.path.join(args.seed_dir, "scores"), debug_chr=None,
            profile_or_counts=[head]))


def git_head(repo):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    # An old git (1.8, no -C): read .git itself.
    gitdir = os.path.join(repo, ".git")
    try:
        head = open(os.path.join(gitdir, "HEAD")).read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        if os.path.isfile(os.path.join(gitdir, ref)):
            return open(os.path.join(gitdir, ref)).read().strip()
        for line in open(os.path.join(gitdir, "packed-refs")):
            if line.rstrip().endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def write_regions(source, path, n, force):
    """Copy the first n lines of source (all of them for n <= 0) to path,
    or check that path already holds exactly those."""
    with open(source) as fh:
        lines = fh.readlines()
    if n > 0:
        lines = lines[:n]
    if os.path.isfile(path):
        with open(path) as fh:
            if fh.read() == "".join(lines):
                return len(lines)
        if not force:
            sys.exit(f"Error: {path} holds other regions (another --n-regions?); "
                     "remove it, or pass --force to rescore every seed")
    # Seeds of one row may start together: each writes its own partial file,
    # and the identical copies replace one another atomically.
    partial = f"{path}.partial.{os.getpid()}"
    with open(partial, "w") as fh:
        fh.writelines(lines)
    os.replace(partial, path)
    return len(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the original no-bias model (.h5)")
    ap.add_argument("--regions-source", required=True,
                    help="regions to take the first --n-regions lines of (10-column BED)")
    ap.add_argument("--n-regions", type=int, default=0,
                    help="score the first N regions only (default 0: all of them)")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True, help="<root>/original/<ASSAY>/<ENCID>/fold_N")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    ap.add_argument("--original-repo", required=True,
                    help="a checkout of kundajelab/chrombpnet (the original package)")
    ap.add_argument("--force", action="store_true", help="rescore seeds that are COMPLETED")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--seed-dir", help=argparse.SUPPRESS)
    ap.add_argument("--regions", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args)

    for label, path in (("model", args.model), ("regions source", args.regions_source),
                        ("genome", args.genome),
                        ("original repo", os.path.join(args.original_repo, "chrombpnet"))):
        if not os.path.exists(path):
            sys.exit(f"Error: {label} not found: {path}")
    os.makedirs(args.out, exist_ok=True)
    regions = os.path.join(args.out, "regions.bed")
    n = write_regions(args.regions_source, regions, args.n_regions, args.force)
    commit = git_head(args.original_repo)
    print(f"model:    {args.model}")
    print(f"regions:  {regions} ({n} of {args.regions_source})")
    print(f"original: {args.original_repo} ({commit or 'commit unknown'})")
    print(f"seeds:    {' '.join(map(str, args.seeds))}; heads: {' '.join(args.heads)}")

    for seed in args.seeds:
        final = os.path.join(args.out, f"seed_{seed}")
        if os.path.isfile(os.path.join(final, "COMPLETED")) and not args.force:
            print(f"have: {final} -- skipped (--force rescores it)")
            continue
        partial = final + ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        os.makedirs(partial)
        started = datetime.datetime.now()
        print(f"seed {seed}: scoring ...", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                        "--seed", str(seed), "--seed-dir", partial, "--regions", regions,
                        "--model", args.model, "--regions-source", args.regions_source,
                        "--genome", args.genome, "--out", args.out,
                        "--original-repo", args.original_repo, "--heads", *args.heads],
                       check=True)
        finished = datetime.datetime.now()
        with open(os.path.join(partial, "run.json"), "w") as fh:
            json.dump({
                "script": os.path.abspath(__file__),
                "started": started.isoformat(timespec="seconds"),
                "finished": finished.isoformat(timespec="seconds"),
                "host": socket.gethostname(),
                "seed": seed,
                "references": "deeplift dinuc_shuffle with rng=np.random.RandomState(seed), "
                              "fresh for each head; 20 per region (the original's default)",
                "heads": args.heads,
                "model": args.model,
                "regions": regions,
                "n_regions": n,
                "original_repo": os.path.realpath(args.original_repo),
                "original_commit": commit,
                "python": sys.executable,
            }, fh, indent=4)
        with open(os.path.join(partial, "COMPLETED"), "w") as fh:
            fh.write(finished.isoformat(timespec="seconds") + "\n")
        shutil.rmtree(final, ignore_errors=True)
        os.replace(partial, final)
        print(f"seed {seed}: done in {(finished - started).total_seconds() / 60:.1f} min -> {final}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
