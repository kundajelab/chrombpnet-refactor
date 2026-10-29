#!/usr/bin/env python
"""Score one bias-free model on a dataset's shared regions: DeepSHAP and predictions.

    python score_shared_regions.py regions --peaks <fold_0 filtered.peaks.bed> ... \
        --n 10000 --seed 1234 --out <dir>/regions.bed
    python score_shared_regions.py score --model <chrombpnet_nobias.pt> --regions <dir>/regions.bed \
        --genome <genome.fa> --out-prefix <dir>/<label> [--device cuda]

regions   the shared regions of a dataset: a --seed draw of --n peaks among
          those (by chromosome and summit) in every given peak file -- the
          five fold models' auxiliary/filtered.peaks.bed -- so that every
          fold model, and every student, is scored on the same peaks.
          Written atomically; the same inputs always give the same file.
score     for one model: its DeepSHAP scores, both heads, as `chrombpnet
          contribs_bw` and the pipeline compute them (seed 1234, 20 shuffled
          references, the profile target's midpoint weighting) -- the
          references depend only on each region's sequence and the seed, so
          every model gets the same ones -- and its predictions:
            <prefix>.counts_scores.h5, <prefix>.profile_scores.h5
            <prefix>.interpreted_regions.bed   (the rows of both)
            <prefix>.predictions.h5            logcounts (n,), probs (n, L): the
                                               bias-free model's log counts and
                                               profile probabilities, same rows
            <prefix>.COMPLETED
          Skipped when <prefix>.COMPLETED exists (FORCE=1 redoes it).

Both run in the chrombpnet-pytorch environment, the scoring on a GPU.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def keys_of(path):
    """(chrom, summit) -> the 10-column line, in file order."""
    out = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 10:
                out.setdefault((f[0], int(f[1]) + int(f[9])), line if line.endswith("\n") else line + "\n")
    return out


def make_regions(args):
    sets = [keys_of(p) for p in args.peaks]
    common = [k for k in sets[0] if all(k in s for s in sets[1:])]
    if len(common) < args.n:
        sys.exit(f"Error: only {len(common)} peaks are in all {len(sets)} files, fewer than --n {args.n}")
    rng = np.random.RandomState(args.seed)
    chosen = sorted(rng.choice(len(common), size=args.n, replace=False))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    partial = f"{args.out}.partial.{os.getpid()}"
    with open(partial, "w") as fh:
        for i in chosen:
            fh.write(sets[0][common[i]])
    os.replace(partial, args.out)
    print(f"wrote {args.out}: {args.n} of the {len(common)} peaks in all {len(sets)} files")


def score(args):
    done = f"{args.out_prefix}.COMPLETED"
    if os.path.isfile(done) and os.environ.get("FORCE", "0") != "1":
        print(f"exists: {done} -- skipped (FORCE=1 rescores)")
        return
    import h5py
    import pandas as pd
    import pyfaidx
    import torch

    from chrombpnet import engines
    from chrombpnet.evaluation.interpret import input_utils, interpret
    from chrombpnet.regions import NARROWPEAK_SCHEMA

    engines.configure("pytorch")
    partial = f"{args.out_prefix}.partial"
    ns = argparse.Namespace(
        model_h5=args.model, regions=args.regions, genome=args.genome, output_prefix=partial,
        profile_or_counts=["counts", "profile"], batch_size=64, interpret_batch_size=None,
        num_shuffles=20, seed=args.seed, device=args.device, debug_chr=None,
        contribs_profile_stopgrad=None)
    interpret.main(ns)

    # Predictions of the same model on the same regions, in the same order.
    engine = engines.for_model(args.model)
    model = engine.load_predictor(args.model, args.device, announce=False, summary=False)
    regions = pd.read_csv(f"{partial}.interpreted_regions.bed", sep="\t", names=NARROWPEAK_SCHEMA)
    genome = pyfaidx.Fasta(args.genome)
    seqs, used = input_utils.get_seq(regions, genome, int(model.input_shape[1]))
    genome.close()
    if not used.all():
        sys.exit("Error: some interpreted regions could not be read again")
    logits, logcounts = model.predict(seqs, batch_size=64)
    probs = torch.softmax(torch.from_numpy(np.asarray(logits, dtype=np.float32)), dim=-1).numpy()
    with h5py.File(f"{partial}.predictions.h5", "w") as f:
        f.create_dataset("logcounts", data=np.asarray(logcounts, dtype=np.float32).reshape(-1),
                         compression="gzip")
        f.create_dataset("probs", data=probs.astype(np.float32), compression="gzip")
        f.attrs["model"] = args.model
    for suffix in (".counts_scores.h5", ".profile_scores.h5", ".interpreted_regions.bed",
                   ".predictions.h5", ".interpret.args.json"):
        os.replace(f"{partial}{suffix}", f"{args.out_prefix}{suffix}")
    with open(done, "w") as fh:
        fh.write(f"{args.model}\n")
    print(f"wrote {args.out_prefix}.*")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("regions")
    r.add_argument("--peaks", nargs="+", required=True)
    r.add_argument("--n", type=int, default=10000)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--regions", required=True)
    s.add_argument("--genome", required=True)
    s.add_argument("--out-prefix", required=True)
    s.add_argument("--seed", type=int, default=1234)
    s.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    (make_regions if args.cmd == "regions" else score)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
