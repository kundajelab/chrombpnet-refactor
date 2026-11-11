#!/usr/bin/env python3
"""Time ``chrombpnet pred_bw`` under each optimization setting, fairly.

Each *arm* is one way of running ``pred_bw`` on the same regions, the same
observed bigWig and the same three models (bias, ChromBPNet, no-bias):

==========  ==================================================================
original    chrombpnet 1.x (TensorFlow 2.8), on the original's ``.h5`` models
none        chrombpnet-refactor at commit 4962484, ``--optimize none``
kit         4962484, ``--optimize 1``   (the optimization kit's changes, no Triton)
ours        4962484, ``--optimize 3``   (this package's own changes)
both        4962484, ``--optimize 1,3`` (the default at 4962484)
kitfk       4962484, ``--optimize 1,2`` (the kit's changes with its Triton forward)
fkonly      4962484, ``--optimize 2``   (the Triton forward alone)
main        the current chrombpnet-refactor, default (should match ``both``)
fk          the current chrombpnet-refactor, ``--predict-fused-kernel`` (fused kernels)
==========  ==================================================================

``--optimize`` was removed after commit 4962484, so the phase arms need a
checkout (or ``git archive``) of that commit. An arm runs only when its inputs
are given; ``--arms`` picks a subset.

The method is what makes the numbers comparable:

* the genome, its index, the observed bigWig and every model are read once
  before anything is timed, so no arm pays for a cold file cache;
* every arm gets one untimed warm-up run on the first ``--warmup-regions``
  regions (a fused-kernel refusal there marks ``fk`` refused and skips it);
* the timed runs come in ``--rounds`` rounds, each running the arms in a
  different order;
* every run is a fresh process started from an empty working directory,
  with ``PYTHONPATH`` pinned to that arm's checkout, after a check that the
  right ``chrombpnet`` is the one imported; only the ``pred_bw`` command
  itself is timed.

Results go to ``OUT_DIR/timings.tsv`` (one row per run); each run's log stays
in ``OUT_DIR/<round>_<arm>/run.log`` and its bulky outputs are deleted.
``summarize.py`` collates one or more of these directories.

Site-specific setup is the caller's: activate nothing, but pass each arm's
interpreter, and anything its framework needs in the environment with
``--env-original`` / ``--env-refactor`` (e.g. ``LD_LIBRARY_PATH`` with CUDA
11.2 and cuDNN 8 for the original's TensorFlow 2.8). Standard library only.
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("original", "none", "kit", "ours", "both", "kitfk", "fkonly", "main", "fk")
PHASE_FLAGS = {"none": "none", "kit": "1", "ours": "3", "both": "1,3", "kitfk": "1,2", "fkonly": "2"}
COLUMNS = ("run", "fold", "gpu", "arch", "arm", "round", "position", "seconds", "status", "detail")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="See README.md for an example.")
    io = p.add_argument_group("inputs and output")
    io.add_argument("--out-dir", required=True, help="where timings.tsv and the run logs go")
    io.add_argument("--genome", required=True, help="genome FASTA (with .fai)")
    io.add_argument("--chrom-sizes", required=True)
    io.add_argument("--observed-bigwig", required=True, help="the observed bigWig given to pred_bw -bw")
    regions = io.add_mutually_exclusive_group(required=True)
    regions.add_argument("--peaks", help="peaks BED to subsample --n-regions regions from")
    regions.add_argument("--regions", help="regions BED to predict as is")
    io.add_argument("--n-regions", type=int, default=20000, help="regions drawn from --peaks (default 20000)")
    io.add_argument("--seed", type=int, default=1234, help="seed of the --peaks subsample")

    models = p.add_argument_group("models")
    models.add_argument("--chrombpnet-pt", help="refactor ChromBPNet model (.pt)")
    models.add_argument("--nobias-pt", help="refactor no-bias model (.pt)")
    models.add_argument("--bias-pt", help="refactor bias model (.pt), e.g. bias_model_scaled.pt")
    models.add_argument("--chrombpnet-h5", help="original ChromBPNet model (.h5)")
    models.add_argument("--nobias-h5", help="original no-bias model (.h5)")
    models.add_argument("--bias-h5", help="original bias model (.h5)")

    code = p.add_argument_group("code and interpreters")
    code.add_argument("--refactor-repo", help="chrombpnet-refactor checkout for the main and fk arms")
    code.add_argument("--refactor-python", default=sys.executable,
                      help="interpreter for the refactor arms (default: this one)")
    code.add_argument("--phase-repo", help="checkout or `git archive` of commit 4962484, for none/kit/ours/both")
    code.add_argument("--phase-python", help="interpreter for the phase arms (default: --refactor-python)")
    code.add_argument("--original-repo", help="chrombpnet 1.x checkout (kundajelab/chrombpnet), for original")
    code.add_argument("--original-python", help="interpreter of the chrombpnet 1.x environment")
    code.add_argument("--env-original", action="append", default=[], metavar="KEY=VALUE",
                      help="environment variable for the original arm (repeatable)")
    code.add_argument("--env-refactor", action="append", default=[], metavar="KEY=VALUE",
                      help="environment variable for the refactor arms (repeatable)")

    run = p.add_argument_group("run")
    run.add_argument("--arms", default=",".join(ARMS),
                     help="comma-separated arms (default: every arm whose inputs are given)")
    run.add_argument("--device", default="cuda", help="--device for the refactor arms (default cuda)")
    run.add_argument("--batch-size", type=int, default=256, help="pred_bw -bs (default 256)")
    run.add_argument("--rounds", type=int, default=3, help="timed rounds (default 3)")
    run.add_argument("--warmup-regions", type=int, default=2000,
                     help="regions in each arm's untimed warm-up run (default 2000)")
    run.add_argument("--label", default="", help="value of the fold column, e.g. fold_0")
    run.add_argument("--run-tag", help="value of the run column (default: the GPU's name)")
    run.add_argument("--keep-outputs", action="store_true", help="keep each run's bigWigs and HDF5s")
    return p.parse_args(argv)


def env_pairs(pairs, option):
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"{option} expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key] = value
    return out


def gpu_identity(device):
    """(name, arch) of the GPU the runs will see, from nvidia-smi."""
    if not device.startswith("cuda"):
        return "CPU", "cpu"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    index = device.split(":", 1)[1] if ":" in device else (visible if visible.isdigit() else "0")
    try:
        out = subprocess.run(["nvidia-smi", "-i", index, "--query-gpu=name,compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True)
        name, cap = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return name, "cuda-" + cap.replace(".", "")
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return "unknown GPU", "unknown"


def write_regions(args, out_dir):
    regions = out_dir / "regions.bed"
    if args.regions:
        shutil.copyfile(args.regions, regions)
    else:
        lines = [line for line in Path(args.peaks).read_text().splitlines() if line.strip()]
        rng = random.Random(args.seed)
        if len(lines) > args.n_regions:
            lines = rng.sample(lines, args.n_regions)
        lines.sort(key=lambda line: (line.split("\t")[0], int(line.split("\t")[1])))
        regions.write_text("\n".join(lines) + "\n")
    warmup = out_dir / "warmup_regions.bed"
    lines = regions.read_text().splitlines()[:args.warmup_regions]
    warmup.write_text("\n".join(lines) + "\n")
    return regions, warmup


def available_arms(args):
    """{arm: why it cannot run} for the arms whose inputs are missing."""
    missing = {}
    pt = all([args.chrombpnet_pt, args.nobias_pt, args.bias_pt])
    h5 = all([args.chrombpnet_h5, args.nobias_h5, args.bias_h5])
    for arm in ARMS:
        if arm == "original" and not (args.original_repo and args.original_python and h5):
            missing[arm] = "needs --original-repo, --original-python and the three .h5 models"
        elif arm in PHASE_FLAGS and not (args.phase_repo and pt):
            missing[arm] = "needs --phase-repo (commit 4962484) and the three .pt models"
        elif arm in ("main", "fk") and not (args.refactor_repo and pt):
            missing[arm] = "needs --refactor-repo and the three .pt models"
    return missing


def arm_spec(arm, args, regions, out):
    """(python, checkout, extra env, pred_bw arguments) of one arm."""
    common = ["-r", str(regions), "-g", args.genome, "-c", args.chrom_sizes,
              "-bw", args.observed_bigwig, "-op", str(out / "pred"), "-os", str(out / "stats"),
              "-bs", str(args.batch_size), "-t", "0"]
    if arm == "original":
        models = ["-cm", args.chrombpnet_h5, "-cmb", args.nobias_h5, "-bm", args.bias_h5]
        return (args.original_python, args.original_repo, env_pairs(args.env_original, "--env-original"),
                ["pred_bw"] + models + common)
    models = ["-cm", args.chrombpnet_pt, "-cmb", args.nobias_pt, "-bm", args.bias_pt]
    extra = env_pairs(args.env_refactor, "--env-refactor")
    device = ["--device", args.device]
    if arm in PHASE_FLAGS:
        return (args.phase_python or args.refactor_python, args.phase_repo, extra,
                ["pred_bw", "--optimize", PHASE_FLAGS[arm]] + models + common + device)
    flags = ["--predict-fused-kernel"] if arm == "fk" else []
    return args.refactor_python, args.refactor_repo, extra, ["pred_bw"] + flags + models + common + device


def run_arm(arm, args, regions, out, neutral):
    """Run one arm in a fresh process; return (status, seconds or None)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    python, checkout, extra, pred_bw = arm_spec(arm, args, regions, out)
    env = dict(os.environ, **extra)
    env["PYTHONPATH"] = str(Path(checkout).resolve())
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TRITON_CACHE_DIR", str(Path(args.out_dir).resolve() / ".triton-cache"))
    check = ("import chrombpnet, os, sys; p = os.path.realpath(chrombpnet.__file__); "
             f"sys.exit(0 if p.startswith(os.path.realpath({str(checkout)!r})) else 'wrong chrombpnet: ' + p)")
    with open(out / "run.log", "w") as log:
        imported = subprocess.run([python, "-c", check], cwd=neutral, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
        if imported.returncode:
            return "failed", None
        log.write("$ " + " ".join([python, "-m", "chrombpnet.CHROMBPNET"] + pred_bw) + "\n")
        log.flush()
        start = time.perf_counter()
        completed = subprocess.run([python, "-m", "chrombpnet.CHROMBPNET"] + pred_bw, cwd=neutral,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - start
    if completed.returncode == 0:
        return "ok", seconds
    if "TritonRouteRefused" in (out / "run.log").read_text(errors="replace"):
        return "refused", None
    return "failed", None


def drop_outputs(out):
    for path in out.rglob("*"):
        if path.is_file() and path.name != "run.log":
            path.unlink()


def warm_file_cache(paths):
    for path in paths:
        with open(path, "rb") as handle:
            while handle.read(64 << 20):
                pass


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neutral = out_dir / "cwd"           # never run from a checkout: -m puts the cwd first
    neutral.mkdir(exist_ok=True)
    gpu, arch = gpu_identity(args.device)
    run_tag = args.run_tag or "".join(c if c.isalnum() else "-" for c in gpu).strip("-")
    missing = available_arms(args)
    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in requested if a not in ARMS]
    if unknown:
        raise SystemExit("unknown arm(s): {} (choose from {})".format(", ".join(unknown), ", ".join(ARMS)))
    explicit = args.arms != ",".join(ARMS)
    arms = []
    for arm in requested:
        if arm in missing:
            if explicit:
                raise SystemExit(f"arm {arm} {missing[arm]}")
            print(f"skipping {arm}: {missing[arm]}")
        else:
            arms.append(arm)
    if not arms:
        raise SystemExit("no arm has its inputs; see --help")

    regions, warmup = write_regions(args, out_dir)
    table = out_dir / "timings.tsv"
    with open(table, "w") as handle:
        handle.write("\t".join(COLUMNS) + "\n")

    def record(arm, round_, position, seconds, status, detail=""):
        with open(table, "a") as handle:
            handle.write("\t".join(str(x) for x in (
                run_tag, args.label, gpu, arch, arm, round_, position,
                "" if seconds is None else f"{seconds:.1f}", status, detail)) + "\n")

    print(f"run {run_tag} ({gpu}, {arch}); {sum(1 for _ in open(regions))} regions; "
          f"{args.rounds} round(s); arms {', '.join(arms)}; output {out_dir}")
    print("warm-up: file cache")
    models = [args.chrombpnet_pt, args.nobias_pt, args.bias_pt,
              args.chrombpnet_h5, args.nobias_h5, args.bias_h5]
    warm_file_cache([args.genome, args.genome + ".fai", args.observed_bigwig]
                    + [m for m in models if m and os.path.exists(m)])

    active = []
    print(f"warm-up: one untimed run per arm on {args.warmup_regions} regions")
    for arm in arms:
        out = out_dir / f"warmup_{arm}"
        status, _ = run_arm(arm, args, warmup, out, neutral)
        print(f"  {arm:<9} {status}")
        drop_outputs(out)
        if status == "ok":
            active.append(arm)
        elif status == "refused":
            record(arm, "-", "-", None, "refused", f"fused kernels refused on {arch}")
        else:
            record(arm, "-", "-", None, "failed", f"warm-up failed: {out / 'run.log'}")
    if not active:
        raise SystemExit("no arm survived the warm-up; see the warmup_*/run.log files")

    shift = math.ceil(len(active) / args.rounds)
    for round_ in range(args.rounds):
        print(f"round {round_ + 1} of {args.rounds}")
        for position in range(len(active)):
            arm = active[(position + round_ * shift) % len(active)]
            out = out_dir / f"r{round_}_{arm}"
            status, seconds = run_arm(arm, args, regions, out, neutral)
            print(f"  {arm:<9} {status:<8} {'' if seconds is None else f'{seconds:.1f} s'}")
            record(arm, round_, position, seconds, status, "" if status == "ok" else str(out / "run.log"))
            if not args.keep_outputs:
                drop_outputs(out)
    shutil.rmtree(neutral, ignore_errors=True)
    print(f"timings: {table}")


if __name__ == "__main__":
    main()
