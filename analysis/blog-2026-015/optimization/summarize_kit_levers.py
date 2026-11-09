#!/usr/bin/env python3
"""Each optimization-kit lever's effect in original chrombpnet, per GPU, from
run_kit_modes.sh runs: stock's stage time (the `stages` arm,
stock_stage_timer.py) against the kit's time for the same stage (its run
record, in each kit arm's opt_manifest.json). Prints one line per lever, ready
for the blog's "Measured effect (in original chrombpnet)" column.

    python summarize_kit_levers.py ANALYSIS_DIR      # the jobs' OUT_ROOT (site.env's OPTIMIZATION_OUT)

The kit runs several stages beside the GPU forward, so its time is what stays
on the critical path (e.g. waiting for the bigWig writer at close, not the
writer's CPU time). Medians over every round of a run tag; runs are never
mixed. Host-side levers are read from the `fast` arm (the kit's default),
the Triton forward from `exact` on an L40S (where `fast` keeps stock's
TensorFlow graph) and from `fast` on an H200.
"""

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("usage: python summarize_kit_levers.py ANALYSIS_DIR  (the jobs' OUT_ROOT: site.env's OPTIMIZATION_OUT)")
ROOT = sys.argv[1]


def find(tree, key):
    """The first value stored under `key` anywhere in a JSON tree."""
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        children = tree.values()
    elif isinstance(tree, list):
        children = tree
    else:
        return None
    for child in children:
        found = find(child, key)
        if found is not None:
            return found
    return None


def kit_stages(manifest_path):
    tree = json.load(open(manifest_path))
    timings = find(tree, "timings_s") or {}
    metrics = find(tree, "metrics_stage_timings_s") or {}
    preimport = find(tree, "preimport") or {}
    out = {
        "featurise": timings.get("featurise"),
        "forward": timings.get("forward"),
        "bigwig_write": timings.get("bigwig_close"),      # the wait at close; the writer itself ran beside the forward
        "h5_write": metrics.get("h5_assemble"),
    }
    finish = metrics.get("finish_stage_s")
    if isinstance(finish, (int, float)) and isinstance(out["h5_write"], (int, float)):
        out["metrics"] = finish - out["h5_write"]
    walls = preimport.get("walls_s") if isinstance(preimport, dict) else None
    out["finish_imports_overlapped"] = sum(walls.values()) if isinstance(walls, dict) else None
    out["_raw"] = {"timings_s": timings, "metrics_stage_timings_s": metrics}
    return out


def stock_stages(directory):
    stages = json.load(open(os.path.join(directory, "stages.json")))
    t0, t1 = (float(x) for x in open(os.path.join(directory, "time")).read().split())
    stages["teardown"] = t1 - stages["main_return_epoch"]
    stages["wall"] = t1 - t0
    return stages


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def cell(gpu, before, after):
    if before is None or after is None:
        return "%s: n/a" % gpu
    ratio = "%.1fx faster" % (before / after) if after > 0.05 else "hidden"
    return "%s: %.1f s -> %.1f s (%s)" % (gpu, before, after, ratio)


def main():
    runs = sorted(glob.glob(os.path.join(ROOT, "*_kit-modes*", "fold_GM12878")))
    if not runs:
        sys.exit("no *_kit-modes*/fold_GM12878 under " + ROOT)
    rows = defaultdict(dict)
    for run in runs:
        tag = run.split(os.sep)[-2]
        gpu = "H200" if "H200" in tag else "L40S" if "L40S" in tag else tag
        stock = [stock_stages(d) for d in glob.glob(os.path.join(run, "r*_stages"))
                 if os.path.exists(os.path.join(d, "stages.json"))]
        kit = {arm: [kit_stages(p) for p in glob.glob(os.path.join(run, "r*_%s" % arm, "opt_manifest.json"))]
               for arm in ("fast", "exact")}
        if not stock or not kit["fast"]:
            print("## %s: no stages arm or no kit run records (rerun run_kit_modes.sh with ARMS including stages)" % tag)
            continue
        print("## %s  (%d stock-stage runs, %d fast, %d exact)" % (tag, len(stock), len(kit["fast"]), len(kit["exact"])))
        s = {k: median([r.get(k) for r in stock]) for k in
             ("featurise", "forward", "bigwig_write", "metrics", "h5_write", "finish_imports", "teardown", "wall")}
        f = {k: median([r.get(k) for r in kit["fast"]]) for k in
             ("featurise", "forward", "bigwig_write", "metrics", "h5_write", "finish_imports_overlapped")}
        e = {k: median([r.get(k) for r in kit["exact"]]) for k in ("forward",)}
        forward_kit = e["forward"] if gpu == "L40S" else f["forward"]
        print("  stock stages (s):", {k: round(v, 1) for k, v in s.items() if v is not None})
        print("  kit fast stages (s):", {k: round(v, 1) for k, v in f.items() if v is not None})
        if any(v is None for v in f.values()):
            print("  (some kit stage missing; raw record of one run:)", json.dumps(kit["fast"][0]["_raw"])[:600])
        rows["prefetch"][gpu] = cell(gpu, s["featurise"], f["featurise"])
        rows["metrics_stage"][gpu] = cell(gpu, s["metrics"], f["metrics"])
        rows["bigwig_writer"][gpu] = cell(gpu, s["bigwig_write"], f["bigwig_write"])
        rows["preimport"][gpu] = cell(gpu, s["finish_imports"], 0.0) + (
            " (%.1f s of imports moved under the forward)" % f["finish_imports_overlapped"]
            if f["finish_imports_overlapped"] is not None else "")
        rows["exit_fast"][gpu] = cell(gpu, s["teardown"], 0.0)
        rows["h5_writer"][gpu] = cell(gpu, s["h5_write"], f["h5_write"])
        rows["forward_route"][gpu] = cell(gpu, s["forward"], forward_kit)
    print("\n## blog cells: stage time, stock -> kit")
    for lever in ("prefetch", "metrics_stage", "bigwig_writer", "preimport", "exit_fast", "forward_route", "h5_writer"):
        print("  %-14s %s" % (lever, "; ".join(rows[lever][g] for g in ("L40S", "H200") if g in rows[lever])))


if __name__ == "__main__":
    main()
