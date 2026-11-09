#!/bin/bash
# Collate the per-fold timings into one table per run, with speedups of the
# fused-kernel arm against the default. Run after the jobs finish:
#
#   ./summarize.sh RESULTS     # the jobs' OUT_ROOT (site.env's OPTIMIZATION_ANALYSIS)
set -euo pipefail
RESULTS="${1:?usage: summarize.sh RESULTS_DIR (the OUT_ROOT of the benchmark jobs, site.env OPTIMIZATION_ANALYSIS)}"
shopt -s nullglob
# Runs live at results/<run tag>/fold_N/ so concurrent sweeps on different
# cards cannot collide. The older flat results/fold_N/ layout is still read.
files=("${RESULTS}"/*/fold_*/timings.tsv "${RESULTS}"/fold_*/timings.tsv)
[[ ${#files[@]} -gt 0 ]] || { echo "no timings.tsv under ${RESULTS}" >&2; exit 1; }

python3 - "${files[@]}" <<'PYEOF'
import collections, statistics, sys

rows = []
for path in sys.argv[1:]:
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            if len(values) == len(header):
                rows.append(dict(zip(header, values)))

# Only these commands carry --predict-fk. contribs_bw does not: DeepLIFT
# needs an autograd graph the fused kernels have not got, so it is timed once
# and any apparent difference there is run-to-run noise (cache warmth, GPU
# clocks) and is parenthesised.
RESPONDS = {"pred_bw", "footprints"}

# Which optimizations each command can actually reach, traced through the
# code. Printed with the table so a flat number is never read as "phase 2
# did nothing" when the command simply cannot use it.
REACH = {
 "pred_bw":     "all of 1, 2 and 3",
 "contribs_bw": "1 (memmap reader) and 3 (bigwig writer); NOT 2 -- deep_lift_shap "
                "needs an autograd graph, so there is no Triton route",
 "footprints":  "1 (one-hot LUT only; it keeps the pyfaidx loop), 2 and 3",
}

# One table per run: averaging an A100 sweep together with an H200 sweep
# would produce a number describing neither.
runs = collections.defaultdict(list)
for row in rows:
    runs[row.get("run") or f"{row.get('gpu','?')} {row.get('arch','?')}"].append(row)

for run_tag, run_rows in sorted(runs.items()):
    by_stage = collections.defaultdict(lambda: collections.defaultdict(list))
    status = collections.defaultdict(set)
    for row in run_rows:
        status[(row["stage"], row["phase"])].add(row["status"])
        if row["seconds"]:
            by_stage[row["stage"]][row["phase"]].append(float(row["seconds"]))

    print("=" * 74)
    print(f"run   : {run_tag}")
    print(f"GPU   : {', '.join(sorted({r.get('gpu','?') for r in run_rows}))}"
          f"   arch: {', '.join(sorted({r.get('arch','?') for r in run_rows}))}")
    print(f"folds : {len({r['fold'] for r in run_rows})}")
    print()

    for stage, phases in by_stage.items():
        base = statistics.mean(phases["default"]) if phases.get("default") else None
        responds = stage in RESPONDS
        note = "" if responds else "   [no --predict-fk option; changes here are noise]"
        print(f"--- {stage} ---{note}")
        print(f"  {'arm':10} {'n':>2} {'mean s':>9} {'sd':>7} {'vs default':>11}   status")
        for phase in sorted(phases, key=lambda p: (p != "default", p)):
            values = phases[phase]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            speedup = f"{base / mean:.2f}x" if base and mean else "-"
            if not responds and phase != "default":
                speedup = f"({speedup})"
            print(f"  {phase:10} {len(values):2d} {mean:9.1f} {sd:7.1f} {speedup:>9}   "
                  f"{','.join(sorted(status[(stage, phase)]))}")
        # Phases that produced no time at all, e.g. Triton refused on A100.
        for (st, phase), states in sorted(status.items()):
            if st == stage and phase not in phases:
                print(f"  {phase:10} {'-':>2} {'-':>9} {'-':>7} {'-':>9}   "
                      f"{','.join(sorted(states))}")
        if stage in REACH:
            print(f"  reaches: {REACH[stage]}")
        print()

print("A phase shown as 'refused' could not run on that card -- phase 2 is refused")
print("on A100, whose kernels have no bitwise route. A phase whose time matches")
print("'none' may simply not reach that command; see the 'reaches' line.")
PYEOF

echo
echo "artifact differences vs the default arm:"
for f in "${RESULTS}"/*/fold_*/artifact_diffs.txt "${RESULTS}"/fold_*/artifact_diffs.txt; do
  [[ -f "$f" ]] || continue
  rel="${f#"${RESULTS}"/}"
  echo "  ${rel%/artifact_diffs.txt}"
  sed 's/^/  /' "$f"
done
