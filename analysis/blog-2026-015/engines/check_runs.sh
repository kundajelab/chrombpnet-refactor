#!/bin/bash
# Which runs of the sweep finished, and why did the others not?
#
#   ./check_runs.sh              # summary + every incomplete run
#   ./check_runs.sh -v           # every run
#   ./check_runs.sh -e jax       # one engine
#
# A run is complete when its COMPLETED marker exists, which run_train_full.sh
# writes only after both pipelines finished and the models and metrics are on
# disk. For the rest, the latest Slurm log of that engine, dataset and fold
# supplies the reason. Exits 0 only when every run is complete.

set -uo pipefail

ENGINE_FILTER=""
VERBOSE=0
while getopts "e:v" opt; do
  case $opt in
    e) ENGINE_FILTER="$OPTARG" ;;
    v) VERBOSE=1 ;;
    *) echo "usage: $0 [-e ENGINE] [-v]" >&2; exit 2 ;;
  esac
done

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

# engine|encid|fold -> "log;state;reason" from the newest log of each.
declare -A LOG_STATE
latest_logs() {   # $1 = log name prefix; newest last
  ls -1tr "${LOG_DIR}/$1".*.out 2>/dev/null
}
reason_from() {   # $1 = .err path
  local reason
  reason=$(grep -hoE "^[A-Za-z_.]*(Error|Exception)[^$]*" "$1" 2>/dev/null | tail -1)
  [[ -n "$reason" ]] || reason=$(grep -hoE "slurmstepd.*(CANCELLED|OUT_OF_MEMORY|TIME LIMIT|DUE TO)[^$]*" "$1" 2>/dev/null | tail -1)
  [[ -n "$reason" ]] || reason=$(grep -hoE "^Error: .*" "${1%.err}.out" 2>/dev/null | tail -1)
  echo "${reason:-no error recorded (still running? killed?)}"
}
while read -r out; do
  engine=$(grep -m1 "^ENGINE:" "$out" 2>/dev/null | awk '{print $2}')
  encid=$(grep -m1 "^ENCID:" "$out" 2>/dev/null | awk '{print $2}')
  fold=$(grep -m1 "^FOLD_NUM:" "$out" 2>/dev/null | awk '{print $2}')
  [[ -n "$engine" && -n "$encid" && -n "$fold" ]] || continue
  if grep -q "^Completed run" "$out"; then state=done; reason=""
  else state=fail; reason="$(reason_from "${out%.out}.err")"; fi
  LOG_STATE["${engine}|${encid}|${fold}"]="$(basename "$out");${state};${reason}"
done < <(latest_logs engines_train)

# engine|task -> RUNNING/PENDING/..., for array tasks still in the queue
# (squeue -r: one line per task). Without squeue (not on sherlock) a run with
# no COMPLETED marker and no error is reported as FAIL, which may mean running.
declare -A QUEUE_STATE
if command -v squeue > /dev/null 2>&1; then
  while read -r name task state; do
    [[ "${name}" == engines_* && "${task}" =~ ^[0-9]+$ ]] || continue
    QUEUE_STATE["${name#engines_}|${task}"]="${state}"
  done < <(squeue -h -r -u "${USER}" -o "%j %K %T" 2>/dev/null)
fi

declare -A PREP_STATE
while read -r out; do
  encid=$(grep -m1 "^ENCID:" "$out" 2>/dev/null | awk '{print $2}')
  fold=$(grep -m1 "^FOLD_NUM:" "$out" 2>/dev/null | awk '{print $2}')
  [[ -n "$encid" && -n "$fold" ]] || continue
  grep -q "^Completed run" "$out" && continue
  PREP_STATE["${encid}|${fold}"]="$(reason_from "${out%.out}.err")"
done < <(latest_logs engines_prep)

printf "%-10s %-12s %-5s %-7s %-4s %-4s %-5s %-4s  %s\n" ENGINE ENCID FOLD STATUS prep bias model eval REASON
printf '%.0s-' {1..112}; echo

total=0; done_n=0; fail_n=0; notrun_n=0; queued_n=0
declare -A DONE_BY EXPECTED_BY
n_rows="$(data_row_count "${IDS_FILE}")" || exit 1
for (( row = 1; row <= n_rows; row++ )); do
  read_row "${row}" || exit 1
  prep=.; [[ -s "$(prep_dir)/negatives.bed.gz" ]] && prep=Y
  for engine in "${ENGINES[@]}"; do
    [[ -n "$ENGINE_FILTER" && "$engine" != "$ENGINE_FILTER" ]] && continue
    total=$((total + 1)); EXPECTED_BY[$engine]=$(( ${EXPECTED_BY[$engine]:-0} + 1 ))
    sfx="$(engine_suffix "$engine")"
    d="${ROOT}/${engine}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"
    bias=.; model=.; ev=.
    [[ -f "${d}/bias_model/models/${ENCID}_bias${sfx}" ]] && bias=Y
    [[ -f "${d}/model/models/chrombpnet${sfx}" ]] && model=Y
    [[ -f "${d}/model/evaluation/chrombpnet_metrics.json" ]] && ev=Y
    entry="${LOG_STATE["${engine}|${ENCID}|${FOLD_NUM}"]:-}"
    queued="${QUEUE_STATE["${engine}|$((row - 1))"]:-}"
    if [[ -f "${d}/COMPLETED" ]]; then
      status=OK; reason=""; done_n=$((done_n + 1)); DONE_BY[$engine]=$(( ${DONE_BY[$engine]:-0} + 1 ))
      [[ "$VERBOSE" -eq 1 ]] || continue
    elif [[ -n "$queued" ]]; then
      status="${queued:0:7}"; queued_n=$((queued_n + 1))
      reason="in the queue"
      [[ -n "$entry" ]] && reason="in the queue; last log ${entry%%;*}"
    elif [[ -z "$entry" ]]; then
      status=NOTRUN; notrun_n=$((notrun_n + 1))
      reason="no log"
      [[ "$prep" == "." && -n "${PREP_STATE["${ENCID}|${FOLD_NUM}"]:-}" ]] && \
        reason="prep failed: ${PREP_STATE["${ENCID}|${FOLD_NUM}"]}"
    else
      IFS=';' read -r log state reason <<< "$entry"
      status=FAIL; fail_n=$((fail_n + 1))
      [[ "$state" == "done" ]] && reason="log says done but no COMPLETED marker (${log})"
      reason="${reason} [${log}]"
    fi
    printf "%-10s %-12s %-5s %-7s %-4s %-4s %-5s %-4s  %s\n" \
           "$engine" "$ENCID" "$FOLD_NUM" "$status" "$prep" "$bias" "$model" "$ev" "${reason:0:70}"
  done
done

echo
echo "=== per engine"
for engine in "${ENGINES[@]}"; do
  [[ -n "$ENGINE_FILTER" && "$engine" != "$ENGINE_FILTER" ]] && continue
  printf "  %-10s %2d / %2d complete\n" "$engine" "${DONE_BY[$engine]:-0}" "${EXPECTED_BY[$engine]:-0}"
done
echo
if command -v squeue > /dev/null 2>&1; then
  echo "=== totals: ${done_n} complete, ${queued_n} running or pending, ${fail_n} failed, ${notrun_n} never started, of ${total}"
else
  echo "=== totals: ${done_n} complete, ${fail_n} failed or running (no squeue here to tell), ${notrun_n} never started, of ${total}"
fi
(( done_n == total )) && { echo "ALL RUNS COMPLETE"; exit 0; }
exit 1
