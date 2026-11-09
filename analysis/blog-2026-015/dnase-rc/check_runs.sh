#!/bin/bash
# Did every run_train_full.sh array task finish, and if not, why?
#
#   ./check_runs.sh              # summary + every incomplete run
#   ./check_runs.sh -v           # every run, complete or not
#   ./check_runs.sh -a right     # one arm only
#
# Two independent sources are cross-checked, because either alone can lie:
#
#   on disk   the artifacts a finished run must leave behind
#   in logs   what Slurm actually recorded, including the failure reason
#
# A run is complete (OK) when its checkpoint and metrics JSON are whole
# (_complete.sh, which run_retrain_full.sh uses too); a missing "Completed run"
# log line is noted beside it, since later steps may not have run.
#
#   ROOT=... LOGS=... IDS_FILE=...   (default: DNASE_RC_ROOT, LOG_DIR, DNASE_RC_IDS)
#
# A run can leave a bias model and no ChromBPNet model (it died in the second
# pipeline), or a model and no metrics (it died in its evaluation). Checking
# only for "the directory exists" would call both of those a success.

set -uo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

ARM_FILTER=""
VERBOSE=0
while getopts "a:v" opt; do
  case $opt in
    a) ARM_FILTER="$OPTARG" ;;
    v) VERBOSE=1 ;;
    *) echo "usage: $0 [-a ARM] [-v]" >&2; exit 2 ;;
  esac
done

[[ -n "${ROOT:-}" ]] || { require_site DNASE_RC_ROOT; ROOT="${DNASE_RC_ROOT}"; }
# The Slurm logs: LOG_DIR (where submit_all.sh sends them), else this folder, where a
# job submitted from it writes them (the #SBATCH --output paths are relative).
LOGS="${LOGS:-${LOG_DIR:-${BLOG_SCRIPTS}/dnase-rc}}"
[[ -n "${IDS_FILE:-}" ]] || { require_site DNASE_RC_IDS; IDS_FILE="${DNASE_RC_IDS}"; }
IDS="${IDS_FILE}"
# shellcheck source=_complete.sh
source "${BLOG_SCRIPTS}/dnase-rc/_complete.sh"

ARMS=(standard fr-only rc-only right left both-half both-full)
DNASE_ONLY=" right left both-half both-full "

# ---------------------------------------------------------------- log index
# arm|encid|fold -> "state;reason". Built from the job output itself, so it
# does not depend on knowing which Slurm job id belonged to which arm. Both
# run_train_full.sh's own jobs and run_retrain_full.sh's (which hand off to it)
# count, and a success sticks: a later failed attempt does not hide it.
declare -A LOG_STATE
for out in "${LOGS}"/dnase_rc_train.*.out "${LOGS}"/dnase_rc_retrain.*.out; do
  [[ -e "$out" ]] || continue
  arm=$(grep -m1 "^DATA_AUG:" "$out" 2>/dev/null | awk '{print $2}')
  encid=$(grep -m1 "^ENCID:" "$out" 2>/dev/null | awk '{print $2}')
  fold=$(grep -m1 "^FOLD_NUM:" "$out" 2>/dev/null | awk '{print $2}')
  [[ -n "$arm" && -n "$encid" && -n "$fold" ]] || continue

  err="${out%.out}.err"
  if grep -q "^Completed run" "$out" 2>/dev/null; then
    state="done"; reason=""
  elif grep -q "^SKIP:" "$out" 2>/dev/null; then
    state="skip"; reason="arm not defined for this assay"
  else
    state="fail"
    reason=$(grep -hoE "^[A-Za-z.]*(Error|Exception)[^$]*" "$err" 2>/dev/null | tail -1)
    [[ -n "$reason" ]] || reason=$(grep -hoE "slurmstepd.*(CANCELLED|OUT_OF_MEMORY|TIME LIMIT)[^$]*" "$err" 2>/dev/null | tail -1)
    [[ -n "$reason" ]] || reason="no error recorded (killed? still running?)"
  fi
  [[ "${LOG_STATE["${arm}|${encid}|${fold}"]:-}" == done* ]] && continue
  LOG_STATE["${arm}|${encid}|${fold}"]="${state};${reason}"
done

# ------------------------------------------------------------------- report
printf "%-11s %-12s %-5s %-6s %-5s %-5s %-5s  %s\n" \
       ARM ENCID FOLD STATUS bias model eval REASON
printf '%.0s-' {1..110}; echo

total=0; done_n=0; fail_n=0; skip_n=0; missing_n=0; unlogged_n=0
declare -A ARM_DONE ARM_EXP

# _rest takes every column after FOLD_NUM (TENSORFLOW_DIR, ...); without it
# `read` appends them to $fold and every run path is wrong.
while read -r assay encid _bam _peaks fold _rest; do
  [[ "$assay" == "ASSAY" ]] && continue
  for arm in "${ARMS[@]}"; do
    [[ -n "$ARM_FILTER" && "$arm" != "$ARM_FILTER" ]] && continue
    # DNASE-only arms legitimately skip ATAC rows; not a failure.
    if [[ "$DNASE_ONLY" == *" $arm "* && "$assay" != "DNASE" ]]; then
      continue
    fi

    total=$((total+1)); ARM_EXP[$arm]=$(( ${ARM_EXP[$arm]:-0} + 1 ))
    d="${ROOT}/${arm}/${assay}/${encid}/fold_${fold}"
    bias=.; model=.; ev=.
    compgen -G "${d}/bias_model/models/*_bias.pt" >/dev/null 2>&1 && bias=Y
    checkpoint_complete "${d}/model/models/chrombpnet.pt" && model=Y
    metrics_complete "${d}/model/evaluation/chrombpnet_metrics.json" && ev=Y

    entry="${LOG_STATE["${arm}|${encid}|${fold}"]:-}"
    log_state="${entry%%;*}"; reason="${entry#*;}"
    [[ "$entry" == "" ]] && { log_state="nolog"; reason="no Slurm log found for this combination"; }

    if [[ "$model" == "Y" && "$ev" == "Y" ]]; then
      status=OK; done_n=$((done_n+1)); ARM_DONE[$arm]=$(( ${ARM_DONE[$arm]:-0} + 1 ))
      if [[ "$log_state" == "done" ]]; then
        reason=""
      else
        unlogged_n=$((unlogged_n+1))
        reason="no 'Completed run' logged: later steps may be missing"
      fi
      [[ "$VERBOSE" -eq 1 ]] || continue
    elif [[ "$log_state" == "skip" ]]; then
      status=SKIP; skip_n=$((skip_n+1))
      [[ "$VERBOSE" -eq 1 ]] || continue
    elif [[ "$log_state" == "nolog" ]]; then
      status=NOTRUN; missing_n=$((missing_n+1))
    elif [[ "$model" == "Y" ]]; then
      # artifacts present but the job never reported success
      status=PARTIAL; fail_n=$((fail_n+1))
    else
      status=FAIL; fail_n=$((fail_n+1))
    fi

    printf "%-11s %-12s %-5s %-6s %-5s %-5s %-5s  %s\n" \
           "$arm" "$encid" "$fold" "$status" "$bias" "$model" "$ev" "${reason:0:52}"
  done
done < <(tr -d '\r' < "$IDS")

echo
echo "=== per arm ==="
for arm in "${ARMS[@]}"; do
  [[ -n "$ARM_FILTER" && "$arm" != "$ARM_FILTER" ]] && continue
  printf "  %-11s %2d / %2d complete\n" "$arm" "${ARM_DONE[$arm]:-0}" "${ARM_EXP[$arm]:-0}"
done

echo
echo "=== totals ==="
echo "  expected          ${total}"
echo "  complete          ${done_n}"
[[ "$unlogged_n" -gt 0 ]] && echo "    of which no 'Completed run' logged: ${unlogged_n} (-v lists them)"
echo "  failed            ${fail_n}"
echo "  never launched    ${missing_n}"
[[ "$skip_n" -gt 0 ]] && echo "  skipped (by design) ${skip_n}"

echo
echo "=== failure reasons in the logs (earlier attempts of runs now complete included) ==="
for out in "${LOGS}"/dnase_rc_train.*.out "${LOGS}"/dnase_rc_retrain.*.out; do
  [[ -e "$out" ]] || continue
  grep -q "^Completed run" "$out" 2>/dev/null && continue
  grep -q "^SKIP:" "$out" 2>/dev/null && continue
  err="${out%.out}.err"
  grep -hoE "^[A-Za-z.]*(Error|Exception)[^$]*" "$err" 2>/dev/null | tail -1
done | sort | uniq -c | sort -rn | head -10

[[ "$done_n" -eq "$total" ]] && { echo; echo "ALL RUNS COMPLETE"; exit 0; }
echo; echo "INCOMPLETE: ${done_n} of ${total}"
exit 1
