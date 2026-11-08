#!/bin/bash
# Submit every arm of both sweeps, in the order their dependencies require.
#
#   ./submit_all.sh sherlock            # submit
#   ./submit_all.sh sherlock --dry-run  # print the sbatch lines only
#
#   run_retrain_full.sh    rows 1-10  (K562)    array 0-9
#   run_train_chrombpnet.sh rows 11-20 (stomach) array 10-19
#
# The stomach runs borrow each arm's K562 fold-0 bias model, which the K562
# run of that same arm produces (array task 0 for ATAC, task 5 for DNase).
# So each stomach job waits on its own arm's K562 job -- `afterany`, not
# `afterok`, because a K562 array legitimately contains tasks that skip
# (DNase-only arms on ATAC rows) or that were already complete.

set -euo pipefail

CLUSTER="${1:-sherlock}"
DRY=""
[[ "${2:-}" == "--dry-run" ]] && DRY="echo"

HERE="$(dirname "$(readlink -f "$0")")"
ARMS=(standard fr-only rc-only right left both-half both-full)

echo "=== K562 (rows 1-10) ==="
declare -A K562_JOB
for arm in "${ARMS[@]}"; do
  if [[ -n "$DRY" ]]; then
    echo "sbatch --parsable ${HERE}/run_retrain_full.sh ${CLUSTER} ${arm}"
    K562_JOB[$arm]="<jobid>"
  else
    K562_JOB[$arm]=$(sbatch --parsable "${HERE}/run_retrain_full.sh" "${CLUSTER}" "${arm}")
    printf "  %-10s job %s\n" "${arm}" "${K562_JOB[$arm]}"
  fi
done

echo
echo "=== stomach (rows 11-20), each waiting on its own arm's K562 job ==="
for arm in "${ARMS[@]}"; do
  dep="afterany:${K562_JOB[$arm]}"
  if [[ -n "$DRY" ]]; then
    echo "sbatch --parsable --dependency=${dep} ${HERE}/run_train_chrombpnet.sh ${CLUSTER} ${arm}"
  else
    jid=$(sbatch --parsable --dependency="${dep}" \
                 "${HERE}/run_train_chrombpnet.sh" "${CLUSTER}" "${arm}")
    printf "  %-10s job %s  (after %s)\n" "${arm}" "${jid}" "${K562_JOB[$arm]}"
  fi
done

echo
echo "Progress:  ./check_runs.sh          Plot:  ./plot_summarize.sh --refresh"
