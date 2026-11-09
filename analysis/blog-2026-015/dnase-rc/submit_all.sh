#!/bin/bash
# Submit every arm of both sweeps, in the order their dependencies require.
#
#   ./submit_all.sh slurm            # submit
#   ./submit_all.sh slurm --dry-run  # print the sbatch lines only
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

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

CLUSTER="${1:-slurm}"
DRY=""
[[ "${2:-}" == "--dry-run" ]] && DRY="echo"

HERE="${BLOG_SCRIPTS}/dnase-rc"
ARMS=(standard fr-only rc-only right left both-half both-full)

# Each job finds this folder through SLURM_SUBMIT_DIR, so submit from it. SBATCH_SITE_ARGS
# (site.env) carries the cluster's own options; with LOG_DIR set the logs go there, named
# as the #SBATCH lines name them (job name, array job, task).
cd "${HERE}"
read -ra SITE_ARGS <<< "${SBATCH_SITE_ARGS:-}"
LOG_ARGS=()
[[ -n "${LOG_DIR:-}" ]] && LOG_ARGS=(-o "${LOG_DIR}/%x.%A_%a.out" -e "${LOG_DIR}/%x.%A_%a.err")
SB=(sbatch --parsable ${SITE_ARGS[@]+"${SITE_ARGS[@]}"} ${LOG_ARGS[@]+"${LOG_ARGS[@]}"})

echo "=== K562 (rows 1-10) ==="
declare -A K562_JOB
for arm in "${ARMS[@]}"; do
  if [[ -n "$DRY" ]]; then
    echo "${SB[*]} ${HERE}/run_retrain_full.sh ${CLUSTER} ${arm}"
    K562_JOB[$arm]="<jobid>"
  else
    K562_JOB[$arm]=$("${SB[@]}" "${HERE}/run_retrain_full.sh" "${CLUSTER}" "${arm}")
    printf "  %-10s job %s\n" "${arm}" "${K562_JOB[$arm]}"
  fi
done

echo
echo "=== stomach (rows 11-20), each waiting on its own arm's K562 job ==="
for arm in "${ARMS[@]}"; do
  dep="afterany:${K562_JOB[$arm]}"
  if [[ -n "$DRY" ]]; then
    echo "${SB[*]} --dependency=${dep} ${HERE}/run_train_chrombpnet.sh ${CLUSTER} ${arm}"
  else
    jid=$("${SB[@]}" --dependency="${dep}" \
                 "${HERE}/run_train_chrombpnet.sh" "${CLUSTER}" "${arm}")
    printf "  %-10s job %s  (after %s)\n" "${arm}" "${jid}" "${K562_JOB[$arm]}"
  fi
done

echo
echo "Progress:  ./check_runs.sh          Then:  ./summarize.sh, ./plot_summarize.sh --results ... --out ..."
