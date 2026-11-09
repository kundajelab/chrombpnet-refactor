#!/bin/bash
# Submit every arm of the sweep: all 20 data rows, K562 (rows 1-10) and
# stomach (rows 11-20), one array per arm.
#
#   ./submit_all.sh slurm            # submit
#   ./submit_all.sh slurm --dry-run  # print the sbatch lines only
#
# Each task runs run_retrain_full.sh, which exits 0 when its run is already
# complete and otherwise hands the row to run_train_full.sh. So resubmitting
# after a partial sweep picks up only what is missing; run_train_full.sh on
# its own would delete and retrain a finished fold. Every row trains its own
# bias model, the stomach rows past the bias-model quality check with
# --ignore-bias-flag (run_train_full.sh's IGNORE_BIAS=1 default), as the
# blog's runs did. Borrowing a K562 bias model for them instead is
# run_train_chrombpnet.sh (README: "An alternative not used for the results").

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

echo "=== all 20 rows (K562 1-10, stomach 11-20), one array per arm ==="
for arm in "${ARMS[@]}"; do
  if [[ -n "$DRY" ]]; then
    echo "${SB[*]} --array=0-19 ${HERE}/run_retrain_full.sh ${CLUSTER} ${arm}"
  else
    jid=$("${SB[@]}" --array=0-19 "${HERE}/run_retrain_full.sh" "${CLUSTER}" "${arm}")
    printf "  %-10s job %s\n" "${arm}" "${jid}"
  fi
done

echo
echo "Progress:  ./check_runs.sh          Then:  ./summarize.sh, ./plot_summarize.sh --results ... --out ..."
