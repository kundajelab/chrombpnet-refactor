#!/bin/bash
# Blog 2026-015 Section 3, L40S column with the kit's Triton TF32 forward: stock, its
# stage timings (the per-lever cells' stock side), the kit's `fast` as shipped (TensorFlow's
# forward on an L40S) and `fast_k1` (the same mode with the Triton TF32 forward, via
# kit_hook_l40s_k1fast/), all in one job, 5 rounds after an untimed warm-up.
#
#   bash submit_l40s_kit_fast_k1.sh
#   python summarize_sections34.py <OPTIMIZATION_OUT>   # the NVIDIA-L40S_cuda-89_sections34-fastk1 section
# site.env: SBATCH_L40S, SBATCH_SITE_ARGS, LOG_DIR.
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"   # sbatch from this folder: the jobs find ../_site.sh through it
if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
# Slurm logs into site.env's LOG_DIR when it names one (else in this folder).
log_args () {  # %j | %A_%a -> LOGS
  LOGS=()
  [[ -n "${LOG_DIR:-}" ]] && LOGS=(-o "${LOG_DIR}/%x.$1.out" -e "${LOG_DIR}/%x.$1.err")
  return 0
}
require_site SBATCH_L40S
log_args %j
# shellcheck disable=SC2086  # the site's options are word-split on purpose
ARMS=stock,stages,fast,fast_k1 RUN_KIND=sections34-fastk1 \
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} "${LOGS[@]}" --time=02:00:00 --job-name=chrombpnet_kit_fastk1 run_sections34.sh slurm
