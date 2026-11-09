#!/bin/bash
# Blog 2026-015, Section 4's Total and the comparison table's pred_bw row, head to head
# at the original's HDF5 compression, with chrombpnet-refactor's own command line:
# chrombpnet 1.x (stock, gzip 4 as it ships), chrombpnet-refactor with
# --h5-gzip-level 4 --no-h5-shuffle (refactor4: the PyTorch TF32 forward; refactor4_fast:
# plus --predict-fused-kernel-fast), and chrombpnet-refactor at its default gzip 9 with
# shuffle (refactor), on an L40S, all in one job, 5 rounds after an untimed warm-up.
#
#   bash submit_gzip4_headtohead.sh
#   python summarize_sections34.py <OPTIMIZATION_OUT>   # the NVIDIA-L40S_cuda-89_sections34-gzip4 section
# site.env: SBATCH_L40S, SBATCH_SITE_ARGS, LOG_DIR. The refactor arms import this checkout.
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
# The refactor arms need the options in the checkout they import.
grep -q -- "--h5-gzip-level" "${CHROMBPNET_REPO}/chrombpnet/parsers.py" || {
  echo "submit_gzip4_headtohead.sh: ${CHROMBPNET_REPO} has no --h5-gzip-level" >&2; exit 2; }
log_args %j
# shellcheck disable=SC2086  # the site's options are word-split on purpose
ARMS=stock,refactor4,refactor4_fast,refactor RUN_KIND=sections34-gzip4 \
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} "${LOGS[@]}" --time=02:00:00 --job-name=chrombpnet_gzip4 run_sections34.sh slurm
