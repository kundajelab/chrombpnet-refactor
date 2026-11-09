#!/bin/bash
# Blog 2026-015 comparison table, the "Best comparison" pred_bw row with
# chrombpnet-refactor on the optimization kit's Triton TF32 kernels: the original
# package and chrombpnet-refactor each predict with the full model over the 20,000
# peaks and score them against the observed bigWig, with no bigWig or predictions
# HDF5 written (headtohead.py), as cherimoya evaluate does. Arms, all in one job
# per fold (array 0-4 = folds 0-4), 3 rounds after an untimed warm-up:
#   orig_predict      chrombpnet 1.x (TensorFlow 2.8)
#   ref_predict       chrombpnet-refactor, standard PyTorch (TF32 convolutions in cuDNN)
#   ref_predict_fast  chrombpnet-refactor with --predict-fused-kernel-fast
# Run tag *_headtohead-fast, so the earlier *_headtohead results (the attribution
# rows) are kept.
#
#   bash submit_headtohead_fast.sh
#   python summarize_rerun.py <OPTIMIZATION_OUT>   # the NVIDIA-L40S_cuda-89_headtohead-fast section
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
log_args %A_%a
# shellcheck disable=SC2086  # the site's options are word-split on purpose
ARMS=orig_predict,ref_predict,ref_predict_fast RUN_KIND=headtohead-fast \
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} "${LOGS[@]}" --time=03:00:00 --job-name=chrombpnet_headtohead_fast run_headtohead.sh slurm
