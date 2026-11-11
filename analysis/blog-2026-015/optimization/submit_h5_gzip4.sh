#!/bin/bash
# Blog 2026-015, Section 4 row 7 at the original's HDF5 compression: one model's
# *_predictions.h5 for the 20,000 peaks written by the original's gzip 4 (h5py's own,
# 4962484's writer), and by chrombpnet-refactor's threaded writer at gzip 4 without shuffle
# (--h5-gzip-level 4 --no-h5-shuffle) and at its default gzip 9 with shuffle, plus gzip 9 on
# one thread: run_section4.sh's h5 stage alone, on an L40S node, in one job (each in a fresh
# process, in a different order each round, 3 rounds). Its own run tag (*_h5-gzip4), so the
# earlier Section 4 job's medians are left as they are.
#
#   bash submit_h5_gzip4.sh
#   python summarize_section4.py <OPTIMIZATION_OUT>   # its last block: *_h5-gzip4
# site.env: SBATCH_L40S, SBATCH_SITE_ARGS, LOG_DIR. Exports code-4962484/ (the original's
# writer, run_section4.sh's baseline) from this repository if it is not there.
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
grep -q -- "--h5-gzip-level" "${CHROMBPNET_REPO}/chrombpnet/parsers.py" || {
  echo "submit_h5_gzip4.sh: ${CHROMBPNET_REPO} has no --h5-gzip-level" >&2; exit 2; }
if [[ ! -f code-4962484/chrombpnet/optimize.py ]]; then
  # (cd ...): Sherlock's git 1.8 has no `git -C`.
  mkdir -p code-4962484 && (cd "${CHROMBPNET_REPO}" && git archive 4962484) | tar -x -C code-4962484
fi
log_args %j
# shellcheck disable=SC2086  # the site's options are word-split on purpose
STAGES=h5 RUN_KIND=h5-gzip4 \
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} "${LOGS[@]}" --time=01:00:00 --job-name=chrombpnet_h5_gzip4 run_section4.sh slurm
