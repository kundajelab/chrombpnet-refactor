#!/bin/bash
# Submits the jobs that fill the last values in blog 2026-015:
#
#   h200  run_section4.sh on an H200 node (one job, ~20 min): the six
#         "H200: pending" parts of Section 4 (shared inputs, fixed-step
#         bigWigs, NumPy writer arguments, batched cast, HDF5 on threads,
#         GC matching).
#   l40s  run_headtohead.sh on an L40S (array 0-4, one task per fold): the four
#         "Pending" cells of the comparison table, and new contribs_bw totals
#         for both packages (the table's 385 s and 256 s predate gzip 9).
#
#   bash submit_blog_remaining.sh            # both
#   bash submit_blog_remaining.sh h200       # Section 4 on H200 only
#   bash submit_blog_remaining.sh l40s       # head-to-head only
#
# Both import the main checkout (CHROMBPNET_REPO) when they run, so don't
# change it while they are queued or running. site.env: SBATCH_H200 and/or
# SBATCH_L40S (the sbatch options that select that GPU), SBATCH_SITE_ARGS, LOG_DIR.
# When they finish:
#   python summarize_section4.py <OPTIMIZATION_OUT>   # Section 4 cells, L40S and H200 side by side
#   python summarize_rerun.py <OPTIMIZATION_OUT>      # head-to-head medians (run tag *_headtohead)

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

which="${1:-both}"
case "${which}" in
  both|h200|l40s) ;;
  *) echo "usage: $0 [both|h200|l40s]" >&2; exit 2 ;;
esac
SITE_NEEDS=()
[[ "${which}" == l40s ]] || SITE_NEEDS+=(SBATCH_H200)
[[ "${which}" == h200 ]] || SITE_NEEDS+=(SBATCH_L40S)
require_site "${SITE_NEEDS[@]}"
# shellcheck disable=SC2086  # the site's options are word-split on purpose
if [[ "${which}" != l40s ]]; then
  echo -n "Section 4, H200: "
  log_args %j
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} "${LOGS[@]}" run_section4.sh slurm
fi
# shellcheck disable=SC2086
if [[ "${which}" != h200 ]]; then
  echo -n "head-to-head, L40S: "
  log_args %A_%a
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} "${LOGS[@]}" run_headtohead.sh slurm
fi
