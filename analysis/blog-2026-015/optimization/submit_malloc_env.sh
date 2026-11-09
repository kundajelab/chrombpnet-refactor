#!/bin/bash
# Submits the runs that fill blog 2026-015's kit-table row 8 (`malloc_env`) on
# the GPU, per GPU (L40S, H200):
#
#   original   run_kit_modes.sh, ARMS=stock,stockmalloc: stock chrombpnet 1.0.1
#              (the kit's pinned package and model, 20,000 peaks, -bs 256) with
#              and without the kit's three glibc variables, 5 rounds; the row's
#              "vs. original chrombpnet" cells (wall time and peak memory).
#              One job, ~25 min. Run tag *_kit-malloc.
#   refactor   run_pred_bw_kit_fused.sh, ARMS=nomalloc,main: pred_bw with glibc's
#              default allocator (--no-malloc-env) and as shipped, which since
#              d407ca8 applies malloc_env itself; 5 folds x 3 rounds; the bracketed
#              chrombpnet-refactor numbers and the peak memory in the "Applied?"
#              cell. Array 0-4, ~15 min a task. Run tag *_kit-fused-malloc.
#
# Each run records peak_rss_mb (peak_rss.py: the peak resident memory of its
# largest process), so the memory now comes from the GPU runs themselves, not
# the CPU check. The existing kit-modes / kit-fused results are not touched.
#
#   bash submit_malloc_env.sh            # both GPUs
#   bash submit_malloc_env.sh l40s       # L40S only
#   bash submit_malloc_env.sh h200       # H200 only (on a shared partition: may wait, may be preempted)
#   bash submit_malloc_env.sh h200 refactor   # only the refactor job (original | refactor | both)
#
# The refactor jobs import the main checkout (CHROMBPNET_REPO) when they run,
# so don't change it while they are queued or running. site.env: SBATCH_L40S
# and/or SBATCH_H200 (the sbatch options that select that GPU), SBATCH_SITE_ARGS
# (e.g. nodes to exclude), LOG_DIR.
# When they finish:
#   python summarize_rerun.py <OPTIMIZATION_OUT>   # sections *_kit-malloc (stockmalloc vs stock) and
#                                                  # *_kit-fused-malloc (main vs nomalloc), with peak GB

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
part="${2:-both}"
case "${which}" in
  both|l40s|h200) ;;
  *) echo "usage: $0 [both|l40s|h200] [both|original|refactor]" >&2; exit 2 ;;
esac
case "${part}" in
  both|original|refactor) ;;
  *) echo "usage: $0 [both|l40s|h200] [both|original|refactor]" >&2; exit 2 ;;
esac

SITE_NEEDS=()
[[ "${which}" == h200 ]] || SITE_NEEDS+=(SBATCH_L40S)
[[ "${which}" == l40s ]] || SITE_NEEDS+=(SBATCH_H200)
require_site "${SITE_NEEDS[@]}"

# shellcheck disable=SC2086  # the site's options are word-split on purpose
submit () {  # label gpu-options
  local label="$1" gpu="$2"
  if [[ "${part}" != refactor ]]; then
    echo -n "original +/- malloc_env, ${label}: "
    log_args %j
    ARMS=stock,stockmalloc RUN_KIND=kit-malloc \
      sbatch ${SBATCH_SITE_ARGS} ${gpu} "${LOGS[@]}" --time=02:00:00 \
      --job-name=chrombpnet_kit_malloc run_kit_modes.sh slurm
  fi
  if [[ "${part}" != original ]]; then
    echo -n "refactor +/- malloc_env, ${label}: "
    log_args %A_%a
    ARMS=nomalloc,main RUN_KIND=kit-fused-malloc \
      sbatch ${SBATCH_SITE_ARGS} ${gpu} "${LOGS[@]}" --time=02:00:00 \
      --job-name=chrombpnet_predbw_malloc run_pred_bw_kit_fused.sh slurm
  fi
}

[[ "${which}" == h200 ]] || submit L40S "${SBATCH_L40S}"
[[ "${which}" == l40s ]] || submit H200 "${SBATCH_H200}"
