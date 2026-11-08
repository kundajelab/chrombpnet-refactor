#!/bin/bash
# Submit the comparison: the new version's 20 runs, the 20 LOFO students, the
# 4 final students (after the LOFO students, whose epochs they reuse), and the
# final students' scoring on each of the 20 folds (after the final students).
#
#   ./submit_all.sh sherlock --container            # everything
#   ./submit_all.sh sherlock --container --dry-run  # print the sbatch lines only
#
#   run_train_new.sh          array 0-19, GPU
#   run_distill.sh lofo       array 0-19, GPU (independent of the new runs: the
#                             teachers are the engines sweep's)
#   run_distill.sh final      array 0-3,  GPU, after the whole lofo array
#                             succeeded (--dependency=afterok)
#   run_distill.sh final-score  array 0-19, GPU, after the whole final array
#
# Then, once all are done:  sbatch run_summarize.sh
#
# Resubmitting is safe: finished tasks skip themselves, and a distill task
# that hit the time limit resumes from its last finished step.
#
# Environment:
#   ONLY="new lofo final final-score"  which arrays (default: all four)
#   ARRAY=0-4             only these rows (new, lofo, final-score; 0-based)
#   GPU_CONSTRAINT=..     --constraint for the training jobs (default none,
#                         as in the engines sweep)
#   DISTILL_GPU_CONSTRAINT=..  for the distill jobs (default none)
#   SBATCH_EXTRA=".."     more sbatch options for every job, word-split

set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

parse_common_flags "$@"
CLUSTER="sherlock"
DRY=""
for arg in ${ARGS[@]+"${ARGS[@]}"}; do
  case "${arg}" in
    --dry-run) DRY=1 ;;
    sherlock|local) CLUSTER="${arg}" ;;
    *) echo "unknown argument '${arg}' (usage: submit_all.sh [sherlock] [--container] [--dry-run])" >&2; exit 2 ;;
  esac
done
CONTAINER_ARG=()
(( USE_CONTAINER )) && CONTAINER_ARG=(--container)
read -r -a ONLY_LIST <<< "${ONLY:-new lofo final final-score}"
wants() { local x; for x in "${ONLY_LIST[@]}"; do [[ "${x}" == "$1" ]] && return 0; done; return 1; }

if [[ ! -e "${ROOT}/preprocess" ]]; then
  echo "Error: ${ROOT}/preprocess is missing; link it to ${ENGINES_ROOT}/preprocess first:" >&2
  echo "  mkdir -p ${ROOT} && ln -s ${ENGINES_ROOT}/preprocess ${ROOT}/preprocess" >&2
  exit 1
fi

read -r -a SBATCH_EXTRA_OPT <<< "${SBATCH_EXTRA:-}"
ARRAY_OPT=(--array="${ARRAY:-0-$(( $(data_row_count "${IDS_FILE}") - 1 ))}")
TRAIN_OPT=(${SBATCH_EXTRA_OPT[@]+"${SBATCH_EXTRA_OPT[@]}"})
[[ -n "${GPU_CONSTRAINT:-}" ]] && TRAIN_OPT+=(--constraint="${GPU_CONSTRAINT}")
DISTILL_GPU_CONSTRAINT="${DISTILL_GPU_CONSTRAINT:-}"
DISTILL_OPT=(${SBATCH_EXTRA_OPT[@]+"${SBATCH_EXTRA_OPT[@]}"})
[[ -n "${DISTILL_GPU_CONSTRAINT}" ]] && DISTILL_OPT+=(--constraint="${DISTILL_GPU_CONSTRAINT}")

submit() {
  if [[ -n "${DRY}" ]]; then
    echo "sbatch --parsable $*" >&2
    echo "<jobid>"
  else
    sbatch --parsable "$@"
  fi
}

if wants new; then
  job="$(submit "${ARRAY_OPT[@]}" ${TRAIN_OPT[@]+"${TRAIN_OPT[@]}"} "${HERE}/run_train_new.sh" "${CLUSTER}" \
                ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "=== new version (profile-only bias, pearsonr-jsd selection): job ${job}"
fi
DEPENDENCY=()
if wants lofo; then
  lofo_job="$(submit "${ARRAY_OPT[@]}" ${DISTILL_OPT[@]+"${DISTILL_OPT[@]}"} --job-name=expand_distill_lofo \
                     "${HERE}/run_distill.sh" "${CLUSTER}" lofo ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "=== LOFO students: job ${lofo_job}"
  DEPENDENCY=(--dependency="afterok:${lofo_job}" --kill-on-invalid-dep=yes)
fi
if wants final; then
  n_datasets="$(dataset_list | wc -l)"
  final_job="$(submit --array="0-$(( n_datasets - 1 ))" ${DISTILL_OPT[@]+"${DISTILL_OPT[@]}"} \
                      ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} --job-name=expand_distill_final \
                      "${HERE}/run_distill.sh" "${CLUSTER}" final ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "=== final students (${n_datasets} datasets): job ${final_job}"
  DEPENDENCY=(--dependency="afterok:${final_job}" --kill-on-invalid-dep=yes)
fi
if wants final-score; then
  score_job="$(submit "${ARRAY_OPT[@]}" ${DISTILL_OPT[@]+"${DISTILL_OPT[@]}"} ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
                      --job-name=expand_distill_final_score \
                      "${HERE}/run_distill.sh" "${CLUSTER}" final-score ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "=== final students scored per fold: job ${score_job}"
fi
echo
echo "Then:  sbatch ${HERE}/run_summarize.sh"
