#!/bin/bash
# Ensemble vs distillation for one teacher set: distill its students, score
# them and its five fold models on shared regions, and compare them.
#
#   ./submit_shared.sh sherlock             # TEACHER_SET=updated (the default here)
#   ./submit_shared.sh sherlock --dry-run   # print the sbatch lines only
#   TEACHER_SET=default ONLY="scores compare" ./submit_shared.sh sherlock
#                                           # the control: the existing students
#
# The jobs run natively, in SHERLOCK_ENV (_common.sh); --container runs them
# in the Ubuntu image with chrombpnet-pytorch instead.
#
#   lofo      run_distill.sh lofo    array 0-19, SCORE=0 (students only, no per-fold scoring)
#   final     run_distill.sh final   array 0-3,  SCORE=0, after the whole lofo array
#   scores    run_shared_scores.sh   array 0-43 (4 datasets x 11 models), after final
#   compare   run_compare_shared.sh  CPU, after scores
#
# Each stage waits for the previous one with --dependency=afterok, so one
# failed task cancels the stages after it (--kill-on-invalid-dep): fix it and
# resubmit with ONLY= the stages still to run; finished tasks skip themselves.
#
# Environment:
#   TEACHER_SET=updated|default   whose students (default: updated)
#   ONLY="lofo final scores compare"  which stages (default: all four)
#   N_REGIONS=10000               shared peaks per dataset
#   GPU_CONSTRAINT=..             --constraint for every GPU job (default none)
#   SBATCH_EXTRA=".."             more sbatch options for every job

set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
TEACHER_SET="${TEACHER_SET:-updated}"
export TEACHER_SET
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

parse_common_flags "$@"
CLUSTER="sherlock"
DRY=""
for arg in ${ARGS[@]+"${ARGS[@]}"}; do
  case "${arg}" in
    --dry-run) DRY=1 ;;
    sherlock|local) CLUSTER="${arg}" ;;
    *) echo "unknown argument '${arg}' (usage: submit_shared.sh [sherlock] [--container] [--dry-run])" >&2; exit 2 ;;
  esac
done
teacher_tree > /dev/null || exit 2
CONTAINER_ARG=()
(( USE_CONTAINER )) && CONTAINER_ARG=(--container)
read -r -a ONLY_LIST <<< "${ONLY:-lofo final scores compare}"
wants() { local x; for x in "${ONLY_LIST[@]}"; do [[ "${x}" == "$1" ]] && return 0; done; return 1; }
N_REGIONS="${N_REGIONS:-10000}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-}"
read -r -a SBATCH_EXTRA_OPT <<< "${SBATCH_EXTRA:-}"
GPU_OPT=(${SBATCH_EXTRA_OPT[@]+"${SBATCH_EXTRA_OPT[@]}"})
[[ -n "${GPU_CONSTRAINT}" ]] && GPU_OPT+=(--constraint="${GPU_CONSTRAINT}")
EXPORT="--export=ALL,TEACHER_SET=${TEACHER_SET},SCORE=0,N_REGIONS=${N_REGIONS}"
n_rows="$(data_row_count "${IDS_FILE}")"
n_datasets="$(dataset_list | wc -l)"

submit() {
  if [[ -n "${DRY}" ]]; then
    echo "sbatch --parsable $*" >&2
    echo "<jobid>"
  else
    sbatch --parsable "$@"
  fi
}
DEPENDENCY=()
after() { DEPENDENCY=(--dependency="afterok:$1" --kill-on-invalid-dep=yes); }

echo "=== ensemble vs distillation: TEACHER_SET=${TEACHER_SET} ($(teacher_tree)), students in $(distill_root)"
if wants lofo; then
  job="$(submit --array="0-$(( n_rows - 1 ))" "${EXPORT}" ${GPU_OPT[@]+"${GPU_OPT[@]}"} \
                --job-name="expand_distill_lofo_${TEACHER_SET}" \
                "${HERE}/run_distill.sh" "${CLUSTER}" lofo ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "  LOFO students:        job ${job}"; after "${job}"
fi
if wants final; then
  job="$(submit --array="0-$(( n_datasets - 1 ))" "${EXPORT}" ${GPU_OPT[@]+"${GPU_OPT[@]}"} \
                ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} --job-name="expand_distill_final_${TEACHER_SET}" \
                "${HERE}/run_distill.sh" "${CLUSTER}" final ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "  final students:       job ${job}"; after "${job}"
fi
if wants scores; then
  job="$(submit --array="0-$(( n_datasets * 11 - 1 ))" "${EXPORT}" ${GPU_OPT[@]+"${GPU_OPT[@]}"} \
                ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} --job-name="expand_shared_${TEACHER_SET}" \
                "${HERE}/run_shared_scores.sh" "${CLUSTER}" ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "  shared-region scores: job ${job}"; after "${job}"
fi
if wants compare; then
  job="$(submit "${EXPORT}" ${SBATCH_EXTRA_OPT[@]+"${SBATCH_EXTRA_OPT[@]}"} ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
                --job-name="expand_compare_shared_${TEACHER_SET}" "${HERE}/run_compare_shared.sh")"
  echo "  comparison:           job ${job}"
fi
echo
echo "Results: ${ROOT}/analysis/fold-mean-${TEACHER_SET}/"
