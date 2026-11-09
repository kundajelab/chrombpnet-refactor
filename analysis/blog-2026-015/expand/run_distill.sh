#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=expand_distill
#SBATCH --output=expand_distill.%A_%a.out
#SBATCH --error=expand_distill.%A_%a.err

# Distill a dataset's five engines-sweep PyTorch fold models (the teachers)
# into one bias-free student, then score the student as the teachers were.
#
#   sbatch --array=0-19 run_distill.sh slurm lofo --container         # one per dataset and fold
#   sbatch --array=0-3  run_distill.sh slurm final --container        # one per dataset, after lofo
#   sbatch --array=0-19 run_distill.sh slurm final-score --container  # after final
#   SLURM_ARRAY_TASK_ID=0 ./run_distill.sh local lofo
#
# lofo   (array over the ids table's rows) student k: trained on fold k's
#        training chromosomes, with early stopping on fold k's validation
#        chromosomes; never sees fold k's test chromosomes. The four teachers
#        j != k did train on them, so its fold-k test scores are mildly
#        optimistic. Inputs are teacher k's own: auxiliary/filtered.peaks.bed,
#        filtered.nonpeaks.bed, data_unstranded.bw and fold_k.json.
#        -> ${ROOT}/distill/lofo/<ASSAY>/<ENCID>/fold_k/
# final  (array over datasets, in the ids table's order) one student per
#        dataset on every chromosome, for the median over its LOFO students
#        of the epochs they kept (EPOCHS=N overrides), on fold 0's inputs
#        (the folds' peak files differ by a few outliers; the bigwig is the
#        same). Training only: final-score scores it.
#        -> ${ROOT}/distill/final/<ASSAY>/<ENCID>/student/
# final-score  (array over the ids table's rows) the final student of the
#        row's dataset, scored on the row's fold k, whose test chromosomes it
#        trained on. A job of its own per fold, because DeepSHAP on the 30K
#        subsample alone takes ~6 h on a TITAN V.
#        -> ${ROOT}/distill/final/<ASSAY>/<ENCID>/fold_k/
#
# Scoring each fold k (evaluate_student in _common.sh): the student joined
# with teacher k's bias_model_scaled.pt, predicted on teacher k's test
# peaks; DeepSHAP on teacher k's 30K subsample; marginal footprints. Each
# fold_k/ is then laid out as a pipeline run's (model/models,
# model/evaluation, model/auxiliary/interpret_subsample, timings.tsv,
# COMPLETED), so ../../expand/run_summarize.sh reads it as another run.
#
# Resubmitting is safe and resumes: a finished fold skips itself, a student
# already trained (model/models/DISTILLED, student/COMPLETED) is not trained
# again, only scored. FORCE=1 redoes everything (with final-score: that
# fold's scoring, never the student); FORCE_EVAL=1 only the scoring.
#
# TEACHER_SET=updated distills chrombpnet-refactor (updated)'s fold models
# (${ROOT}/new) into ${ROOT}/distill-updated/ instead (_common.sh); the paths
# below are then under distill-updated. SCORE=0 trains the students only
# (lofo, final), without scoring them on each fold.
#
# Environment: DRY_RUN=1, ALLOW_CPU=1, EXTRA_ARGS=".." (appended to
# `chrombpnet distill`, e.g. "--epochs 1"), EPOCHS=N (final: the epochs),
# SCORE_FOLDS="0 .." (final: also score these folds in the same job, a quick
# check).

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
EXPAND_DIR="${EXPAND_DIR:-${BLOG_SCRIPTS}/expand}"
[[ -f "${EXPAND_DIR}/_common.sh" ]] || { echo "Error: no _common.sh in ${EXPAND_DIR}; set EXPAND_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${EXPAND_DIR}/_common.sh"

parse_common_flags "$@"
CLUSTER_ARG=""; MODE=""
for a in ${ARGS[@]+"${ARGS[@]}"}; do
  case "${a}" in
    slurm|sherlock|local) CLUSTER_ARG="${a}" ;;
    lofo|final|final-score) MODE="${a}" ;;
    *) echo "unknown argument '${a}' (usage: run_distill.sh [slurm|local] lofo|final|final-score [--container])" >&2; exit 2 ;;
  esac
done
[[ -n "${MODE}" ]] || { echo "usage: run_distill.sh [slurm|local] lofo|final|final-score [--container]" >&2; exit 2; }
resolve_cluster "${CLUSTER_ARG}"
DISTILL_ROOT="$(distill_root)" || exit 2
SCORE="${SCORE:-1}"
export SCORE
if [[ "${SCORE}" == "0" && "${MODE}" == "final-score" ]]; then
  echo "Error: SCORE=0 with final-score scores nothing" >&2; exit 2
fi
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
export SLURM_ARRAY_TASK_ID   # through the container re-exec
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Error: SLURM_ARRAY_TASK_ID is unset" >&2; exit 2; }
if [[ "${MODE}" == "lofo" ]]; then
  select_row
  TRAIN_FOLD="${FOLD_NUM}"
  DATASET_DIR="${DISTILL_ROOT}/lofo/${ASSAY}/${ENCID}"
  STUDENT_RUN="${DATASET_DIR}/fold_${FOLD_NUM}"
  EVAL_FOLDS=("${FOLD_NUM}")
else
  if [[ "${MODE}" == "final" ]]; then
    select_dataset "${SLURM_ARRAY_TASK_ID}"
    EVAL_FOLDS=()
    [[ -n "${SCORE_FOLDS:-}" ]] && read -r -a EVAL_FOLDS <<< "${SCORE_FOLDS}"
  else
    select_row
    EVAL_FOLDS=("${FOLD_NUM}")
    # The student is shared by the dataset's five rows: never retrain it here.
    if [[ "${FORCE:-0}" == "1" ]]; then FORCE=0; FORCE_EVAL=1; fi
  fi
  TRAIN_FOLD=0
  DATASET_DIR="${DISTILL_ROOT}/final/${ASSAY}/${ENCID}"
  STUDENT_RUN="${DATASET_DIR}/student"
fi
# SCORE=0: train only, score no fold.
[[ "${SCORE}" == "0" ]] && EVAL_FOLDS=()
teacher_paths || exit 1
TEACHER_TRAIN="$(old_run_dir "${TRAIN_FOLD}")"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

echo "MODE:     ${MODE}"
echo "TEACHERS: ${TEACHER_SET} ($(teacher_tree)) -> ${DISTILL_ROOT}${SCORE:+, SCORE=${SCORE}}"
echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "STUDENT:  ${STUDENT_RUN}"
echo "TEACHERS: ${TEACHERS[*]}"
echo "INPUTS:   ${TEACHER_TRAIN}/model/auxiliary (fold ${TRAIN_FOLD})"
echo "SCORED:   folds ${EVAL_FOLDS[*]+${EVAL_FOLDS[*]}}"
echo "EXTRA_ARGS: ${EXTRA_ARGS:-<none>}"

# The student's checkpoint prefix: <run>/model/models/chrombpnet_nobias.{pt,log}
# for a lofo run (scored in place), <student>/chrombpnet_nobias.* for final.
if [[ "${MODE}" == "lofo" ]]; then PREFIX="${STUDENT_RUN}/model/models/chrombpnet_nobias"
else PREFIX="${STUDENT_RUN}/chrombpnet_nobias"; fi
DONE_MARK="$(dirname "${PREFIX}")/DISTILLED"
[[ "${MODE}" != "lofo" ]] && DONE_MARK="${STUDENT_RUN}/COMPLETED"

finished() {
  local k
  [[ -f "${DONE_MARK}" ]] || return 1
  for k in ${EVAL_FOLDS[@]+"${EVAL_FOLDS[@]}"}; do
    [[ -f "${DATASET_DIR}/fold_${k}/COMPLETED" ]] || return 1
  done
}
if [[ "${FORCE:-0}" != "1" && "${FORCE_EVAL:-0}" != "1" ]] && finished; then
  echo "exists: the student and every fold it scores are COMPLETED -- skipped (FORCE=1 redoes it)"
  exit 0
fi
if [[ "${MODE}" == "final-score" && ! -f "${DONE_MARK}" && -z "${DRY_RUN:-}" ]]; then
  echo "Error: the final student is not trained yet (${DONE_MARK}); run run_distill.sh final first" >&2
  exit 1
fi

distill_cmd() {
  DISTILL_CMD=(chrombpnet distill -t "${TEACHERS[@]}" -g "${GENOME_PATH}"
               -bw "${TEACHER_TRAIN}/model/auxiliary/data_unstranded.bw"
               -p "${TEACHER_TRAIN}/model/auxiliary/filtered.peaks.bed"
               -n "${TEACHER_TRAIN}/model/auxiliary/filtered.nonpeaks.bed"
               -fl "${REFERENCE_DIR}/folds/fold_${TRAIN_FOLD}.json" -op "${PREFIX}" -d "${ASSAY}"
               --device cuda "$@" ${EXTRA[@]+"${EXTRA[@]}"})
}

if [[ -n "${DRY_RUN:-}" ]]; then
  case "${MODE}" in
    final) distill_cmd --all-chromosomes --epochs "${EPOCHS:-<median LOFO epoch kept>}"
           echo "DRY_RUN: ${DISTILL_CMD[*]}" ;;
    lofo) distill_cmd; echo "DRY_RUN: ${DISTILL_CMD[*]}" ;;
    final-score) echo "DRY_RUN: the student: ${PREFIX}.pt" ;;
  esac
  for k in ${EVAL_FOLDS[@]+"${EVAL_FOLDS[@]}"}; do echo "DRY_RUN: score fold ${k} -> ${DATASET_DIR}/fold_${k}"; done
  exit 0
fi

if needs_container; then
  export EXPAND_DIR
  container_run pytorch "${CLUSTER}" bash "${EXPAND_DIR}/run_distill.sh" "${CLUSTER}" "${MODE}"
  exit $?
fi

activate_engine pytorch "${CLUSTER}"
gpu_preflight pytorch
runtime_check pytorch

## DISTILL ---------------------------------------------------------------------
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -rf "${STUDENT_RUN}"
  for k in ${EVAL_FOLDS[@]+"${EVAL_FOLDS[@]}"}; do rm -rf "${DATASET_DIR}/fold_${k}"; done
fi
if [[ -f "${DONE_MARK}" ]]; then
  echo "exists: ${DONE_MARK} -- the student is trained; scoring only"
else
  if [[ "${MODE}" == "final" ]]; then
    if [[ -z "${EPOCHS:-}" ]]; then
      # The median over the dataset's LOFO students of the epoch kept, +1:
      # how many epochs each ran before the one it would have stopped on.
      EPOCHS="$(python - "${DISTILL_ROOT}/lofo/${ASSAY}/${ENCID}" <<'PY'
import statistics, sys, torch
kept = []
for k in range(5):
    path = f"{sys.argv[1]}/fold_{k}/model/models/chrombpnet_nobias.pt"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        sys.exit(f"Error: LOFO student {k} has not been trained ({path}); set EPOCHS=N to go on without it")
    kept.append(int(payload["provenance"]["best_epoch"]) + 1)
print(int(statistics.median(kept)))
print("LOFO students' epochs kept:", kept, file=sys.stderr)
PY
)"
    fi
    echo "EPOCHS: ${EPOCHS} (all chromosomes)"
    distill_cmd --all-chromosomes --epochs "${EPOCHS}"
  else
    distill_cmd
  fi
  rm -rf "${STUDENT_RUN}"
  mkdir -p "$(dirname "${PREFIX}")"
  write_provenance "${STUDENT_RUN}/provenance.txt"
  echo "command: ${DISTILL_CMD[*]}" >> "${STUDENT_RUN}/provenance.txt"
  timed "${STUDENT_RUN}" distill "${DISTILL_CMD[@]}"
  [[ -f "${PREFIX}.pt" ]] || { echo "Error: ${PREFIX}.pt was not written" >&2; exit 1; }
  date '+%Y-%m-%d %H:%M:%S' > "${DONE_MARK}"
fi

## SCORE -----------------------------------------------------------------------
for k in ${EVAL_FOLDS[@]+"${EVAL_FOLDS[@]}"}; do
  RUN="${DATASET_DIR}/fold_${k}"
  if [[ -f "${RUN}/COMPLETED" && "${FORCE_EVAL:-0}" != "1" ]]; then
    echo "exists: ${RUN}/COMPLETED -- skipped"
    continue
  fi
  if [[ "${MODE}" == "lofo" ]]; then
    # Keep the student; redo the scoring from a clean slate.
    rm -rf "${RUN}/model/evaluation" "${RUN}/model/auxiliary" "${RUN}/model/models/chrombpnet.pt" "${RUN}/COMPLETED"
    awk -F'\t' 'NR == 1 || $1 == "distill"' "${RUN}/timings.tsv" > "${RUN}/timings.tsv.tmp" \
      && mv "${RUN}/timings.tsv.tmp" "${RUN}/timings.tsv"
  else
    rm -rf "${RUN}"
    mkdir -p "${RUN}/model/models"
    write_provenance "${RUN}/provenance.txt"
    echo "student: ${PREFIX}.pt (all chromosomes; fold ${k}'s test chromosomes were seen)" >> "${RUN}/provenance.txt"
    ln -s "${PREFIX}.pt" "${RUN}/model/models/chrombpnet_nobias.pt"
    ln -s "${PREFIX}.log" "${RUN}/model/models/chrombpnet_nobias.log"
  fi
  evaluate_student "${RUN}" "${k}"
  date '+%Y-%m-%d %H:%M:%S' > "${RUN}/COMPLETED"
  echo "Output: ${RUN}"
done
echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
