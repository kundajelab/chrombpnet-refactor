#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=expand_shared
#SBATCH --output=/path/to/workdir/sherlock/logs/expand_shared.%A_%a.out
#SBATCH --error=/path/to/workdir/sherlock/logs/expand_shared.%A_%a.err
#SBATCH --array=0-43

# Score one bias-free model on its dataset's shared regions: DeepSHAP (both
# heads) and predictions (analysis/expand/score_shared_regions.py).
#
#   TEACHER_SET=updated sbatch run_shared_scores.sh sherlock   # 44 tasks, natively (_common.sh)
#   TEACHER_SET=updated SLURM_ARRAY_TASK_ID=0 ./run_shared_scores.sh local
#
# Task i is dataset i / 11 (the ids table's order) and model i % 11:
#   0-4   fold_0 .. fold_4   the TEACHER_SET's fold models (the teachers)
#   5-9   lofo_0 .. lofo_4   its LOFO students (run_distill.sh lofo)
#   10    final              its final student (run_distill.sh final)
# TEACHER_SET (_common.sh): updated (${ROOT}/new, students in distill-updated)
# or default (the engines sweep, students in distill).
#
# The shared regions, ${OUT}/regions.bed: N_REGIONS (default 10000) peaks
# drawn with seed 1234 from those in all five fold models'
# auxiliary/filtered.peaks.bed. Whichever task comes first writes them; the
# draw is deterministic, so every task has the same file.
#
# Output, ${ROOT}/shared/${TEACHER_SET}/<ASSAY>/<ENCID>/<model>.* (SHARED_ROOT=...
# replaces ${ROOT}/shared, e.g. for a quick check with N_REGIONS=200); a model
# already scored (<model>.COMPLETED) is skipped, FORCE=1 rescores it.
# Environment: DRY_RUN=1, ALLOW_CPU=1, N_REGIONS=N.

set -euo pipefail

_find_here() {
  local candidate
  for candidate in "${EXPAND_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/expand" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/_common.sh" && -f "${candidate}/run_shared_scores.sh" ]] \
      && { echo "${candidate}"; return 0; }
  done
  return 1
}
EXPAND_DIR="$(_find_here)" || { echo "Error: cannot locate expand/_common.sh; set EXPAND_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${EXPAND_DIR}/_common.sh"

parse_common_flags "$@"
resolve_cluster "${ARGS[0]:-}"
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
export SLURM_ARRAY_TASK_ID
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Error: SLURM_ARRAY_TASK_ID is unset" >&2; exit 2; }
N_REGIONS="${N_REGIONS:-10000}"
export N_REGIONS
TREE="$(teacher_tree)" || exit 2
DISTILL_ROOT="$(distill_root)" || exit 2

MODELS=(fold_0 fold_1 fold_2 fold_3 fold_4 lofo_0 lofo_1 lofo_2 lofo_3 lofo_4 final)
select_dataset $(( SLURM_ARRAY_TASK_ID / ${#MODELS[@]} ))
LABEL="${MODELS[$(( SLURM_ARRAY_TASK_ID % ${#MODELS[@]} ))]}"
k="${LABEL#*_}"
case "${LABEL}" in
  fold_*) MODEL="${TREE}/${ASSAY}/${ENCID}/fold_${k}/model/models/chrombpnet_nobias.pt"
          READY="${TREE}/${ASSAY}/${ENCID}/fold_${k}/COMPLETED" ;;
  lofo_*) MODEL="${DISTILL_ROOT}/lofo/${ASSAY}/${ENCID}/fold_${k}/model/models/chrombpnet_nobias.pt"
          READY="$(dirname "${MODEL}")/DISTILLED" ;;
  final)  MODEL="${DISTILL_ROOT}/final/${ASSAY}/${ENCID}/student/chrombpnet_nobias.pt"
          READY="${DISTILL_ROOT}/final/${ASSAY}/${ENCID}/student/COMPLETED" ;;
esac
OUT="${SHARED_ROOT:-${ROOT}/shared}/${TEACHER_SET}/${ASSAY}/${ENCID}"
PEAKS=()
for j in "${FOLDS[@]}"; do PEAKS+=("${TREE}/${ASSAY}/${ENCID}/fold_${j}/model/auxiliary/filtered.peaks.bed"); done
SCORER="${CHROMBPNET_DIR}/analysis/expand/score_shared_regions.py"

echo "TEACHER_SET: ${TEACHER_SET} (${TREE}; students in ${DISTILL_ROOT})"
echo "DATASET:  ${ASSAY} ${ENCID}"
echo "MODEL:    ${LABEL} ${MODEL}"
echo "REGIONS:  ${OUT}/regions.bed (${N_REGIONS} shared peaks)"

if [[ -f "${OUT}/${LABEL}.COMPLETED" && "${FORCE:-0}" != "1" ]]; then
  echo "exists: ${OUT}/${LABEL}.COMPLETED -- skipped (FORCE=1 rescores)"
  exit 0
fi
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: model ready: $([[ -f "${READY}" && -f "${MODEL}" ]] && echo yes || echo "not yet (${READY})")"
  echo "DRY_RUN: python ${SCORER} regions --peaks ${PEAKS[*]} --n ${N_REGIONS} --out ${OUT}/regions.bed"
  echo "DRY_RUN: python ${SCORER} score --model ${MODEL} --regions ${OUT}/regions.bed --genome ${GENOME_PATH} --out-prefix ${OUT}/${LABEL}"
  exit 0
fi
[[ -f "${READY}" && -f "${MODEL}" ]] || { echo "Error: the model is not ready (${READY}, ${MODEL})" >&2; exit 1; }
for f in "${PEAKS[@]}"; do [[ -f "${f}" ]] || { echo "Error: missing ${f}" >&2; exit 1; }; done

if needs_container; then
  export EXPAND_DIR
  container_run pytorch "${CLUSTER}" bash "${EXPAND_DIR}/run_shared_scores.sh" "${CLUSTER}"
  exit $?
fi

activate_engine pytorch "${CLUSTER}"
gpu_preflight pytorch

mkdir -p "${OUT}"
[[ -s "${OUT}/regions.bed" ]] || python "${SCORER}" regions --peaks "${PEAKS[@]}" --n "${N_REGIONS}" \
  --out "${OUT}/regions.bed"
start="$(date +%s)"
python "${SCORER}" score --model "${MODEL}" --regions "${OUT}/regions.bed" --genome "${GENOME_PATH}" \
  --out-prefix "${OUT}/${LABEL}" --device cuda
echo "=== scored in $(( ($(date +%s) - start) / 60 )) min on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
