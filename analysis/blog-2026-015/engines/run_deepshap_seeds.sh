#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --job-name=engines_deepshap_seeds
#SBATCH --output=engines_deepshap_seeds.%A_%a.out
#SBATCH --error=engines_deepshap_seeds.%A_%a.err
#SBATCH --array=0-19

# The original chrombpnet's DeepSHAP on one row's original model, once per
# seed: how far DeepSHAP on the same model moves with its shuffled references
# alone (run_deepshap_seeds.py says how, and why).
#
#   ./submit_all.sh slurm --deepshap-seeds        # every row, one array per seed
#   sbatch <SBATCH_SHARED_GPU_ARGS> run_deepshap_seeds.sh slurm
#   SLURM_ARRAY_TASK_ID=1 ./run_deepshap_seeds.sh local
#
# Rows as for run_train_full.sh. It runs in the chrombpnet-tensorflow env
# (TF 2.8 and the kundajelab shap fork, which the original needs), with the
# original package imported from ORIGINAL_REPO, not chrombpnet-refactor.
# run_summarize.sh then adds them to contributions.tsv (the original against
# itself across seeds, and against the stored scores), and plot_summarize.sh
# draws them as lines over the engines' bars.
#
# Environment:
#   SEEDS="1 2 3"      seeds to score (default: 1 2 3); COMPLETED ones are skipped
#   N_REGIONS=N        only the first N of the 30K subsample's peaks (default: all;
#                      about 4 h per seed on a P100 or an L40S)
#   HEADS="counts"     heads to score (default: counts profile)
#   FORCE=1            rescore COMPLETED seeds
#   DRY_RUN=1          resolve and print, run nothing
#   ALLOW_CPU=1        skip the GPU preflight check
#   SEEDS_ROOT=..      output tree (default: ${ROOT}/original)
#
# From site.env (each can be overridden from the environment too):
#   ORIGINAL_REPO      the original package, kundajelab/chrombpnet at 09938fd
#   ENCODE_MODELS      where the original models are: the model is
#                      <dir>/<ASSAY>/<ENCID>/models/fold_N/model.chrombpnet_nobias.fold_N.<ENCID>.h5,
#                      the ENCODE release, whose genome-wide scores are the
#                      originals run_summarize.py compares against
#
# Output: ${SEEDS_ROOT}/<ASSAY>/<ENCID>/fold_N/regions.bed and
#   .../seed_<s>/scores.{counts,profile}_scores.h5 (+ run.json, COMPLETED)

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
ENGINES_DIR="${ENGINES_DIR:-${BLOG_SCRIPTS}/engines}"
[[ -f "${ENGINES_DIR}/run_deepshap_seeds.py" ]] || { echo "Error: no run_deepshap_seeds.py in ${ENGINES_DIR}; set ENGINES_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${ENGINES_DIR}/_common.sh"

parse_common_flags "$@"
resolve_cluster "${ARGS[0]:-}"
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
select_row

# The original runs in the TensorFlow engine's env.
ENGINE="tensorflow"
require_site ORIGINAL_REPO ENCODE_MODELS
ORIGINAL_MODEL="${ENCODE_MODELS}/${ASSAY}/${ENCID}/models/fold_${FOLD_NUM}/model.chrombpnet_nobias.fold_${FOLD_NUM}.${ENCID}.h5"
SUBSAMPLE="${ROOT}/pytorch/${ASSAY}/${ENCID}/fold_${FOLD_NUM}/model/auxiliary/interpret_subsample/chrombpnet_nobias.interpreted_regions.bed"
SEEDS_ROOT="${SEEDS_ROOT:-${ROOT}/original}"
OUT_DIR="${SEEDS_ROOT}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3}"
read -r -a HEAD_LIST <<< "${HEADS:-counts profile}"
N_REGIONS="${N_REGIONS:-0}"
PY_ARGS=(--model "${ORIGINAL_MODEL}" --regions-source "${SUBSAMPLE}" --n-regions "${N_REGIONS}"
         --genome "${GENOME_PATH}" --out "${OUT_DIR}" --seeds "${SEED_LIST[@]}"
         --heads "${HEAD_LIST[@]}" --original-repo "${ORIGINAL_REPO}")
[[ "${FORCE:-0}" == "1" ]] && PY_ARGS+=(--force)

echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "MODEL:    ${ORIGINAL_MODEL}"
echo "REGIONS:  $( (( N_REGIONS > 0 )) && echo "first ${N_REGIONS} of" || echo "all of") ${SUBSAMPLE}"
echo "SEEDS:    ${SEED_LIST[*]}"
echo "HEADS:    ${HEAD_LIST[*]}"
echo "ORIGINAL: ${ORIGINAL_REPO}"
echo "OUT_DIR:  ${OUT_DIR}"
echo "CONTAINER: $( (( USE_CONTAINER )) && echo "yes (${CONTAINER_IMAGE})" || echo no)"

for f in "${ORIGINAL_MODEL}" "${SUBSAMPLE}" "${ORIGINAL_REPO}/chrombpnet/evaluation/interpret/interpret.py"; do
  [[ -e "${f}" ]] || { echo "Error: not found: ${f}" >&2; exit 2; }
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: python ${ENGINES_DIR}/run_deepshap_seeds.py ${PY_ARGS[*]}"
  exit 0
fi

if needs_container; then
  export ENGINES_DIR
  container_run "${ENGINE}" "${CLUSTER}" bash "${ENGINES_DIR}/run_deepshap_seeds.sh" "${CLUSTER}"
  exit $?
fi

activate_engine "${ENGINE}" "${CLUSTER}"
gpu_preflight "${ENGINE}"

start="$(date +%s)"
python "${ENGINES_DIR}/run_deepshap_seeds.py" "${PY_ARGS[@]}"
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Completed run in $(( ($(date +%s) - start) / 60 )) min. ($(date '+%Y-%m-%d %H:%M:%S'))"
