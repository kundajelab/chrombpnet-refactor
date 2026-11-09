#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=engines_contribution
#SBATCH --output=engines_contribution.%A_%a.out
#SBATCH --error=engines_contribution.%A_%a.err
#SBATCH --array=0-19

# Score one finished run's 30K peak subsample again with DeepSHAP and keep the
# scores (run_contribution.py says how, and why).
#
#   ./submit_all.sh slurm --container --contribution   # every engine, every row
#   sbatch run_contribution.sh slurm <engine> [--container]
#   SLURM_ARRAY_TASK_ID=5 ./run_contribution.sh local jax
#
# Arguments as for run_train_full.sh. Runs that have not finished are
# skipped; heads already scored are skipped (FORCE=1 rescores them).
# submit_all.sh --contribution adds what sbatch needs per engine: site.env's
# JAX_GPU_CONSTRAINT for JAX; TF_GPU_CONSTRAINT and SBATCH_SHARED_GPU_ARGS for
# TensorFlow (its DeepSHAP runs in a second process on the GPU).
#
# Environment:
#   HEADS="counts"   heads to score (default: counts profile; existing ones skipped)
#   PROFILE_WEIGHTING=midpoint|stopgrad
#                    rescore the profile head (only) with that weighting of the
#                    profile target, into chrombpnet_nobias.<weighting>.profile_scores.h5;
#                    the pipeline's scores are left alone. midpoint: chrombpnet's
#                    default and chrombpnet 1.x's; stopgrad: --contribs-profile-stopgrad.
#                    Needs a chrombpnet with that option (main since 2026-10-10).
#                    Compare with: OUT_DIR=<dir> sbatch run_summarize.sh --profile-tag <weighting>
#   FORCE=1          rescore heads whose scores exist
#   DRY_RUN=1        resolve and print, run nothing
#   ALLOW_CPU=1      skip the GPU preflight check
#   INTERPRET_BATCH_SIZE=N   contribution-scoring batch size (default: what the
#                    pipeline used; 8 for PyTorch and JAX on a GPU under 16 GB)
#
# Output, beside the pipeline's own, in
# ${ROOT}/<engine>/<ASSAY>/<ENCID>/fold_N/model/auxiliary/interpret_subsample/:
#   chrombpnet_nobias.counts_scores.h5    (and .profile_scores.h5)
#   chrombpnet_nobias.contribution.args.json
# or, with PROFILE_WEIGHTING=<weighting>:
#   chrombpnet_nobias.<weighting>.profile_scores.h5
#   chrombpnet_nobias.<weighting>.contribution.args.json

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
ENGINES_DIR="${ENGINES_DIR:-${BLOG_SCRIPTS}/engines}"
[[ -f "${ENGINES_DIR}/run_contribution.py" ]] || { echo "Error: no run_contribution.py in ${ENGINES_DIR}; set ENGINES_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${ENGINES_DIR}/_common.sh"

parse_common_flags "$@"
resolve_args "${ARGS[0]:-}" "${ARGS[1]:-}"
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
select_row

FOLD_DIR="${ROOT}/${ENGINE}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"
if [[ -n "${PROFILE_WEIGHTING:-}" ]]; then
  read -r -a HEAD_LIST <<< "${HEADS:-profile}"
else
  read -r -a HEAD_LIST <<< "${HEADS:-counts profile}"
fi
PY_ARGS=(--run-dir "${FOLD_DIR}" --heads "${HEAD_LIST[@]}" --device cuda)
[[ -n "${PROFILE_WEIGHTING:-}" ]] && PY_ARGS+=(--profile-weighting "${PROFILE_WEIGHTING}")
[[ "${FORCE:-0}" == "1" ]] && PY_ARGS+=(--force)

# Contribution-scoring batch size. The pipeline's (16 at -bs 64) fits a 12 GB
# GPU for every engine, but here nothing else holds the GPU, so on a small
# GPU (_common.sh) PyTorch and JAX use SMALL_GPU_INTERPRET_BATCH_SIZE to
# leave room; the scores do not depend on it beyond float rounding.
# TensorFlow explains one example at a time whatever it is told.
# INTERPRET_BATCH_SIZE=N sets it for every engine.
SMALL_GPU_INTERPRET_BATCH_SIZE="${SMALL_GPU_INTERPRET_BATCH_SIZE:-8}"
if [[ -n "${INTERPRET_BATCH_SIZE:-}" ]]; then
  PY_ARGS+=(--interpret-batch-size "${INTERPRET_BATCH_SIZE}")
elif [[ "${ENGINE}" != "tensorflow" ]] && small_gpu; then
  PY_ARGS+=(--interpret-batch-size "${SMALL_GPU_INTERPRET_BATCH_SIZE}")
fi

echo "ENGINE:   ${ENGINE}"
echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "FOLD_DIR: ${FOLD_DIR}"
echo "HEADS:    ${HEAD_LIST[*]}"
echo "PROFILE_WEIGHTING: ${PROFILE_WEIGHTING:-none (the pipeline files)}"
echo "CONTAINER: $( (( USE_CONTAINER )) && echo "yes (${CONTAINER_IMAGE})" || echo no)"

if [[ ! -f "${FOLD_DIR}/COMPLETED" ]]; then
  echo "SKIP: ${FOLD_DIR} has not completed (no COMPLETED marker)."
  exit 0
fi
check_native "${ENGINE}" "${CLUSTER}" || exit 2

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: python ${ENGINES_DIR}/run_contribution.py ${PY_ARGS[*]}"
  exit 0
fi

if needs_container; then
  export ENGINES_DIR
  container_run "${ENGINE}" "${CLUSTER}" bash "${ENGINES_DIR}/run_contribution.sh" "${CLUSTER}" "${ENGINE}"
  exit $?
fi

activate_engine "${ENGINE}" "${CLUSTER}"
gpu_preflight "${ENGINE}"

start="$(date +%s)"
python "${ENGINES_DIR}/run_contribution.py" "${PY_ARGS[@]}"
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Completed run in $(( ($(date +%s) - start) / 60 )) min. ($(date '+%Y-%m-%d %H:%M:%S'))"
