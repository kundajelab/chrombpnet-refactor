#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=engines_train
#SBATCH --output=engines_train.%A_%a.out
#SBATCH --error=engines_train.%A_%a.err
#SBATCH --array=0-19

# Train one engine's bias model and ChromBPNet model for one dataset and fold.
#
#   sbatch run_train_full.sh slurm <engine> [--container]   # all 20 rows
#   SLURM_ARRAY_TASK_ID=5 ./run_train_full.sh local jax
#
# <engine> is pytorch, tensorflow or jax. The cluster may be omitted (it is
# inferred from $SLURM_JOB_ID), and the engine may come from ENGINE= instead.
# --container runs the env inside the Ubuntu 22.04 image (_common.sh); JAX on
# a glibc older than 2.27 needs it, and it gives every engine the same system
# tools.
#
# Each task runs, as dnase-rc/run_train_full.sh does for one arm:
#
#   chrombpnet bias pipeline --engine <engine>   -> bias_model/
#   chrombpnet pipeline      --engine <engine>   -> model/
#
# on the nonpeaks run_prep_nonpeaks.sh wrote for this dataset and fold, which
# every engine shares (submit_all.sh makes each task wait for its own prep
# task). Everything else is each command's default, the same for every
# engine: --dnase-convention both-half on DNase, --data-aug augment, --seed 1234,
# the bias threshold factor (0.5 ATAC, 0.8 DNase, as in dnase-rc).
#
# Environment:
#   DRY_RUN=1      resolve and print every path and command, run nothing
#   FORCE=1        retrain a run that already completed (default: skip it)
#   IGNORE_BIAS=0  enforce `pipeline`'s bias-model quality check (default 1:
#                  a warning, as in dnase-rc -- the stomach samples' own bias
#                  models fail it in every configuration)
#   EXTRA_ARGS=".." appended to both pipelines, e.g. for a quick check:
#                  EXTRA_ARGS="--epochs 1 --interpret-subsample 500"
#   ALLOW_CPU=1    skip the GPU preflight check
#   BATCH_SIZE=N   -bs N for both pipelines (default: the pipeline's 64)
#
# Output, under ${ROOT}/<engine>/<ASSAY>/<ENCID>/fold_N/:
#   bias_model/ model/   the two pipelines' output directories
#   provenance.txt       commit, env, host and GPU the run used
#   timings.tsv          wall time of each step, in seconds
#   COMPLETED            written last; a run without it is incomplete

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
ENGINES_DIR="${ENGINES_DIR:-${BLOG_SCRIPTS}/engines}"
[[ -f "${ENGINES_DIR}/_common.sh" ]] || { echo "Error: no _common.sh in ${ENGINES_DIR}; set ENGINES_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${ENGINES_DIR}/_common.sh"

parse_common_flags "$@"
resolve_args "${ARGS[0]:-}" "${ARGS[1]:-}"
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
select_row

SFX="$(engine_suffix "${ENGINE}")"
FOLD_DIR="${ROOT}/${ENGINE}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"
BIAS_DIR="${FOLD_DIR}/bias_model"
MODEL_DIR="${FOLD_DIR}/model"
BIAS_MODEL_PATH="${BIAS_DIR}/models/${ENCID}_bias${SFX}"
NEG_BED_GZ="$(prep_dir)/negatives.bed.gz"

IGNORE_BIAS="${IGNORE_BIAS:-1}"
BIAS_FLAG=()
[[ "${IGNORE_BIAS}" == "1" ]] && BIAS_FLAG=(--ignore-bias-flag)
# Word-split on purpose: EXTRA_ARGS holds whole options.
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

echo "ENGINE:   ${ENGINE}"
echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "FOLD_DIR: ${FOLD_DIR}"
echo "IGNORE_BIAS: ${IGNORE_BIAS} (bias-model quality check is $([[ "${IGNORE_BIAS}" == "1" ]] && echo 'a WARNING' || echo 'ENFORCED'))"
echo "EXTRA_ARGS: ${EXTRA_ARGS:-<none>}"
echo "CONTAINER: $( (( USE_CONTAINER )) && echo "yes (${CONTAINER_IMAGE})" || echo no)"

if [[ -f "${FOLD_DIR}/COMPLETED" && "${FORCE:-0}" != "1" ]]; then
  echo "exists: ${FOLD_DIR}/COMPLETED -- skipped (FORCE=1 retrains)"
  echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
  exit 0
fi
for f in "${BAM_PATH}" "${PEAKS_BED_PATH}" "${GENOME_PATH}" "${CHROM_PATH}" "${FOLD_JSON_PATH}"; do
  [[ -f "${f}" ]] || { echo "Error: missing ${f}" >&2; exit 1; }
done
if [[ ! -s "${NEG_BED_GZ}" && -z "${DRY_RUN:-}" ]]; then
  echo "Error: no nonpeaks at ${NEG_BED_GZ}. Run run_prep_nonpeaks.sh for this row first." >&2
  exit 1
fi

# Batch size: the pipeline's default (-bs 64) for every engine, or
# BATCH_SIZE=N for both stages. TensorFlow needs a GPU of 16 GB or more
# (submit_all.sh asks for one): its DeepSHAP runs in a second process on the
# GPU while the first still holds the memory it trained with, and on a 12 GB
# TITAN Xp that worker ran out of memory at -bs 32 as well as at 64.
BIAS_BS=()
MODEL_BS=()
if [[ -n "${BATCH_SIZE:-}" ]]; then
  BIAS_BS=(-bs "${BATCH_SIZE}")
  MODEL_BS=(-bs "${BATCH_SIZE}")
  BATCH_NOTE="-bs ${BATCH_SIZE} for both stages (BATCH_SIZE)"
fi
echo "BATCH_SIZE: ${BATCH_NOTE:-the pipeline defaults}"

COMMON=(-ibam "${BAM_PATH}" -d "${ASSAY}" -g "${GENOME_PATH}" -c "${CHROM_PATH}"
        -p "${PEAKS_BED_PATH}" -n "${NEG_BED_GZ}" -fl "${FOLD_JSON_PATH}"
        --engine "${ENGINE}" --device cuda)
BIAS_CMD=(chrombpnet bias pipeline "${COMMON[@]}" -b "${BIAS_THRESHOLD_FACTOR}"
          -o "${BIAS_DIR}" -fp "${ENCID}" ${BIAS_BS[@]+"${BIAS_BS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
MODEL_CMD=(chrombpnet pipeline "${COMMON[@]}" -b "${BIAS_MODEL_PATH}"
           -o "${MODEL_DIR}" ${BIAS_FLAG[@]+"${BIAS_FLAG[@]}"} ${MODEL_BS[@]+"${MODEL_BS[@]}"}
           ${EXTRA[@]+"${EXTRA[@]}"})

check_native "${ENGINE}" "${CLUSTER}" || exit 2

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: ${BIAS_CMD[*]}"
  echo "DRY_RUN: ${MODEL_CMD[*]}"
  echo "DRY_RUN set -- resolved successfully, stopping before any work."
  exit 0
fi

# --container: this same script again, inside the container (_common.sh).
# Its exit status is the job's.
if needs_container; then
  export ENGINES_DIR
  container_run "${ENGINE}" "${CLUSTER}" bash "${ENGINES_DIR}/run_train_full.sh" "${CLUSTER}" "${ENGINE}"
  exit $?
fi

activate_engine "${ENGINE}" "${CLUSTER}"
gpu_preflight "${ENGINE}"
runtime_check "${ENGINE}"

## PIPELINE --------------------------------------------------------------------
if [[ -d "${FOLD_DIR}" ]]; then
  echo "Removing incomplete run: ${FOLD_DIR}"
  rm -rf "${FOLD_DIR}"
fi
mkdir -p "${FOLD_DIR}"
write_provenance "${FOLD_DIR}/provenance.txt"
printf 'step\tseconds\tstart\tend\n' > "${FOLD_DIR}/timings.tsv"

# Runs a step and appends its wall time to timings.tsv.
step() {
  local name="$1"; shift
  local start end
  start="$(date +%s)"
  echo "=== ${name}: $(date '+%Y-%m-%d %H:%M:%S')"
  "$@"
  end="$(date +%s)"
  printf '%s\t%s\t%s\t%s\n' "${name}" "$((end - start))" "${start}" "${end}" >> "${FOLD_DIR}/timings.tsv"
  echo "=== ${name}: done in $(( (end - start) / 60 )) min"
}

step bias_pipeline "${BIAS_CMD[@]}"
[[ -f "${BIAS_MODEL_PATH}" ]] || { echo "Error: ${BIAS_MODEL_PATH} was not written" >&2; exit 1; }

step chrombpnet_pipeline "${MODEL_CMD[@]}"
for f in "${MODEL_DIR}/models/chrombpnet${SFX}" "${MODEL_DIR}/models/chrombpnet_nobias${SFX}" \
         "${MODEL_DIR}/evaluation/chrombpnet_metrics.json"; do
  [[ -f "${f}" ]] || { echo "Error: ${f} was not written" >&2; exit 1; }
done

date '+%Y-%m-%d %H:%M:%S' > "${FOLD_DIR}/COMPLETED"
echo "Output: ${FOLD_DIR}"
echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
