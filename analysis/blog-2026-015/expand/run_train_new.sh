#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=expand_new
#SBATCH --output=expand_new.%A_%a.out
#SBATCH --error=expand_new.%A_%a.err
#SBATCH --array=0-19

# Train the new version for one dataset and fold: a profile-only bias model,
# then ChromBPNet with the epoch kept by validation counts r + 1 - median JSD.
#
#   sbatch run_train_new.sh slurm --container      # all 20 rows
#   SLURM_ARRAY_TASK_ID=5 ./run_train_new.sh local    # one row, here
#
# Each task runs the engines sweep's two commands (../engines/run_train_full.sh)
# with the PyTorch engine, plus the two new options:
#
#   chrombpnet bias pipeline ... --bias-profile-only          -> bias_model/
#   chrombpnet pipeline      ... --model-selection pearsonr-jsd -> model/
#
# on the engines sweep's nonpeaks for this dataset and fold (${ROOT}/preprocess
# links to them), so old and new see the same regions. Everything else is
# each command's default, as there: --seed 1234, --data-aug augment,
# --dnase-convention both-half on DNase, the bias threshold factor (0.5 ATAC,
# 0.8 DNase), --ignore-bias-flag (moot here: a profile-only bias model has no
# counts to check). One draw differs from the old runs': their code drew the
# ChromBPNet training negatives with a fixed seed of 1, this one with --seed
# (1234), so filtered.nonpeaks.bed is another subset of the same pool; every
# other input (bigwig, filtered peaks, bias regions) is identical. Afterwards the subsample's profile scores, which the
# pipeline now computes with the original's midpoint weighting, are linked
# under the name run_summarize.py reads (chrombpnet_nobias.midpoint.*).
#
# Environment: DRY_RUN=1, FORCE=1, EXTRA_ARGS="..", ALLOW_CPU=1, BATCH_SIZE=N
# as in ../engines/run_train_full.sh. A quick check:
#   EXTRA_ARGS="--epochs 1 --interpret-subsample 2000" ROOT=<scratch root> ...
#
# Output, under ${ROOT}/new/<ASSAY>/<ENCID>/fold_N/: bias_model/ model/
# provenance.txt timings.tsv COMPLETED, as in the engines sweep.

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
resolve_cluster "${ARGS[0]:-}"
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
export SLURM_ARRAY_TASK_ID   # through the container re-exec
select_row

FOLD_DIR="${ROOT}/new/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"
BIAS_DIR="${FOLD_DIR}/bias_model"
MODEL_DIR="${FOLD_DIR}/model"
BIAS_MODEL_PATH="${BIAS_DIR}/models/${ENCID}_bias.pt"
NEG_BED_GZ="$(prep_dir)/negatives.bed.gz"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "FOLD_DIR: ${FOLD_DIR}"
echo "EXTRA_ARGS: ${EXTRA_ARGS:-<none>}"
echo "CONTAINER: $( (( USE_CONTAINER )) && echo "yes (${CONTAINER_IMAGE})" || echo no)"

if [[ -f "${FOLD_DIR}/COMPLETED" && "${FORCE:-0}" != "1" ]]; then
  echo "exists: ${FOLD_DIR}/COMPLETED -- skipped (FORCE=1 retrains)"
  exit 0
fi
for f in "${BAM_PATH}" "${PEAKS_BED_PATH}" "${GENOME_PATH}" "${CHROM_PATH}" "${FOLD_JSON_PATH}"; do
  [[ -f "${f}" ]] || { echo "Error: missing ${f}" >&2; exit 1; }
done
[[ -s "${NEG_BED_GZ}" ]] || { echo "Error: no nonpeaks at ${NEG_BED_GZ} (is ${ROOT}/preprocess linked?)" >&2; exit 1; }

BS=()
if [[ -n "${BATCH_SIZE:-}" ]]; then
  BS=(-bs "${BATCH_SIZE}")
  BATCH_NOTE="-bs ${BATCH_SIZE} for both stages (BATCH_SIZE)"
fi

COMMON=(-ibam "${BAM_PATH}" -d "${ASSAY}" -g "${GENOME_PATH}" -c "${CHROM_PATH}"
        -p "${PEAKS_BED_PATH}" -n "${NEG_BED_GZ}" -fl "${FOLD_JSON_PATH}"
        --engine pytorch --device cuda)
BIAS_CMD=(chrombpnet bias pipeline "${COMMON[@]}" -b "${BIAS_THRESHOLD_FACTOR}" --bias-profile-only
          -o "${BIAS_DIR}" -fp "${ENCID}" ${BS[@]+"${BS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
MODEL_CMD=(chrombpnet pipeline "${COMMON[@]}" -b "${BIAS_MODEL_PATH}" --model-selection pearsonr-jsd
           -o "${MODEL_DIR}" --ignore-bias-flag ${BS[@]+"${BS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: ${BIAS_CMD[*]}"
  echo "DRY_RUN: ${MODEL_CMD[*]}"
  exit 0
fi

if needs_container; then
  export EXPAND_DIR
  container_run pytorch "${CLUSTER}" bash "${EXPAND_DIR}/run_train_new.sh" "${CLUSTER}"
  exit $?
fi

activate_engine pytorch "${CLUSTER}"
gpu_preflight pytorch
runtime_check pytorch

if [[ -d "${FOLD_DIR}" ]]; then
  echo "Removing incomplete run: ${FOLD_DIR}"
  rm -rf "${FOLD_DIR}"
fi
mkdir -p "${FOLD_DIR}"
write_provenance "${FOLD_DIR}/provenance.txt"

timed "${FOLD_DIR}" bias_pipeline "${BIAS_CMD[@]}"
[[ -f "${BIAS_MODEL_PATH}" ]] || { echo "Error: ${BIAS_MODEL_PATH} was not written" >&2; exit 1; }
timed "${FOLD_DIR}" chrombpnet_pipeline "${MODEL_CMD[@]}"
for f in "${MODEL_DIR}/models/chrombpnet.pt" "${MODEL_DIR}/models/chrombpnet_nobias.pt" \
         "${MODEL_DIR}/evaluation/chrombpnet_metrics.json"; do
  [[ -f "${f}" ]] || { echo "Error: ${f} was not written" >&2; exit 1; }
done
link_midpoint_scores "${MODEL_DIR}/auxiliary/interpret_subsample"

date '+%Y-%m-%d %H:%M:%S' > "${FOLD_DIR}/COMPLETED"
echo "Output: ${FOLD_DIR}"
echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
