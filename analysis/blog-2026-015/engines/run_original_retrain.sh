#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=engines_original_retrain
#SBATCH --output=engines_original_retrain.%A_%a.out
#SBATCH --error=engines_original_retrain.%A_%a.err
#SBATCH --array=0-19

# The original chrombpnet, retrained on one row's sweep inputs, then scored by
# its own DeepSHAP as run_deepshap_seeds.sh scores the ENCODE model: the
# "original (new model)" comparison. How close do two models trained by the
# same original code come to the stored scores, against the engines?
#
#   ./submit_all.sh slurm --original-retrain      # every row
#   sbatch <SBATCH_SHARED_GPU_ARGS> run_original_retrain.sh slurm
#   SLURM_ARRAY_TASK_ID=1 ./run_original_retrain.sh local
#
# Rows as for run_train_full.sh, and the same inputs as every engine's run of
# the row: the BAM, peaks, nonpeaks (run_prep_nonpeaks.sh's), fold and bias
# threshold factor. The code is kundajelab/chrombpnet (ORIGINAL_REPO, at
# 09938fd) in its own conda env (ORIGINAL_ENV, chrombpnet: an editable install
# of that checkout with the original's pinned requirements -- tfmodisco
# 0.5.16.0, which its read-shift estimate imports, pyfaidx 0.6.1, modisco-lite
# 2.0.7; for DeepSHAP the same TensorFlow 2.8.0, kundajelab shap fork,
# deeplift, numpy and h5py as chrombpnet-tensorflow, where
# run_deepshap_seeds.sh scored the ENCODE models), through
# run_original_retrain.py:
#
#   bias_train        original `chrombpnet bias train`  -> bias_model/
#   chrombpnet_train  original `chrombpnet train` on that bias model -> model/
#   predict           the original's test-set predictions, the call its
#                     `pipeline` makes after training -> model/evaluation/
#   deepshap          run_deepshap_seeds.py on model/models/chrombpnet_nobias.h5:
#                     the engines' 30K peak subsample, references seeded as
#                     there -> seed_<s>/
#
# `train`, not `pipeline`: the model is the same, and the rest of `pipeline`
# (footprints, DeepSHAP on its own subsample, MoDISco, reports) is hours that
# nothing here reads. What the original does differently from the engines is
# its own: its read shifts (DNase: the 0/+1 cut convention, not both-half), its
# own bias model, no RC data augmentation option. One change to it, as in the
# sweep: IGNORE_BIAS=1 (default) makes its bias-model check -- an assert in the
# original -- the engines' --ignore-bias-flag warning; the stomach samples'
# bias models fail it (run_original_retrain.py).
#
# A step that finished is not run again: a job that hits its time limit is
# resubmitted and goes on from the next step. A step that runs again clears
# every step after it.
#
# Natively only, not --container. The original's `bias train` and `train` end
# by writing their report PDF with weasyprint, which loads pango and cairo
# from the system (the original's Dockerfile installs them with apt); the env
# does not bring them. Under Slurm this runs site.env's hook report_libraries
# for them (on Sherlock: module load system; pango; system cairo, as in the
# chrombpnet runs); the lab's GPU hosts have them in the OS; the container
# image has neither.
#
# Environment:
#   SEEDS="1"          DeepSHAP seeds (default: 1, the seed run_deepshap_seeds.sh
#                      scored the ENCODE model with: same references, so the
#                      two models compare on the model alone)
#   N_REGIONS=N        only the first N of the 30K subsample's peaks (default: all)
#   HEADS="counts"     heads to score (default: counts profile)
#   IGNORE_BIAS=0      keep the original's assert on the bias model (default 1)
#   EXTRA_ARGS=".."    appended to both training commands, e.g. "-e 1"
#   BATCH_SIZE=N       -bs N for training and predict (default: the original's 64)
#   FORCE=1            start the row again from scratch
#   DRY_RUN=1          resolve and print, run nothing
#   ALLOW_CPU=1        skip the GPU preflight check
#   ORIGINAL_REPO=..   the original package, kundajelab/chrombpnet at 09938fd
#                      (default: site.env's)
#   ORIGINAL_ENV=..    the conda env that imports it (default: chrombpnet)
#   RETRAIN_ROOT=..    output tree (default: ${ROOT}/original_retrained)
#
# Output, under ${RETRAIN_ROOT}/<ASSAY>/<ENCID>/fold_N/:
#   bias_model/ model/          the original's two output directories
#   regions.bed seed_<s>/       run_deepshap_seeds.py's, as under ${ROOT}/original
#   provenance.txt timings.tsv  one block / one line per job and step
#   COMPLETED                   written last

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
ENGINES_DIR="${ENGINES_DIR:-${BLOG_SCRIPTS}/engines}"
[[ -f "${ENGINES_DIR}/run_original_retrain.py" ]] || { echo "Error: no run_original_retrain.py in ${ENGINES_DIR}; set ENGINES_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${ENGINES_DIR}/_common.sh"

parse_common_flags "$@"
resolve_cluster "${ARGS[0]:-}"
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
select_row

# The original's own env, set up for TensorFlow 2.8 as the TensorFlow
# engine's is (CUDA 11.2, cuDNN 8.1).
ENGINE="tensorflow"
require_site ORIGINAL_REPO
ORIGINAL_ENV="${ORIGINAL_ENV:-chrombpnet}"
RETRAIN_ROOT="${RETRAIN_ROOT:-${ROOT}/original_retrained}"
FOLD_DIR="${RETRAIN_ROOT}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"
BIAS_DIR="${FOLD_DIR}/bias_model"
MODEL_DIR="${FOLD_DIR}/model"
BIAS_MODEL_PATH="${BIAS_DIR}/models/${ENCID}_bias.h5"
NOBIAS_MODEL="${MODEL_DIR}/models/chrombpnet_nobias.h5"
NEG_BED_GZ="$(prep_dir)/negatives.bed.gz"
SUBSAMPLE="${ROOT}/pytorch/${ASSAY}/${ENCID}/fold_${FOLD_NUM}/model/auxiliary/interpret_subsample/chrombpnet_nobias.interpreted_regions.bed"
read -r -a SEED_LIST <<< "${SEEDS:-1}"
read -r -a HEAD_LIST <<< "${HEADS:-counts profile}"
N_REGIONS="${N_REGIONS:-0}"
IGNORE_BIAS="${IGNORE_BIAS:-1}"
BIAS_FLAG=()
[[ "${IGNORE_BIAS}" == "1" ]] && BIAS_FLAG=(--ignore-bias-flag)
# Word-split on purpose: EXTRA_ARGS holds whole options.
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
BS=()
if [[ -n "${BATCH_SIZE:-}" ]]; then
  BS=(-bs "${BATCH_SIZE}")
  BATCH_NOTE="-bs ${BATCH_SIZE} for training and predict (BATCH_SIZE)"
fi

ORIGINAL=(python "${ENGINES_DIR}/run_original_retrain.py" --original-repo "${ORIGINAL_REPO}")
COMMON=(-ibam "${BAM_PATH}" -d "${ASSAY}" -g "${GENOME_PATH}" -c "${CHROM_PATH}"
        -p "${PEAKS_BED_PATH}" -n "${NEG_BED_GZ}" -fl "${FOLD_JSON_PATH}")
BIAS_CMD=("${ORIGINAL[@]}" chrombpnet bias train "${COMMON[@]}" -b "${BIAS_THRESHOLD_FACTOR}"
          -o "${BIAS_DIR}" -fp "${ENCID}" ${BS[@]+"${BS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
MODEL_CMD=("${ORIGINAL[@]}" ${BIAS_FLAG[@]+"${BIAS_FLAG[@]}"} chrombpnet train "${COMMON[@]}"
           -b "${BIAS_MODEL_PATH}" -o "${MODEL_DIR}" ${BS[@]+"${BS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
# As the original's `pipeline` calls it after training: args.batch_size is
# the training command's (64), not predict.py's own default (512).
PREDICT_CMD=("${ORIGINAL[@]}" predict -g "${GENOME_PATH}" -b "${MODEL_DIR}/auxiliary/data_unstranded.bw"
             -p "${MODEL_DIR}/auxiliary/filtered.peaks.bed" -n None -fl "${FOLD_JSON_PATH}"
             -m "${MODEL_DIR}/models/chrombpnet.h5" -o "${MODEL_DIR}/evaluation/chrombpnet"
             -bs "${BATCH_SIZE:-64}")
DEEPSHAP_CMD=(python "${ENGINES_DIR}/run_deepshap_seeds.py" --model "${NOBIAS_MODEL}"
              --regions-source "${SUBSAMPLE}" --n-regions "${N_REGIONS}" --genome "${GENOME_PATH}"
              --out "${FOLD_DIR}" --seeds "${SEED_LIST[@]}" --heads "${HEAD_LIST[@]}"
              --original-repo "${ORIGINAL_REPO}")

echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "FOLD_DIR: ${FOLD_DIR}"
echo "ORIGINAL: ${ORIGINAL_REPO} (env ${ORIGINAL_ENV})"
echo "IGNORE_BIAS: ${IGNORE_BIAS} (the original's bias-model assert is $([[ "${IGNORE_BIAS}" == "1" ]] && echo 'a WARNING' || echo 'KEPT'))"
echo "SEEDS:    ${SEED_LIST[*]}; HEADS: ${HEAD_LIST[*]}; REGIONS: $( (( N_REGIONS > 0 )) && echo "first ${N_REGIONS}" || echo all)"
echo "EXTRA_ARGS: ${EXTRA_ARGS:-<none>}"
echo "BATCH_SIZE: ${BATCH_NOTE:-the defaults of the original}"
echo "CONTAINER: $( (( USE_CONTAINER )) && echo "yes (${CONTAINER_IMAGE})" || echo no)"

if [[ -f "${FOLD_DIR}/COMPLETED" && "${FORCE:-0}" != "1" ]]; then
  echo "exists: ${FOLD_DIR}/COMPLETED -- skipped (FORCE=1 retrains)"
  exit 0
fi
for f in "${BAM_PATH}" "${PEAKS_BED_PATH}" "${GENOME_PATH}" "${CHROM_PATH}" "${FOLD_JSON_PATH}" \
         "${SUBSAMPLE}" "${ORIGINAL_REPO}/chrombpnet/CHROMBPNET.py"; do
  [[ -f "${f}" ]] || { echo "Error: missing ${f}" >&2; exit 1; }
done
if [[ ! -s "${NEG_BED_GZ}" && -z "${DRY_RUN:-}" ]]; then
  echo "Error: no nonpeaks at ${NEG_BED_GZ}. Run run_prep_nonpeaks.sh for this row first." >&2
  exit 1
fi

check_native "${ENGINE}" "${CLUSTER}" || exit 2

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: ${BIAS_CMD[*]}"
  echo "DRY_RUN: ${MODEL_CMD[*]}"
  echo "DRY_RUN: ${PREDICT_CMD[*]}"
  echo "DRY_RUN: ${DEEPSHAP_CMD[*]}"
  echo "DRY_RUN set -- resolved successfully, stopping before any work."
  exit 0
fi

if needs_container || in_container; then
  echo "Error: run_original_retrain.sh runs natively, not in the container: the original's report" \
       "PDF needs the system's pango, which the image does not have (site.env: report_libraries)." >&2
  exit 2
fi

activate_engine "${ENGINE}" "${CLUSTER}" "${ORIGINAL_ENV}" "${ORIGINAL_REPO}"
if [[ "${CLUSTER}" == "slurm" ]] && declare -F site_report_libraries > /dev/null; then
  site_hook report_libraries || exit 1
  echo "site.env: report_libraries (pango and cairo, for weasyprint)"
fi
gpu_preflight "${ENGINE}"
runtime_check "${ENGINE}"

## STEPS -----------------------------------------------------------------------
[[ "${FORCE:-0}" == "1" ]] && rm -rf "${FOLD_DIR}"
mkdir -p "${FOLD_DIR}"
ENGINE=original CHROMBPNET_DIR="${ORIGINAL_REPO}" write_provenance "${FOLD_DIR}/provenance.job"
{
  cat "${FOLD_DIR}/provenance.job"
  echo "ignore_bias: ${IGNORE_BIAS}"
  echo "seeds: ${SEED_LIST[*]}; heads: ${HEAD_LIST[*]}; n_regions: ${N_REGIONS}"
  echo
} >> "${FOLD_DIR}/provenance.txt"
rm -f "${FOLD_DIR}/provenance.job"
[[ -f "${FOLD_DIR}/timings.tsv" ]] || printf 'step\tseconds\tstart\tend\n' > "${FOLD_DIR}/timings.tsv"

STEPS=(bias_train chrombpnet_train predict deepshap)
# Each step's outputs, removed before it runs again (and so after any step
# before it ran again).
outputs_of() {
  case "$1" in
    bias_train) echo "${BIAS_DIR}" ;;
    chrombpnet_train) echo "${MODEL_DIR}" ;;
    predict) echo "${MODEL_DIR}/evaluation/chrombpnet_metrics.json" ;;
    deepshap) echo "${FOLD_DIR}/regions.bed"; ls -d "${FOLD_DIR}"/seed_* 2>/dev/null || true ;;
  esac
}

# Runs a step unless it finished before; appends its wall time to timings.tsv.
step() {
  local name="$1"; shift
  if [[ -f "${FOLD_DIR}/.done.${name}" ]]; then
    echo "=== ${name}: done before -- skipped"
    return 0
  fi
  local later=0 s start end
  for s in "${STEPS[@]}"; do
    [[ "${s}" == "${name}" ]] && later=1
    if (( later )); then
      rm -f "${FOLD_DIR}/.done.${s}"
      # The deepshap step keeps COMPLETED seeds of this model; it is cleared
      # only when a step before it runs again.
      [[ "${s}" == deepshap && "${name}" == deepshap ]] && continue
      outputs_of "${s}" | while read -r p; do [[ -n "${p}" ]] && rm -rf "${p}"; done
    fi
  done
  start="$(date +%s)"
  echo "=== ${name}: $(date '+%Y-%m-%d %H:%M:%S')"
  "$@"
  end="$(date +%s)"
  printf '%s\t%s\t%s\t%s\n' "${name}" "$((end - start))" "${start}" "${end}" >> "${FOLD_DIR}/timings.tsv"
  date '+%Y-%m-%d %H:%M:%S' > "${FOLD_DIR}/.done.${name}"
  echo "=== ${name}: done in $(( (end - start) / 60 )) min"
}
need() {
  local f
  for f in "$@"; do [[ -f "${f}" ]] || { echo "Error: ${f} was not written" >&2; exit 1; }; done
}

step bias_train "${BIAS_CMD[@]}"
need "${BIAS_MODEL_PATH}"
step chrombpnet_train "${MODEL_CMD[@]}"
need "${MODEL_DIR}/models/chrombpnet.h5" "${NOBIAS_MODEL}"
step predict "${PREDICT_CMD[@]}"
need "${MODEL_DIR}/evaluation/chrombpnet_metrics.json"
step deepshap "${DEEPSHAP_CMD[@]}"
for seed in "${SEED_LIST[@]}"; do need "${FOLD_DIR}/seed_${seed}/COMPLETED"; done

date '+%Y-%m-%d %H:%M:%S' > "${FOLD_DIR}/COMPLETED"
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Output: ${FOLD_DIR}"
echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
