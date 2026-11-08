#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=chrombpnet_optimization_tf
#SBATCH --output=/path/to/workdir/sherlock/logs/chrombpnet_optimization_tf.%x.%A_%a.out
#SBATCH --error=/path/to/workdir/sherlock/logs/chrombpnet_optimization_tf.%x.%A_%a.err
#SBATCH --array=0-4
#
# The stock TensorFlow ChromBPNet baseline for run_chrombpnet_optimization_
# benchmark.sh: the same three commands, the same fold, the same region
# subsample, timed the same way -- so "what did the PyTorch port and its
# optimizations actually buy" has a floor to be measured against.
#
#   ./list_gpu_skus.sh                  # what the partitions offer
#   sbatch --constraint="GPU_SKU:<name>" run_chrombpnet_optimization_tensorflow.sh sherlock
#   ./summarize.sh                      # reads this alongside the PyTorch runs
#
# Deliberately NOT matched to the PyTorch runs:
#
#   * No --predict-fk. Stock has no such flag; this is the unoptimized floor,
#     and it records itself as arm 'tensorflow' so the tables line up.
#   * No determinism settings. run/run_chrombpnet_benchmark_tensorflow.sh
#     pins TF's determinism because it is checking numerical parity; that
#     costs throughput and would understate stock. A timing baseline should
#     be stock as shipped, which is also what the PyTorch default is.
#   * No --device. Stock takes whatever GPU is visible.
#
# Two differences in the commands themselves, from reading stock's parsers:
#
#   * stock `contribs_bw` has no --batch-size at all, so none is passed.
#   * stock `pred_bw -d/--debug-chr` is broken upstream in 1.0.1, so it is
#     never used here (the PyTorch runs do not use it either).
#
# Usage: run_chrombpnet_optimization_tensorflow.sh CLUSTER [FOLD_NUM] [N_REGIONS]

set -euo pipefail

CLUSTER="${1:-sherlock}"
FOLD_NUM="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
N_REGIONS="${3:-20000}"
# DeepLIFT costs far more per region than a forward; matches the PyTorch job.
N_CONTRIB="${N_CONTRIB:-2000}"

OAK_DIR="/path/to/workdir"
TF_REPO="${BENCHMARK_TF_REPO:-${OAK_DIR}/software/chrombpnet}"
REFERENCE_DIR="${OAK_DIR}/chromatlas/reference"
GENOME_PATH="${REFERENCE_DIR}/hg38/hg38.fa"
CHROM_PATH="${REFERENCE_DIR}/hg38/hg38.chrom.sizes"
FOLD_JSON="${REFERENCE_DIR}/folds/fold_${FOLD_NUM}.json"

# The TensorFlow-trained folds, laid out exactly like the PyTorch ones.
ASSAY="${ASSAY:-ATAC}"
ENCID="${ENCID:-ENCSR868FGK}"
MODELS_ROOT="${TF_MODELS_ROOT:-${OAK_DIR}/chromatlas/test/chrombpnet-tf/${ASSAY}/${ENCID}}"
MODEL_DIR="${MODELS_ROOT}/fold_${FOLD_NUM}/model"

OUT_ROOT="${OAK_DIR}/chromatlas/test/chrombpnet-refactor-optimization/analysis"

CONDA_ENV="${CONDA_ENV:-chrombpnet}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
if [[ "${CLUSTER}" == "sherlock" ]]; then
  # Mirrors run/run_chrombpnet_benchmark_tensorflow.sh; TF 2.8 needs this pair.
  module load cuda/11.2.0
  module load cudnn/8.1.1.33
  module load system
  module load pango
  module load system cairo
fi

# Stock must be first on sys.path, and the PyTorch refactor must not be
# anywhere near it: both packages are called `chrombpnet`, so whichever is
# found first wins. Python puts the *current directory* at the head of
# sys.path for `-m`, so running from the refactor checkout silently imports
# the wrong package -- which fails confusingly on `import torch`. Run from a
# neutral directory and pin PYTHONPATH.
export PYTHONPATH="${TF_REPO}"
NEUTRAL_CWD="$(mktemp -d)"
trap 'rm -rf "${NEUTRAL_CWD}"' EXIT
cd "${NEUTRAL_CWD}"

CHROMBPNET=(python -m chrombpnet.CHROMBPNET)

# --------------------------------------------------------------- preflight
echo "### preflight"
preflight_fail () { echo "PREFLIGHT FAILED: $1" >&2; exit 1; }
python -c "import chrombpnet, sys; p=chrombpnet.__file__;
import os; assert os.path.realpath(p).startswith(os.path.realpath('${TF_REPO}')), p; print('  package:', p)" \
  || preflight_fail "the 'chrombpnet' that imports is not the TensorFlow one at ${TF_REPO}.
  The PyTorch refactor uses the same package name; check PYTHONPATH and that
  no editable install of the refactor is in env '${CONDA_ENV}'."
python -c "import tensorflow as tf; print('  tensorflow:', tf.__version__)" 2>/dev/null \
  || preflight_fail "tensorflow does not import in env '${CONDA_ENV}'"
"${CHROMBPNET[@]}" pred_bw --help >/dev/null 2>&1 \
  || preflight_fail "'chrombpnet pred_bw --help' fails; the stock CLI is broken in this env"
# `set -e` would abort on the probe itself, before the diagnosis below could
# run, so the exit status has to be captured explicitly.
set +e
python - > "${NEUTRAL_CWD}/tf_gpu.log" 2>&1 <<'PYEOF'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUS:", [g.name for g in gpus] or "NONE")
sys.exit(0 if gpus else 1)
PYEOF
tf_gpu_rc=$?
set -e
if [[ ${tf_gpu_rc} -ne 0 ]]; then
  # TF fails silently to CPU when a CUDA library will not load, and the run
  # would then time a CPU forward -- a meaningless baseline. Name the
  # library, because the usual one is specific and fixable.
  missing="$(grep -oE "Could not load dynamic library '[^']+'" "${NEUTRAL_CWD}/tf_gpu.log" \
            | sed "s/.*'\(.*\)'/\1/" | sort -u | paste -sd', ')"
  hint=""
  if [[ "${missing}" == *libcudnn.so.8* ]]; then
    hint="
  TensorFlow 2.8 needs cuDNN 8, not 9. On Sherlock that is
  'module load cudnn/8.1.1.33' (this script does that for CLUSTER=sherlock).
  On a workstation, point LD_LIBRARY_PATH at a cuDNN 8 install; a cuDNN 9
  path will not satisfy it."
  fi
  preflight_fail "TensorFlow sees no GPU, so this would time a CPU forward.
  Libraries it could not load: ${missing:-none reported}${hint}
  Full log: ${NEUTRAL_CWD}/tf_gpu.log (also check CUDA_VISIBLE_DEVICES
  against nvidia-smi -L)."
fi
echo "  GPUs visible to TF: $(grep -oE '^GPUS: .*' "${NEUTRAL_CWD}/tf_gpu.log" | cut -d' ' -f2-)"

VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${VISIBLE_GPUS}" -gt 0 ]]; then
  for index in ${CUDA_VISIBLE_DEVICES//,/ }; do
    if [[ "${index}" =~ ^[0-9]+$ && "${index}" -ge "${VISIBLE_GPUS}" ]]; then
      preflight_fail "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names GPU ${index}, but
  $(hostname) has ${VISIBLE_GPUS} GPU(s), numbered 0..$((VISIBLE_GPUS - 1))."
    fi
  done
fi

# --------------------------------------------------------------- inputs
CHROMBPNET_H5="${MODEL_DIR}/models/chrombpnet.h5"
NOBIAS_H5="${MODEL_DIR}/models/chrombpnet_nobias.h5"
BIAS_H5="${MODEL_DIR}/models/bias_model_scaled.h5"
PEAKS_BED="${MODEL_DIR}/auxiliary/filtered.peaks.bed"
NONPEAKS_BED="${MODEL_DIR}/auxiliary/filtered.nonpeaks.bed"
OBSERVED_BW="${MODEL_DIR}/auxiliary/data_unstranded.bw"
MOTIFS_TSV="${MODEL_DIR}/auxiliary/motif_to_pwm.tsv"

for required in "${CHROMBPNET_H5}" "${NOBIAS_H5}" "${BIAS_H5}" "${PEAKS_BED}" \
                "${NONPEAKS_BED}" "${OBSERVED_BW}" "${MOTIFS_TSV}" \
                "${FOLD_JSON}" "${GENOME_PATH}" "${CHROM_PATH}"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 1; }
done

# --------------------------------------------------------------- identity
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
ARCH="cuda-${GPU_CC//./}"
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
DEVICE_SLUG="dev${CUDA_VISIBLE_DEVICES:-all}"
DEVICE_SLUG="${DEVICE_SLUG//,/-}"
# "TF-stock" prefix keeps this a distinct run in summarize.sh rather than
# being averaged in with the PyTorch sweeps.
RUN_TAG="${RUN_TAG:-TF-stock_${GPU_SLUG}_${ARCH}_${DEVICE_SLUG}}"

WORK="${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
if [[ -d "${WORK}" ]] && ! rm -rf "${WORK}" 2>/dev/null; then
  stale="${WORK}.stale.$(date +%s)"
  echo "note: could not remove ${WORK}; moving it to ${stale}"
  mv "${WORK}" "${stale}"
fi
mkdir -p "${WORK}"
SUMMARY="${WORK}/timings.tsv"
printf "run\tfold\tgpu\tarch\tstage\tphase\tseconds\tstatus\tdetail\n" > "${SUMMARY}"

echo "=================================================================="
echo " framework       : stock TensorFlow ChromBPNet"
echo " repo            : ${TF_REPO}"
echo " conda env       : ${CONDA_ENV}"
echo " run tag         : ${RUN_TAG}"
echo " fold            : ${FOLD_NUM}"
echo " GPU             : ${GPU_NAME}  (compute ${GPU_CC} -> ${ARCH})"
echo " models          : ${MODEL_DIR}/models"
echo " peaks available : $(wc -l < "${PEAKS_BED}")"
echo " regions used    : ${N_REGIONS}"
echo " output dir      : ${WORK}"
echo "=================================================================="

# Same deterministic subsample as the PyTorch job, so the two are comparable.
REGIONS="${WORK}/regions.bed"
if [[ "${N_REGIONS}" == "all" ]]; then
  cp "${PEAKS_BED}" "${REGIONS}"
else
  awk -v n="${N_REGIONS}" 'BEGIN{srand(1234)} {if (NR<=n) r[NR]=$0;
       else {i=int(rand()*NR)+1; if (i<=n) r[i]=$0}}
       END{for (j=1;j<=n && j<=NR;j++) print r[j]}' "${PEAKS_BED}" \
    | sort -k1,1 -k2,2n > "${REGIONS}"
fi
echo "regions to predict: $(wc -l < "${REGIONS}")"

record () {  # stage seconds status detail
  printf "%s\t%s\t%s\t%s\t%s\ttensorflow\t%s\t%s\t%s\n" \
    "${RUN_TAG}" "${FOLD_NUM}" "${GPU_NAME}" "${ARCH}" "$1" "$2" "$3" "$4" >> "${SUMMARY}"
}

time_stage () {  # name outdir -- remaining args are the chrombpnet command
  local name="$1"; local outdir="$2"; shift 2
  rm -rf "${outdir}" 2>/dev/null || mv "${outdir}" "${outdir}.stale.$(date +%s)"
  mkdir -p "${outdir}"
  local start; start=$(date +%s.%N)
  set +e
  "${CHROMBPNET[@]}" "$@" > "${outdir}/run.log" 2>&1
  local rc=$?
  set -e
  local elapsed; elapsed=$(awk -v a="$(date +%s.%N)" -v b="${start}" 'BEGIN{printf "%.1f", a-b}')
  if [[ ${rc} -eq 0 ]]; then
    record "${name}" "${elapsed}" ok ""
    printf "  %-12s %8.1fs\n" "${name}" "${elapsed}"
  else
    record "${name}" "" failed "see ${outdir}/run.log"
    printf "  %-12s FAILED (exit %d)\n" "${name}" "${rc}"
    tail -20 "${outdir}/run.log" | sed 's/^/  | /'
  fi
}

echo
echo "### 1/3  pred_bw"
time_stage pred_bw "${WORK}/pred_bw" pred_bw \
  -cm "${CHROMBPNET_H5}" -cmb "${NOBIAS_H5}" -bm "${BIAS_H5}" \
  -r "${REGIONS}" -g "${GENOME_PATH}" -c "${CHROM_PATH}" \
  -bw "${OBSERVED_BW}" -op "${WORK}/pred_bw/pred" -os "${WORK}/pred_bw/stats" \
  -bs 256 -t 0

echo
CONTRIB_REGIONS="${WORK}/contrib_regions.bed"
head -n "${N_CONTRIB}" "${REGIONS}" > "${CONTRIB_REGIONS}"
echo "### 2/3  contribs_bw ($(wc -l < "${CONTRIB_REGIONS}") regions; stock takes no --batch-size)"
time_stage contribs_bw "${WORK}/contribs_bw" contribs_bw \
  -m "${NOBIAS_H5}" -r "${CONTRIB_REGIONS}" \
  -g "${GENOME_PATH}" -c "${CHROM_PATH}" \
  -op "${WORK}/contribs_bw/contrib" -pc profile -t 0

echo
echo "### 3/3  footprints"
time_stage footprints "${WORK}/footprints" footprints \
  -m "${NOBIAS_H5}" -r "${NONPEAKS_BED}" \
  -g "${GENOME_PATH}" -fl "${FOLD_JSON}" \
  -op "${WORK}/footprints/fp" -pwm_f "${MOTIFS_TSV}" -bs 256

echo
echo "=================================================================="
echo " summary: ${SUMMARY}"
echo "=================================================================="
column -t -s $'\t' "${SUMMARY}"
