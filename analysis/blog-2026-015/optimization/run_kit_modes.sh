#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=chrombpnet_kit_modes
#SBATCH --output=/path/to/workdir/sherlock/logs/chrombpnet_kit_modes.%x.%j.out
#SBATCH --error=/path/to/workdir/sherlock/logs/chrombpnet_kit_modes.%x.%j.err
#
# The optimization kit's own modes on stock chrombpnet 1.0.1: what the kit's
# changes are worth in the original package, for the blog's "in original
# chrombpnet" column. Same method as run_pred_bw_rerun.sh: the genome, bigWig
# and model read once before anything is timed, an untimed warm-up run of each
# arm, ROUNDS rounds in a different order each, only the pred_bw command timed.
#
#   bash setup_kit_env.sh                                    # once
#   sbatch --constraint="GPU_SKU:L40S" run_kit_modes.sh sherlock
#   sbatch --constraint="GPU_SKU:<H200 name>" run_kit_modes.sh sherlock
#   python summarize_rerun.py
#
# Or as a plain bash script on a lab GPU host (run tag *_kit-modes_<host>):
#
#   CUDA_VISIBLE_DEVICES=<idle GPU> nohup bash run_kit_modes.sh \
#     > /path/to/workdir/sherlock/logs/kit_modes.$(hostname -s).log 2>&1 &
#
# Arms (the kit's run.sh pred_bw, one model -cm as the kit runs it):
#   stock     --mode off          stock chrombpnet 1.0.1 as shipped
#   stockdet  --mode off --det 1  stock with TensorFlow's determinism settings
#   exact     --mode exact        every kit change, the fp32 Triton forward (K1)
#   fast      --mode fast         every kit change; on H100/H200 the Triton forward
#                                 in TF32, on L40S stock's TensorFlow graph (the
#                                 kit's default route per card)
#   stockmalloc  --mode off with the kit's malloc_env lever (the three glibc
#             variables its fast/exact modes set: MALLOC_MMAP_MAX_=0,
#             MALLOC_TRIM_THRESHOLD_=-1, MALLOC_TOP_PAD_=536870912), so stock
#             vs stockmalloc is that lever alone in the original package
#   stages    stock chrombpnet 1.0.1 under stock_stage_timer.py: the same run,
#             each stage a kit lever replaces timed (featurise, forward, bigWig
#             write, metrics, HDF5 write, finish-stage imports, teardown)
# So on an L40S, fast vs stock is the kit's changes without its fused kernels
# and exact vs fast is the fused kernels; on an H200 both modes use them. The
# kit arms keep their opt_manifest.json (the kit's run record, its own stage
# timings) beside run.log; summarize_kit_levers.py pairs them with `stages`.
# Every run also records peak_rss_mb (peak_rss.py: the peak resident memory of
# its largest process).
#
# malloc_env alone (its own run tag, so the kit-modes results stay as they are):
#   ARMS=stock,stockmalloc RUN_KIND=kit-malloc sbatch --constraint="GPU_SKU:L40S" run_kit_modes.sh sherlock
#
# The model is the kit's (ENCODE ENCFF142IOR, GM12878 ATAC-seq fold 0: its
# install step checks the digests); the regions are the 20,000 peaks the
# refactor runs predict, so the region count matches.
#
# Usage: run_kit_modes.sh [CLUSTER] [REGIONS_BED]
#   CLUSTER  sherlock | local; inferred from the Slurm job if omitted.
#            local: no module loads; the GPU is the first of CUDA_VISIBLE_DEVICES
#            and must be idle; TensorFlow 2.8 gets /usr/local's CUDA 11.2 and
#            cuDNN 8.1 (or TF_CUDA_DIRS)
#   env: ROUNDS (5), ARMS (stock,stockdet,exact,fast,stages; also stockmalloc), RUN_KIND
#        (kit-modes: the run tag's suffix), BATCH_SIZE (256, as the
#        refactor runs; the kit itself uses 1024), RUN_TAG, OUT_ROOT, KIT_ENV,
#        KIT_HOME, DEVICE (cuda; cpu for a smoke test), TF_CUDA_DIRS, ALLOW_BUSY_GPU

set -eo pipefail

OAK_DIR="/path/to/workdir"
HERE="${OAK_DIR}/software/chrombpnet-refactor/analysis/blog-2026-015/optimization"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"
resolve_cluster "${1:-}"
DEVICE="${DEVICE:-cuda}"
REGIONS_SRC="${2:-${OAK_DIR}/chromatlas/test/chrombpnet-refactor-optimization/analysis-rerun/NVIDIA-L40S_cuda-89/fold_0/regions.bed}"
ROUNDS="${ROUNDS:-5}"
ARMS_ARG="${ARMS:-stock,stockdet,exact,fast,stages}"
# An unknown arm would reach run.sh without --mode, i.e. as the kit's default mode.
for arm in ${ARMS_ARG//,/ }; do
  case "${arm}" in
    stock|stockdet|exact|fast|stages|stockmalloc) ;;
    *) echo "unknown arm '${arm}' (expected stock, stockdet, exact, fast, stages, stockmalloc)" >&2; exit 2 ;;
  esac
done
BATCH_SIZE="${BATCH_SIZE:-256}"
KIT="${KIT_ROOT:-${OAK_DIR}/software/uplifting-biomolecular-modeling}/chrombpnet"
KIT_ENV="${KIT_ENV:-chrombpnet-kit}"
KIT_HOME="${KIT_HOME:-${OAK_DIR}/chromatlas/test/chrombpnet-kit}"
GENOME="${OAK_DIR}/chromatlas/reference/hg38/hg38.fa"
CHROM_SIZES="${OAK_DIR}/chromatlas/reference/hg38/hg38.chrom.sizes"
WEIGHTS="${KIT_HOME}/weights"
MODEL="${WEIGHTS}/GM12878_ATAC/fold_0/chrombpnet_recompiled.h5"
BIGWIG="${KIT_HOME}/data/ENCFF180XQC.bigWig"
OUT_ROOT="${OUT_ROOT:-${OAK_DIR}/chromatlas/test/chrombpnet-refactor-optimization/analysis-rerun}"

for required in "${KIT}/run.sh" "${MODEL}" "${GENOME}" "${GENOME}.fai" "${CHROM_SIZES}" \
                "${BIGWIG}" "${REGIONS_SRC}"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required} (run setup_kit_env.sh first?)" >&2; exit 1; }
done

eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate "${KIT_ENV}"
# TensorFlow 2.8 loads CUDA 11.2 and cuDNN 8.1 from the system; the Triton
# forward's torch brings its own CUDA 12.4 wheels (so locally no other CUDA
# directory stays on LD_LIBRARY_PATH: ~/.bashrc's CUDA 12 would shadow them).
if [[ "${CLUSTER}" == "sherlock" ]]; then
  # The modules' CUDA 11.2 / cuDNN 8.1 only, not the submitting shell's CUDA or
  # cuDNN (2026-10-21: torch here reported cuDNN 9.4, not the env's 9.1, and
  # exact mode died loading cuDNN). And a C11 compiler: Triton builds a
  # launcher for every kernel the kit's shipped cache lacks (exact mode's), and
  # the nodes' gcc 4.8 has no stdatomic.h.
  export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
  module load gcc/14.2.0
  module load cuda/11.2.0
  module load cudnn/8.1.1.33
else
  tf_local_libraries
fi
export CHROMBPNET_OPT_WEIGHTS="${WEIGHTS}" CHROMBPNET_OPT_DATA="${KIT_HOME}/data"
export PYTHONHASHSEED=0 CFLAGS=-g0 PYTHONNOUSERSITE=1
unset PYTHONPATH
# Triton's compiled helpers are built against the host glibc, and Sherlock's
# GPU nodes run different ones (see run_pred_bw_rerun.sh): one root per glibc;
# on a lab host, one per host.
GLIBC_VERSION="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
CACHE_KEY="glibc-${GLIBC_VERSION:-unknown}"
[[ "${CLUSTER}" == local ]] && CACHE_KEY="$(hostname -s)-${CACHE_KEY}"
export MODEL_OPT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-${KIT_HOME}/jit/${CACHE_KEY}}"
mkdir -p "${MODEL_OPT_JIT_ROOT}"

select_gpu
ARCH="cuda-${GPU_CC//./}"
case "${GPU_NAME}" in
  *H200*) CONFIG=h200 ;;
  *A100*) CONFIG=a100 ;;
  *)      CONFIG=h100 ;;   # H100, and cards the kit has no config for (the route follows the card found)
esac
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}_${RUN_KIND:-kit-modes}$(host_suffix)}"
start_results "${OUT_ROOT}/${RUN_TAG}/fold_GM12878"
SUMMARY="${WORK}/timings.tsv"
printf "run\tfold\tgpu\tarch\tarm\tround\tposition\tseconds\tstatus\tdetail\n" > "${SUMMARY}"
REGIONS="${WORK}/regions.bed"
cp "${REGIONS_SRC}" "${REGIONS}"
WARM_REGIONS="${WORK}/warmup_regions.bed"
head -n 2000 "${REGIONS}" > "${WARM_REGIONS}"

echo "=================================================================="
echo " run tag : ${RUN_TAG}    host $(hostname) (${CLUSTER})    GPU ${GPU_ID}: ${GPU_NAME} (${ARCH})"
echo " load    : $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) CPUs"
echo " config  : ${CONFIG}    arms: ${ARMS_ARG}    rounds: ${ROUNDS}    -bs ${BATCH_SIZE}"
echo " regions : $(wc -l < "${REGIONS}")    model: ${MODEL}"
echo " output  : ${WORK}"
echo "=================================================================="

arm_flags () {
  case "$1" in
    stock|stockmalloc) echo "--mode off" ;;
    stockdet) echo "--mode off --det 1" ;;
    exact)    echo "--mode exact" ;;
    fast)     echo "--mode fast" ;;
    stages)   echo "" ;;
    *)        echo "unknown arm $1" >&2; return 2 ;;
  esac
}

run_arm () {  # arm regions outdir -> "status seconds"
  local arm="$1" regions="$2" out="$3" rc flags malloc=()
  [[ "${arm}" == stockmalloc ]] && malloc=(env MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=-1 MALLOC_TOP_PAD_=536870912)
  rm -rf "${out}" 2>/dev/null || mv "${out}" "${out}.stale.$(date +%s)"
  mkdir -p "${out}"
  flags="$(arm_flags "${arm}")"
  local t0 t1
  t0=$(date +%s.%N)
  if [[ "${arm}" == stages ]]; then
    # Stock's own pred_bw, as the kit's `--mode off` runs it (no kit variable
    # set), with stock_stage_timer.py's timers around its stages.
    ( cd "${WORK}" && unset $(compgen -e | grep '^CHROMBPNET_OPT') && \
      python -I "${HERE}/peak_rss.py" "${out}/peak_rss_mb" \
      python -I "${HERE}/stock_stage_timer.py" "${out}/stages.json" pred_bw \
        -cm "${MODEL}" -g "${GENOME}" -c "${CHROM_SIZES}" -r "${regions}" -bw "${BIGWIG}" \
        -bs "${BATCH_SIZE}" -op "${out}/pred" -os "${out}/stats" ) > "${out}/run.log" 2>&1 && rc=0 || rc=$?
  else
    # shellcheck disable=SC2086
    ( cd "${KIT}" && python -I "${HERE}/peak_rss.py" "${out}/peak_rss_mb" \
        "${malloc[@]}" bash run.sh pred_bw --config "${CONFIG}" ${flags} \
        -cm "${MODEL}" -g "${GENOME}" -c "${CHROM_SIZES}" -r "${regions}" -bw "${BIGWIG}" \
        -bs "${BATCH_SIZE}" -op "${out}/pred" -os "${out}/stats" ) > "${out}/run.log" 2>&1 && rc=0 || rc=$?
  fi
  t1=$(date +%s.%N)
  echo "${t0} ${t1}" > "${out}/time"   # the stages arm's teardown: t1 minus when main() returned
  if [[ ${rc} -eq 0 ]]; then
    echo "ok $(awk -v a="${t1}" -v b="${t0}" 'BEGIN{printf "%.1f", a-b}')"
  elif [[ ${rc} -eq 3 ]] && grep -q "NOT ACTIVE" "${out}/run.log"; then
    echo "refused -"
  else
    echo "failed -"
  fi
}

record () {  # arm round position seconds status detail
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${RUN_TAG}" "GM12878_fold0" "${GPU_NAME}" \
    "${ARCH}" "$1" "$2" "$3" "$4" "$5" "$6" >> "${SUMMARY}"
}

if [[ "${DEVICE}" != "cpu" ]]; then
  # TensorFlow 2.8 without its CUDA libraries runs on the CPU, silently.
  echo "### GPU check"
  ( cd "${WORK}"; check_framework_gpu tensorflow "${WORK}/gpu_check_tf.log" ) || exit 1
  ( cd "${WORK}"; check_framework_gpu torch "${WORK}/gpu_check_pt.log" ) || exit 1
  echo "  ${KIT_ENV}: TensorFlow and PyTorch see the GPU"
fi

echo "### warm-up: file cache"
cat "${GENOME}" "${GENOME}.fai" "${BIGWIG}" "${WEIGHTS}"/GM12878_ATAC/fold_0/*.h5 > /dev/null

IFS=',' read -r -a ARMS_LIST <<< "${ARMS_ARG}"
ACTIVE=()
echo "### warm-up: one untimed run per arm on $(wc -l < "${WARM_REGIONS}") regions"
for arm in "${ARMS_LIST[@]}"; do
  read -r status seconds <<< "$(run_arm "${arm}" "${WARM_REGIONS}" "${WORK}/warmup_${arm}")"
  printf "  %-9s %s\n" "${arm}" "${status}"
  case "${status}" in
    ok)      ACTIVE+=("${arm}") ;;
    refused) record "${arm}" "-" "-" "" refused "$(grep -m1 -o 'NOT ACTIVE.*' "${WORK}/warmup_${arm}/run.log" | cut -c1-160)" ;;
    *)       record "${arm}" "-" "-" "" failed "warm-up failed: ${WORK}/warmup_${arm}/run.log"
             tail -15 "${WORK}/warmup_${arm}/run.log" | sed 's/^/  | /' ;;
  esac
  find "${WORK}/warmup_${arm}" -type f ! -name run.log ! -name time ! -name opt_manifest.json ! -name stages.json ! -name peak_rss_mb -delete 2>/dev/null || true
done
[[ ${#ACTIVE[@]} -gt 0 ]] || { echo "no arm survived the warm-up" >&2; exit 1; }

n=${#ACTIVE[@]}
shift_by=$(( (n + ROUNDS - 1) / ROUNDS ))
(( shift_by > 0 )) || shift_by=1
for (( round = 0; round < ROUNDS; round++ )); do
  echo "### round $((round + 1)) of ${ROUNDS}"
  for (( i = 0; i < n; i++ )); do
    arm="${ACTIVE[$(( (i + round * shift_by) % n ))]}"
    read -r status seconds <<< "$(run_arm "${arm}" "${REGIONS}" "${WORK}/r${round}_${arm}")"
    printf "  %-9s %-8s %s\n" "${arm}" "${status}" "${seconds}"
    if [[ "${status}" == ok ]]; then
      record "${arm}" "${round}" "${i}" "${seconds}" ok ""
    else
      record "${arm}" "${round}" "${i}" "" "${status}" "${WORK}/r${round}_${arm}/run.log"
    fi
    find "${WORK}/r${round}_${arm}" -type f ! -name run.log ! -name time ! -name opt_manifest.json ! -name stages.json ! -name peak_rss_mb -delete 2>/dev/null || true
  done
done
echo "timings: ${SUMMARY}"
finish_results
