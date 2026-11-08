#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=chrombpnet_predbw_rerun
#SBATCH --output=/path/to/workdir/sherlock/logs/chrombpnet_predbw_rerun.%x.%A_%a.out
#SBATCH --error=/path/to/workdir/sherlock/logs/chrombpnet_predbw_rerun.%x.%A_%a.err
#SBATCH --array=0-4
#
# Re-measures pred_bw for the blog post's speedup tables. The September sweep
# (run_chrombpnet_optimization_benchmark.sh, phase matrix) always ran the
# unoptimized arm first, on a cold file cache, which inflated it. Here the
# genome, bigWig and models are read once before anything is timed, every
# arm gets an untimed warm-up run, and each round runs the arms in a
# different order. Every arm predicts the same regions against the same
# observed bigWig.
#
#   ./list_gpu_skus.sh                       # the exact GPU_SKU names
#   sbatch --constraint="GPU_SKU:<H200 name>" run_pred_bw_rerun.sh sherlock
#   PT_ENV=chrombpnet-pytorch-sherlock \
#     sbatch --constraint="GPU_SKU:<L40S name>" run_pred_bw_rerun.sh sherlock   # glibc 2.17
#   python summarize_rerun.py                # once the arrays finish
#
# Arms (pred_bw with -bw, three models: bias, ChromBPNet, no-bias):
#   original  chrombpnet 1.x, TensorFlow 2.8 (env `chrombpnet`), the fold's TF models
#   none      chrombpnet-refactor at 693d162, --optimize none (no optimizations)
#   kit       693d162 --optimize 1     (the optimization kit's changes, no Triton)
#   ours      693d162 --optimize 3     (this package's own changes)
#   both      693d162 --optimize 1,3   (the default at 693d162)
#   main      current main, default    (should match `both`)
#   fk        current main, --predict-fk (refused on cards with no route, e.g. L40S)
# The refactor arms run the fold's PyTorch models, all in PT_ENV (default
# chrombpnet-pytorch, torch 2.14.1): the comparison is between code paths, so
# one env serves all. On the L40S nodes (glibc 2.17) set
# PT_ENV=chrombpnet-pytorch-sherlock (torch 2.5.1, built on Sherlock).
#
# Usage: run_pred_bw_rerun.sh CLUSTER [FOLD_NUM] [N_REGIONS]
#   CLUSTER    sherlock | local (local: no module loads; DEVICE=cpu for a smoke test)
#   env:       ROUNDS (3), ARMS (original,none,kit,ours,both,main,fk),
#              PT_ENV, TF_ENV, RUN_TAG, DEVICE (cuda)

set -eo pipefail

CLUSTER="${1:-sherlock}"
FOLD_NUM="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
N_REGIONS="${3:-20000}"
ROUNDS="${ROUNDS:-3}"
ARMS_ARG="${ARMS:-original,none,kit,ours,both,main,fk}"
DEVICE="${DEVICE:-cuda}"

OAK_DIR="/path/to/workdir"
HERE="${OAK_DIR}/software/chrombpnet-refactor/analysis/blog-2026-015/optimization"
CODE_OLD="${HERE}/code-693d162"            # git archive 693d162 (Sherlock's git has no worktrees)
CODE_MAIN="${CHROMBPNET_REPO:-${OAK_DIR}/software/chrombpnet-refactor}"
TF_REPO="${OAK_DIR}/software/chrombpnet"
GENOME="${OAK_DIR}/chromatlas/reference/hg38/hg38.fa"
CHROM_SIZES="${OAK_DIR}/chromatlas/reference/hg38/hg38.chrom.sizes"
PT_MODEL="${OAK_DIR}/chromatlas/test/chrombpnet-refactor/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
TF_MODEL="${OAK_DIR}/chromatlas/test/chrombpnet-tf/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
OBSERVED_BW="${PT_MODEL}/auxiliary/data_unstranded.bw"
PEAKS="${PT_MODEL}/auxiliary/filtered.peaks.bed"
OUT_ROOT="${OAK_DIR}/chromatlas/test/chrombpnet-refactor-optimization/analysis-rerun"

# chrombpnet-pytorch: torch 2.14.1, needs glibc >= 2.28. On a glibc 2.17 node
# (the L40S ones) set PT_ENV=chrombpnet-pytorch-sherlock: torch 2.5.1, built on
# Sherlock, the env of the earlier runs.
PT_ENV="${PT_ENV:-chrombpnet-pytorch}"
TF_ENV="${TF_ENV:-chrombpnet}"
# Triton caches the helpers it compiles (cuda_utils.so) alongside the kernels,
# built against the host's glibc. Sherlock's GPU nodes do not all run the same
# OS: a build on an H200 node (glibc 2.34) fails to load on an L40S node (2.17),
# which is what failed every fused-kernel run on the L40S. One cache per glibc
# version, shared by the array's tasks.
GLIBC_VERSION="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OAK_DIR}/chromatlas/test/chrombpnet-refactor-optimization/.triton-cache/glibc-${GLIBC_VERSION:-unknown}}"
mkdir -p "${TRITON_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# chrombpnet-pytorch (PyTorch 2.14) needs glibc >= 2.28 (engines/envs/glibc_scan.py:
# torch and igraph); some of sherlock's GPU nodes (the L40S ones) run 2.17. Stop
# here on such a node rather than failing at the first import (ARMS=original
# never enters PT_ENV).
GLIBC_FLOOR="${GLIBC_FLOOR:-2.28}"
if [[ "${PT_ENV}" != "chrombpnet-pytorch-sherlock" && ",${ARMS_ARG}," != ",original," ]]; then
  NODE_GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
  if [[ -n "${NODE_GLIBC}" && "$(printf '%s\n%s\n' "${GLIBC_FLOOR}" "${NODE_GLIBC}" | sort -V | head -1)" != "${GLIBC_FLOOR}" ]]; then
    echo "Error: ${PT_ENV} needs glibc >= ${GLIBC_FLOOR}; this node ($(hostname)) has ${NODE_GLIBC}." >&2
    echo "       Resubmit with --exclude=$(hostname), or set PT_ENV=chrombpnet-pytorch-sherlock." >&2
    exit 1
  fi
fi

for required in "${CODE_OLD}/chrombpnet/CHROMBPNET.py" "${CODE_MAIN}/chrombpnet/CHROMBPNET.py" \
                "${TF_REPO}/chrombpnet/CHROMBPNET.py" "${GENOME}" "${CHROM_SIZES}" \
                "${OBSERVED_BW}" "${PEAKS}" \
                "${PT_MODEL}/models/chrombpnet.pt" "${PT_MODEL}/models/chrombpnet_nobias.pt" \
                "${PT_MODEL}/models/bias_model_scaled.pt" \
                "${TF_MODEL}/models/chrombpnet.h5" "${TF_MODEL}/models/chrombpnet_nobias.h5" \
                "${TF_MODEL}/models/bias_model_scaled.h5"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 1; }
done

eval "$(conda shell.bash hook)"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
[[ "${DEVICE}" == "cpu" ]] && { GPU_NAME="CPU"; GPU_CC="0"; }
ARCH="cuda-${GPU_CC//./}"
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}}"
WORK="${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
if [[ -d "${WORK}" ]] && ! rm -rf "${WORK}" 2>/dev/null; then
  mv "${WORK}" "${WORK}.stale.$(date +%s)"
fi
mkdir -p "${WORK}"
NEUTRAL="${WORK}/cwd"     # run every command from here: never from a checkout
mkdir -p "${NEUTRAL}"
SUMMARY="${WORK}/timings.tsv"
printf "run\tfold\tgpu\tarch\tarm\tround\tposition\tseconds\tstatus\tdetail\n" > "${SUMMARY}"

REGIONS="${WORK}/regions.bed"
awk -v n="${N_REGIONS}" 'BEGIN{srand(1234)} {if (NR<=n) r[NR]=$0;
     else {i=int(rand()*NR)+1; if (i<=n) r[i]=$0}}
     END{for (j=1;j<=n && j<=NR;j++) print r[j]}' "${PEAKS}" \
  | sort -k1,1 -k2,2n > "${REGIONS}"
WARM_REGIONS="${WORK}/warmup_regions.bed"
head -n 2000 "${REGIONS}" > "${WARM_REGIONS}"

echo "=================================================================="
echo " run tag   : ${RUN_TAG}    fold ${FOLD_NUM}    host $(hostname)"
echo " GPU       : ${GPU_NAME} (${ARCH})"
echo " regions   : $(wc -l < "${REGIONS}") (warm-up: $(wc -l < "${WARM_REGIONS}"))"
echo " rounds    : ${ROUNDS}    arms: ${ARMS_ARG}"
echo " envs      : ${PT_ENV} (refactor), ${TF_ENV} (original)"
echo " output    : ${WORK}"
echo "=================================================================="

# --- environments, each entered in its own subshell -------------------------
setup_pt () {
  if [[ "${CLUSTER}" == "sherlock" && "${PT_ENV}" == "chrombpnet-pytorch-sherlock" ]]; then
    module load gcc/14.2.0
  fi
  conda activate "${PT_ENV}"
}
setup_tf () {
  if [[ "${CLUSTER}" == "sherlock" ]]; then
    # TF 2.8 needs CUDA 11.2 and cuDNN 8; mirrors run_chrombpnet_optimization_tensorflow.sh.
    module load cuda/11.2.0
    module load cudnn/8.1.1.33
    module load system
    module load pango
    module load system cairo
  fi
  conda activate "${TF_ENV}"
}

# The command line of one arm, given its regions and output directory.
arm_command () {  # arm regions outdir
  local arm="$1" regions="$2" out="$3"
  local common="-r ${regions} -g ${GENOME} -c ${CHROM_SIZES} -bw ${OBSERVED_BW} -op ${out}/pred -os ${out}/stats -bs 256 -t 0"
  local pt="-cm ${PT_MODEL}/models/chrombpnet.pt -cmb ${PT_MODEL}/models/chrombpnet_nobias.pt -bm ${PT_MODEL}/models/bias_model_scaled.pt"
  case "${arm}" in
    original) echo "TF ${TF_REPO} pred_bw -cm ${TF_MODEL}/models/chrombpnet.h5 -cmb ${TF_MODEL}/models/chrombpnet_nobias.h5 -bm ${TF_MODEL}/models/bias_model_scaled.h5 ${common}" ;;
    none)     echo "PT ${CODE_OLD} pred_bw --optimize none ${pt} ${common} --device ${DEVICE}" ;;
    kit)      echo "PT ${CODE_OLD} pred_bw --optimize 1 ${pt} ${common} --device ${DEVICE}" ;;
    ours)     echo "PT ${CODE_OLD} pred_bw --optimize 3 ${pt} ${common} --device ${DEVICE}" ;;
    both)     echo "PT ${CODE_OLD} pred_bw --optimize 1,3 ${pt} ${common} --device ${DEVICE}" ;;
    main)     echo "PT ${CODE_MAIN} pred_bw ${pt} ${common} --device ${DEVICE}" ;;
    fk)       echo "PT ${CODE_MAIN} pred_bw --predict-fk ${pt} ${common} --device ${DEVICE}" ;;
    *)        echo "unknown arm ${arm}" >&2; return 2 ;;
  esac
}

# Runs one arm in a fresh process; prints "status seconds". Only the python
# command is timed, not the module loads or the conda activation.
run_arm () {  # arm regions outdir
  local arm="$1" regions="$2" out="$3" rc kind code rest
  rm -rf "${out}" 2>/dev/null || mv "${out}" "${out}.stale.$(date +%s)"
  mkdir -p "${out}"
  read -r kind code rest <<< "$(arm_command "${arm}" "${regions}" "${out}")"
  (
    set +e
    if [[ "${kind}" == "TF" ]]; then setup_tf; else setup_pt; fi
    export PYTHONPATH="${code}"
    cd "${NEUTRAL}"
    python -c "import chrombpnet, os, sys; p = os.path.realpath(chrombpnet.__file__);
sys.exit(0 if p.startswith(os.path.realpath('${code}')) else 'wrong chrombpnet: ' + p)" || exit 3
    t0=$(date +%s.%N)
    # shellcheck disable=SC2086
    python -m chrombpnet.CHROMBPNET ${rest}
    rc=$?
    t1=$(date +%s.%N)
    echo "${rc} ${t0} ${t1}" > "${out}/time"
    exit ${rc}
  ) > "${out}/run.log" 2>&1 && rc=0 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    read -r _ t0 t1 < "${out}/time"
    echo "ok $(awk -v a="${t1}" -v b="${t0}" 'BEGIN{printf "%.1f", a-b}')"
  elif grep -q "TritonRouteRefused" "${out}/run.log"; then
    echo "refused -"
  else
    echo "failed -"
  fi
}

record () {  # arm round position seconds status detail
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${RUN_TAG}" "${FOLD_NUM}" "${GPU_NAME}" \
    "${ARCH}" "$1" "$2" "$3" "$4" "$5" "$6" >> "${SUMMARY}"
}

# --- warm the file cache and every arm ---------------------------------------
echo "### warm-up: file cache"
cat "${GENOME}" "${GENOME}.fai" "${OBSERVED_BW}" "${PT_MODEL}"/models/*.pt "${TF_MODEL}"/models/*.h5 > /dev/null

IFS=',' read -r -a ARMS_LIST <<< "${ARMS_ARG}"
ACTIVE=()
echo "### warm-up: one untimed run per arm on $(wc -l < "${WARM_REGIONS}") regions"
for arm in "${ARMS_LIST[@]}"; do
  read -r status seconds <<< "$(run_arm "${arm}" "${WARM_REGIONS}" "${WORK}/warmup_${arm}")"
  printf "  %-9s %s\n" "${arm}" "${status}"
  find "${WORK}/warmup_${arm}" -type f ! -name run.log ! -name time -delete 2>/dev/null || true
  case "${status}" in
    ok)      ACTIVE+=("${arm}") ;;
    refused) record "${arm}" "-" "-" "" refused "no Triton route on ${ARCH}" ;;
    *)       record "${arm}" "-" "-" "" failed "warm-up failed: ${WORK}/warmup_${arm}/run.log"
             tail -15 "${WORK}/warmup_${arm}/run.log" | sed 's/^/  | /' ;;
  esac
done
[[ ${#ACTIVE[@]} -gt 0 ]] || { echo "no arm survived the warm-up" >&2; exit 1; }

# --- timed rounds, rotating the order ----------------------------------------
n=${#ACTIVE[@]}
shift_by=$(( (n + ROUNDS - 1) / ROUNDS ))
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
    # Keep only the logs: three bigWigs per run add up across rounds.
    find "${WORK}/r${round}_${arm}" -type f ! -name run.log ! -name time -delete 2>/dev/null || true
  done
done
rm -rf "${NEUTRAL}"
echo "timings: ${SUMMARY}"
