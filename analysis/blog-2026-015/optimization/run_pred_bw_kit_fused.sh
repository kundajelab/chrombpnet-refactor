#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --job-name=chrombpnet_predbw_kit_fused
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --array=0-4
#
# What the fused kernels cost without this package's own optimizations: the
# kit's changes with and without its Triton forward (--optimize 1 vs 1,2 at
# 4962484), against the unoptimized and default runs of the same job. Same
# method as run_pred_bw_rerun.sh (a copy of it, with two more arms and its own
# run tag, so it never touches that job's results):
#
# From this folder, with site.env's cluster options and GPU constraints:
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_pred_bw_kit_fused.sh slurm
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_pred_bw_kit_fused.sh slurm
#   python summarize_rerun.py <OPTIMIZATION_OUT>
#
# Or as a plain bash script on a GPU host (an L40S host, an H200 host):
# the five folds one after another on one GPU, run tag *_kit-fused_<host>.
#
#   CUDA_VISIBLE_DEVICES=<idle GPU> nohup bash run_pred_bw_kit_fused.sh \
#     > <log dir>/kit_fused.$(hostname -s).log 2>&1 &
#
# Re-measures pred_bw for the blog post's speedup tables. The September sweep
# (run_chrombpnet_optimization_benchmark.sh, phase matrix) always ran the
# unoptimized arm first, on a cold file cache, which inflated it. Here the
# genome, bigWig and models are read once before anything is timed, every
# arm gets an untimed warm-up run, and each round runs the arms in a
# different order. Every arm predicts the same regions against the same
# observed bigWig.
#
#   ./list_gpu_skus.sh <partitions>          # the cluster's GPU feature names
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_pred_bw_rerun.sh slurm
#   PT_ENV=<glibc 2.17 build> \
#     sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_pred_bw_rerun.sh slurm   # glibc 2.17
#   python summarize_rerun.py <OPTIMIZATION_OUT>   # once the arrays finish
#
# Arms (pred_bw with -bw, three models: bias, ChromBPNet, no-bias):
#   original  chrombpnet 1.x, TensorFlow 2.8 (env `chrombpnet`), the fold's TF models
#   none      chrombpnet-refactor at 4962484, --optimize none (no optimizations)
#   kit       4962484 --optimize 1     (the optimization kit's changes, no Triton)
#   ours      4962484 --optimize 3     (this package's own changes)
#   both      4962484 --optimize 1,3   (the default at 4962484)
#   kitfk     4962484 --optimize 1,2   (the kit's changes with its Triton forward)
#   fkonly    4962484 --optimize 2     (the Triton forward alone)
#   main      current main, default    (should match `both`)
#   fk        current main, --predict-fused-kernel (everything plus the fused kernels)
#   nomalloc  current main with --no-malloc-env: glibc's default allocator (since
#             fa45de5 the default applies malloc_env itself, so `main` is tuned)
#   mallocenv current main, default, with the kit's malloc_env lever: glibc never
#             mmaps or trims the heap, and pads it by 512 MiB (MALLOC_MMAP_MAX_=0,
#             MALLOC_TRIM_THRESHOLD_=-1, MALLOC_TOP_PAD_=536870912, which the
#             kit sets for itself and its helper processes)
# The refactor arms run the fold's PyTorch models, all in PT_ENV: the
# comparison is between code paths, so one env serves all. Under Slurm it
# defaults to site.env's PT_ENV_GLIBC217 if set (the blog's runs: torch 2.5.1,
# built for glibc 2.17, which the cluster's L40S and many of its H200 nodes
# run), else and elsewhere to chrombpnet-pytorch (torch 2.14.1, glibc >= 2.28).
#
# Usage: run_pred_bw_kit_fused.sh [CLUSTER] [FOLDS] [N_REGIONS]
#   CLUSTER    slurm | local; inferred from the Slurm job if omitted.
#              local: no site hooks; the GPU is the first of
#              CUDA_VISIBLE_DEVICES and must be idle; TensorFlow 2.8 gets
#              /usr/local's CUDA 11.2 and cuDNN 8.1 (or TF_CUDA_DIRS)
#   FOLDS      a fold, a comma-separated list, or all (default: the array task
#              under Slurm, else all), run one after another
#   env:       ROUNDS (3), ARMS (none,kit,kitfk,fkonly,both,fk; all nine:
#              original,none,kit,ours,both,main,kitfk,fkonly,fk), PT_ENV,
#              TF_ENV, RUN_TAG, OUT_ROOT (site.env's OPTIMIZATION_OUT), TRITON_CACHE_DIR
#              (.triton-cache/<glibc> beside OUT_ROOT), DEVICE (cuda; cpu for a smoke
#              test), TF_CUDA_DIRS, ALLOW_BUSY_GPU
#   site.env:  ORIGINAL_REPO, GENOME, CHROM_SIZES, REFACTOR_MODELS, ORIGINAL_MODELS,
#              OPTIMIZATION_OUT (unless OUT_ROOT is set); optional PT_ENV_GLIBC217 and
#              the hooks pytorch_compiler, tf28_libraries, report_libraries (under Slurm)

set -eo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
HERE="${BLOG_SCRIPTS}/optimization"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

resolve_cluster "${1:-}"
FOLDS="${2:-${SLURM_ARRAY_TASK_ID:-all}}"
N_REGIONS="${3:-20000}"
ROUNDS="${ROUNDS:-3}"
ARMS_ARG="${ARMS:-none,kit,kitfk,fkonly,both,fk}"
DEVICE="${DEVICE:-cuda}"
for arm in ${ARMS_ARG//,/ }; do
  case "${arm}" in
    original|none|kit|ours|both|kitfk|fkonly|main|fk|mallocenv|nomalloc) ;;
    *) echo "unknown arm '${arm}' (expected original, none, kit, ours, both, kitfk, fkonly, main, fk, mallocenv, nomalloc)" >&2; exit 2 ;;
  esac
done

# Several folds: this script once per fold, in turn (what the array does under
# Slurm, without the parallelism, which would share one host's GPU and CPUs).
[[ "${FOLDS}" == all ]] && FOLDS=0,1,2,3,4
if [[ "${FOLDS}" == *,* ]]; then
  IFS=',' read -r -a FOLD_LIST <<< "${FOLDS}"
  failed=()
  for fold in "${FOLD_LIST[@]}"; do
    bash "${BASH_SOURCE[0]}" "${CLUSTER}" "${fold}" "${N_REGIONS}" || failed+=("${fold}")
  done
  [[ ${#failed[@]} -eq 0 ]] || { echo "folds that failed: ${failed[*]}" >&2; exit 1; }
  exit 0
fi
FOLD_NUM="${FOLDS}"

SITE_NEEDS=(ORIGINAL_REPO GENOME CHROM_SIZES REFACTOR_MODELS ORIGINAL_MODELS)
[[ -n "${OUT_ROOT:-}" ]] || SITE_NEEDS+=(OPTIMIZATION_OUT)
require_site "${SITE_NEEDS[@]}"
CODE_OLD="${HERE}/code-4962484"            # git archive 4962484 (no git worktree needed)
CODE_MAIN="${CHROMBPNET_REPO}"
TF_REPO="${ORIGINAL_REPO}"
PT_MODEL="${REFACTOR_MODELS}/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
TF_MODEL="${ORIGINAL_MODELS}/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
OBSERVED_BW="${PT_MODEL}/auxiliary/data_unstranded.bw"
PEAKS="${PT_MODEL}/auxiliary/filtered.peaks.bed"
OUT_ROOT="${OUT_ROOT:-${OPTIMIZATION_OUT}}"

# chrombpnet-pytorch: torch 2.14.1, needs glibc >= 2.28; a cluster whose GPU
# nodes mostly run 2.17 names a build for them in site.env (PT_ENV_GLIBC217:
# torch 2.5.1 for the blog's runs), the default under Slurm.
PT_ENV="$(default_pt_env)"
TF_ENV="${TF_ENV:-chrombpnet}"
# Triton caches the helpers it compiles (cuda_utils.so) alongside the kernels,
# built against the host's glibc. A cluster's GPU nodes need not all run the
# same OS: a build on an H200 node (glibc 2.34) failed to load on an L40S node
# (2.17), which is what failed every fused-kernel run on the L40S. One cache per
# glibc version, shared by the array's tasks; on a GPU host, one per host.
GLIBC_VERSION="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
CACHE_KEY="glibc-${GLIBC_VERSION:-unknown}"
[[ "${CLUSTER}" == local ]] && CACHE_KEY="$(hostname -s)-${CACHE_KEY}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(dirname "${OUT_ROOT}")/.triton-cache/${CACHE_KEY}}"
mkdir -p "${TRITON_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# chrombpnet-pytorch (PyTorch 2.14) needs glibc >= 2.28 (engines/envs/glibc_scan.py:
# torch and igraph); some clusters' GPU nodes (the blog's L40S ones) run 2.17.
# Stop here on such a node rather than failing at the first import
# (ARMS=original never enters PT_ENV).
GLIBC_FLOOR="${GLIBC_FLOOR:-2.28}"
if ! is_glibc217_env "${PT_ENV}" && [[ ",${ARMS_ARG}," != ",original," ]]; then
  NODE_GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
  if [[ -n "${NODE_GLIBC}" && "$(printf '%s\n%s\n' "${GLIBC_FLOOR}" "${NODE_GLIBC}" | sort -V | head -1)" != "${GLIBC_FLOOR}" ]]; then
    echo "Error: ${PT_ENV} needs glibc >= ${GLIBC_FLOOR}; this node ($(hostname)) has ${NODE_GLIBC}." >&2
    echo "       Resubmit with --exclude=$(hostname), or set PT_ENV to a glibc 2.17 build (site.env PT_ENV_GLIBC217)." >&2
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

eval "$("$(conda_exe)" shell.bash hook)"

select_gpu
ARCH="cuda-${GPU_CC//./}"
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}_${RUN_KIND:-kit-fused}$(host_suffix)}"
start_results "${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
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
echo " run tag   : ${RUN_TAG}    fold ${FOLD_NUM}    host $(hostname) (${CLUSTER})"
echo " GPU       : ${GPU_ID}: ${GPU_NAME} (${ARCH})    load average $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) CPUs"
echo " regions   : $(wc -l < "${REGIONS}") (warm-up: $(wc -l < "${WARM_REGIONS}"))"
echo " rounds    : ${ROUNDS}    arms: ${ARMS_ARG}"
echo " envs      : ${PT_ENV} (refactor), ${TF_ENV} (original)"
echo " output    : ${WORK}"
echo "=================================================================="

# --- environments, each entered in its own subshell -------------------------
setup_pt () {
  # PyTorch's own CUDA and cuDNN wheels only: no CUDA or cuDNN directory from
  # a login shell or the submitting shell.
  export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
  if [[ "${CLUSTER}" == slurm ]] && is_glibc217_env "${PT_ENV}"; then
    site_hook pytorch_compiler
  fi
  conda activate "${PT_ENV}"
}
setup_tf () {
  if [[ "${CLUSTER}" == slurm ]]; then
    # TF 2.8 needs CUDA 11.2 and cuDNN 8; mirrors run_chrombpnet_optimization_tensorflow.sh.
    # The site hooks' copies only, not the submitting shell's CUDA or cuDNN.
    export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
    site_hook tf28_libraries
    site_hook report_libraries
  else
    tf_local_libraries
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
    kitfk)    echo "PT ${CODE_OLD} pred_bw --optimize 1,2 ${pt} ${common} --device ${DEVICE}" ;;
    fkonly)   echo "PT ${CODE_OLD} pred_bw --optimize 2 ${pt} ${common} --device ${DEVICE}" ;;
    main)     echo "PT ${CODE_MAIN} pred_bw ${pt} ${common} --device ${DEVICE}" ;;
    nomalloc) echo "PT ${CODE_MAIN} pred_bw ${pt} ${common} --device ${DEVICE} --no-malloc-env" ;;
    fk)       echo "PT ${CODE_MAIN} pred_bw --predict-fused-kernel ${pt} ${common} --device ${DEVICE}" ;;
    mallocenv) echo "PTMALLOC ${CODE_MAIN} pred_bw ${pt} ${common} --device ${DEVICE}" ;;
    *)        echo "unknown arm ${arm}" >&2; return 2 ;;
  esac
}

# Runs one arm in a fresh process; prints "status seconds". Only the python
# command is timed, not the site hooks or the conda activation.
run_arm () {  # arm regions outdir
  local arm="$1" regions="$2" out="$3" rc kind code rest
  rm -rf "${out}" 2>/dev/null || mv "${out}" "${out}.stale.$(date +%s)"
  mkdir -p "${out}"
  read -r kind code rest <<< "$(arm_command "${arm}" "${regions}" "${out}")"
  (
    set +e
    if [[ "${kind}" == "TF" ]]; then setup_tf; else setup_pt; fi
    if [[ "${kind}" == "PTMALLOC" ]]; then
      export MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=-1 MALLOC_TOP_PAD_=536870912
    fi
    export PYTHONPATH="${code}"
    cd "${NEUTRAL}"
    python -c "import chrombpnet, os, sys; p = os.path.realpath(chrombpnet.__file__);
sys.exit(0 if p.startswith(os.path.realpath('${code}')) else 'wrong chrombpnet: ' + p)" || exit 3
    t0=$(date +%s.%N)
    # shellcheck disable=SC2086
    python -I "${HERE}/peak_rss.py" "${out}/peak_rss_mb" python -m chrombpnet.CHROMBPNET ${rest}
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

# Each env reaches the GPU (TensorFlow 2.8 would otherwise run on the CPU, silently).
if [[ "${DEVICE}" != "cpu" ]]; then
  echo "### GPU check"
  if [[ ",${ARMS_ARG}," == *,original,* ]]; then
    ( setup_tf; cd "${NEUTRAL}"; check_framework_gpu tensorflow "${WORK}/gpu_check_tf.log" ) || exit 1
    echo "  ${TF_ENV}: TensorFlow sees the GPU"
  fi
  if [[ ",${ARMS_ARG}," != ",original," ]]; then
    ( setup_pt; cd "${NEUTRAL}"; check_framework_gpu torch "${WORK}/gpu_check_pt.log" ) || exit 1
    echo "  ${PT_ENV}: PyTorch sees the GPU"
  fi
fi

ACTIVE=()
echo "### warm-up: one untimed run per arm on $(wc -l < "${WARM_REGIONS}") regions"
for arm in "${ARMS_LIST[@]}"; do
  read -r status seconds <<< "$(run_arm "${arm}" "${WARM_REGIONS}" "${WORK}/warmup_${arm}")"
  printf "  %-9s %s\n" "${arm}" "${status}"
  find "${WORK}/warmup_${arm}" -type f ! -name run.log ! -name time ! -name peak_rss_mb -delete 2>/dev/null || true
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
    find "${WORK}/r${round}_${arm}" -type f ! -name run.log ! -name time ! -name peak_rss_mb -delete 2>/dev/null || true
  done
done
rm -rf "${NEUTRAL}"
echo "timings: ${SUMMARY}"
finish_results
