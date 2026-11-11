#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --job-name=chrombpnet_sections34
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#
# Blog 2026-015, Sections 3 and 4 head to head: chrombpnet 1.x, 1.x with the
# optimization kit, and chrombpnet-refactor, on one dataset and fold
# (ENCSR868FGK fold 0, K562 ATAC-seq), the full model only (-cm), the same
# 20,000 peaks, observed bigWig and batch size, in one job: each arm gets an
# untimed warm-up run, then ROUNDS rounds in a different order each, only the
# pred_bw command timed (and its peak memory recorded, peak_rss.py).
#
# From this folder, with site.env's cluster options and GPU constraints:
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_sections34.sh slurm
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_sections34.sh slurm
#   bash run_sections34.sh local                       # on a GPU host
#   python summarize_sections34.py <OPTIMIZATION_OUT>  # the blog cells, per GPU
#
# The default arms are the blog tables' (Section 3 against the kit's `fast`
# mode; Section 4 at the original's compression, gzip 4, with the inference
# step on each forward route), under run tag *_sections34-fast. On an L40S the
# kit's `fast` keeps TensorFlow's forward (its rule: files bitwise equal to
# stock's default run), so only the H200 column has its Triton forward; the
# forward_fast / refactor4_fast arms run this package's TF32 kernels on both.
# ARMS=... RUN_KIND=... picks others, e.g. the earlier L40S job's:
# ARMS=stock,stages,exact,refactor,refactor4 RUN_KIND=sections34.
#
# Arms:
#   stock      chrombpnet 1.x as the kit pins it (1.0.1, TensorFlow 2.8), the kit's
#              --mode off: the baseline of both sections
#   stages     stock under stock_stage_timer.py: the stage each kit lever replaces,
#              stock's side (Section 3's per-row cells)
#   fast       1.x with the kit, --mode fast, its default (on an H200: every lever,
#              its TF32 Triton forward): Section 3
#   stockmalloc  stock with the kit's malloc_env lever (MALLOC_MMAP_MAX_=0,
#              MALLOC_TRIM_THRESHOLD_=-1, MALLOC_TOP_PAD_=536870912, as
#              run_kit_modes.sh): that lever alone in the original package
#   refactor4  chrombpnet-refactor's default pred_bw (PyTorch, TF32 convolutions) at
#              the original's HDF5 compression, gzip 4 without shuffle
#              (--h5-gzip-level 4 --no-h5-shuffle): Section 4
#   refactor4_fast   refactor4 with --predict-fused-kernel-fast (the kit's TF32 kernels)
#   refactor4_nomalloc  refactor4 with --no-malloc-env: glibc's default allocator
#   refactor4_prev  refactor4 at fa45de5 (code-fa45de5/, an export of that commit,
#              which predates --h5-gzip-level: the same filters via refactor_gzip4.py):
#              before within-model streaming, the incremental bigWig writer and the
#              faster observed read (dfaa33b..8b07542), so prev -> refactor4 is their
#              effect in this job
#   forward_eager / forward_fast / forward_exact  chrombpnet-refactor's forward alone
#              over the regions on each route (forward_timer.py, 3 repeats a run):
#              Section 4's inference row
#   (not by default)
#   fast_k1    the kit's `fast` with its Triton TF32 forward on an L40S, where the kit's
#              own table keeps TensorFlow's: kit_hook_l40s_k1fast/sitecustomize.py
#              (KIT_L40S_K1_FAST=1) overrides that one table answer; nothing else changes
#   exact      1.x with the kit, --mode exact (its fp32 Triton forward)
#   refactor   chrombpnet-refactor at its own default compression (gzip 9 + shuffle)
#   refactor4_exact  refactor4 with --predict-fused-kernel-exact (the kit's fp32 kernels)
#   refactor_nomalloc  refactor with --no-malloc-env
#   original   chrombpnet at 09938fd in the `chrombpnet` env, as the earlier
#              Section 4 job ran 1.x; a check that `stock` is the same 1.x
# The 1.x arms run the fold's 1.x .h5 model. The kit's Triton route reads that
# model's components beside -cm, as bias_scaled.h5 and nobias.h5, so -cm points
# into a directory of links with those names. chrombpnet-refactor's PyTorch
# engine reads .pt files: it runs its own model of the same dataset and fold,
# with the same architecture (512 filters, 8 dilated layers, 2,114 bp in).
#
# Usage: run_sections34.sh [CLUSTER] [FOLD] [N_REGIONS]
#   CLUSTER    slurm | local; inferred from the Slurm job if omitted
#   FOLD       0 (default)
#   env: ROUNDS (5), ARMS (the default arms above), BATCH_SIZE (256),
#        RUN_KIND (sections34-fast: the run tag's suffix), OUT_ROOT (site.env's
#        OPTIMIZATION_OUT), PT_ENV, TF_ENV, KIT_ENV, KIT_HOME, CHROMBPNET_PREV_REPO,
#        DEVICE (cuda; cpu for a smoke test), ALLOW_BUSY_GPU
#   site.env: ORIGINAL_REPO, KIT_ROOT, KIT_HOME, GENOME, CHROM_SIZES, REFACTOR_MODELS,
#        ORIGINAL_MODELS, OPTIMIZATION_OUT (unless OUT_ROOT is set); optional PT_ENV_GLIBC217
#        and the hooks kit_compiler, tf28_libraries, pytorch_compiler (under Slurm)

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
FOLD_NUM="${2:-0}"
N_REGIONS="${3:-20000}"
ROUNDS="${ROUNDS:-5}"
ARMS_ARG="${ARMS:-stock,stages,fast,stockmalloc,refactor4,refactor4_fast,refactor4_nomalloc,refactor4_prev,forward_eager,forward_fast,forward_exact}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"
for arm in ${ARMS_ARG//,/ }; do
  case "${arm}" in
    stock|stages|exact|fast|fast_k1|stockmalloc|refactor|refactor4|refactor4_fast|refactor4_exact|refactor_nomalloc|refactor4_nomalloc|refactor4_prev|forward_eager|forward_fast|forward_exact|original) ;;
    *) echo "unknown arm '${arm}' (expected stock, stages, exact, fast, fast_k1, stockmalloc, refactor, refactor4, refactor4_fast, refactor4_exact, refactor_nomalloc, refactor4_nomalloc, refactor4_prev, forward_eager, forward_fast, forward_exact, original)" >&2; exit 2 ;;
  esac
done

SITE_NEEDS=(ORIGINAL_REPO KIT_ROOT KIT_HOME GENOME CHROM_SIZES REFACTOR_MODELS ORIGINAL_MODELS)
[[ -n "${OUT_ROOT:-}" ]] || SITE_NEEDS+=(OPTIMIZATION_OUT)
require_site "${SITE_NEEDS[@]}"
CODE_MAIN="${CHROMBPNET_REPO}"
CODE_PREV="${CHROMBPNET_PREV_REPO:-${HERE}/code-fa45de5}"
TF_REPO="${ORIGINAL_REPO}"
KIT="${KIT_ROOT}/chrombpnet"
KIT_ENV="${KIT_ENV:-chrombpnet-kit}"
PT_MODEL="${REFACTOR_MODELS}/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
TF_MODEL="${ORIGINAL_MODELS}/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
OBSERVED_BW="${PT_MODEL}/auxiliary/data_unstranded.bw"
PEAKS="${PT_MODEL}/auxiliary/filtered.peaks.bed"
OUT_ROOT="${OUT_ROOT:-${OPTIMIZATION_OUT}}"
PT_ENV="$(default_pt_env)"   # under Slurm, PT_ENV_GLIBC217 (built for glibc 2.17 nodes) if site.env names one
TF_ENV="${TF_ENV:-chrombpnet}"

for required in "${KIT}/run.sh" "${CODE_MAIN}/chrombpnet/CHROMBPNET.py" "${TF_REPO}/chrombpnet/CHROMBPNET.py" \
                "${GENOME}" "${GENOME}.fai" "${CHROM_SIZES}" "${OBSERVED_BW}" "${PEAKS}" \
                "${PT_MODEL}/models/chrombpnet.pt" "${TF_MODEL}/models/chrombpnet.h5" \
                "${TF_MODEL}/models/chrombpnet_nobias.h5" "${TF_MODEL}/models/bias_model_scaled.h5" \
                "${HERE}/stock_stage_timer.py" "${HERE}/peak_rss.py" "${HERE}/refactor_gzip4.py" \
                "${HERE}/forward_timer.py" "${CODE_PREV}/chrombpnet/CHROMBPNET.py" \
                "${HERE}/kit_hook_l40s_k1fast/sitecustomize.py"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 1; }
done

eval "$("$(conda_exe)" shell.bash hook)"

# The kit's compiled Triton helpers are built against the host glibc, and a
# cluster's GPU nodes may run different ones: one cache root per glibc (as
# run_kit_modes.sh); on a GPU host, one per host.
GLIBC_VERSION="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
CACHE_KEY="glibc-${GLIBC_VERSION:-unknown}"
[[ "${CLUSTER}" == local ]] && CACHE_KEY="$(hostname -s)-${CACHE_KEY}"
KIT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-${KIT_HOME}/jit/${CACHE_KEY}}"
mkdir -p "${KIT_JIT_ROOT}"

select_gpu
ARCH="cuda-${GPU_CC//./}"
case "${GPU_NAME}" in
  *H200*) CONFIG=h200 ;;
  *A100*) CONFIG=a100 ;;
  *)      CONFIG=h100 ;;   # H100, and cards the kit has no config for (its route follows the card found)
esac
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}_${RUN_KIND:-sections34-fast}$(host_suffix)}"
start_results "${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
NEUTRAL="${WORK}/cwd"     # every command runs from here: never from a checkout
mkdir -p "${NEUTRAL}"
SUMMARY="${WORK}/timings.tsv"
printf "run\tfold\tgpu\tarch\tarm\tround\tposition\tseconds\tstatus\tdetail\n" > "${SUMMARY}"

# The 1.x model with its components beside it, under the names the kit's
# Triton route looks for.
MODEL_1X="${WORK}/model_1x"
mkdir -p "${MODEL_1X}"
ln -sf "${TF_MODEL}/models/chrombpnet.h5" "${MODEL_1X}/chrombpnet.h5"
ln -sf "${TF_MODEL}/models/bias_model_scaled.h5" "${MODEL_1X}/bias_scaled.h5"
ln -sf "${TF_MODEL}/models/chrombpnet_nobias.h5" "${MODEL_1X}/nobias.h5"

# The same 20,000 peaks as the earlier pred_bw jobs (run_pred_bw_kit_fused.sh).
REGIONS="${WORK}/regions.bed"
awk -v n="${N_REGIONS}" 'BEGIN{srand(1234)} {if (NR<=n) r[NR]=$0;
     else {i=int(rand()*NR)+1; if (i<=n) r[i]=$0}}
     END{for (j=1;j<=n && j<=NR;j++) print r[j]}' "${PEAKS}" \
  | sort -k1,1 -k2,2n > "${REGIONS}"
WARM_REGIONS="${WORK}/warmup_regions.bed"
head -n 2000 "${REGIONS}" > "${WARM_REGIONS}"

echo "=================================================================="
echo " run tag : ${RUN_TAG}    fold ${FOLD_NUM}    host $(hostname) (${CLUSTER}, glibc ${GLIBC_VERSION})"
echo " GPU     : ${GPU_ID}: ${GPU_NAME} (${ARCH}), kit config ${CONFIG}    $(nproc) CPUs"
echo " regions : $(wc -l < "${REGIONS}") (warm-up $(wc -l < "${WARM_REGIONS}"))    -bs ${BATCH_SIZE}"
echo " arms    : ${ARMS_ARG}    rounds: ${ROUNDS}"
echo " envs    : ${KIT_ENV} (stock, stages, stockmalloc, exact, fast), ${PT_ENV} (refactor*, forward_*), ${TF_ENV} (original)"
echo " code    : ${CODE_MAIN} at $(git -C "${CODE_MAIN}" rev-parse --short HEAD 2>/dev/null)$(git -C "${CODE_MAIN}" diff --quiet HEAD 2>/dev/null || echo ' + uncommitted changes')"
echo " output  : ${WORK}"
echo "=================================================================="

# --- environments, each entered in its arm's subshell --------------------------
setup_kit () {
  # TensorFlow 2.8 loads CUDA 11.2 and cuDNN 8.1 from the system; the Triton
  # forward's torch brings its own CUDA 12.4 wheels. The site hooks' copies only,
  # not the submitting shell's; a C11 compiler for the launchers Triton builds.
  if [[ "${CLUSTER}" == slurm ]]; then
    export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
    site_hook kit_compiler
    site_hook tf28_libraries
  else
    tf_local_libraries
  fi
  conda activate "${KIT_ENV}"
  export CHROMBPNET_OPT_WEIGHTS="${KIT_HOME}/weights" CHROMBPNET_OPT_DATA="${KIT_HOME}/data"
  export MODEL_OPT_JIT_ROOT="${KIT_JIT_ROOT}"
  export PYTHONHASHSEED=0 CFLAGS=-g0 PYTHONNOUSERSITE=1
  unset PYTHONPATH
}
setup_pt () {
  # PyTorch's own CUDA and cuDNN wheels only.
  export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
  if [[ "${CLUSTER}" == slurm ]] && is_glibc217_env "${PT_ENV}"; then
    site_hook pytorch_compiler
  fi
  conda activate "${PT_ENV}"
  export PYTHONPATH="${CODE_MAIN}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}
setup_tf () {
  if [[ "${CLUSTER}" == slurm ]]; then
    export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
    site_hook tf28_libraries
  else
    tf_local_libraries
  fi
  conda activate "${TF_ENV}"
  export PYTHONPATH="${TF_REPO}"
}

# Runs one arm in a fresh process; prints "status seconds". Only the pred_bw
# command is timed, not the site hooks or the conda activation. `time` holds
# "t0 t1" (stock_stage_timer's teardown is t1 minus when main() returned).
run_arm () {  # arm regions outdir
  local arm="$1" regions="$2" out="$3" rc
  rm -rf "${out}" 2>/dev/null || mv "${out}" "${out}.stale.$(date +%s)"
  mkdir -p "${out}"
  local io="-g ${GENOME} -c ${CHROM_SIZES} -r ${regions} -bw ${OBSERVED_BW} -bs ${BATCH_SIZE} -op ${out}/pred -os ${out}/stats -t 0"
  local peak="python -I ${HERE}/peak_rss.py ${out}/peak_rss_mb"
  (
    set +e
    case "${arm}" in
      stock|stockmalloc|exact|fast|fast_k1)
        setup_kit
        mode="${arm}"; [[ "${arm}" == stock || "${arm}" == stockmalloc ]] && mode=off
        if [[ "${arm}" == fast_k1 ]]; then
          mode=fast
          export PYTHONPATH="${HERE}/kit_hook_l40s_k1fast" KIT_L40S_K1_FAST=1
        fi
        malloc=(); [[ "${arm}" == stockmalloc ]] && malloc=(env MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=-1 MALLOC_TOP_PAD_=536870912)
        cd "${KIT}"
        t0=$(date +%s.%N)
        # shellcheck disable=SC2086
        ${peak} "${malloc[@]}" bash run.sh pred_bw --config "${CONFIG}" --mode "${mode}" -cm "${MODEL_1X}/chrombpnet.h5" ${io}
        rc=$? ;;
      stages)
        setup_kit
        # Stock's own pred_bw, as --mode off runs it (no kit variable set),
        # with stock_stage_timer.py's timers around its stages.
        unset $(compgen -e | grep '^CHROMBPNET_OPT')
        cd "${NEUTRAL}"
        t0=$(date +%s.%N)
        # shellcheck disable=SC2086
        ${peak} python -I "${HERE}/stock_stage_timer.py" "${out}/stages.json" pred_bw -cm "${MODEL_1X}/chrombpnet.h5" ${io}
        rc=$? ;;
      refactor|refactor4|refactor4_fast|refactor4_exact|refactor_nomalloc|refactor4_nomalloc|refactor4_prev)
        setup_pt
        code="${CODE_MAIN}"; [[ "${arm}" == *_prev ]] && code="${CODE_PREV}"
        export PYTHONPATH="${code}"
        cd "${NEUTRAL}"
        python -c "import chrombpnet, os, sys; p = os.path.realpath(chrombpnet.__file__);
sys.exit(0 if p.startswith(os.path.realpath('${code}')) else 'wrong chrombpnet: ' + p)" || exit 3
        cli="python -m chrombpnet.CHROMBPNET"; extra=""
        if [[ "${arm}" == refactor4_prev ]]; then
          # fa45de5 predates --h5-gzip-level; refactor_gzip4.py sets the same filters.
          cli="python ${HERE}/refactor_gzip4.py"
        elif [[ "${arm}" == refactor4* ]]; then
          # The original's HDF5 compression: gzip 4, no byte-shuffle.
          extra="--h5-gzip-level 4 --no-h5-shuffle"
        fi
        case "${arm}" in
          *_nomalloc) extra="${extra} --no-malloc-env" ;;
          *_fast)     extra="${extra} --predict-fused-kernel-fast" ;;
          *_exact)    extra="${extra} --predict-fused-kernel-exact" ;;
        esac
        t0=$(date +%s.%N)
        # shellcheck disable=SC2086
        ${peak} ${cli} pred_bw -cm "${PT_MODEL}/models/chrombpnet.pt" ${io} --device "${DEVICE}" ${extra}
        rc=$?
        # A fused-kernel arm whose kernels could not run fell back to the eager
        # forward (chrombpnet.fused_kernels.FALLBACK_MARKER): it timed the wrong route.
        if [[ ${rc} -eq 0 && ( "${arm}" == *_fast || "${arm}" == *_exact ) ]] &&
           grep -q "fused Triton kernels did not engage" "${out}/run.log"; then
          echo "REFUSED: the fused kernels fell back to the eager forward (warning above)"; rc=3
        fi ;;
      forward_eager|forward_fast|forward_exact)
        setup_pt
        cd "${NEUTRAL}"
        t0=$(date +%s.%N)
        python "${HERE}/forward_timer.py" "${out}/forward.json" "${PT_MODEL}/models/chrombpnet.pt" \
          "${regions}" "${GENOME}" "${CHROM_SIZES}" "${arm#forward_}" "${BATCH_SIZE}" 3
        rc=$? ;;
      original)
        setup_tf
        cd "${NEUTRAL}"
        t0=$(date +%s.%N)
        # shellcheck disable=SC2086
        ${peak} python -m chrombpnet.CHROMBPNET pred_bw -cm "${TF_MODEL}/models/chrombpnet.h5" ${io}
        rc=$? ;;
    esac
    t1=$(date +%s.%N)
    echo "${t0} ${t1}" > "${out}/time"
    exit ${rc}
  ) > "${out}/run.log" 2>&1 && rc=0 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    read -r t0 t1 < "${out}/time"
    echo "ok $(awk -v a="${t1}" -v b="${t0}" 'BEGIN{printf "%.1f", a-b}')"
  elif [[ ${rc} -eq 3 ]] && grep -q "NOT ACTIVE\|REFUSED" "${out}/run.log"; then
    echo "refused -"
  else
    echo "failed -"
  fi
}

record () {  # arm round position seconds status detail
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${RUN_TAG}" "${FOLD_NUM}" "${GPU_NAME}" \
    "${ARCH}" "$1" "$2" "$3" "$4" "$5" "$6" >> "${SUMMARY}"
}
keep_logs () {  # outdir: the logs and run records, not the bigWigs and HDF5 files
  find "$1" -type f ! -name run.log ! -name time ! -name peak_rss_mb ! -name opt_manifest.json \
       ! -name stages.json ! -name forward.json -delete 2>/dev/null || true
}

IFS=',' read -r -a ARMS_LIST <<< "${ARMS_ARG}"

if [[ "${DEVICE}" != "cpu" ]]; then
  # Each environment reaches the GPU (TensorFlow 2.8 would otherwise run on the CPU, silently).
  echo "### GPU check"
  if [[ ",${ARMS_ARG}," =~ ,(stock|stages|stockmalloc|exact|fast|fast_k1), ]]; then
    ( setup_kit; cd "${NEUTRAL}"; check_framework_gpu tensorflow "${WORK}/gpu_check_kit_tf.log" ) || exit 1
    ( setup_kit; cd "${NEUTRAL}"; check_framework_gpu torch "${WORK}/gpu_check_kit_pt.log" ) || exit 1
    echo "  ${KIT_ENV}: TensorFlow and PyTorch see the GPU"
  fi
  if [[ ",${ARMS_ARG}," =~ ,(refactor[a-z0-9_]*|forward_[a-z]+), ]]; then
    ( setup_pt; cd "${NEUTRAL}"; check_framework_gpu torch "${WORK}/gpu_check_pt.log" ) || exit 1
    echo "  ${PT_ENV}: PyTorch sees the GPU"
  fi
  if [[ ",${ARMS_ARG}," == *,original,* ]]; then
    ( setup_tf; cd "${NEUTRAL}"; check_framework_gpu tensorflow "${WORK}/gpu_check_tf.log" ) || exit 1
    echo "  ${TF_ENV}: TensorFlow sees the GPU"
  fi
fi

echo "### warm-up: file cache"
cat "${GENOME}" "${GENOME}.fai" "${OBSERVED_BW}" "${PT_MODEL}/models/chrombpnet.pt" "${TF_MODEL}"/models/*.h5 > /dev/null

ACTIVE=()
echo "### warm-up: one untimed run per arm on $(wc -l < "${WARM_REGIONS}") regions"
for arm in "${ARMS_LIST[@]}"; do
  read -r status seconds <<< "$(run_arm "${arm}" "${WARM_REGIONS}" "${WORK}/warmup_${arm}")"
  printf "  %-9s %s\n" "${arm}" "${status}"
  keep_logs "${WORK}/warmup_${arm}"
  case "${status}" in
    ok)      ACTIVE+=("${arm}") ;;
    refused) record "${arm}" "-" "-" "" refused "$(grep -m1 -o 'NOT ACTIVE.*\|REFUSED.*' "${WORK}/warmup_${arm}/run.log" | cut -c1-160)" ;;
    *)       record "${arm}" "-" "-" "" failed "warm-up failed: ${WORK}/warmup_${arm}/run.log"
             tail -15 "${WORK}/warmup_${arm}/run.log" | sed 's/^/  | /' ;;
  esac
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
    keep_logs "${WORK}/r${round}_${arm}"
  done
done
rm -rf "${NEUTRAL}"
echo "timings: ${SUMMARY}"
finish_results
