#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --job-name=chrombpnet_headtohead
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --array=0-4
#
# The blog's comparison table, split so cherimoya is compared only on work it
# also does: the original package and chrombpnet-refactor each run their own
# CLI command with the writers cherimoya has no equivalent for switched off
# (headtohead.py), next to the full command. Same folds, peaks and method as
# run_pred_bw_kit_fused.sh (untimed warm-up of each arm, ROUNDS rounds in a
# rotating order, only the command timed); cherimoya's side is run_cherimoya.sh.
#
# From this folder, with site.env's cluster options and L40S constraint:
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_headtohead.sh slurm
#   python summarize_rerun.py <OPTIMIZATION_OUT>   # run tag <gpu>_<arch>_headtohead
#
# Arms:
#   orig_predict   original pred_bw, the fold's full model (-cm) only, the
#                  20,000 peaks scored against the observed bigWig, no bigWig
#                  or predictions HDF5 written               (= cherimoya evaluate)
#   ref_predict    the same with chrombpnet-refactor
#   ref_predict_fast  ref_predict with --predict-fused-kernel-fast (the optimization
#                  kit's Triton TF32 kernels; not by default)
#   orig_attr      original contribs_bw, the no-bias model, profile head, the
#                  first 2,000 peaks, no scores HDF5 or bigWig written
#                                                              (= cherimoya attribute)
#   ref_attr       the same with chrombpnet-refactor
#   orig_contribs  original contribs_bw as is (scores HDF5 and bigWig)
#   ref_contribs   chrombpnet-refactor's, as is
# Both packages' DeepSHAP use 20 dinucleotide-shuffled references per sequence,
# as cherimoya attribute does. pred_bw's full three-model times come from the
# kit-fused job.
#
# Usage: run_headtohead.sh [CLUSTER] [FOLDS]
#   env: ROUNDS (3), ARMS (all six), PT_ENV, TF_ENV, RUN_TAG, RUN_KIND (headtohead:
#        the run tag's suffix), OUT_ROOT (site.env's OPTIMIZATION_OUT),
#        DEVICE (cuda; cpu for a smoke test), N_REGIONS (20000), N_CONTRIB (2000),
#        TF_CUDA_DIRS, ALLOW_BUSY_GPU
#   site.env: ORIGINAL_REPO, GENOME, CHROM_SIZES, REFACTOR_MODELS, ORIGINAL_MODELS,
#        OPTIMIZATION_OUT (unless OUT_ROOT is set); optional PT_ENV_GLIBC217 and the hooks
#        pytorch_compiler, tf28_libraries, report_libraries (under Slurm)

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
ROUNDS="${ROUNDS:-3}"
ARMS_ARG="${ARMS:-orig_predict,ref_predict,orig_attr,ref_attr,orig_contribs,ref_contribs}"
DEVICE="${DEVICE:-cuda}"
N_REGIONS="${N_REGIONS:-20000}"
N_CONTRIB="${N_CONTRIB:-2000}"
for arm in ${ARMS_ARG//,/ }; do
  case "${arm}" in
    orig_predict|ref_predict|ref_predict_fast|orig_attr|ref_attr|orig_contribs|ref_contribs) ;;
    *) echo "unknown arm '${arm}'" >&2; exit 2 ;;
  esac
done

[[ "${FOLDS}" == all ]] && FOLDS=0,1,2,3,4
if [[ "${FOLDS}" == *,* ]]; then
  IFS=',' read -r -a FOLD_LIST <<< "${FOLDS}"
  failed=()
  for fold in "${FOLD_LIST[@]}"; do
    bash "${BASH_SOURCE[0]}" "${CLUSTER}" "${fold}" || failed+=("${fold}")
  done
  [[ ${#failed[@]} -eq 0 ]] || { echo "folds that failed: ${failed[*]}" >&2; exit 1; }
  exit 0
fi
FOLD_NUM="${FOLDS}"

SITE_NEEDS=(ORIGINAL_REPO GENOME CHROM_SIZES REFACTOR_MODELS ORIGINAL_MODELS)
[[ -n "${OUT_ROOT:-}" ]] || SITE_NEEDS+=(OPTIMIZATION_OUT)
require_site "${SITE_NEEDS[@]}"
CODE_MAIN="${CHROMBPNET_REPO}"
TF_REPO="${ORIGINAL_REPO}"
PT_MODEL="${REFACTOR_MODELS}/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
TF_MODEL="${ORIGINAL_MODELS}/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model"
OBSERVED_BW="${PT_MODEL}/auxiliary/data_unstranded.bw"
PEAKS="${PT_MODEL}/auxiliary/filtered.peaks.bed"
OUT_ROOT="${OUT_ROOT:-${OPTIMIZATION_OUT}}"
PT_ENV="$(default_pt_env)"
TF_ENV="${TF_ENV:-chrombpnet}"
for required in "${CODE_MAIN}/chrombpnet/CHROMBPNET.py" "${TF_REPO}/chrombpnet/CHROMBPNET.py" "${GENOME}" \
    "${OBSERVED_BW}" "${PEAKS}" "${PT_MODEL}/models/chrombpnet.pt" "${PT_MODEL}/models/chrombpnet_nobias.pt" \
    "${TF_MODEL}/models/chrombpnet.h5" "${TF_MODEL}/models/chrombpnet_nobias.h5"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 1; }
done
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

eval "$("$(conda_exe)" shell.bash hook)"
select_gpu
ARCH="cuda-${GPU_CC//./}"
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}_${RUN_KIND:-headtohead}$(host_suffix)}"
start_results "${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
NEUTRAL="${WORK}/cwd"
mkdir -p "${NEUTRAL}"
SUMMARY="${WORK}/timings.tsv"
printf "run\tfold\tgpu\tarch\tarm\tround\tposition\tseconds\tstatus\tdetail\n" > "${SUMMARY}"

# The 20,000 peaks the pred_bw jobs (and run_cherimoya.sh) draw, and their first 2,000.
REGIONS="${WORK}/regions.bed"
awk -v n="${N_REGIONS}" 'BEGIN{srand(1234)} {if (NR<=n) r[NR]=$0;
     else {i=int(rand()*NR)+1; if (i<=n) r[i]=$0}}
     END{for (j=1;j<=n && j<=NR;j++) print r[j]}' "${PEAKS}" \
  | sort -k1,1 -k2,2n > "${REGIONS}"
CONTRIB="${WORK}/contrib_regions.bed"
head -n "${N_CONTRIB}" "${REGIONS}" > "${CONTRIB}"
WARM="${WORK}/warmup_regions.bed"
head -n 200 "${REGIONS}" > "${WARM}"

echo "=================================================================="
echo " run tag   : ${RUN_TAG}    fold ${FOLD_NUM}    host $(hostname) (${CLUSTER})"
echo " GPU       : ${GPU_ID}: ${GPU_NAME} (${ARCH})    load average $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) CPUs"
echo " regions   : $(wc -l < "${REGIONS}") (attribution: $(wc -l < "${CONTRIB}"))    rounds: ${ROUNDS}"
echo " arms      : ${ARMS_ARG}"
echo " envs      : ${PT_ENV} (refactor), ${TF_ENV} (original)"
echo " output    : ${WORK}"
echo "=================================================================="

setup_pt () {
  export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
  if [[ "${CLUSTER}" == slurm ]] && is_glibc217_env "${PT_ENV}"; then
    site_hook pytorch_compiler
  fi
  conda activate "${PT_ENV}"
}
setup_tf () {
  if [[ "${CLUSTER}" == slurm ]]; then
    export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
    site_hook tf28_libraries
    site_hook report_libraries
  else
    tf_local_libraries
  fi
  conda activate "${TF_ENV}"
}

arm_command () {  # arm regions contrib outdir -> "PT|TF code slice args..."
  local arm="$1" regions="$2" contrib="$3" out="$4" dev=""
  [[ "${arm}" == ref_* ]] && dev="--device ${DEVICE}"
  local genome="-g ${GENOME} -c ${CHROM_SIZES}"
  case "${arm}" in
    orig_predict)  echo "TF ${TF_REPO} predict pred_bw -cm ${TF_MODEL}/models/chrombpnet.h5 -r ${regions} ${genome} -bw ${OBSERVED_BW} -op ${out}/pred -bs 256 -t 0" ;;
    ref_predict)   echo "PT ${CODE_MAIN} predict pred_bw -cm ${PT_MODEL}/models/chrombpnet.pt -r ${regions} ${genome} -bw ${OBSERVED_BW} -op ${out}/pred -bs 256 -t 0 ${dev}" ;;
    ref_predict_fast) echo "PT ${CODE_MAIN} predict pred_bw -cm ${PT_MODEL}/models/chrombpnet.pt -r ${regions} ${genome} -bw ${OBSERVED_BW} -op ${out}/pred -bs 256 -t 0 ${dev} --predict-fused-kernel-fast" ;;
    orig_attr)     echo "TF ${TF_REPO} attribute contribs_bw -m ${TF_MODEL}/models/chrombpnet_nobias.h5 -r ${contrib} ${genome} -op ${out}/contrib -pc profile -t 0" ;;
    ref_attr)      echo "PT ${CODE_MAIN} attribute contribs_bw -m ${PT_MODEL}/models/chrombpnet_nobias.pt -r ${contrib} ${genome} -op ${out}/contrib -pc profile -t 0 ${dev}" ;;
    orig_contribs) echo "TF ${TF_REPO} none contribs_bw -m ${TF_MODEL}/models/chrombpnet_nobias.h5 -r ${contrib} ${genome} -op ${out}/contrib -pc profile -t 0" ;;
    ref_contribs)  echo "PT ${CODE_MAIN} none contribs_bw -m ${PT_MODEL}/models/chrombpnet_nobias.pt -r ${contrib} ${genome} -op ${out}/contrib -pc profile -t 0 ${dev}" ;;
  esac
}

run_arm () {  # arm regions contrib outdir -> "status seconds"
  local arm="$1" regions="$2" contrib="$3" out="$4" rc kind code slice rest
  rm -rf "${out}" 2>/dev/null || mv "${out}" "${out}.stale.$(date +%s)"
  mkdir -p "${out}"
  read -r kind code slice rest <<< "$(arm_command "${arm}" "${regions}" "${contrib}" "${out}")"
  (
    set +e
    if [[ "${kind}" == "TF" ]]; then setup_tf; else setup_pt; fi
    export PYTHONPATH="${code}"
    cd "${NEUTRAL}"
    python -c "import chrombpnet, os, sys; p = os.path.realpath(chrombpnet.__file__);
sys.exit(0 if p.startswith(os.path.realpath('${code}')) else 'wrong chrombpnet: ' + p)" || exit 3
    t0=$(date +%s.%N)
    # shellcheck disable=SC2086
    python "${HERE}/headtohead.py" "${slice}" ${rest}
    rc=$?
    t1=$(date +%s.%N)
    echo "${rc} ${t0} ${t1}" > "${out}/time"
    exit ${rc}
  ) > "${out}/run.log" 2>&1 && rc=0 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    read -r _ t0 t1 < "${out}/time"
    echo "ok $(awk -v a="${t1}" -v b="${t0}" 'BEGIN{printf "%.1f", a-b}')"
  else
    echo "failed -"
  fi
}

record () {  # arm round position seconds status detail
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${RUN_TAG}" "${FOLD_NUM}" "${GPU_NAME}" \
    "${ARCH}" "$1" "$2" "$3" "$4" "$5" "$6" >> "${SUMMARY}"
}

if [[ "${DEVICE}" != "cpu" ]]; then
  echo "### GPU check"
  ( setup_tf; cd "${NEUTRAL}"; check_framework_gpu tensorflow "${WORK}/gpu_check_tf.log" ) || exit 1
  ( setup_pt; cd "${NEUTRAL}"; check_framework_gpu torch "${WORK}/gpu_check_pt.log" ) || exit 1
  echo "  ${TF_ENV} and ${PT_ENV} see the GPU"
fi

echo "### warm-up: file cache"
cat "${GENOME}" "${GENOME}.fai" "${OBSERVED_BW}" "${PT_MODEL}"/models/*.pt "${TF_MODEL}"/models/*.h5 > /dev/null

IFS=',' read -r -a ARMS_LIST <<< "${ARMS_ARG}"
ACTIVE=()
echo "### warm-up: one untimed run per arm on 200 regions"
for arm in "${ARMS_LIST[@]}"; do
  read -r status seconds <<< "$(run_arm "${arm}" "${WARM}" "${WARM}" "${WORK}/warmup_${arm}")"
  printf "  %-14s %s\n" "${arm}" "${status}"
  if [[ "${status}" == ok ]]; then
    ACTIVE+=("${arm}")
  else
    record "${arm}" "-" "-" "" failed "warm-up failed: ${WORK}/warmup_${arm}/run.log"
    tail -15 "${WORK}/warmup_${arm}/run.log" | sed 's/^/  | /'
  fi
  find "${WORK}/warmup_${arm}" -type f ! -name run.log ! -name time -delete 2>/dev/null || true
done
[[ ${#ACTIVE[@]} -gt 0 ]] || { echo "no arm survived the warm-up" >&2; exit 1; }

n=${#ACTIVE[@]}
shift_by=$(( (n + ROUNDS - 1) / ROUNDS ))
for (( round = 0; round < ROUNDS; round++ )); do
  echo "### round $((round + 1)) of ${ROUNDS}"
  for (( i = 0; i < n; i++ )); do
    arm="${ACTIVE[$(( (i + round * shift_by) % n ))]}"
    read -r status seconds <<< "$(run_arm "${arm}" "${REGIONS}" "${CONTRIB}" "${WORK}/r${round}_${arm}")"
    printf "  %-14s %-8s %s\n" "${arm}" "${status}" "${seconds}"
    if [[ "${status}" == ok ]]; then
      record "${arm}" "${round}" "${i}" "${seconds}" ok ""
    else
      record "${arm}" "${round}" "${i}" "" "${status}" "${WORK}/r${round}_${arm}/run.log"
      tail -10 "${WORK}/r${round}_${arm}/run.log" | sed 's/^/  | /'
    fi
    find "${WORK}/r${round}_${arm}" -type f ! -name run.log ! -name time -delete 2>/dev/null || true
  done
done
rm -rf "${NEUTRAL}"
echo "timings: ${SUMMARY}"
finish_results
