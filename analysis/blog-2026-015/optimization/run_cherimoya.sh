#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=chrombpnet_cherimoya
#SBATCH --output=/path/to/workdir/sherlock/logs/chrombpnet_cherimoya.%x.%A_%a.out
#SBATCH --error=/path/to/workdir/sherlock/logs/chrombpnet_cherimoya.%x.%A_%a.err
#SBATCH --array=0-4
#
# Cherimoya (software/cherimoya: Jacob Schreiber's PyTorch successor to
# ChromBPNet) on the blog's "Against the original ChromBPNet" rows, for its
# cherimoya column: the same folds, peaks, non-peaks and motifs as those rows,
# timed with the method of run_pred_bw_kit_fused.sh (an untimed warm-up of each
# command, ROUNDS rounds in a rotating order, only the command timed).
#
#   sbatch --constraint="GPU_SKU:L40S" run_cherimoya.sh sherlock
#   sbatch -p owners --constraint="GPU_SKU:H200_SXM5" run_cherimoya.sh sherlock
#   python summarize_rerun.py      # run tag <gpu>_<arch>_cherimoya
#
# Arms (cherimoya's own commands, `cherimoya <command> -p <json>`):
#   evaluate     the 20,000 peaks of pred_bw's row, predicted and scored
#                against the fold's observed bigWig                  (pred_bw)
#   attribute    DeepLIFT/SHAP on the profile head, the first 2,000 of them,
#                the full 2,114 bp window, 20 shuffled references (contribs_bw)
#   marginalize  the fold's 5 motifs inserted into each test-chromosome
#                non-peak                                         (footprints)
# The work is not identical, and the blog's footnote says so: cherimoya has one
# model per fold (no bias factorization: ATAC models are trained on the
# observed signal), evaluate writes metrics but no predicted bigWig, attribute
# writes NPZ, and marginalize inserts each motif once where footprints also
# inserts its reverse complement.
#
# Models: CATv1 (huggingface.co/programmable-genomics/CATv1), ENCSR868FGK
# fold_k, the same chromosome folds as ChromBPNet's ENCODE models, under
# chromatlas/test/cherimoya/ATAC/ENCSR868FGK/. Inputs come from the fold's
# chrombpnet-refactor files (cherimoya_inputs.py, run on the host).
#
# Environment: conda env `cherimoya` (cherimoya 0.3.1's uv.lock: torch 2.13.0,
# installed from the CUDA 12.6 index, triton 3.7.1, tangermeme 1.6.0, plus
# conda-forge gcc for Triton's launchers). torch >= 2.9 and triton >= 3.5 are
# built for glibc >= 2.28 only (PyPI and conda-forge alike), so on Sherlock
# (glibc 2.17) each command runs in the Ubuntu 22.04 container the expand and
# engines jobs use; on a lab host, natively (CONTAINER=1 forces the container).
#
# Usage: run_cherimoya.sh [CLUSTER] [FOLDS]
#   env: ROUNDS (3), ARMS (evaluate,attribute,marginalize), BATCH_SIZE (256, as
#        pred_bw and footprints ran; attribute keeps cherimoya's 64), COMPILE
#        (unset = cherimoya's defaults: evaluate and marginalize compiled,
#        attribute not; 0 or 1 for all), N_REGIONS (20000), N_CONTRIB (2000),
#        N_NONPEAKS (all; a cap for smoke tests),
#        RUN_TAG, OUT_ROOT, DEVICE (cuda; cpu for a smoke test), CONTAINER,
#        ALLOW_BUSY_GPU

set -eo pipefail

OAK_DIR="/path/to/workdir"
HERE="${OAK_DIR}/software/chrombpnet-refactor/analysis/blog-2026-015/optimization"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

resolve_cluster "${1:-}"
FOLDS="${2:-${SLURM_ARRAY_TASK_ID:-all}}"
ROUNDS="${ROUNDS:-3}"
ARMS_ARG="${ARMS:-evaluate,attribute,marginalize}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_REGIONS="${N_REGIONS:-20000}"
N_CONTRIB="${N_CONTRIB:-2000}"
for arm in ${ARMS_ARG//,/ }; do
  case "${arm}" in
    evaluate|attribute|marginalize) ;;
    *) echo "unknown arm '${arm}' (expected evaluate, attribute, marginalize)" >&2; exit 2 ;;
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

CHERIMOYA_ENV="${CHERIMOYA_ENV:-${OAK_DIR}/software/miniconda3/envs/cherimoya}"
INPUT_PY="${INPUT_PY:-${OAK_DIR}/software/miniconda3/envs/chrombpnet-pytorch-sherlock/bin/python}"
IMAGE="${CONTAINER_IMAGE:-${OAK_DIR}/software/containers/ubuntu-22.04.sif}"
GENOME="${OAK_DIR}/chromatlas/reference/hg38/hg38.fa"
FOLD_JSON="${OAK_DIR}/chromatlas/reference/folds/fold_${FOLD_NUM}.json"
AUX="${OAK_DIR}/chromatlas/test/chrombpnet-refactor/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/model/auxiliary"
MODEL="${OAK_DIR}/chromatlas/test/cherimoya/ATAC/ENCSR868FGK/fold_${FOLD_NUM}/cherimoya.torch"
OUT_ROOT="${OUT_ROOT:-${OAK_DIR}/chromatlas/test/chrombpnet-refactor-optimization/analysis-rerun}"
for required in "${CHERIMOYA_ENV}/bin/cherimoya" "${INPUT_PY}" "${GENOME}" "${FOLD_JSON}" "${MODEL}" \
                "${AUX}/filtered.peaks.bed" "${AUX}/filtered.nonpeaks.bed" "${AUX}/data_unstranded.bw" \
                "${AUX}/motif_to_pwm.tsv"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 1; }
done

# The container on Sherlock (glibc 2.17: the env's torch needs 2.28), natively elsewhere.
if [[ -z "${CONTAINER:-}" ]]; then
  [[ "${CLUSTER}" == sherlock ]] && CONTAINER=1 || CONTAINER=0
fi
if [[ "${CONTAINER}" == 1 ]]; then
  RUNTIME="$(command -v apptainer || command -v singularity || true)"
  [[ -n "${RUNTIME}" ]] || { echo "Error: neither apptainer nor singularity is on PATH" >&2; exit 1; }
  [[ -f "${IMAGE}" ]] || { echo "Error: no container image at ${IMAGE}" >&2; exit 1; }
  BINDS=(--bind /oak)
  [[ -d /scratch ]] && BINDS+=(--bind /scratch)
  if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" && "${TMPDIR}" != /tmp* ]]; then BINDS+=(--bind "${TMPDIR}"); fi
  CACHE_KEY="container-ubuntu-22.04"
else
  CACHE_KEY="$(hostname -s)"
fi
# PyTorch's own CUDA and cuDNN wheels only, not the submitting shell's.
export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

select_gpu
ARCH="cuda-${GPU_CC//./}"
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}_cherimoya$(host_suffix)}"
start_results "${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
# Compiled kernels (Triton's, and torch.compile's: cherimoya compiles evaluate
# and marginalize by default) persist across this task's runs, in a cache of
# its own. With one cache shared by the array's tasks (2026-10-27, L40S), 5 of
# 45 timed runs took 10-15x longer while other tasks were compiling into it.
CACHE_ROOT="${OAK_DIR}/chromatlas/test/cherimoya/.cache/${CACHE_KEY}/${RUN_TAG}/fold_${FOLD_NUM}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/inductor}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
SUMMARY="${WORK}/timings.tsv"
printf "run\tfold\tgpu\tarch\tarm\tround\tposition\tseconds\tstatus\tdetail\n" > "${SUMMARY}"

read -r n_regions n_contrib n_nonpeaks <<< "$("${INPUT_PY}" -I "${HERE}/cherimoya_inputs.py" "${WORK}/inputs" \
    "${AUX}/filtered.peaks.bed" "${AUX}/filtered.nonpeaks.bed" "${FOLD_JSON}" "${AUX}/motif_to_pwm.tsv" \
    "${N_REGIONS}" "${N_CONTRIB}")"
# The refactor and original arms' own sample, where that job ran on this GPU type: the same peaks, or say so.
for other in "${OUT_ROOT}/${GPU_SLUG}_${ARCH}_kit-fused/fold_${FOLD_NUM}/regions.bed"; do
  if [[ -f "${other}" ]]; then
    cmp -s "${other}" "${WORK}/inputs/regions.bed" && echo " regions: identical to ${other}" \
      || echo " WARNING: regions differ from ${other} (another awk?)"
  fi
done

echo "=================================================================="
echo " run tag   : ${RUN_TAG}    fold ${FOLD_NUM}    host $(hostname) (${CLUSTER}; container ${CONTAINER})"
echo " GPU       : ${GPU_ID}: ${GPU_NAME} (${ARCH})    load average $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) CPUs"
echo " inputs    : ${n_regions} peaks, ${n_contrib} for attribution, ${n_nonpeaks} test non-peaks, 5 motifs"
echo " rounds    : ${ROUNDS}    arms: ${ARMS_ARG}    batch ${BATCH_SIZE}    compile ${COMPILE:-cherimoya default}"
echo " model     : ${MODEL}"
echo " output    : ${WORK}"
echo "=================================================================="

# Runs "$@" with the cherimoya env (in the container on Sherlock), timing the
# command alone; writes "rc t0 t1" to $1.
in_env () {  # timefile command...
  local timefile="$1"; shift
  local inner='export PATH="${CHERIMOYA_ENV}/bin:${PATH}"
for cc in "${CHERIMOYA_ENV}"/bin/x86_64-conda-linux-gnu-gcc; do [ -x "${cc}" ] && export CC="${cc}"; done
for cxx in "${CHERIMOYA_ENV}"/bin/x86_64-conda-linux-gnu-g++; do [ -x "${cxx}" ] && export CXX="${cxx}"; done
[ -d /.singularity.d/libs ] && export TRITON_LIBCUDA_PATH=/.singularity.d/libs
t0=$(date +%s.%N); "$@"; rc=$?; t1=$(date +%s.%N)
echo "${rc} ${t0} ${t1}" > "${TIMEFILE}"; exit ${rc}'
  if [[ "${CONTAINER}" == 1 ]]; then
    CHERIMOYA_ENV="${CHERIMOYA_ENV}" TIMEFILE="${timefile}" LANG=C.UTF-8 LC_ALL= \
      "${RUNTIME}" exec --nv "${BINDS[@]}" "${IMAGE}" bash -c "${inner}" bash "$@"
  else
    CHERIMOYA_ENV="${CHERIMOYA_ENV}" TIMEFILE="${timefile}" bash -c "${inner}" bash "$@"
  fi
}

# The parameters JSON of one arm on one input size (full or warm-up).
write_params () {  # arm outdir warm
  local arm="$1" out="$2" warm="$3" loci n compile=""
  [[ -n "${COMPILE:-}" ]] && compile="$([[ "${COMPILE}" == 1 ]] && echo true || echo false)"
  "${INPUT_PY}" -I - "$arm" "$out" "$warm" "${WORK}/inputs" "${MODEL}" "${GENOME}" \
      "${AUX}/data_unstranded.bw" "${BATCH_SIZE}" "${DEVICE}" "${compile}" "${N_NONPEAKS:-0}" <<'EOF'
import json, sys
arm, out, warm, inputs, model, genome, signal, batch, device, compile_, cap = sys.argv[1:12]
warm = warm == "1"
def loci(name, n):
    # The compiled commands warm up on the full input, so every kernel and
    # batch shape the timed runs use is compiled (and cached) beforehand;
    # attribute is not compiled and warms up on a few regions.
    path = "%s/%s.centered.bed" % (inputs, name)
    if not warm or arm != "attribute":
        return path
    small = "%s/warmup.%s.bed" % (out, name)
    with open(path) as source, open(small, "w") as target:
        target.writelines(line for i, line in zip(range(n), source))
    return small
common = {"model": model, "sequences": genome, "chroms": None, "device": device}
if arm == "evaluate":
    p = dict(common, loci=loci("regions", 2000), signals=[signal], batch_size=int(batch),
             performance_filename=out + "/performance.tsv")
elif arm == "attribute":
    p = dict(common, loci=loci("contrib_regions", 200), output="profile", attr_window=2114, n_shuffles=20,
             ohe_filename=out + "/attr.ohe.npz", attr_filename=out + "/attr.attr.npz", idx_filename=out + "/attr.idx.npy")
else:
    path = loci("test_nonpeaks", 2000)
    n = sum(1 for _ in open(path))
    if int(cap):
        n = min(n, int(cap))
    p = dict(common, loci=path, motifs=inputs + "/motifs.meme", n_loci=n, batch_size=int(batch),
             output_filename=out + "/marginalize/")
if compile_:
    p["compile"] = compile_ == "true"
json.dump(p, open(out + "/params.json", "w"), indent=1)
EOF
}

run_arm () {  # arm outdir warm -> "status seconds"
  local arm="$1" out="$2" warm="$3" rc
  rm -rf "${out}" 2>/dev/null || mv "${out}" "${out}.stale.$(date +%s)"
  mkdir -p "${out}"
  write_params "${arm}" "${out}" "${warm}"
  ( cd "${out}" && in_env "${out}/time" cherimoya "${arm}" -p "${out}/params.json" ) > "${out}/run.log" 2>&1 && rc=0 || rc=$?
  if [[ ${rc} -eq 0 && -f "${out}/time" ]]; then
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
  in_env "${WORK}/gpu_check.time" python -I "${HERE}/gpu_check.py" torch > "${WORK}/gpu_check_pt.log" 2>&1 \
    || { echo "Error: PyTorch in ${CHERIMOYA_ENV} failed its GPU check:" >&2; tail -8 "${WORK}/gpu_check_pt.log" | sed 's/^/  | /' >&2; exit 1; }
  sed -n 1,2p "${WORK}/gpu_check_pt.log" | sed 's/^/  /'
fi

echo "### warm-up: file cache"
cat "${GENOME}" "${GENOME}.fai" "${AUX}/data_unstranded.bw" "${MODEL}" > /dev/null

IFS=',' read -r -a ARMS_LIST <<< "${ARMS_ARG}"
ACTIVE=()
echo "### warm-up: one untimed run per arm (evaluate and marginalize on the full input: fills the kernel cache)"
for arm in "${ARMS_LIST[@]}"; do
  read -r status seconds <<< "$(run_arm "${arm}" "${WORK}/warmup_${arm}" 1)"
  printf "  %-12s %s %s\n" "${arm}" "${status}" "${seconds}"
  if [[ "${status}" == ok ]]; then
    ACTIVE+=("${arm}")
  else
    record "${arm}" "-" "-" "" failed "warm-up failed: ${WORK}/warmup_${arm}/run.log"
    tail -15 "${WORK}/warmup_${arm}/run.log" | sed 's/^/  | /'
  fi
  find "${WORK}/warmup_${arm}" -type f ! -name run.log ! -name params.json ! -name time -delete 2>/dev/null || true
done
[[ ${#ACTIVE[@]} -gt 0 ]] || { echo "no arm survived the warm-up" >&2; exit 1; }

n=${#ACTIVE[@]}
shift_by=$(( (n + ROUNDS - 1) / ROUNDS ))
(( shift_by > 0 )) || shift_by=1
for (( round = 0; round < ROUNDS; round++ )); do
  echo "### round $((round + 1)) of ${ROUNDS}"
  for (( i = 0; i < n; i++ )); do
    arm="${ACTIVE[$(( (i + round * shift_by) % n ))]}"
    read -r status seconds <<< "$(run_arm "${arm}" "${WORK}/r${round}_${arm}" 0)"
    printf "  %-12s %-8s %s\n" "${arm}" "${status}" "${seconds}"
    if [[ "${status}" == ok ]]; then
      record "${arm}" "${round}" "${i}" "${seconds}" ok ""
    else
      record "${arm}" "${round}" "${i}" "" "${status}" "${WORK}/r${round}_${arm}/run.log"
      tail -10 "${WORK}/r${round}_${arm}/run.log" | sed 's/^/  | /'
    fi
    # Keep the logs, parameters and evaluate's metrics; the arrays add up.
    find "${WORK}/r${round}_${arm}" -type f ! -name run.log ! -name params.json ! -name time \
      ! -name performance.tsv -delete 2>/dev/null || true
  done
done
echo "timings: ${SUMMARY}"
finish_results
