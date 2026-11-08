# Sourced by run_pred_bw_kit_fused.sh and run_kit_modes.sh: what differs between
# a Slurm job on Sherlock and a plain `bash` run on one of the lab's GPU hosts
# (the lab GPU host, another lab host, ...).

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CLUSTER: sherlock or local; inferred from the Slurm job when not given.
resolve_cluster () {
  CLUSTER="${1:-}"
  if [[ -z "${CLUSTER}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then CLUSTER=sherlock; else CLUSTER=local; fi
  fi
  case "${CLUSTER}" in
    sherlock|local) ;;
    *) echo "unknown cluster '${CLUSTER}' (expected sherlock or local)" >&2; exit 2 ;;
  esac
}

# Host part of a local run's tag, so a run on a lab host never replaces a
# Sherlock job's results for the same GPU model (or another host's).
host_suffix () {
  [[ "${CLUSTER}" == local ]] && echo "_$(hostname -s)"
  return 0
}

# Sets GPU_ID, GPU_NAME and GPU_CC for the one GPU the run uses. Under Slurm,
# the job's GPU. Locally, the first entry of CUDA_VISIBLE_DEVICES (0 if unset),
# made the only visible one (TensorFlow would otherwise take memory on every
# visible GPU), with CUDA numbering GPUs by PCI bus as nvidia-smi does (CUDA's
# default puts the fastest first, which differs on a host with mixed cards).
# Locally a GPU that already runs something is refused: its timings would not
# be comparable (ALLOW_BUSY_GPU=1 overrides). DEVICE=cpu hides every GPU.
select_gpu () {
  if [[ "${DEVICE:-cuda}" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
    GPU_ID="-"; GPU_NAME="CPU"; GPU_CC="0"
    return 0
  fi
  GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
  GPU_ID="${GPU_ID%%,*}"
  if [[ "${CLUSTER}" == local ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  fi
  GPU_NAME="$(nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  GPU_CC="$(nvidia-smi -i "${GPU_ID}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  if [[ -z "${GPU_NAME}" ]]; then   # an id nvidia-smi does not take: the first GPU, as before
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
  fi
  if [[ "${CLUSTER}" == local && "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    local apps
    apps="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${apps}" ]]; then
      echo "Error: GPU ${GPU_ID} (${GPU_NAME}) is already in use, so its timings would not be comparable:" >&2
      echo "${apps}" | sed 's/^/  /' >&2
      echo "  Pick an idle one with CUDA_VISIBLE_DEVICES=<index> (nvidia-smi), or set ALLOW_BUSY_GPU=1." >&2
      exit 1
    fi
  fi
}

# LD_LIBRARY_PATH without its CUDA and cuDNN directories. On the lab hosts
# ~/.bashrc puts a system CUDA 12 / cuDNN 9 there, and a Sherlock job inherits
# the submitting shell's; either outranks the RUNPATH by which PyTorch's cuDNN
# 9 finds its own sub-libraries (on Sherlock the kit env's torch reported cuDNN
# 9.4, not its own 9.1, and the kit's exact mode died loading cuDNN).
ld_path_without_cuda () {
  printf '%s' "${LD_LIBRARY_PATH:-}" \
    | awk -v RS=: -v ORS=: 'NF && tolower($0) !~ /cuda|cudnn/' | sed 's/:$//'
}

# TensorFlow 2.8's CUDA 11.2 and cuDNN 8.1 on a lab host (Sherlock loads them
# as modules), colon-separated: TF_CUDA_DIRS if set, else the /usr/local copies.
local_tf_cuda_dirs () {
  if [[ -n "${TF_CUDA_DIRS:-}" ]]; then echo "${TF_CUDA_DIRS}"; return 0; fi
  local d found=()
  for d in /usr/local/cuda-11.2/lib64 /usr/local/cudnn*8.1*/lib64 /usr/local/cudnn*8.1*/lib; do
    [[ -d "${d}" ]] && found+=("${d}")
  done
  ( IFS=:; echo "${found[*]}" )
}

# The library path for a TensorFlow 2.8 process on a lab host: CUDA 11.2 and
# cuDNN 8.1, and none of the other CUDA directories.
tf_local_libraries () {
  local dirs rest
  dirs="$(local_tf_cuda_dirs)"
  rest="$(ld_path_without_cuda)"
  export LD_LIBRARY_PATH="${dirs}${rest:+${dirs:+:}${rest}}"
}

# TensorFlow 2.8 has no machine code for Hopper (H100/H200): the driver
# compiles its PTX when each process starts. With the default 1 GiB JIT cache
# that does not all stay, and every run would compile again; at the maximum,
# the untimed warm-up run fills the cache for the timed ones.
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"

# Exits unless the active env's framework reaches the GPU (gpu_check.py:
# TensorFlow 2.8 without its CUDA libraries runs on the CPU without an error;
# PyTorch must run its own cuDNN). Run in a subshell that has set up the env.
check_framework_gpu () {  # tensorflow|torch log
  local framework="$1" log="$2"
  if ! { echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
         python -I "${COMMON_DIR}/gpu_check.py" "${framework}"; } > "${log}" 2>&1; then
    echo "Error: ${framework} in $(basename "${CONDA_PREFIX:-?}") failed its check on GPU ${GPU_ID} (${GPU_NAME}); log ${log}:" >&2
    tail -8 "${log}" | sed 's/^/  | /' >&2
    if [[ "${framework}" == tensorflow && "${CLUSTER}" == local ]]; then
      echo "  TensorFlow 2.8 needs CUDA 11.2 and cuDNN 8.1; looked in: ${LD_LIBRARY_PATH:-(nothing)}" >&2
      echo "  Set TF_CUDA_DIRS=<cuda-11.2>/lib64:<cudnn-8.1>/lib64 for this host." >&2
    fi
    return 1
  fi
}

# A run's results go to a hidden partial directory beside the final one and
# replace the previous run's only when this run finishes, so a preempted or
# failed run never deletes the last complete one (owners preemption cost the
# first H200 kit-modes results that way). Sets WORK; call finish_results last.
start_results () {  # final directory
  WORK_FINAL="$1"
  WORK="$(dirname "$1")/.partial_$(basename "$1")"
  if [[ -d "${WORK}" ]] && ! rm -rf "${WORK}" 2>/dev/null; then mv "${WORK}" "${WORK}.stale.$(date +%s)"; fi
  mkdir -p "${WORK}"
}
finish_results () {
  if [[ -d "${WORK_FINAL}" ]] && ! rm -rf "${WORK_FINAL}" 2>/dev/null; then
    mv "${WORK_FINAL}" "$(dirname "${WORK_FINAL}")/.replaced_$(basename "${WORK_FINAL}").$(date +%s)"
  fi
  mv "${WORK}" "${WORK_FINAL}"
  echo "results: ${WORK_FINAL}"
}
