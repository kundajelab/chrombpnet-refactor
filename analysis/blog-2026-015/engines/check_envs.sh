#!/bin/bash
# Check that each engine's conda env can run the pipeline here, before
# submitting 60 jobs: it activates, imports chrombpnet from the engines
# checkout, has the CLI, runs a convolution on the GPU, and loads and
# exercises everything else the pipeline uses (runtime_check in _common.sh).
#
#   srun -p <partitions> --gpus=1 --time=0:30:00 --gpu_cmode=shared \
#        -C "GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0" ./check_envs.sh sherlock --container
#   ./check_envs.sh local
#   ./check_envs.sh sherlock jax --container    # one engine
#
# --container runs each check inside the container (see _common.sh), as the
# jobs will with submit_all.sh --container; the image is pulled here the
# first time. On sherlock JAX fails without it. The -C constraint asks for a GPU the
# JAX env's cuDNN has kernels for (compute capability 7.5+); PyTorch and
# TensorFlow also run on older ones. --gpu_cmode=shared: the TensorFlow
# check refuses a GPU in sherlock's default Exclusive_Process mode, which
# the TF engine's DeepSHAP process cannot use (submit_all.sh asks for shared
# mode for the TensorFlow jobs).
#
# Exits non-zero if any engine fails; each failure says why.

set -uo pipefail

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"
parse_common_flags "$@"
REQUESTED="${ARGS[1]:-}"
resolve_cluster "${ARGS[0]:-}"

SELECTED=("${ENGINES[@]}")
[[ -n "${REQUESTED}" ]] && SELECTED=("${REQUESTED}")

echo "host: $(hostname)$(in_container && echo ' (in the container)')"
echo "glibc: $(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null \
  | sed 's/^/gpu:  /' || echo "gpu:  nvidia-smi not available"

failed=()
for engine in "${SELECTED[@]}"; do
  echo
  echo "=== ${engine}"
  if needs_container; then
    # The same check, run inside the container.
    if ensure_image && container_run "${engine}" "${CLUSTER}" bash "${HERE}/check_envs.sh" "${CLUSTER}" "${engine}"; then
      echo "${engine}: OK (in the container ${CONTAINER_IMAGE})"
    else
      echo "${engine}: FAILED"
      failed+=("${engine}")
    fi
    continue
  fi
  if ! check_native "${engine}" "${CLUSTER}"; then
    echo "${engine}: FAILED"
    failed+=("${engine}")
    continue
  fi
  # A subshell per engine, so one env's modules and variables cannot leak
  # into the next check. Chained with &&: set -e does not apply inside an
  # `if` condition.
  if ( activate_engine "${engine}" "${CLUSTER}" &&
       chrombpnet --help > /dev/null &&
       echo "cli: $(command -v chrombpnet)" &&
       gpu_preflight "${engine}" &&
       runtime_check "${engine}" ); then
    echo "${engine}: OK"
  else
    echo "${engine}: FAILED"
    failed+=("${engine}")
  fi
done

echo
if (( ${#failed[@]} )); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "All engines ready: ${SELECTED[*]}"
