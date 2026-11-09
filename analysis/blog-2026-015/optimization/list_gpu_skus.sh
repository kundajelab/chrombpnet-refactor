#!/bin/bash
# Which GPU node features (the names a --constraint takes) actually exist, and
# in which partitions. Run on the cluster's login node. Slurm rejects a
# --constraint that matches no node in the requested partitions with
# BadConstraints, and the job then pends forever, so use a name from this list
# verbatim (e.g. in site.env's SBATCH_L40S / SBATCH_H200 / SBATCH_H100).
#
#   ./list_gpu_skus.sh PARTITIONS [FEATURE_PATTERN]
#   PARTITIONS       comma-separated partitions to look in
#   FEATURE_PATTERN  grep -i pattern for the GPU features (default: every feature of
#                    the GPU nodes; e.g. '^GPU_SKU:' on a cluster that names them so)
PARTITIONS="${1:?usage: list_gpu_skus.sh PARTITIONS [FEATURE_PATTERN]}"
PATTERN="${2:-.}"

gpu_features () {
  sinfo -h -p "${PARTITIONS}" -o "%G|%f" 2>/dev/null \
    | awk -F'|' '$1 ~ /gpu/ {print $2}' | tr ',' '\n' | grep -i -- "${PATTERN}" | sort -u
}

echo "=== GPU node features present in: ${PARTITIONS} ==="
gpu_features | sed 's/^/  /'

echo
echo "=== nodes with GPUs, their features and counts ==="
printf "  %-18s %-10s %s\n" NODES GRES FEATURES
sinfo -h -p "${PARTITIONS}" -o "%N|%G|%f" 2>/dev/null | while IFS='|' read -r nodes gres feats; do
  [[ "${gres}" == *gpu* ]] || continue
  printf "  %-18s %-10s %s\n" "${nodes}" "${gres}" \
    "$(tr ',' '\n' <<< "${feats}" | grep -i -- "${PATTERN}" | paste -sd, -)"
done | sort -u

echo
echo "Use one verbatim, e.g.:"
gpu_features | head -3 \
  | sed 's|^|  sbatch --constraint="|; s|$|" run_chrombpnet_optimization_benchmark.sh slurm|'
