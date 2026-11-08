#!/bin/bash
# Which GPU_SKU values actually exist, and in which partitions. Run on a
# Sherlock login node. Slurm rejects a --constraint that matches no node in
# the requested partitions with BadConstraints, and the job then pends
# forever, so use a name from this list verbatim.
PARTITIONS="${1:-<partitions>}"

echo "=== GPU_SKU features present in: ${PARTITIONS} ==="
sinfo -h -p "${PARTITIONS}" -o "%f" 2>/dev/null \
  | tr ',' '\n' | grep -i "^GPU_SKU:" | sort -u | sed 's/^/  /'

echo
echo "=== nodes with GPUs, their features and counts ==="
printf "  %-18s %-10s %s\n" NODES GRES FEATURES
sinfo -h -p "${PARTITIONS}" -o "%N|%G|%f" 2>/dev/null | while IFS='|' read -r nodes gres feats; do
  [[ "${gres}" == *gpu* ]] || continue
  printf "  %-18s %-10s %s\n" "${nodes}" "${gres}" \
    "$(tr ',' '\n' <<< "${feats}" | grep -i GPU_SKU | paste -sd, -)"
done | sort -u

echo
echo "Use one verbatim, e.g.:"
sinfo -h -p "${PARTITIONS}" -o "%f" 2>/dev/null \
  | tr ',' '\n' | grep -i "^GPU_SKU:" | sort -u | head -3 \
  | sed 's|^|  sbatch --constraint="|; s|$|" run_chrombpnet_optimization_benchmark.sh sherlock|'
