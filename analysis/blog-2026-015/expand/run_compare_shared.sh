#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=expand_compare_shared
#SBATCH --output=/path/to/workdir/sherlock/logs/expand_compare_shared.%j.out
#SBATCH --error=/path/to/workdir/sherlock/logs/expand_compare_shared.%j.err

# The students against their teachers' five-fold mean, on the shared regions
# run_shared_scores.sh scored (analysis/expand/compare_fold_means.py): DeepSHAP
# and predictions.
#
#   TEACHER_SET=updated sbatch run_compare_shared.sh
#   TEACHER_SET=updated ./run_compare_shared.sh        # here
#
# Output: ${ROOT}/analysis/fold-mean-${TEACHER_SET}/ (OUT_DIR=... overrides; SHARED_ROOT=...
# reads run_shared_scores.sh's tree from there instead of ${ROOT}/shared).

set -euo pipefail

_find_here() {
  local candidate
  for candidate in "${EXPAND_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/expand" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/_common.sh" && -f "${candidate}/run_compare_shared.sh" ]] \
      && { echo "${candidate}"; return 0; }
  done
  return 1
}
EXPAND_DIR="$(_find_here)" || { echo "Error: cannot locate expand/_common.sh; set EXPAND_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${EXPAND_DIR}/_common.sh"

case "${TEACHER_SET}" in
  updated) LABEL="chrombpnet-refactor (updated)" ;;
  default) LABEL="chrombpnet-refactor (default)" ;;
  *) echo "Error: TEACHER_SET must be default or updated" >&2; exit 2 ;;
esac
OUT_DIR="${OUT_DIR:-${ROOT}/analysis/fold-mean-${TEACHER_SET}}"
CONDA_ENV="${CONDA_ENV:-/path/to/workdir/software/miniconda3/envs/chrombpnet-pytorch}"
export PATH="${CONDA_ENV}/bin:${PATH}"
ANALYSIS="${CHROMBPNET_DIR}/analysis"
echo "TEACHER_SET: ${TEACHER_SET} (${LABEL}) -> ${OUT_DIR}"
python "${ANALYSIS}/expand/compare_fold_means.py" --scores "${SHARED_ROOT:-${ROOT}/shared}/${TEACHER_SET}" --label "${LABEL}" \
  --out-dir "${OUT_DIR}" --biosamples "${ANALYSIS}/engine/biosamples.tsv"
