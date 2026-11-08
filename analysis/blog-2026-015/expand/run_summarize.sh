#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=expand_summarize
#SBATCH --output=/path/to/workdir/sherlock/logs/expand_summarize.%j.out
#SBATCH --error=/path/to/workdir/sherlock/logs/expand_summarize.%j.err

# Collate and plot old vs new vs the students, with the engines sweep's
# scripts from this checkout (analysis/engine/run_summarize.py,
# plot_summarize.py) and analysis/expand/plot_expand.py:
#
#   runs     chrombpnet-refactor (default)  the engines sweep's pytorch runs (and the teachers)
#            chrombpnet-refactor (updated)  ${ROOT}/new: profile-only bias, pearsonr-jsd selection
#            student-lofo                   ${ROOT}/distill/lofo
#            student-final                  ${ROOT}/distill/final (saw every test chromosome)
#   versus   each student vs chrombpnet-refactor (default): student fold k against
#            teacher k (predictions, DeepSHAP)
#   plus     original, original_retrained from the engines sweep's tree
#   JSD      the default runs, original and original_retrained report median_jsd in base e,
#            the current code in base 2: --jsd-base-e puts all on base 2
#
#   sbatch run_summarize.sh                 # CPU node; the first run reads ~11 GB
#                                           # of original scores per dataset and fold
#   ./run_summarize.sh --no-contributions   # here, quickly
#
#   OUT_DIR=...   (default: ${ROOT}/analysis/profile-stopgrad-off)
#   CONDA_ENV=... (default: chrombpnet-pytorch)
#   WORKERS=N     (default: the job's CPUs, at most 4)
# Extra arguments go to run_summarize.py.

set -euo pipefail

_find_here() {
  local candidate
  for candidate in "${EXPAND_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/expand" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/_common.sh" && -f "${candidate}/run_summarize.sh" ]] \
      && { echo "${candidate}"; return 0; }
  done
  return 1
}
EXPAND_DIR="$(_find_here)" || { echo "Error: cannot locate expand/_common.sh; set EXPAND_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${EXPAND_DIR}/_common.sh"

OUT_DIR="${OUT_DIR:-${ROOT}/analysis/profile-stopgrad-off}"
mkdir -p "${OUT_DIR}"
CONDA_ENV="${CONDA_ENV:-/path/to/workdir/software/miniconda3/envs/chrombpnet-pytorch}"
export PATH="${CONDA_ENV}/bin:${PATH}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
(( WORKERS > 4 )) && WORKERS=4
ANALYSIS="${CHROMBPNET_DIR}/analysis"
DEFAULT="chrombpnet-refactor (default)"
UPDATED="chrombpnet-refactor (updated)"

python "${ANALYSIS}/engine/run_summarize.py" --root "${ENGINES_ROOT}" --ids "${IDS_FILE}" \
  --biosamples "${ANALYSIS}/engine/biosamples.tsv" --out-dir "${OUT_DIR}" --workers "${WORKERS}" \
  --profile-stopgrad off \
  --runs "${DEFAULT}=${ENGINES_ROOT}/pytorch" "${UPDATED}=${ROOT}/new" \
         "student-lofo=${ROOT}/distill/lofo" "student-final=${ROOT}/distill/final" \
  --versus "student-lofo=${DEFAULT}" "student-final=${DEFAULT}" \
  --jsd-base-e "${DEFAULT}" original original_retrained "$@"
python "${ANALYSIS}/engine/plot_summarize.py" --dir "${OUT_DIR}" --reference "${DEFAULT}" --noun model
python "${ANALYSIS}/expand/plot_expand.py" --dir "${OUT_DIR}"
