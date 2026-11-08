#!/bin/bash
# Plot the tables run_summarize.sh wrote.
#
#   ./plot_summarize.sh                     # from the existing TSVs
#   ./plot_summarize.sh --refresh           # re-run run_summarize.sh first
#   ./plot_summarize.sh --reference original   # compare against the old TF model
#   ./plot_summarize.sh --no-original
#
# Reads and writes each of the run tree's analysis/profile-*/ that
# run_summarize.sh has written (profile-stopgrad-on, profile-stopgrad-off), or
# only --dir (e.g. --dir profile-stopgrad-off), each figure as .png, .pdf and .tsv:
#   summary_scores      each engine's model on its own test set
#   summary_agreement   how closely each engine's model agrees with the original's
#   summary_contributions_pearson, summary_contributions_cosine
#                       each engine's DeepSHAP scores vs the original's
#   summary_runtime     wall time per run and step
# Datasets run left to right in the same ENCID order as dnase-rc's plots.
# All options are passed to plot_summarize.py; see --help there.

set -euo pipefail

_find_engines_dir() {
  local candidate
  for candidate in "${ENGINES_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/engines" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/plot_summarize.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
ENGINES_DIR="$(_find_engines_dir)" || { echo "Error: cannot locate plot_summarize.py; set ENGINES_DIR." >&2; exit 1; }

# shellcheck source=_common.sh
source "${ENGINES_DIR}/_common.sh"
# The tables and figures live in the run tree's analysis/<dir> (run_summarize.sh).
export ANALYSIS_DIR="${ANALYSIS_DIR:-${ROOT}/analysis}"
CONDA_ENV="${CONDA_ENV:-/path/to/workdir/software/miniconda3/envs/chrombpnet-pytorch}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export MPLBACKEND=Agg

exec python "${ENGINES_DIR}/plot_summarize.py" "$@"
