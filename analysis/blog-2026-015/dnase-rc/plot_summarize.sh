#!/bin/bash
# Plot the scores collated by summarize.sh.
#
#   ./plot_summarize.sh                    # uses the existing results.tsv
#   ./plot_summarize.sh --refresh          # re-run summarize.sh first
#   ./plot_summarize.sh --arm right --arm standard
#   ./plot_summarize.sh --arm standard_tensorflow --arm standard   # old vs refactor
#   ./plot_summarize.sh --source headtohead
#
# standard_tensorflow (the old TensorFlow ChromBPNet) is plotted by default;
# while it is, each other arm's means also show their difference from it.
# On the figures it is labelled "original", and standard "original-refactor";
# --arm still takes the arm names.
#
# Writes to the run tree's analysis/ directory, each as .png, .pdf and .tsv:
#   summary_scores                         each model on its own test set
#   summary_scores_aug_on_standard_tests   arm models on the standard models'
#                                          test sets       (from run_compare.sh)
#     ..._original, ..._refactor           its two halves, one figure each
#   summary_scores_standard_on_aug_tests   standard models on each arm's test
#                                          set             (from run_compare.sh)
#     ..._original, ..._refactor           its two halves, one figure each
# All options are passed through to plot_summarize.py; see --help there.

set -euo pipefail

# Not `dirname "$0"`: under sbatch that is the Slurm spool copy, which holds
# nothing but the script. Same resolution order as the run_*.sh scripts.
_find_dnase_rc_dir() {
  local candidate
  for candidate in "${DNASE_RC_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/dnase-rc" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/plot_summarize.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
DNASE_RC_DIR="$(_find_dnase_rc_dir)" || {
  echo "Error: cannot locate plot_summarize.py. Set DNASE_RC_DIR." >&2; exit 1; }

CONDA_ENV="${CONDA_ENV:-/path/to/workdir/software/miniconda3/envs/chrombpnet-pytorch}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export MPLBACKEND=Agg

exec python "${DNASE_RC_DIR}/plot_summarize.py" "$@"
