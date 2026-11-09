#!/bin/bash
# Plot the scores collated by summarize.sh.
#
#   ./plot_summarize.sh --results <run tree>/analysis/results.tsv --out <dir>/summary_scores
#   ./plot_summarize.sh --results ... --out ... --fill-original ENCSR868FGK:2   # as the blog
#   ./plot_summarize.sh --results ... --out ... --arm right --arm standard
#   ./plot_summarize.sh --results ... --out ... --source headtohead
#
# --results is summarize.sh's TSV; CONDA_ENV (default ${CONDA_ROOT}/envs/chrombpnet-pytorch)
# is the environment it runs in.
#
# standard_tensorflow (the old TensorFlow ChromBPNet) is plotted by default;
# while it is, each other arm's means also show their difference from it.
# On the figures it is labelled "original", and standard "original-refactor";
# --arm still takes the arm names.
#
# Writes, under the --out prefix, each as .png, .pdf and .tsv:
#   summary_scores                         each model on its own test set
#   summary_scores_aug_on_standard_tests   arm models on the standard models'
#                                          test sets       (from run_compare.sh)
#     ..._original, ..._refactor           its two halves, one figure each
#   summary_scores_standard_on_aug_tests   standard models on each arm's test
#                                          set             (from run_compare.sh)
#     ..._original, ..._refactor           its two halves, one figure each
# All options are passed through to plot_summarize.py; see --help there.

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

# Not `dirname "$0"`: under sbatch that is the Slurm spool copy, which holds
# nothing but the script. DNASE_RC_DIR, else ${BLOG_SCRIPTS}/dnase-rc, as in
# the run_*.sh scripts.
_find_dnase_rc_dir() {
  local candidate
  for candidate in "${DNASE_RC_DIR:-}" "${BLOG_SCRIPTS}/dnase-rc"; do
    [[ -n "${candidate}" && -f "${candidate}/plot_summarize.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
DNASE_RC_DIR="$(_find_dnase_rc_dir)" || {
  echo "Error: cannot locate plot_summarize.py. Set DNASE_RC_DIR." >&2; exit 1; }

[[ -n "${CONDA_ENV:-}" ]] || { require_site CONDA_ROOT; CONDA_ENV="${CONDA_ROOT}/envs/chrombpnet-pytorch"; }
export PATH="${CONDA_ENV}/bin:${PATH}"
export MPLBACKEND=Agg

exec python "${DNASE_RC_DIR}/plot_summarize.py" "$@"
