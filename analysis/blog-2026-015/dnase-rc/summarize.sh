#!/bin/bash
# Collate every arm's test metrics, and the old TensorFlow ChromBPNet's, into
# one TSV, for all datasets and folds.
#
#   ./summarize.sh                      # -> results.tsv on stdout and to disk
#   ./summarize.sh | column -t -s $'\t' # readable
#
#   ROOT=...      arm trees to read     (default: DNASE_RC_ROOT from site.env)
#   OUT=...       TSV to write          (default: ${ROOT}/analysis/results.tsv)
#   IDS_FILE=...  ids table whose TENSORFLOW_DIR column locates the old models
#                 (default: DNASE_RC_IDS)
#   CONDA_ENV=... environment directory (default: ${CONDA_ROOT}/envs/chrombpnet-pytorch)
#
# What each source means, and the headline printed to stderr, are described
# in summarize.py.

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
    [[ -n "${candidate}" && -f "${candidate}/summarize.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
DNASE_RC_DIR="$(_find_dnase_rc_dir)" || {
  echo "Error: cannot locate summarize.py. Set DNASE_RC_DIR." >&2; exit 1; }

[[ -n "${ROOT:-}" ]] || { require_site DNASE_RC_ROOT; ROOT="${DNASE_RC_ROOT}"; }
OUT="${OUT:-${ROOT}/analysis/results.tsv}"
[[ -n "${IDS_FILE:-}" ]] || { require_site DNASE_RC_IDS; IDS_FILE="${DNASE_RC_IDS}"; }
mkdir -p "$(dirname "${OUT}")"

[[ -n "${CONDA_ENV:-}" ]] || { require_site CONDA_ROOT; CONDA_ENV="${CONDA_ROOT}/envs/chrombpnet-pytorch"; }
export PATH="${CONDA_ENV}/bin:${PATH}"

exec python "${DNASE_RC_DIR}/summarize.py" --root "${ROOT}" --out "${OUT}" \
  --ids "${IDS_FILE}" --biosamples "${DNASE_RC_DIR}/biosamples.tsv"
