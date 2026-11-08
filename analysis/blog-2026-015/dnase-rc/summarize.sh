#!/bin/bash
# Collate every arm's test metrics, and the old TensorFlow ChromBPNet's, into
# one TSV, for all datasets and folds.
#
#   ./summarize.sh                      # -> results.tsv on stdout and to disk
#   ./summarize.sh | column -t -s $'\t' # readable
#
#   ROOT=...      arm trees to read     (default: the dnase-rc test tree)
#   OUT=...       TSV to write          (default: ${ROOT}/analysis/results.tsv)
#   IDS_FILE=...  ids table whose TENSORFLOW_DIR column locates the old models
#
# What each source means, and the headline printed to stderr, are described
# in summarize.py.

set -euo pipefail

# Not `dirname "$0"`: under sbatch that is the Slurm spool copy, which holds
# nothing but the script. Same resolution order as the run_*.sh scripts.
_find_dnase_rc_dir() {
  local candidate
  for candidate in "${DNASE_RC_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/dnase-rc" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/summarize.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
DNASE_RC_DIR="$(_find_dnase_rc_dir)" || {
  echo "Error: cannot locate summarize.py. Set DNASE_RC_DIR." >&2; exit 1; }

ROOT="${ROOT:-/path/to/workdir/chromatlas/test/chrombpnet-refactor-dnase-rc}"
OUT="${OUT:-${ROOT}/analysis/results.tsv}"
mkdir -p "$(dirname "${OUT}")"
IDS_FILE="${IDS_FILE:-/path/to/workdir/sherlock/ids/chromatlas/chrombpnet-refactor/ids_chrombpnet_refactor_benchmark_data_aug.txt}"

CONDA_ENV="${CONDA_ENV:-/path/to/workdir/software/miniconda3/envs/chrombpnet-pytorch}"
export PATH="${CONDA_ENV}/bin:${PATH}"

exec python "${DNASE_RC_DIR}/summarize.py" --root "${ROOT}" --out "${OUT}" \
  --ids "${IDS_FILE}" --biosamples "${DNASE_RC_DIR}/biosamples.tsv"
