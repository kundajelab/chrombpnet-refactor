#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=<partitions>
#SBATCH --job-name=engines_summarize
#SBATCH --output=/path/to/workdir/sherlock/logs/engines_summarize.%j.out
#SBATCH --error=/path/to/workdir/sherlock/logs/engines_summarize.%j.err

# Collate the engines sweep into results.tsv, agreement.tsv,
# contributions.tsv, inputs.tsv and timings.tsv in ${ROOT}/analysis/<dir>, and
# print a headline. Which engines' profile scores it compares picks <dir>:
#
#   --profile-stopgrad off   (the default) PROFILE_WEIGHTING=midpoint's
#                            rescoring, chrombpnet_nobias.midpoint.profile_scores.h5:
#                            the original's weighting   -> profile-stopgrad-off
#   --profile-stopgrad on    the pipeline's chrombpnet_nobias.profile_scores.h5:
#                            this sweep ran before the midpoint default, so they
#                            honour the stop-gradient   -> profile-stopgrad-on
#
#   sbatch run_summarize.sh            # on a CPU node: the first run reads ~11 GB
#                                      # of original contribution scores per
#                                      # dataset and fold (later runs use the cache)
#   sbatch run_summarize.sh --profile-stopgrad on
#   ./run_summarize.sh                 # here
#   ./run_summarize.sh --no-contributions   # skip the contribution comparison
#   ./run_summarize.sh --check-bigwigs      # also confirm the engines' bigwigs match
#   column -t -s $'\t' contributions.tsv | less -S
#
#   ROOT=...      run tree to read     (default: the engines test tree)
#   OUT_DIR=...   where the TSVs go    (default: ${ROOT}/analysis/<dir>, above; a
#                 relative path is taken inside ${ROOT}/analysis)
#   CONDA_ENV=... python env           (default: chrombpnet-pytorch)
#   WORKERS=N     datasets x folds compared at once (default: the job's CPUs, at most 4)
#
# What each table means is described in run_summarize.py.

set -euo pipefail

_find_engines_dir() {
  local candidate
  for candidate in "${ENGINES_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/engines" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/run_summarize.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
ENGINES_DIR="$(_find_engines_dir)" || { echo "Error: cannot locate run_summarize.py; set ENGINES_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${ENGINES_DIR}/_common.sh"

ANALYSIS_DIR="${ANALYSIS_DIR:-${ROOT}/analysis}"
STOPGRAD="off"
for (( i = 1; i <= $#; i++ )); do
  case "${!i}" in
    --profile-stopgrad) j=$(( i + 1 )); STOPGRAD="${!j:-}" ;;
    --profile-stopgrad=*) STOPGRAD="${!i#--profile-stopgrad=}" ;;
  esac
done
case "${STOPGRAD}" in
  on|off) DEFAULT_OUT="profile-stopgrad-${STOPGRAD}" ;;
  *) echo "Error: --profile-stopgrad takes on or off, not '${STOPGRAD}'" >&2; exit 2 ;;
esac
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT}}"
[[ "${OUT_DIR}" == /* ]] || OUT_DIR="${ANALYSIS_DIR}/${OUT_DIR}"
mkdir -p "${OUT_DIR}"
CONDA_ENV="${CONDA_ENV:-/path/to/workdir/software/miniconda3/envs/chrombpnet-pytorch}"
export PATH="${CONDA_ENV}/bin:${PATH}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
(( WORKERS > 4 )) && WORKERS=4

exec python "${ENGINES_DIR}/run_summarize.py" --root "${ROOT}" --ids "${IDS_FILE}" \
  --biosamples "${ENGINES_DIR}/biosamples.tsv" --out-dir "${OUT_DIR}" --workers "${WORKERS}" "$@"
