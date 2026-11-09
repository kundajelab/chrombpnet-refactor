#!/bin/bash
# Re-run only the array tasks that did not finish.
#
#   sbatch run_retrain_full.sh slurm right
#   ./run_retrain_full.sh local right          # one row, set SLURM_ARRAY_TASK_ID
#
# A task exits 0 immediately when its run is already complete, so the whole
# array can be resubmitted after a partial sweep and it will pick up only
# what is missing. Nothing already finished is recomputed or deleted.
#
# "Complete" needs both halves, because either alone is misleading:
#
#   the artifacts   a finished run leaves models/chrombpnet.pt and
#                   evaluation/chrombpnet_metrics.json
#   the log         the job printed "Completed run"
#
# A run that died in QC leaves the model but never printed that line, and its
# evaluation directory is half-written. Treating it as done would put a
# partially-scored fold into summarize.sh. Treating a *finished* run as
# incomplete would throw away a GPU-day, so the check has to be exact rather
# than conservative in one direction.
#
# Training itself is delegated to run_train_full.sh -- this script only decides
# whether to invoke it -- so the two can never drift apart.
#
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=dnase_rc_retrain
#SBATCH --output=dnase_rc_retrain.%A_%a.out
#SBATCH --error=dnase_rc_retrain.%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

VALID_ARMS=(standard fr-only rc-only right left both-half both-full)
DNASE_ONLY_ARMS=(right left both-half both-full)
contains() { local n="$1"; shift; for x in "$@"; do [[ "$x" == "$n" ]] && return 0; done; return 1; }

# Where this project's scripts live.
#
# NOT `dirname "$0"`. Slurm copies the batch script to
# /var/spool/slurmd/job<N>/slurm_script and runs it from there, so under
# sbatch $0 is that copy and its directory holds nothing else -- sourcing a
# sibling file through it fails on the cluster while working perfectly when
# run by hand, which is exactly how it got shipped. So it is DNASE_RC_DIR, or
# ${BLOG_SCRIPTS}/dnase-rc -- which the lines above find through
# SLURM_SUBMIT_DIR under sbatch, so submit from this folder -- and the
# directory is verified before it is used.
_find_dnase_rc_dir() {
  local candidate
  for candidate in "${DNASE_RC_DIR:-}" "${BLOG_SCRIPTS}/dnase-rc"; do
    [[ -n "${candidate}" && -f "${candidate}/_args.sh" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
DNASE_RC_DIR="$(_find_dnase_rc_dir)" || {
  echo "Error: cannot locate _args.sh. Set DNASE_RC_DIR to the dnase-rc directory." >&2
  exit 1; }

# shellcheck source=_args.sh
source "${DNASE_RC_DIR}/_args.sh"
resolve_args "${1:-}" "${2:-}"

HERE="${DNASE_RC_DIR}"
[[ -n "${IDS_FILE:-}" ]] || { require_site DNASE_RC_IDS; IDS_FILE="${DNASE_RC_IDS}"; }
require_site DNASE_RC_ROOT
ROOT="${DNASE_RC_ROOT}"
# The Slurm logs: LOG_DIR (where submit_all.sh sends them), else this folder, where a
# job submitted from it writes them (the #SBATCH --output paths are relative).
LOGS="${LOGS:-${LOG_DIR:-${DNASE_RC_DIR}}}"

VALID_ARMS=(standard fr-only rc-only right left both-half both-full)
DNASE_ONLY_ARMS=(right left both-half both-full)
contains() { local n="$1"; shift; for x in "$@"; do [[ "$x" == "$n" ]] && return 0; done; return 1; }
contains "${DATA_AUG}" "${VALID_ARMS[@]}" || {
  echo "unknown --data-aug ${DATA_AUG@Q}; expected one of: ${VALID_ARMS[*]}" >&2; exit 2; }

: "${SLURM_ARRAY_TASK_ID:=0}"

# SLURM_ARRAY_TASK_ID is 0-based; data row 1 is the first line after the header.
DATA_ROW=$((SLURM_ARRAY_TASK_ID + 1))

# DATA_ROW is bounded by the ids table's own length
N_DATA_ROWS="$(data_row_count "${IDS_FILE}")" || exit 1
if (( DATA_ROW < 1 || DATA_ROW > N_DATA_ROWS )); then
  echo "SKIP: array task ${SLURM_ARRAY_TASK_ID} is data row ${DATA_ROW}, but"
  echo "      ${IDS_FILE##*/} has ${N_DATA_ROWS} data rows. Nothing to do."
  exit 0
fi

line=$(sed -n "$((DATA_ROW + 1))p" "$IDS_FILE")
[[ -n "${line}" ]] || { echo "Error: no data on row ${DATA_ROW} of ${IDS_FILE}"; exit 1; }
ASSAY=$(echo ${line} | cut -d " " -f 1)
ENCID=$(echo ${line} | cut -d " " -f 2)
FOLD_NUM=$(echo ${line} | cut -d " " -f 5)

echo "=== retrain check ==="
echo "  arm:   ${DATA_AUG}"
echo "  row:   ${DATA_ROW}  (array task ${SLURM_ARRAY_TASK_ID})"
echo "  data:  ${ASSAY} ${ENCID} fold ${FOLD_NUM}"

# An arm that is not defined for this assay is not incomplete, it is absent.
if contains "${DATA_AUG}" "${DNASE_ONLY_ARMS[@]}" && [[ "${ASSAY}" != "DNASE" ]]; then
  echo "SKIP: --data-aug ${DATA_AUG} is defined only for DNASE; ${ENCID} is ${ASSAY}."
  exit 0
fi

FOLD_DIR="${ROOT}/${DATA_AUG}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"
MODEL="${FOLD_DIR}/model/models/chrombpnet.pt"
METRICS="${FOLD_DIR}/model/evaluation/chrombpnet_metrics.json"

have_model=no; [[ -s "${MODEL}" ]] && have_model=yes
have_metrics=no; [[ -s "${METRICS}" ]] && have_metrics=yes

# Did any previous job for exactly this arm/dataset/fold report success? The
# logs carry their own identity, so this does not depend on knowing which
# Slurm job id belonged to which arm.
logged_done=no
for out in "${LOGS}"/dnase_rc_train.*.out "${LOGS}"/dnase_rc_retrain.*.out; do
  [[ -e "$out" ]] || continue
  grep -q "^Completed run" "$out" 2>/dev/null || continue
  [[ "$(grep -m1 '^DATA_AUG:' "$out" | awk '{print $2}')" == "${DATA_AUG}" ]] || continue
  [[ "$(grep -m1 '^ENCID:'    "$out" | awk '{print $2}')" == "${ENCID}"   ]] || continue
  [[ "$(grep -m1 '^FOLD_NUM:' "$out" | awk '{print $2}')" == "${FOLD_NUM}" ]] || continue
  logged_done=yes
  echo "  prior success logged in: $(basename "$out")"
  break
done

echo "  model: ${have_model}   metrics: ${have_metrics}   logged 'Completed run': ${logged_done}"

if [[ "${have_model}" == "yes" && "${have_metrics}" == "yes" && "${logged_done}" == "yes" ]]; then
  echo "ALREADY COMPLETE -- exiting without retraining."
  exit 0
fi

# Anything else is a retry. Say which of the three was missing, so a rerun
# that keeps failing the same way is visible in the logs rather than silent.
echo "INCOMPLETE -- retraining. Missing:"
[[ "${have_model}"   == "yes" ]] || echo "    models/chrombpnet.pt"
[[ "${have_metrics}" == "yes" ]] || echo "    evaluation/chrombpnet_metrics.json"
[[ "${logged_done}"  == "yes" ]] || echo "    a log reporting 'Completed run'"

# run_train_full.sh removes the fold directory itself before rebuilding it, so a
# half-written tree from the failed attempt does not need clearing here --
# and is left untouched if the exec below never happens.
#
# NOTE for rows 11-20 (stomach). This path trains each dataset its OWN bias
# model, which anti-correlates with observed peak counts at r ~ -0.5 and is
# only accepted because --ignore-bias-flag downgrades the check. That is a
# different experiment from run_train_chrombpnet.sh, which borrows the K562
# fold-0 bias model of the same arm (r ~ +0.52 on these peaks) and is the
# better-founded of the two. Both write to the same
# <arm>/<ASSAY>/<ENCID>/fold_N tree, so they do not compose: whichever
# completes first wins, and this script will then report the other as
# ALREADY COMPLETE and leave it alone. Pick one provenance per arm.
echo
echo "=== handing off to run_train_full.sh ==="
exec env SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}" \
     "${HERE}/run_train_full.sh" "${CLUSTER}" "${DATA_AUG}"
