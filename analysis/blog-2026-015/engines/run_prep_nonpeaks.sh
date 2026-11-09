#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=engines_prep
#SBATCH --output=engines_prep.%A_%a.out
#SBATCH --error=engines_prep.%A_%a.err
#SBATCH --array=0-19

# GC-matched nonpeaks for one dataset and fold, written once and shared by
# all three engines, so that every engine trains and is tested on the same
# regions. CPU only; `chrombpnet prep nonpeaks` touches no model.
#
#   sbatch run_prep_nonpeaks.sh slurm [--container]
#   SLURM_ARRAY_TASK_ID=0 ./run_prep_nonpeaks.sh local
#
#   --container runs it inside the Ubuntu 22.04 image (_common.sh), as
#   submit_all.sh --container does for every step
#   FORCE=1     rebuild even if negatives.bed.gz exists (default: skip)
#   DRY_RUN=1   resolve and print, run nothing
#   PREP_ENGINE=jax   which engine's env runs it (default pytorch; any will
#               do, the command touches no model)
#
# Output: ${ROOT}/prep/<ASSAY>/<ENCID>/fold_N/negatives.bed.gz, plus the
# command's own nonpeaks_negatives.bed and nonpeaks_auxiliary/. The .gz is
# moved into place last, so it exists only if the step finished.

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
ENGINES_DIR="${ENGINES_DIR:-${BLOG_SCRIPTS}/engines}"
[[ -f "${ENGINES_DIR}/_common.sh" ]] || { echo "Error: no _common.sh in ${ENGINES_DIR}; set ENGINES_DIR." >&2; exit 1; }
# shellcheck source=_common.sh
source "${ENGINES_DIR}/_common.sh"

parse_common_flags "$@"
resolve_cluster "${ARGS[0]:-}"
[[ "${CLUSTER}" == "local" ]] && : "${SLURM_ARRAY_TASK_ID:=0}"
select_row

PREP_DIR="$(prep_dir)"
NEG_BED_GZ="${PREP_DIR}/negatives.bed.gz"

echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "PREP_DIR: ${PREP_DIR}"

if [[ -s "${NEG_BED_GZ}" && "${FORCE:-0}" != "1" ]]; then
  echo "exists: ${NEG_BED_GZ} -- skipped (FORCE=1 rebuilds)"
  echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
  exit 0
fi
for f in "${PEAKS_BED_PATH}" "${GENOME_PATH}" "${CHROM_PATH}" "${FOLD_JSON_PATH}" "${BLACKLIST_PATH}"; do
  [[ -f "${f}" ]] || { echo "Error: missing ${f}" >&2; exit 1; }
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: chrombpnet prep nonpeaks -g ${GENOME_PATH} -p ${PEAKS_BED_PATH} -c ${CHROM_PATH}" \
       "-fl ${FOLD_JSON_PATH} -br ${BLACKLIST_PATH} -o ${PREP_DIR}/nonpeaks"
  exit 0
fi

check_native "${PREP_ENGINE:-pytorch}" "${CLUSTER}" || exit 2
if needs_container; then
  export ENGINES_DIR
  container_run "${PREP_ENGINE:-pytorch}" "${CLUSTER}" bash "${ENGINES_DIR}/run_prep_nonpeaks.sh" "${CLUSTER}"
  exit $?
fi

# Any env will do, since no model is involved; the PyTorch one is the default.
activate_engine "${PREP_ENGINE:-pytorch}" "${CLUSTER}"

rm -rf "${PREP_DIR}"
mkdir -p "${PREP_DIR}"
chrombpnet prep nonpeaks \
  -g "${GENOME_PATH}" \
  -p "${PEAKS_BED_PATH}" \
  -c "${CHROM_PATH}" \
  -fl "${FOLD_JSON_PATH}" \
  -br "${BLACKLIST_PATH}" \
  -o "${PREP_DIR}/nonpeaks"

NEG_BED="${PREP_DIR}/nonpeaks_negatives.bed"
[[ -s "${NEG_BED}" ]] || { echo "Error: ${NEG_BED} was not written" >&2; exit 1; }
bgzip -f -@ 4 -l 9 -c "${NEG_BED}" > "${NEG_BED_GZ}.partial"
mv "${NEG_BED_GZ}.partial" "${NEG_BED_GZ}"
echo "nonpeaks: $(zcat "${NEG_BED_GZ}" | wc -l) regions -> ${NEG_BED_GZ}"
echo "Completed run. ($(date '+%Y-%m-%d %H:%M:%S'))"
