#!/bin/bash
# Produce exactly one `alignments_merged.bam` per ENCID, in that ENCID's
# chromatin-atlas downloads directory, for the four datasets the --data-aug
# experiments use.
#
#   DNASE ENCSR000EOT   two replicates -> samtools merge
#   DNASE ENCSR163PKT   one file
#   ATAC  ENCSR337UIU   one file
#   ATAC  ENCSR868FGK   already downloaded elsewhere -> symlink, not a copy
#                       (${RAW_BAM_DIR}/ENCSR868FGK/merged.bam)
#
# Every ENCID ends with `alignments_merged.bam` + `.bai` in
#   ${ATLAS_DIR}/{DNASE,ATAC}/<ENCID>/preprocessing/downloads/
#
# Idempotent: an ENCID whose final BAM already passes `samtools quickcheck` is
# skipped, so a re-run after an interrupted download costs nothing. Partial
# downloads resume (`wget -c`).
#
# Usage:
#   ./00_prepare_bams.sh                 # all four, sequentially (local)
#   ./00_prepare_bams.sh ENCSR000EOT     # just one
#   sbatch 00_prepare_bams.sh            # Slurm array, one ENCID per task
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --job-name=dnase_rc_bams
#SBATCH --output=dnase_rc_bams.%A_%a.out
#SBATCH --error=dnase_rc_bams.%A_%a.err
#SBATCH --array=1-4

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

# samtools ships inside the conda env, not on PATH.
[[ -n "${CONDA_ENV:-}" ]] || { require_site CONDA_ROOT; CONDA_ENV="${CONDA_ROOT}/envs/chrombpnet-pytorch"; }
export PATH="${CONDA_ENV}/bin:${PATH}"
command -v samtools >/dev/null || { echo "samtools not found in ${CONDA_ENV}/bin"; exit 1; }

require_site ATLAS_DIR RAW_BAM_DIR
ATLAS="${ATLAS_DIR}"
SCRATCH="${RAW_BAM_DIR}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ENCID | ASSAY | source: either a comma-separated URL list, or local:<path>
ROWS=(
  "ENCSR000EOT|DNASE|https://www.encodeproject.org/files/ENCFF425WDA/@@download/ENCFF425WDA.bam,https://www.encodeproject.org/files/ENCFF205FNC/@@download/ENCFF205FNC.bam"
  "ENCSR163PKT|DNASE|https://www.encodeproject.org/files/ENCFF620MUD/@@download/ENCFF620MUD.bam"
  "ENCSR337UIU|ATAC|https://www.encodeproject.org/files/ENCFF601PGM/@@download/ENCFF601PGM.bam"
  "ENCSR868FGK|ATAC|local:${SCRATCH}/ENCSR868FGK/merged.bam"
)

prepare_one() {
  local encid="$1" assay="$2" source="$3"
  local outdir="${ATLAS}/${assay}/${encid}/preprocessing/downloads"
  local final="${outdir}/alignments_merged.bam"

  echo "=== ${encid} (${assay}) ==="
  [[ -d "${outdir}" ]] || { echo "  atlas dir missing: ${outdir}"; return 1; }

  if [[ -e "${final}" ]] && samtools quickcheck "${final}" 2>/dev/null; then
    echo "  already present and valid, skipping: ${final}"
    [[ -f "${final}.bai" ]] || { echo "  indexing"; samtools index -@ "${THREADS}" "${final}"; }
    return 0
  fi

  # An ENCID we already hold elsewhere is linked, not copied -- ENCSR868FGK is
  # 22.7 GB and duplicating it buys nothing.
  if [[ "${source}" == local:* ]]; then
    local src="${source#local:}"
    [[ -f "${src}" ]] || { echo "  local source missing: ${src}"; return 1; }
    echo "  symlinking ${src}"
    ln -sfn "${src}" "${final}"
    samtools quickcheck "${final}" || { echo "  quickcheck FAILED on ${src}"; return 1; }
    [[ -f "${final}.bai" ]] || samtools index -@ "${THREADS}" "${final}"
    return 0
  fi

  # Download each part. ENCODE serves presigned S3 URLs that accept GET only,
  # so a HEAD returning 403 is normal and not a failure; -c resumes.
  local work="${outdir}/.dnase_rc_work"
  mkdir -p "${work}"
  local parts=() url name
  IFS=',' read -ra urls <<< "${source}"
  for url in "${urls[@]}"; do
    name="$(basename "${url%/@@download/*}").bam"
    echo "  downloading ${name}"
    wget -c -q --show-progress -O "${work}/${name}" "${url}"
    samtools quickcheck "${work}/${name}" || { echo "  quickcheck FAILED on ${name}"; return 1; }
    parts+=("${work}/${name}")
  done

  local merged="${work}/merged.bam"
  if [[ "${#parts[@]}" -gt 1 ]]; then
    echo "  merging ${#parts[@]} replicates"
    samtools merge -f -@ "${THREADS}" "${merged}" "${parts[@]}"
  else
    merged="${parts[0]}"
  fi

  echo "  sorting"
  samtools sort -@ "${THREADS}" -T "${work}/sort" -o "${work}/sorted.bam" "${merged}"
  samtools quickcheck "${work}/sorted.bam" || { echo "  quickcheck FAILED after sort"; return 1; }

  # Move into place only once the sorted file is known good, so an interrupted
  # run never leaves a truncated alignments_merged.bam behind.
  mv "${work}/sorted.bam" "${final}"
  echo "  indexing"
  samtools index -@ "${THREADS}" "${final}"

  rm -rf "${work}"
  echo "  done: ${final}"
}

main() {
  local selected=("$@")
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${#selected[@]}" -eq 0 ]]; then
    selected=("$(cut -d'|' -f1 <<< "${ROWS[$((SLURM_ARRAY_TASK_ID - 1))]}")")
  fi

  local row encid assay source
  for row in "${ROWS[@]}"; do
    IFS='|' read -r encid assay source <<< "${row}"
    if [[ "${#selected[@]}" -gt 0 ]]; then
      local want=0 s
      for s in "${selected[@]}"; do [[ "${s}" == "${encid}" ]] && want=1; done
      [[ "${want}" -eq 1 ]] || continue
    fi
    prepare_one "${encid}" "${assay}" "${source}"
  done

  echo
  echo "=== summary ==="
  for row in "${ROWS[@]}"; do
    IFS='|' read -r encid assay source <<< "${row}"
    local f="${ATLAS}/${assay}/${encid}/preprocessing/downloads/alignments_merged.bam"
    if [[ -e "${f}" ]] && samtools quickcheck "${f}" 2>/dev/null; then
      printf '  %-12s %-6s OK   %s reads\n' "${encid}" "${assay}" \
        "$(samtools idxstats "${f}" 2>/dev/null | awk '{s+=$3} END {print s+0}')"
    else
      printf '  %-12s %-6s MISSING\n' "${encid}" "${assay}"
    fi
  done
}

main "$@"
