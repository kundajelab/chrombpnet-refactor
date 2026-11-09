#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=dnase_rc_train
#SBATCH --output=dnase_rc_train.%A_%a.out
#SBATCH --error=dnase_rc_train.%A_%a.err
#SBATCH --array=10-19

# Train one --data-aug arm across all four datasets and five folds.
#
#   ./run_train_full.sh <cluster> <data-aug>
#   sbatch run_train_full.sh slurm right
#
# <data-aug> is one of:
#   standard    today's behaviour: DNase 0/+1, RC 50% of train, no shift
#   fr-only     no reverse complement anywhere (train, valid AND test)
#   rc-only     reverse complement everything (train, valid AND test)
#   right       DNase 0/+1 with the +1 post-reversal shift        <- the fix
#   left        DNase -1/0 with the -1 post-reversal shift        <- the mirror
#   both-half   deposit 0.5 on both bases flanking the cut; no shift needed
#   both-full   deposit 1.0 on both bases flanking the cut; no shift needed
#
# standard, fr-only and rc-only are assay-agnostic and run on every row.
# right, left and both-* are DNase-only by construction -- ATAC's +4/-4 signal
# is centred on a base, so there is no bond to index or to split, and the CLI
# raises rather than silently doing nothing. ATAC rows are therefore skipped
# for those arms, and the skip is announced.

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


# The glibc 2.17 build of the environment (envs/chrombpnet-pytorch-glibc.yml), by
# name; site.env's site_pytorch_compiler hook runs before it is activated on a Slurm node.
PT_ENV_GLIBC217="${PT_ENV_GLIBC217:-chrombpnet-pytorch-glibc}"
if [[ "$CLUSTER" == "slurm" ]]; then
	# chrombpnet-pytorch by default; CONDA_ENV=${PT_ENV_GLIBC217} (torch
	# 2.5.1, built for glibc 2.17) reproduces earlier runs.
	CONDA_ENV="${CONDA_ENV:-chrombpnet-pytorch}"
	if [[ "${CONDA_ENV}" == "${PT_ENV_GLIBC217}" ]]; then
		site_hook pytorch_compiler
	fi
else
	: "${SLURM_ARRAY_TASK_ID:=0}"  # Manually set row when running by hand
	CONDA_ENV="${CONDA_ENV:-chrombpnet-pytorch}"
fi

echo "conda environment: ${CONDA_ENV}"
# chrombpnet-pytorch (PyTorch 2.14) needs glibc >= 2.28 (engines/envs/glibc_scan.py:
# torch and igraph); some clusters' GPU nodes still run glibc 2.17. Stop
# here on such a node rather than failing at the first import.
GLIBC_FLOOR="${GLIBC_FLOOR:-2.28}"
if [[ "${CONDA_ENV}" != "${PT_ENV_GLIBC217}" ]]; then
	NODE_GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
	if [[ -n "${NODE_GLIBC}" && "$(printf '%s\n%s\n' "${GLIBC_FLOOR}" "${NODE_GLIBC}" | sort -V | head -1)" != "${GLIBC_FLOOR}" ]]; then
		echo "Error: ${CONDA_ENV} needs glibc >= ${GLIBC_FLOOR}; this node ($(hostname)) has ${NODE_GLIBC}." >&2
		echo "       Resubmit with --exclude=$(hostname), or set CONDA_ENV=${PT_ENV_GLIBC217}." >&2
		exit 1
	fi
fi
eval "$("${CONDA_ROOT:+${CONDA_ROOT}/bin/}conda" shell.bash hook)"
conda activate "${CONDA_ENV}"

[[ -n "${IDS_FILE:-}" ]] || { require_site DNASE_RC_IDS; IDS_FILE="${DNASE_RC_IDS}"; }

require_site GENOME CHROM_SIZES REFERENCE_DIR DNASE_RC_ROOT

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
BAM_PATH=$(echo ${line} | cut -d " " -f 3)
PEAKS_BED_PATH=$(echo ${line} | cut -d " " -f 4)
FOLD_NUM=$(echo ${line} | cut -d " " -f 5)

# The reference run script hardcodes 0.5; this ids file mixes both assays.
if [[ "${ASSAY}" == "ATAC" ]]; then
  BIAS_THRESHOLD_FACTOR=0.5
else
  BIAS_THRESHOLD_FACTOR=0.8
fi

if contains "${DATA_AUG}" "${DNASE_ONLY_ARMS[@]}" && [[ "${ASSAY}" != "DNASE" ]]; then
  echo "SKIP: --data-aug ${DATA_AUG} is defined only for DNASE; ${ENCID} is ${ASSAY}."
  echo "      (ATAC's +4/-4 signal is base-centred, so there is no bond to shift.)"
  exit 0
fi

# The chrombpnet flags this arm stands for (_args.sh).
arm_flags "${DATA_AUG}" "${ASSAY}" || exit 2

GENOME_PATH=${GENOME}
CHROM_PATH=${CHROM_SIZES}
BLACKLIST_PATH=${REFERENCE_DIR}/blacklist.bed.gz
FOLD_JSON_PATH=${REFERENCE_DIR}/folds/fold_${FOLD_NUM}.json

# Each arm gets its own tree, so arms never overwrite one another and
# run_compare.sh can find the standard model alongside.
ARM_DIR=${DNASE_RC_ROOT}/${DATA_AUG}
ENCID_DIR=${ARM_DIR}/${ASSAY}/${ENCID}
FOLD_DIR=${ENCID_DIR}/fold_${FOLD_NUM}
DATA_DIR=${FOLD_DIR}/data/output/${ENCID}
BIAS_DIR=${FOLD_DIR}/bias_model/
MODEL_DIR=${FOLD_DIR}/model/

echo "DATA_AUG: ${DATA_AUG}  (chrombpnet flags: ${ARM_FLAGS[*]:-none})"
echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "FOLD_DIR: ${FOLD_DIR}"
# --ignore-bias-flag downgrades `chrombpnet pipeline`'s bias-model quality
# check from an error to a warning. Resolved here, beside the other run
# settings, rather than next to the call that uses it: the call comes hours
# later, and a setting that only announces itself deep in a two-day log is a
# setting nobody audits. It is also what makes DRY_RUN able to show it.
#
# On by default because these sweeps include samples whose peak and
# background counts overlap so much that no threshold can select a
# representative nonpeak set -- the check then measures the data, not the
# model, and no retraining fixes it.
#
# IGNORE_BIAS=0 restores the hard failure. Use that for any run whose bias
# correction you intend to trust quantitatively.
IGNORE_BIAS="${IGNORE_BIAS:-1}"
BIAS_FLAG=()
if [[ "${IGNORE_BIAS}" == "1" ]]; then
  BIAS_FLAG=(--ignore-bias-flag)
fi
echo "IGNORE_BIAS: ${IGNORE_BIAS} (bias-model quality check is $([[ "${IGNORE_BIAS}" == "1" ]] && echo 'a WARNING' || echo 'ENFORCED'))"

if [[ ! -f "${BAM_PATH}" ]]; then
  echo "Error: BAM not found: ${BAM_PATH}"
  echo "Run ./00_prepare_bams.sh first."
  exit 1
fi

# DRY_RUN=1 stops here: everything above is resolution and validation,
# everything below deletes and rebuilds the fold. Lets the whole
# run_retrain_full.sh -> run_train_full.sh chain be exercised without
# starting a training run.
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN set -- resolved successfully, stopping before any work."
  exit 0
fi

## PIPELINE --------------------------------------------------------------------------------------
if [[ -d "${FOLD_DIR}" ]]; then
  echo "Removing existing directory: ${FOLD_DIR}"
  rm -rf "${FOLD_DIR}" || { echo "Failed to remove existing directory: ${FOLD_DIR}"; exit 1; }
fi

echo "Creating directory: ${MODEL_DIR}"
mkdir -p "${DATA_DIR}" "${BIAS_DIR}" "${MODEL_DIR}" || { echo "Failed to create directories"; exit 1; }

# Prepare nonpeaks
echo "Preparing nonpeaks at: ${DATA_DIR}"
chrombpnet prep nonpeaks \
  -g ${GENOME_PATH} \
  -p ${PEAKS_BED_PATH} \
  -c ${CHROM_PATH} \
  -fl ${FOLD_JSON_PATH} \
  -br ${BLACKLIST_PATH} \
  -o ${DATA_DIR}

NEG_BED_PATH="${DATA_DIR}_negatives.bed"
if [[ ! -f "${NEG_BED_PATH}" ]]; then
  echo "Error: NEG_BED_PATH not found: ${NEG_BED_PATH}"
  exit 1
fi

NEG_BED_GZ_PATH="${DATA_DIR}/negatives.bed.gz"
echo "Compressing NEG_BED_PATH to: ${NEG_BED_GZ_PATH}"
bgzip -f -@ 4 -l 9 -c "${NEG_BED_PATH}" > "${NEG_BED_GZ_PATH}"

# Train a bias model
echo "Training bias model at: ${BIAS_DIR}"
chrombpnet bias pipeline \
	-ibam ${BAM_PATH} \
	-d ${ASSAY} \
	-g ${GENOME_PATH} \
	-c ${CHROM_PATH} \
	-p ${PEAKS_BED_PATH} \
	-n ${NEG_BED_GZ_PATH} \
	-fl ${FOLD_JSON_PATH} \
	-b ${BIAS_THRESHOLD_FACTOR} \
	-o ${BIAS_DIR} \
	-fp ${ENCID} \
	${ARM_FLAGS[@]+"${ARM_FLAGS[@]}"}

BIAS_MODEL_PATH="${BIAS_DIR}/models/${ENCID}_bias.pt"
if [[ ! -f "${BIAS_MODEL_PATH}" ]]; then
  echo "Error: BIAS_MODEL_PATH not found: ${BIAS_MODEL_PATH}"
  exit 1
fi


# Run chrombpnet pipeline
echo "Running chrombpnet pipeline at: ${MODEL_DIR}"
chrombpnet pipeline \
	-ibam ${BAM_PATH} \
	-d ${ASSAY} \
	-g ${GENOME_PATH} \
	-c ${CHROM_PATH} \
	-p ${PEAKS_BED_PATH} \
	-n ${NEG_BED_GZ_PATH} \
	-fl ${FOLD_JSON_PATH} \
	-b ${BIAS_MODEL_PATH} \
	-o ${MODEL_DIR} \
	${ARM_FLAGS[@]+"${ARM_FLAGS[@]}"} \
	"${BIAS_FLAG[@]}"

echo "Output: ${MODEL_DIR}"
echo "Completed run. ($(date +%Y-%m-%d\ %H:%M:%S))"
