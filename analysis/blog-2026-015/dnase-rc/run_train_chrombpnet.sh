#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=dnase_rc_chrombpnet
#SBATCH --output=dnase_rc_chrombpnet.%A_%a.out
#SBATCH --error=dnase_rc_chrombpnet.%A_%a.err
#SBATCH --array=10-19

# Train the ChromBPNet model for the two primary stomach-tissue datasets
# (data rows 11-20), borrowing a K562 bias model instead of training one.
#
# The bias model paths are given on the command line -- this script does not
# derive them from the ids table and does not train one:
#
#   sbatch run_train_chrombpnet.sh slurm right \
#          --bias-model-dnase /path/to/dnase_bias.pt \
#          --bias-model-atac  /path/to/atac_bias.pt
#
#   sbatch run_train_chrombpnet.sh right --bias-model /path/to/one_bias.pt
#
# One submission covers rows 11-20, which span both assays, so the per-assay
# options select by the row's assay. --bias-model is the single-path
# shorthand. BIAS_MODEL_DNASE / BIAS_MODEL_ATAC / BIAS_MODEL work too.
# `run_train_bias.sh` produces a suitable model.
#
# WHY THIS SCRIPT EXISTS
#
# The stomach datasets cannot train their own bias model. Their peak and
# background count distributions overlap -- the weakest 1% of peaks have
# fewer reads than a median background region -- so the bias threshold
# (0.8 x peak 1st-percentile) falls below the nonpeak median and only the
# bottom quarter of nonpeaks qualifies for training. That is selection on
# the outcome variable: the surviving regions systematically lack the
# sequence features that predict high counts, so the model learns an
# inverted relation and ChromBPNet rejects it at r ~ -0.5. See README.
#
# A bias model captures enzyme sequence preference, which is a property of
# the assay rather than the biosample, so a model trained on a high-quality
# K562 experiment of the SAME assay is a reasonable stand-in, and
# `chrombpnet pipeline` rescales it to the target read depth
# (`bias_model_scaled.pt`). Matching the arm matters as much as matching the
# assay: `left` and `both-*` change the bigwig itself, so a bias model from
# a different arm would have been fitted to differently-registered counts.
#
#   ENCSR163PKT (DNASE, stomach)  <- ENCSR000EOT (DNASE, K562) fold 0
#   ENCSR337UIU (ATAC,  stomach)  <- ENCSR868FGK (ATAC,  K562) fold 0
#
# TWO CAVEATS, both real:
#
#  1. `chrombpnet pipeline` still runs `check_bias_model_quality` on the
#     borrowed model against THIS dataset's peaks. Borrowing bypasses the
#     bias *training* problem, not the guard. If the K562 model also
#     anti-correlates on stomach peaks the job will still stop, and that
#     would be telling you the peaks themselves are the problem.
#  2. Folds 1-4 use a fold-0 bias model, so that model saw their test
#     chromosomes while training. For a bias model -- nonpeaks only, no
#     locus-specific accessibility -- the leakage is mild and sharing one
#     bias model across folds is common practice, but it is not zero, and
#     these models are therefore not strictly comparable to the K562 runs,
#     which each trained their own.
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
# Bias model paths come from the command line, not from the ids table.
# Strip them out before resolve_args reads the positionals, so the options
# can appear in any order:
#
#   run_train_chrombpnet.sh right --bias-model-dnase /path/dnase_bias.pt \
#                                 --bias-model-atac  /path/atac_bias.pt
#   run_train_chrombpnet.sh slurm right --bias-model /path/one_bias.pt
#
# Equivalent environment variables are accepted too -- BIAS_MODEL_DNASE,
# BIAS_MODEL_ATAC, BIAS_MODEL -- because sbatch argument passing has been a
# repeated source of silent failure here and an export is harder to drop.
BIAS_MODEL="${BIAS_MODEL:-}"
BIAS_MODEL_DNASE="${BIAS_MODEL_DNASE:-}"
BIAS_MODEL_ATAC="${BIAS_MODEL_ATAC:-}"

_need_value() { [[ -n "${2:-}" ]] || { echo "Error: $1 needs a path" >&2; exit 2; }; }
POSITIONAL=()
while (( $# )); do
  case "$1" in
    --bias-model)         _need_value "$1" "${2:-}"; BIAS_MODEL="$2"; shift 2 ;;
    --bias-model-dnase)   _need_value "$1" "${2:-}"; BIAS_MODEL_DNASE="$2"; shift 2 ;;
    --bias-model-atac)    _need_value "$1" "${2:-}"; BIAS_MODEL_ATAC="$2"; shift 2 ;;
    --bias-model=*)       BIAS_MODEL="${1#*=}"; shift ;;
    --bias-model-dnase=*) BIAS_MODEL_DNASE="${1#*=}"; shift ;;
    --bias-model-atac=*)  BIAS_MODEL_ATAC="${1#*=}"; shift ;;
    *)                    POSITIONAL+=("$1"); shift ;;
  esac
done
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"}
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

# Which supplied path applies to this row. The array spans both assays, so
# the per-assay options let one submission cover all of rows 11-20;
# --bias-model is the single-path shorthand when the array is restricted to
# one assay, or when the same model is wanted for both.
case "${ASSAY}" in
  DNASE) BIAS_MODEL_PATH="${BIAS_MODEL_DNASE:-${BIAS_MODEL}}" ;;
  ATAC)  BIAS_MODEL_PATH="${BIAS_MODEL_ATAC:-${BIAS_MODEL}}" ;;
  *)     echo "Error: unknown assay ${ASSAY@Q}"; exit 1 ;;
esac

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
MODEL_DIR=${FOLD_DIR}/model/

# The donor lives in the same arm's tree, so an arm whose bigwig differs
# (left, both-*) gets a bias model fitted to the same count registration.


echo "DATA_AUG: ${DATA_AUG}  (chrombpnet flags: ${ARM_FLAGS[*]:-none})"
echo "ASSAY:    ${ASSAY}"
echo "ENCID:    ${ENCID}"
echo "FOLD_NUM: ${FOLD_NUM}"
echo "FOLD_DIR: ${FOLD_DIR}"
echo "BIAS:     ${BIAS_MODEL_PATH:-<none given>}"
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

if [[ -z "${BIAS_MODEL_PATH}" ]]; then
  echo "Error: no bias model given for ${ASSAY}."
  echo
  echo "This script does not train one -- pass a path:"
  echo "  --bias-model-dnase PATH   applies to the DNASE rows"
  echo "  --bias-model-atac  PATH   applies to the ATAC rows"
  echo "  --bias-model       PATH   applies to whichever row runs"
  echo "(or export BIAS_MODEL_DNASE / BIAS_MODEL_ATAC / BIAS_MODEL)"
  echo
  echo "A K562 fold-0 model of the same assay AND the same --data-aug arm is"
  echo "the intended choice: left and both-* change the bigwig, so a model"
  echo "from another arm was fitted to differently-registered counts."
  exit 2
fi
# Borrowing this dataset's own model would defeat the purpose -- that model
# is the one that fails the quality check.
if [[ "${BIAS_MODEL_PATH}" == *"/${ENCID}/"* ]]; then
  echo "Error: the given bias model lives under ${ENCID}'s own tree:"
  echo "  ${BIAS_MODEL_PATH}"
  echo "Borrowing is meant to bring in a model from a different dataset."
  exit 2
fi
if [[ ! -f "${BIAS_MODEL_PATH}" ]]; then
  echo "Error: the bias model given for ${ASSAY} does not exist:"
  echo "  ${BIAS_MODEL_PATH}"
  echo
  echo "run_train_bias.sh produces one without training a ChromBPNet model."
  echo "For a K562 fold-0 model of this arm (${DATA_AUG}):"
  echo "  sbatch --array=$([[ ${ASSAY} == ATAC ]] && echo 0 || echo 5) \\"
  echo "         run_train_bias.sh ${CLUSTER} ${DATA_AUG}"
  echo "(array task 0 = ATAC ENCSR868FGK fold 0; task 5 = DNASE ENCSR000EOT fold 0)"
  exit 1
fi

if [[ ! -f "${BAM_PATH}" ]]; then
  echo "Error: BAM not found: ${BAM_PATH}"
  echo "Run ./00_prepare_bams.sh first."
  exit 1
fi

# DRY_RUN=1 stops here: everything above is resolution and validation, and
# everything below deletes and rebuilds the fold. Useful for checking that an
# array index maps to the dataset and donor you expect before committing a
# GPU-day to it.
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
mkdir -p "${DATA_DIR}" "${MODEL_DIR}" || { echo "Failed to create directories"; exit 1; }

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

# No bias model is trained. Record where the borrowed one came from, so a
# run is self-describing without cross-referencing the Slurm log.
cat > "${FOLD_DIR}/borrowed_bias_model.txt" <<EOF
bias model borrowed instead of trained
  assay      ${ASSAY}
  arm        ${DATA_AUG}
  path       ${BIAS_MODEL_PATH}
  reason     ${ENCID} peak and background counts overlap too much to train
             a bias model of its own (see README)
  caveat     a borrowed model trained on other folds has seen this fold's
             test chromosomes; mild for a bias model (nonpeaks only, no
             locus-specific accessibility) but not zero
EOF
cat "${FOLD_DIR}/borrowed_bias_model.txt"


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
