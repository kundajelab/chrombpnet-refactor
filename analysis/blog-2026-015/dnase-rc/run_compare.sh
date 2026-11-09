#!/bin/bash
# Cross-test arm X against the two standard models, in both directions:
#
#   the standard and standard_tensorflow models on arm X's test set
#   arm X's model on the standard and standard_tensorflow test sets
#
# always on the same fold. Arm X's model on its own test set is NOT re-run:
# the training pipeline already wrote model/evaluation/chrombpnet_metrics.json,
# and summarize.py reads it from there.
#
#   ./run_compare.sh <cluster> <data-aug>
#   sbatch run_compare.sh slurm right
#
#   DRY_RUN=1            resolve and print every path and command, run nothing
#   FORCE=1              recompute results that already exist (default: skip them)
#   KEEP_PREDICTIONS=1   keep each run's ~200 MB *_predictions.h5 (default: the
#                        metrics are kept and the predictions removed)
#   TF_CONDA_ENV=...     env for the TensorFlow model (default: chrombpnet)
#   TF_CHROMBPNET_DIR=.. the original ChromBPNet checkout it runs from
#                        (default: ORIGINAL_REPO from site.env)
#   TF_BATCH_SIZE=...    its prediction batch size (default 256; the original's
#                        512 runs out of memory on 11-12 GB GPUs)
#
# Results, under comparisons/<arm>/<ASSAY>/<ENCID>/fold_N/:
#
#   standard_metrics.json                     standard model     on X's test set
#   standard_tensorflow_metrics.json          TensorFlow model   on X's test set
#   on_standard/arm_metrics.json              X's model on standard's test set
#   on_standard_tensorflow/arm_metrics.json   X's model on the TensorFlow test set
#
# A test set is one model's own (regions, signal) pair, restricted to the
# fold's test chromosomes and scored on peaks only (-n None):
#
#   arm, standard        <fold>/model/auxiliary/filtered.peaks.bed + data_unstranded.bw
#   standard_tensorflow  filtered.peaks.bed of the run in the ids table's
#                        TENSORFLOW_DIR + the chromatin-atlas bigwig it trained on
#
# The refactor's models go through `chrombpnet predict`, which runs only the
# prediction + metrics stage -- `chrombpnet qc` would additionally run
# attribution and MoDISco, hours per run, for numbers that are already written
# after the first stage. The TensorFlow .h5 (which the refactor cannot load)
# goes through the original package's own evaluation, `python -m
# chrombpnet.training.predict` from TF_CHROMBPNET_DIR -- the code that wrote
# each TensorFlow run's own chrombpnet_metrics.json.
#
# The arm changes `chrombpnet predict` for one value only: rc-only
# (--data-aug revcomp-only) reverse-complements every test example (and
# reverses its profile). So a refactor model scored on X's test set gets X's
# flags (arm_flags in _args.sh), and X's model on another test set gets
# `standard`'s. The original package has no such option
# and never flips: on rc-only's test set the TensorFlow model sees it
# forward, while the standard model sees it reverse-complemented.
#
# HOW TO READ THE RESULT
#
#   fr-only, rc-only, right
#       The bigwig is byte-identical to `standard`'s, so standard and arm see
#       exactly the same targets. A clean model-vs-model comparison: the
#       difference is attributable to the augmentation alone.
#
#   left, both-half, both-full
#       These arms CHANGE the bigwig. The standard models are therefore being
#       scored against a shifted (left) or re-weighted (both-*) target, and
#       the arm's model against targets it was not trained for; both will
#       lose for reasons that have nothing to do with reverse complementation.
#       Read these arms as evidence about their own convention, NOT as
#       head-to-head tests against standard.
#
#   standard_tensorflow
#       Its test set has its own peak filtering and a bigwig from the older
#       preprocessing, so even fr-only is not a byte-identical comparison.
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --job-name=dnase_rc_cmp
#SBATCH --output=dnase_rc_cmp.%A_%a.out
#SBATCH --error=dnase_rc_cmp.%A_%a.err
#SBATCH --array=0-19

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

# `standard` is the reference every other arm is compared against, so it is
# not itself comparable.
VALID_ARMS=(fr-only rc-only right left both-half both-full)
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

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
KEEP_PREDICTIONS="${KEEP_PREDICTIONS:-0}"
TF_CONDA_ENV="${TF_CONDA_ENV:-chrombpnet}"
[[ -n "${TF_CHROMBPNET_DIR:-}" ]] || { require_site ORIGINAL_REPO; TF_CHROMBPNET_DIR="${ORIGINAL_REPO}"; }
TF_BATCH_SIZE="${TF_BATCH_SIZE:-256}"

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
	: "${SLURM_ARRAY_TASK_ID:=0}"
	CONDA_ENV="${CONDA_ENV:-chrombpnet-pytorch}"
fi

echo "conda environment: ${CONDA_ENV}  (TensorFlow model: ${TF_CONDA_ENV}, ${TF_CHROMBPNET_DIR})"
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
FOLD_NUM=$(echo ${line} | cut -d " " -f 5)
TF_DIR=$(echo ${line} | cut -d " " -f 6)
TF_DIR="${TF_DIR%/}"

if contains "${DATA_AUG}" "${DNASE_ONLY_ARMS[@]}" && [[ "${ASSAY}" != "DNASE" ]]; then
  echo "SKIP: --data-aug ${DATA_AUG} is defined only for DNASE; ${ENCID} is ${ASSAY}."
  exit 0
fi

GENOME_PATH=${GENOME}
CHROM_PATH=${CHROM_SIZES}
FOLD_JSON_PATH=${REFERENCE_DIR}/folds/fold_${FOLD_NUM}.json

ROOT=${DNASE_RC_ROOT}
ARM_FOLD=${ROOT}/${DATA_AUG}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}
STD_FOLD=${ROOT}/standard/${ASSAY}/${ENCID}/fold_${FOLD_NUM}

ARM_MODEL=${ARM_FOLD}/model/models/chrombpnet.pt
STD_MODEL=${STD_FOLD}/model/models/chrombpnet.pt
ARM_BIGWIG=${ARM_FOLD}/model/auxiliary/data_unstranded.bw
ARM_PEAKS=${ARM_FOLD}/model/auxiliary/filtered.peaks.bed
STD_BIGWIG=${STD_FOLD}/model/auxiliary/data_unstranded.bw
STD_PEAKS=${STD_FOLD}/model/auxiliary/filtered.peaks.bed

# The TensorFlow runs come in two layouts: the newer `chrombpnet pipeline`
# keeps models/ and auxiliary/ subdirectories, the older atlas runs keep
# everything flat. First match wins; none means there is no such run.
first_file() { local f; for f in "$@"; do [[ -f "${f}" ]] && { echo "${f}"; return 0; }; done; return 1; }
TF_MODEL=""; TF_PEAKS=""
if [[ -n "${TF_DIR}" ]]; then
  TF_MODEL="$(first_file "${TF_DIR}/models/chrombpnet.h5" "${TF_DIR}/chrombpnet.h5")" || TF_MODEL=""
  TF_PEAKS="$(first_file "${TF_DIR}/auxiliary/filtered.peaks.bed" "${TF_DIR}/auxiliary/filtered.peaks.bed.gz" \
                         "${TF_DIR}/filtered.peaks.bed" "${TF_DIR}/filtered.peaks.bed.gz")" || TF_PEAKS=""
fi
# The older atlas runs were saved under Python 3.7, whose Lambda bytecode the
# chrombpnet env (3.8) cannot load. resave_tf_models.sh writes a checked copy
# here; when it exists it stands in for the original. Peaks stay the run's own.
TF_RESAVED=${ROOT}/standard_tensorflow/${ASSAY}/${ENCID}/fold_${FOLD_NUM}/chrombpnet.h5
[[ -f "${TF_RESAVED}" ]] && TF_MODEL="${TF_RESAVED}"
# Every TensorFlow run in the ids table trained on the atlas bigwig (checked
# against each run's chrombpnet.args.json; one names a /scratch copy of it).
if [[ -z "${TF_BIGWIG:-}" ]]; then
  require_site ATLAS_DIR
  TF_BIGWIG="${ATLAS_DIR}/${ASSAY}/${ENCID}/preprocessing/bigWigs/${ENCID}.bigWig"
fi

OUT_DIR=${ROOT}/comparisons/${DATA_AUG}/${ASSAY}/${ENCID}/fold_${FOLD_NUM}

echo "ARM:      ${DATA_AUG}"
echo "ASSAY:    ${ASSAY}   ENCID: ${ENCID}   FOLD: ${FOLD_NUM}"
echo "TF run:   ${TF_DIR:-<none in ids table>}"
[[ "${TF_MODEL}" == "${TF_RESAVED}" ]] && echo "TF model: the Python 3.8 re-save, ${TF_RESAVED}"

missing=0
for f in "${ARM_MODEL}" "${ARM_BIGWIG}" "${ARM_PEAKS}"; do
  [[ -f "${f}" ]] || { echo "Error: missing ${f}"; missing=1; }
done
if [[ "${missing}" -ne 0 ]]; then
  echo "Run ./run_train_full.sh ${CLUSTER} ${DATA_AUG} first."
  exit 1
fi
HAVE_STD=1
for f in "${STD_MODEL}" "${STD_BIGWIG}" "${STD_PEAKS}"; do
  [[ -f "${f}" ]] || { echo "note: missing ${f} -- skipping the standard comparisons"; HAVE_STD=0; }
done
HAVE_TF=1
for f in "${TF_MODEL:-<no chrombpnet.h5 under ${TF_DIR:-TENSORFLOW_DIR}>}" \
         "${TF_PEAKS:-<no filtered.peaks.bed under ${TF_DIR:-TENSORFLOW_DIR}>}" "${TF_BIGWIG}"; do
  [[ -f "${f}" ]] || { echo "note: missing ${f} -- skipping the standard_tensorflow comparisons"; HAVE_TF=0; }
done

# `standard` and `${DATA_AUG}` must have been preprocessed from the same reads;
# if the arm does not change the bigwig, prove it rather than assume it.
if [[ "${HAVE_STD}" -eq 1 ]]; then
  if cmp -s "${ARM_BIGWIG}" "${STD_BIGWIG}"; then
    echo "bigwig: IDENTICAL to standard -- clean model-vs-model comparison"
  else
    echo "bigwig: DIFFERS from standard -- this arm changes the target;"
    echo "        each model is scored against a target it was not trained for"
  fi
fi

# The original TensorFlow ChromBPNet, from its own checkout. TF 2.8 needs
# CUDA 11.2 + cuDNN 8.1 to see a GPU, which site.env's site_tf28_libraries
# hook provides (module loads, or LD_LIBRARY_PATH). Fails if
# `chrombpnet` would import from anywhere but TF_CHROMBPNET_DIR.
tensorflow_env() {
  conda activate "${TF_CONDA_ENV}" || return 1
  export PYTHONPATH="${TF_CHROMBPNET_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  # TF otherwise reserves the whole card up front; on a shared GPU, take
  # only what the model needs.
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  site_hook tf28_libraries || return 1
  local where
  where="$(python -c 'import chrombpnet, os; print(os.path.dirname(os.path.dirname(os.path.realpath(chrombpnet.__file__))))')" || return 1
  [[ "${where}" == "$(realpath "${TF_CHROMBPNET_DIR}")" ]] || {
    echo "    Error: chrombpnet imports from ${where}, not ${TF_CHROMBPNET_DIR}"; return 1; }
}

# One evaluation: <who> <engine: refactor|tensorflow> <model> <bigwig> <peaks> <data-aug> <output prefix>
FAILED=()
evaluate() {
  local who="$1" engine="$2" model="$3" bigwig="$4" peaks="$5" aug="$6" prefix="$7"
  local cmd
  if [[ "${engine}" == "tensorflow" ]]; then
    aug="n/a, never flipped"
    cmd=(python -m chrombpnet.training.predict -m "${model}" -g "${GENOME_PATH}"
         -b "${bigwig}" -p "${peaks}" -n None -fl "${FOLD_JSON_PATH}" -o "${prefix}"
         -bs "${TF_BATCH_SIZE}")
  else
    # The arm's chrombpnet flags (_args.sh): only rc-only's --data-aug
    # revcomp-only changes what predict does.
    arm_flags "${aug}" "${ASSAY}" || { FAILED+=("${who}"); return 0; }
    cmd=(chrombpnet predict -m "${model}" -d "${ASSAY}" -g "${GENOME_PATH}" -c "${CHROM_PATH}"
         -r "${peaks}" -n None -fl "${FOLD_JSON_PATH}" -bw "${bigwig}"
         -op "${prefix}" ${ARM_FLAGS[@]+"${ARM_FLAGS[@]}"})
  fi
  echo
  echo "--- ${who}"
  echo "    model:  ${model}"
  echo "    test:   ${peaks}"
  echo "            ${bigwig}   (--data-aug ${aug})"
  if [[ -s "${prefix}_metrics.json" && "${FORCE}" != "1" ]]; then
    echo "    exists: ${prefix}_metrics.json -- skipped (FORCE=1 recomputes)"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "    [dry run, ${engine}] ${cmd[*]}"
    return 0
  fi
  mkdir -p "$(dirname "${prefix}")"
  # A subshell, so switching env for the TensorFlow model leaves this one alone.
  if ( if [[ "${engine}" == "tensorflow" ]]; then tensorflow_env || exit 1; fi
       "${cmd[@]}" ); then
    [[ "${KEEP_PREDICTIONS}" == "1" ]] || rm -f "${prefix}_predictions.h5"
  else
    echo "    FAILED: ${who}"
    FAILED+=("${who}")
  fi
}

if [[ "${HAVE_STD}" -eq 1 ]]; then
  evaluate "standard model on ${DATA_AUG}'s test set" refactor "${STD_MODEL}" \
    "${ARM_BIGWIG}" "${ARM_PEAKS}" "${DATA_AUG}" "${OUT_DIR}/standard"
  evaluate "${DATA_AUG}'s model on standard's test set" refactor "${ARM_MODEL}" \
    "${STD_BIGWIG}" "${STD_PEAKS}" standard "${OUT_DIR}/on_standard/arm"
fi
if [[ "${HAVE_TF}" -eq 1 ]]; then
  evaluate "standard_tensorflow model on ${DATA_AUG}'s test set" tensorflow "${TF_MODEL}" \
    "${ARM_BIGWIG}" "${ARM_PEAKS}" "${DATA_AUG}" "${OUT_DIR}/standard_tensorflow"
  evaluate "${DATA_AUG}'s model on standard_tensorflow's test set" refactor "${ARM_MODEL}" \
    "${TF_BIGWIG}" "${TF_PEAKS}" standard "${OUT_DIR}/on_standard_tensorflow/arm"
fi

[[ "${DRY_RUN}" == "1" ]] && { echo; echo "Dry run: nothing was evaluated."; exit 0; }

# Every model on every test set this fold has, own-test scores included, so
# the log reads as one table. The own-test rows come from training.
echo
echo "=== ${DATA_AUG} cross-tests (${ENCID} fold ${FOLD_NUM}, peaks) ==="
TF_OWN="$(first_file "${TF_DIR}/evaluation/chrombpnet_metrics.json" "${TF_DIR}/chrombpnet_metrics.json" 2>/dev/null)" || TF_OWN=""
python - "${DATA_AUG}" \
  "${DATA_AUG}" "${DATA_AUG}" "${ARM_FOLD}/model/evaluation/chrombpnet_metrics.json" \
  standard "${DATA_AUG}" "${OUT_DIR}/standard_metrics.json" \
  standard_tensorflow "${DATA_AUG}" "${OUT_DIR}/standard_tensorflow_metrics.json" \
  "${DATA_AUG}" standard "${OUT_DIR}/on_standard/arm_metrics.json" \
  standard standard "${STD_FOLD}/model/evaluation/chrombpnet_metrics.json" \
  "${DATA_AUG}" standard_tensorflow "${OUT_DIR}/on_standard_tensorflow/arm_metrics.json" \
  standard_tensorflow standard_tensorflow "${TF_OWN:-<none>}" <<'PY'
import json, sys
arm, rest = sys.argv[1], sys.argv[2:]
print(f"{'model':<20} {'test set':<20} {'counts r':>9} {'profile JSD':>12}")
for model, test, path in zip(rest[0::3], rest[1::3], rest[2::3]):
    try:
        m = json.load(open(path))
        r = m["counts_metrics"]["peaks"]["pearsonr"]
        jsd = m["profile_metrics"]["peaks"]["median_jsd"]
        print(f"{model:<20} {test:<20} {r:>9.4f} {jsd:>12.4f}")
    except (OSError, ValueError, KeyError):
        print(f"{model:<20} {test:<20} {'-':>9} {'-':>12}")
PY

if (( ${#FAILED[@]} )); then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "Completed. ($(date +%Y-%m-%d\ %H:%M:%S))"
