#!/bin/bash
# Re-save the old (Python 3.7) TensorFlow ChromBPNet models as copies the
# `chrombpnet` env can load; run_compare.sh then uses the copies.
#
#   ./resave_tf_models.sh            # reference -> resave -> verify
#   ./resave_tf_models.sh --force    # redo models that are already re-saved
#
# Writes <root>/standard_tensorflow/<ASSAY>/<ENCID>/fold_<N>/; the originals
# are only read. Why, and what each step checks: resave_tf_models.py.
#
# The verify step needs a GPU with TensorFlow 2.8's CUDA 11.2 / cuDNN 8.1, which
# site.env's site_tf28_libraries hook provides (module loads, or LD_LIBRARY_PATH).
# Paths: DNASE_RC_ROOT, DNASE_RC_IDS (IDS_FILE overrides), ATLAS_DIR, GENOME,
# REFERENCE_DIR, and ORIGINAL_REPO (TF_CHROMBPNET_DIR overrides).

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

_find_dnase_rc_dir() {
  local candidate
  for candidate in "${DNASE_RC_DIR:-}" "${BLOG_SCRIPTS}/dnase-rc"; do
    [[ -n "${candidate}" && -f "${candidate}/resave_tf_models.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
DNASE_RC_DIR="$(_find_dnase_rc_dir)" || {
  echo "Error: cannot locate resave_tf_models.py. Set DNASE_RC_DIR." >&2; exit 1; }
SCRIPT="${DNASE_RC_DIR}/resave_tf_models.py"
[[ -n "${TF_CHROMBPNET_DIR:-}" ]] || { require_site ORIGINAL_REPO; TF_CHROMBPNET_DIR="${ORIGINAL_REPO}"; }
[[ -n "${IDS_FILE:-}" ]] || { require_site DNASE_RC_IDS; IDS_FILE="${DNASE_RC_IDS}"; }
require_site DNASE_RC_ROOT ATLAS_DIR GENOME REFERENCE_DIR
PATHS=(--ids "${IDS_FILE}" --root "${DNASE_RC_ROOT}")

eval "$("${CONDA_ROOT:+${CONDA_ROOT}/bin/}conda" shell.bash hook)"

echo "=== reference: the originals, under Python 3.7 (env bpnet, CPU)"
( conda activate bpnet && CUDA_VISIBLE_DEVICES="" python "${SCRIPT}" reference "${PATHS[@]}" "$@" )

echo "=== resave: rebuild and check, under Python 3.8 (env chrombpnet, CPU)"
( conda activate chrombpnet && CUDA_VISIBLE_DEVICES="" python "${SCRIPT}" resave "${PATHS[@]}" "$@" )

echo "=== verify: score each copy on its run's own test set (env chrombpnet, GPU)"
(
  conda activate chrombpnet
  export PYTHONPATH="${TF_CHROMBPNET_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  site_hook tf28_libraries
  python "${SCRIPT}" verify "${PATHS[@]}" --atlas-dir "${ATLAS_DIR}" --genome "${GENOME}" \
    --reference-dir "${REFERENCE_DIR}" "$@"
)
