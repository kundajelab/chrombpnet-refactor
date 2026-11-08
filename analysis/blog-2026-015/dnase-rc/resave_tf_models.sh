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
# Runs where the GPU libraries are: on sherlock (inside a GPU job) it loads
# the CUDA 11.2 / cuDNN 8.1 modules, elsewhere it uses the system copies.

set -euo pipefail

_find_dnase_rc_dir() {
  local candidate
  for candidate in "${DNASE_RC_DIR:-}" "${SLURM_SUBMIT_DIR:-}" \
      "/path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/dnase-rc" \
      "$(dirname "$(readlink -f "$0")")"; do
    [[ -n "${candidate}" && -f "${candidate}/resave_tf_models.py" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}
DNASE_RC_DIR="$(_find_dnase_rc_dir)" || {
  echo "Error: cannot locate resave_tf_models.py. Set DNASE_RC_DIR." >&2; exit 1; }
SCRIPT="${DNASE_RC_DIR}/resave_tf_models.py"
TF_CHROMBPNET_DIR="${TF_CHROMBPNET_DIR:-/path/to/workdir/software/chrombpnet}"

eval "$(conda shell.bash hook)"

echo "=== reference: the originals, under Python 3.7 (env bpnet, CPU)"
( conda activate bpnet && CUDA_VISIBLE_DEVICES="" python "${SCRIPT}" reference "$@" )

echo "=== resave: rebuild and check, under Python 3.8 (env chrombpnet, CPU)"
( conda activate chrombpnet && CUDA_VISIBLE_DEVICES="" python "${SCRIPT}" resave "$@" )

echo "=== verify: score each copy on its run's own test set (env chrombpnet, GPU)"
(
  conda activate chrombpnet
  export PYTHONPATH="${TF_CHROMBPNET_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load cuda/11.2.0 cudnn/8.1.1.33
  else
    for d in /usr/local/cuda-11.2/lib64 /usr/local/cudnn-linux-x86_64-8.1.1_cuda11-archive/lib64; do
      [[ -d "${d}" ]] && export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    done
  fi
  python "${SCRIPT}" verify "$@"
)
