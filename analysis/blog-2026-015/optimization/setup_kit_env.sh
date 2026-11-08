#!/bin/bash
# One-time setup for run_kit_modes.sh: Anthropic's ChromBPNet optimization kit
# (github.com/anthropics/uplifting-biomolecular-modeling, chrombpnet/) in a conda
# env, with its weights and data. Follows the kit's STOCK.md 'Stack' route for a
# fresh Python 3.8 environment, except that the torch stack goes into the env
# itself rather than /opt/torch (the kit imports torch from the default path
# first; /opt/torch needs root). Built with CONDA_OVERRIDE_GLIBC=2.17 and only
# manylinux2014-era wheels, so the env also runs on Sherlock's glibc 2.17 nodes.
#
#   bash setup_kit_env.sh        # idempotent: skips what is already there
#
# Needs network access (conda, PyPI, download.pytorch.org, encodeproject.org).

set -euo pipefail

OAK_DIR="/path/to/workdir"
CONDA="${CONDA:-${OAK_DIR}/software/miniconda3/bin/conda}"
ENV_NAME="${KIT_ENV:-chrombpnet-kit}"
ENV_DIR="$(dirname "$(dirname "${CONDA}")")/envs/${ENV_NAME}"
KIT="${KIT_ROOT:-${OAK_DIR}/software/uplifting-biomolecular-modeling}/chrombpnet"
STOCK_SRC="${KIT_STOCK_SRC:-${OAK_DIR}/software/uplifting-biomolecular-modeling-stock}"
KIT_HOME="${KIT_HOME:-${OAK_DIR}/chromatlas/test/chrombpnet-kit}"   # weights/ and data/
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

[[ -f "${KIT}/run.sh" ]] || git clone --depth 1 https://github.com/anthropics/uplifting-biomolecular-modeling.git "$(dirname "${KIT}")"

# --- the environment --------------------------------------------------------
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  CONDA_OVERRIDE_GLIBC=2.17 "${CONDA}" create -y -n "${ENV_NAME}" python=3.8 pip
fi
PY="${ENV_DIR}/bin/python"
PIP=(env -u PYTHONPATH PYTHONNOUSERSITE=1 "${PY}" -m pip install --no-cache-dir)
if ! "${PY}" -c "import tensorflow, chrombpnet, torch, triton, chrombpnet_opt" 2>/dev/null; then
  grep -E '^(pip|setuptools)==' "${KIT}/environment/requirements.lock" > "${WORK}/base.txt"
  "${PIP[@]}" -r "${WORK}/base.txt"
  grep -v -E '^(#|$)' "${KIT}/environment/requirements.lock" > "${WORK}/stack.txt"
  "${PIP[@]}" --no-deps -r "${WORK}/stack.txt"
  # The seven the kit's image takes from Ubuntu packages (STOCK.md 'Stack').
  "${PIP[@]}" six==1.14.0 requests==2.22.0 urllib3==1.25.8 idna==2.8 chardet==3.0.4 \
      certifi==2019.11.28 wheel==0.34.2
  # Stock chrombpnet 1.0.1, from the kit's source archive; keep the source
  # directory, which the kit's pin check reads.
  if [[ ! -d "${STOCK_SRC}/chrombpnet" ]]; then
    mkdir -p "${STOCK_SRC}"
    tar -xzf "${KIT}/stock/chrombpnet-eaa0fe58.tar.gz" -C "${STOCK_SRC}"
    mv "${STOCK_SRC}/chrombpnet-eaa0fe58" "${STOCK_SRC}/chrombpnet"
  fi
  "${PIP[@]}" --no-deps "${STOCK_SRC}/chrombpnet"
  # The Triton forward's stack (torch 2.4.1+cu124, triton 3.0.0), into the env.
  grep -v -E '^(#|$)' "${KIT}/environment/requirements-opt-torch.lock" > "${WORK}/torch.txt"
  "${PIP[@]}" --no-deps -r "${WORK}/torch.txt"
  # pip takes wheels for the building host's glibc: Pillow's (matplotlib's PNG
  # writer, which pred_bw's metrics use) then needs glibc 2.27. Take the same
  # version's manylinux2014 build, which runs on Sherlock's 2.17.
  PILLOW="$("${PY}" -c 'import PIL; print(PIL.__version__)')"
  env -u PYTHONPATH "${PY}" -m pip download --no-deps --only-binary=:all: --platform manylinux2014_x86_64 \
      --python-version 3.8 --implementation cp -d "${WORK}/pillow" "pillow==${PILLOW}"
  "${PIP[@]}" --no-deps --force-reinstall "${WORK}"/pillow/*.whl
fi

# --- weights: ENCODE ENCFF142IOR, fold 0 (GM12878 ATAC-seq) ------------------
W="${KIT_HOME}/weights/GM12878_ATAC/fold_0"
mkdir -p "${W}" "${KIT_HOME}/data"
if [[ ! -f "${W}/bias_scaled.h5" || ! -f "${W}/nobias.h5" ]]; then
  curl -L -o "${WORK}/ENCFF142IOR.tar.gz" \
      https://www.encodeproject.org/files/ENCFF142IOR/@@download/ENCFF142IOR.tar.gz
  tar -xzf "${WORK}/ENCFF142IOR.tar.gz" -C "${WORK}"
  cp "$(find "${WORK}" -name 'model.bias_scaled.fold_0.ENCSR637XSC.h5' | head -1)" "${W}/bias_scaled.h5"
  cp "$(find "${WORK}" -name 'model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5' | head -1)" "${W}/nobias.h5"
fi
# --- data: the observed bigWig the kit's configs name (GM12878 ATAC-seq) ------
[[ -f "${KIT_HOME}/data/ENCFF180XQC.bigWig" ]] || curl -L -o "${KIT_HOME}/data/ENCFF180XQC.bigWig" \
    https://www.encodeproject.org/files/ENCFF180XQC/@@download/ENCFF180XQC.bigWig

# --- the kit: editable install, pin check, the -cm model built and checked ---
cd "${KIT}"
env -u PYTHONPATH PATH="${ENV_DIR}/bin:${PATH}" PYTHONHASHSEED=0 CFLAGS=-g0 \
    bash run.sh install --weights "${KIT_HOME}/weights"
echo "kit env ready: ${ENV_DIR}; weights and data under ${KIT_HOME}"
