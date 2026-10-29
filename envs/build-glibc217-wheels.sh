#!/bin/bash
# Builds the two wheels chrombpnet-pytorch-sherlock.yml needs that PyPI has
# for glibc 2.28 and newer only -- igraph 1.0.0 and leidenalg 0.11.0, which
# modisco pins -- for glibc 2.17 (manylinux2014), in the PyPA manylinux2014
# image (CentOS 7, gcc 10) under apptainer or singularity. Needs network
# access to PyPI and GitHub.
#
#   envs/build-glibc217-wheels.sh <wheel dir>
#   IMAGE=<local .sif> envs/build-glibc217-wheels.sh <wheel dir>
#
# leidenalg links a shared igraph C core and libleidenalg, which its sdist
# does not include: its scripts/build_igraph.sh clones the core from git,
# whose parsers need flex and bison, which the image lacks. The igraph sdist
# vendors the same core (1.0.0) with the parsers generated, so the core is
# built from there with build_igraph.sh's options. auditwheel copies both
# libraries into the leidenalg wheel.
set -euo pipefail
OUT="${1:?usage: build-glibc217-wheels.sh <wheel dir>}"
IMAGE="${IMAGE:-docker://quay.io/pypa/manylinux2014_x86_64}"
mkdir -p "${OUT}"
OUT="$(realpath "${OUT}")"
RUNNER="$(command -v apptainer || command -v singularity)" || { echo "Error: needs apptainer or singularity" >&2; exit 1; }
WORK="$(mktemp -d "${TMPDIR:-/tmp}/glibc217-wheels.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

cat > "${WORK}/build.sh" <<'EOF'
set -euo pipefail
export HOME=/work TMPDIR=/work PIP_CACHE_DIR=/work/pip-cache
PY=/opt/python/cp311-cp311/bin/python
cd /work
"${PY}" -m pip download --no-deps --no-binary igraph,leidenalg igraph==1.0.0 leidenalg==0.11.0 -d sdist
tar xzf sdist/igraph-1.0.0.tar.gz
tar xzf sdist/leidenalg-0.11.0.tar.gz
DEPS=/work/leidenalg-0.11.0/build-deps
cmake -S igraph-1.0.0/vendor/source/igraph -B core-build \
  -DCMAKE_INSTALL_PREFIX="${DEPS}/install" -DBUILD_SHARED_LIBS=ON \
  -DIGRAPH_GLPK_SUPPORT=OFF -DIGRAPH_GRAPHML_SUPPORT=OFF -DIGRAPH_OPENMP_SUPPORT=OFF \
  -DIGRAPH_USE_INTERNAL_BLAS=ON -DIGRAPH_USE_INTERNAL_LAPACK=ON -DIGRAPH_USE_INTERNAL_ARPACK=ON \
  -DIGRAPH_USE_INTERNAL_GLPK=OFF -DIGRAPH_USE_INTERNAL_GMP=ON -DIGRAPH_WARNINGS_AS_ERRORS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build core-build --config Release -j "$(nproc)"
cmake --build core-build --target install --config Release
(cd leidenalg-0.11.0 && bash scripts/build_libleidenalg.sh)
"${PY}" -m pip wheel --no-deps --no-binary igraph -w raw igraph==1.0.0 ./leidenalg-0.11.0
export LD_LIBRARY_PATH="${DEPS}/install/lib:${DEPS}/install/lib64"
for wheel in raw/*.whl; do
  auditwheel repair --plat manylinux2014_x86_64 -w /out "${wheel}"
done
EOF
"${RUNNER}" exec --cleanenv --bind "${WORK}:/work" --bind "${OUT}:/out" "${IMAGE}" bash /work/build.sh
ls -l "${OUT}"
