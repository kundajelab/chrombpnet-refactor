#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --job-name=chrombpnet_optimization
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --array=0-4
#
# Measure what --predict-fk is worth, on the five folds trained by
# run/run_chrombpnet_refactor_benchmark_260911.sh.
#
# Pick the card on the sbatch line (the September sweep ran on H100s: site.env's
# SBATCH_H100), from this folder:
#
#   ./list_gpu_skus.sh <partitions>     # what the partitions actually offer
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H100} run_chrombpnet_optimization_benchmark.sh slurm
#
# Use a feature name that command prints, verbatim. Slurm does not fail a submission
# whose constraint matches no node -- it accepts the job and leaves it
# pending forever with reason BadConstraints, which looks like a queue wait
# rather than a mistake. Check with `squeue -u $USER` after submitting; if
# the reason column says BadConstraints, the feature name is wrong or that
# card is not in --partition.
#
# The card matters for the 'fk' arm and only for it. The Triton kernels ship
# a per-architecture tile table, and on A100 (cuda-80) the route REFUSES by
# design: the kit's own table records that sm_80 has no bitwise route, because
# K1's fp32 kernels do not reproduce that card's log-counts at every batch
# shape. So expect 'fk' to be recorded as "refused" on A100 and to run on
# H100/H200. That is the comparison, not a failure -- the script records the
# refusal reason and carries on.
#
# Everything here runs on real models and real data: this fold's own three
# .pt checkpoints and its own peak regions.
#
#   pred_bw      timed per arm; reaches the fused forward through
#                PredictionAdapter.
#   contribs_bw  timed ONCE. It has no --predict-fk: deep_lift_shap hooks the
#                ReLU modules of the raw model and needs an autograd graph,
#                which the fused kernels have not got.
#   footprints   timed per arm; also reaches the fused forward.
#
# Usage:  sbatch [--constraint=...] run_chrombpnet_optimization_benchmark.sh \
#             CLUSTER [FOLD_NUM] [N_REGIONS] [ARMS]
#   CLUSTER    slurm (default) | local
#   FOLD_NUM   0-4, which fold to run (default SLURM_ARRAY_TASK_ID)
#   N_REGIONS  regions to subsample from the fold's peaks (default 20000;
#              'all' uses every peak, ~270k, which is roughly 13x slower)
#   ARMS       comma-separated list of arms to time: 'default' (how the
#              package now runs) and 'fk' (--predict-fk, the fused Triton
#              forward). The stepwise --optimize phase matrix this script
#              used to sweep was removed from the CLI once its measurement
#              was complete; to re-run that sweep, check out 693d162.
#              (default "none,1,2,3,1|3,all"; '|' stands in for ',' within an
#              element, so 1|3 means --optimize 1,3)
#   env: CONDA_ENV, BENCHMARK_LEVEL (lite), ASSAY (ATAC), ENCID (ENCSR868FGK), N_CONTRIB
#        (2000), MODELS_ROOT, OUT_ROOT (site.env's OPTIMIZATION_ANALYSIS), TRITON_CACHE_DIR
#        (.triton-cache/<glibc> beside OUT_ROOT)
#   site.env: GENOME, CHROM_SIZES, REFERENCE_DIR, BENCHMARK_RUNS (unless MODELS_ROOT is
#        set), OPTIMIZATION_ANALYSIS (unless OUT_ROOT is set); optional PT_ENV_GLIBC217
#        and the hook pytorch_compiler (under Slurm)

set -euo pipefail

CLUSTER="${1:-slurm}"
FOLD_NUM="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
N_REGIONS="${3:-20000}"
# DeepLIFT costs far more per region than a forward, so contribs_bw gets its
# own, smaller subsample taken from the head of the same region list.
N_CONTRIB="${N_CONTRIB:-2000}"
ARMS_ARG="${4:-default,fk}"

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
SITE_NEEDS=(GENOME CHROM_SIZES REFERENCE_DIR)
[[ -n "${MODELS_ROOT:-}" ]] || SITE_NEEDS+=(BENCHMARK_RUNS)
[[ -n "${OUT_ROOT:-}" ]] || SITE_NEEDS+=(OPTIMIZATION_ANALYSIS)
require_site "${SITE_NEEDS[@]}"
REPO="${CHROMBPNET_REPO}"
GENOME_PATH="${GENOME}"
CHROM_PATH="${CHROM_SIZES}"

# Where the benchmark run left its trained folds. Mirrors that script's
# ENCID_DIR: <BENCHMARK_RUNS>/chrombpnet-refactor-${BENCHMARK_LEVEL}-cleanup/${ASSAY}/${ENCID}
BENCHMARK_LEVEL="${BENCHMARK_LEVEL:-lite}"
ASSAY="${ASSAY:-ATAC}"
ENCID="${ENCID:-ENCSR868FGK}"
MODELS_ROOT="${MODELS_ROOT:-${BENCHMARK_RUNS}/chrombpnet-refactor-${BENCHMARK_LEVEL}-cleanup/${ASSAY}/${ENCID}}"

OUT_ROOT="${OUT_ROOT:-${OPTIMIZATION_ANALYSIS}}"

# Environment (CONDA_ENV overrides):
#
#   chrombpnet-pytorch            torch 2.14.1+cu126  the default; needs glibc >= 2.28
#                                                     and a CUDA >= 12.6 driver
#   PT_ENV_GLIBC217 (site.env)   torch 2.5.1+cu118   a build for glibc 2.17 nodes; earlier runs
#
# This used to pick chrombpnet-refactor (torch 2.12 / CUDA 13) on drivers that
# support CUDA 13; that env is gone.
DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
DRIVER_CUDA_MAJOR="${DRIVER_CUDA%%.*}"
if [[ -z "${CONDA_ENV:-}" ]]; then
  CONDA_ENV="chrombpnet-pytorch"
fi
# chrombpnet-pytorch (PyTorch 2.14) needs glibc >= 2.28 (engines/envs/glibc_scan.py:
# torch and igraph); some clusters' GPU nodes (the blog's L40S ones) run 2.17.
# Stop here on such a node rather than failing at the first import.
is_glibc217_env () { [[ -n "${PT_ENV_GLIBC217:-}" && "$1" == "${PT_ENV_GLIBC217}" ]]; }
GLIBC_FLOOR="${GLIBC_FLOOR:-2.28}"
if ! is_glibc217_env "${CONDA_ENV}"; then
  NODE_GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
  if [[ -n "${NODE_GLIBC}" && "$(printf '%s\n%s\n' "${GLIBC_FLOOR}" "${NODE_GLIBC}" | sort -V | head -1)" != "${GLIBC_FLOOR}" ]]; then
    echo "Error: ${CONDA_ENV} needs glibc >= ${GLIBC_FLOOR}; this node ($(hostname)) has ${NODE_GLIBC}." >&2
    echo "       Resubmit with --exclude=$(hostname), or set CONDA_ENV to a glibc 2.17 build (site.env PT_ENV_GLIBC217)." >&2
    exit 1
  fi
fi
if [[ "${CLUSTER}" == slurm ]] && is_glibc217_env "${CONDA_ENV}"; then
  site_hook pytorch_compiler
fi

FOLD_DIR="${MODELS_ROOT}/fold_${FOLD_NUM}"
MODEL_DIR="${FOLD_DIR}/model"

echo "conda environment: ${CONDA_ENV}"
eval "$("${CONDA_EXE:-${CONDA_ROOT:+${CONDA_ROOT}/bin/}conda}" shell.bash hook)"
conda activate "${CONDA_ENV}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Per-job Triton cache on scratch. The kit's precompiled caches were built for
# torch 2.4.1 / triton 3.0.0 and will not match this stack, so 'fk' JITs its
# kernels on first use. A shared cache means only the first job of a sweep pays
# it; a $HOME cache would have array tasks racing each other.
# Triton caches the helpers it compiles (cuda_utils.so) alongside the kernels,
# built against the host's glibc. A cluster's GPU nodes need not all run the
# same OS: a build on an H200 node (glibc 2.34) failed to load on an L40S node
# (2.17), which is what failed every fused-kernel run on the L40S. One cache per
# glibc version, shared by the array's tasks.
GLIBC_VERSION="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(dirname "${OUT_ROOT}")/.triton-cache/glibc-${GLIBC_VERSION:-unknown}}"
mkdir -p "${TRITON_CACHE_DIR}"

# --------------------------------------------------------------- preflight
# One clear failure beats six identical ones. Everything below depends on the
# package importing, the CLI responding and CUDA being usable.
echo "### preflight"
preflight_fail () { echo "PREFLIGHT FAILED: $1" >&2; exit 1; }
python -c "import chrombpnet" 2>/dev/null || preflight_fail \
  "'import chrombpnet' fails in env ${CONDA_ENV}. Is the package installed there?"
python -m chrombpnet.CHROMBPNET pred_bw --help >/dev/null 2>&1 || preflight_fail \
  "'chrombpnet pred_bw --help' fails; the CLI is broken in this environment"
python -m chrombpnet.CHROMBPNET pred_bw --help 2>&1 | grep -q -- "--predict-fk" || preflight_fail \
  "this chrombpnet has no --predict-fk flag; the repo at ${REPO} predates it, or a\n  different chrombpnet is on the path (python -c 'import chrombpnet; print(chrombpnet.__file__)')"
# A CUDA_VISIBLE_DEVICES that names a device this host has not got is the
# single most confusing way for this to fail: torch reports zero devices and
# an empty arch list, which looks exactly like a broken build. Check it
# before blaming the environment.
VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${VISIBLE_GPUS}" -gt 0 ]]; then
  for index in ${CUDA_VISIBLE_DEVICES//,/ }; do
    if [[ "${index}" =~ ^[0-9]+$ && "${index}" -ge "${VISIBLE_GPUS}" ]]; then
      preflight_fail "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' names GPU ${index},
  but $(hostname) has ${VISIBLE_GPUS} GPU(s), numbered 0..$((VISIBLE_GPUS - 1)).
  torch will report zero devices and an empty arch list, which looks like a
  broken install but is not. Set CUDA_VISIBLE_DEVICES to a valid index --
  often this comes from a per-host branch in ~/.bashrc that has no entry for
  this machine and falls through to a default."
    fi
  done
fi

TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '?')"
python - <<'PYEOF' || preflight_fail "CUDA is not usable from environment '${CONDA_ENV}'.
  torch here is built for CUDA ${TORCH_CUDA}; this machine's driver supports
  CUDA ${DRIVER_CUDA:-unknown}. If the first is newer than the second, that is
  the cause: torch then reports no devices at all rather than erroring. If
  they match, look instead for a GPU that is busy, hidden by
  CUDA_VISIBLE_DEVICES, or otherwise unavailable to this process.
  Note cu118 ships sm_90 kernels, so an H100/H200 is NOT the problem.
  Run ./diagnose_cuda.sh in this environment: it reports the cause that
  torch.cuda.is_available() swallows (driver library, device visibility,
  compute mode, or the real init error)."
import chrombpnet, torch, sys
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False", file=sys.stderr); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
PYEOF
echo "  ok: chrombpnet imports, CLI has --predict-fk, CUDA usable"
echo "  package: $(python -c 'import chrombpnet; print(chrombpnet.__file__)')"

# --------------------------------------------------------------- inputs
CHROMBPNET_PT="${MODEL_DIR}/models/chrombpnet.pt"
NOBIAS_PT="${MODEL_DIR}/models/chrombpnet_nobias.pt"
BIAS_PT="${MODEL_DIR}/models/bias_model_scaled.pt"
PEAKS_BED="${MODEL_DIR}/auxiliary/filtered.peaks.bed"
NONPEAKS_BED="${MODEL_DIR}/auxiliary/filtered.nonpeaks.bed"
OBSERVED_BW="${MODEL_DIR}/auxiliary/data_unstranded.bw"
MOTIFS_TSV="${MODEL_DIR}/auxiliary/motif_to_pwm.tsv"
FOLD_JSON="${REFERENCE_DIR}/folds/fold_${FOLD_NUM}.json"

for required in "${CHROMBPNET_PT}" "${NOBIAS_PT}" "${BIAS_PT}" "${PEAKS_BED}" \
                "${NONPEAKS_BED}" "${OBSERVED_BW}" "${MOTIFS_TSV}" \
                "${FOLD_JSON}" "${GENOME_PATH}" "${CHROM_PATH}"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 1; }
done

# Clear the previous run's directory. On NFS a file still open elsewhere
# leaves a .nfsXXXX silly-rename behind that cannot be unlinked, and under
# `set -e` that would take the whole benchmark down before it starts. Move
# the remains aside instead of failing.
# --------------------------------------------------------------- identity
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
ARCH="cuda-${GPU_CC//./}"

# Every run gets its own directory. Without this, two sweeps of the same fold
# on different cards both resolve to results/fold_N -- and since the run
# clears that directory on startup, whichever started second would delete the
# first one's results and logs while it was still writing them.
#
# The tag is the card, its architecture and the device index, which is what
# actually distinguishes concurrent runs. Override RUN_TAG to name a run
# yourself (e.g. RUN_TAG=h200-baseline).
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
DEVICE_SLUG="dev${CUDA_VISIBLE_DEVICES:-all}"
DEVICE_SLUG="${DEVICE_SLUG//,/-}"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}_${DEVICE_SLUG}}"

WORK="${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
if [[ -d "${WORK}" ]] && ! rm -rf "${WORK}" 2>/dev/null; then
  stale="${WORK}.stale.$(date +%s)"
  echo "note: could not remove ${WORK} (files still open?); moving it to ${stale}"
  mv "${WORK}" "${stale}"
fi
mkdir -p "${WORK}"
SUMMARY="${WORK}/timings.tsv"
printf "run\tfold\tgpu\tarch\tstage\tarm\tseconds\tstatus\tdetail\n" > "${SUMMARY}"

echo "=================================================================="
echo " run tag         : ${RUN_TAG}"
echo " fold            : ${FOLD_NUM}"
echo " GPU             : ${GPU_NAME}  (compute ${GPU_CC} -> ${ARCH})"
echo " requested SKU   : ${SLURM_JOB_CONSTRAINTS:-<none; not under slurm>}"
echo " peaks available : $(wc -l < "${PEAKS_BED}")"
echo " regions used    : ${N_REGIONS}"
echo " output dir      : ${WORK}"
echo " repo            : ${REPO}"
echo " driver CUDA     : ${DRIVER_CUDA:-unknown}"
echo " conda env       : ${CONDA_ENV}"
echo " torch           : $(python -c 'import torch; print(torch.__version__, "for CUDA", torch.version.cuda)' 2>/dev/null || echo unavailable)"
echo " triton cache    : ${TRITON_CACHE_DIR}"
echo "=================================================================="

# Subsample the peaks deterministically. The full set is ~270k regions, which
# at three models is on the order of a couple of hours per arm; 20k keeps the
# sweep to roughly ten minutes while staying far above the point
# where per-run overhead would dominate.
REGIONS="${WORK}/regions.bed"
if [[ "${N_REGIONS}" == "all" ]]; then
  cp "${PEAKS_BED}" "${REGIONS}"
else
  awk -v n="${N_REGIONS}" 'BEGIN{srand(1234)} {if (NR<=n) r[NR]=$0;
       else {i=int(rand()*NR)+1; if (i<=n) r[i]=$0}}
       END{for (j=1;j<=n && j<=NR;j++) print r[j]}' "${PEAKS_BED}" \
    | sort -k1,1 -k2,2n > "${REGIONS}"
fi
echo "regions to predict: $(wc -l < "${REGIONS}")"

IFS=',' read -r -a ARMS <<< "${ARMS_ARG}"

# An arm name -> the flags it adds. 'default' adds none: phases 1 and 3 are
# now simply how the package runs.
arm_flags () {
  case "$1" in
    default) printf '' ;;
    fk)      printf -- '--predict-fk' ;;
    *)       echo "unknown arm '$1' (expected: default, fk)" >&2; exit 2 ;;
  esac
}

record () {  # stage arm seconds status detail
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${RUN_TAG}" "${FOLD_NUM}" "${GPU_NAME}" "${ARCH}" "$1" "$2" "$3" "$4" "$5" >> "${SUMMARY}"
}

# ============================================================ 1. pred_bw
echo
echo "### 1/3  pred_bw on fold ${FOLD_NUM}'s own models and peaks"
for arm in "${ARMS[@]}"; do
  read -r -a FLAGS <<< "$(arm_flags "${arm}")"
  outdir="${WORK}/pred_bw/${arm}"
  rm -rf "${outdir}" 2>/dev/null || mv "${outdir}" "${outdir}.stale.$(date +%s)"
  mkdir -p "${outdir}"

  start=$(date +%s.%N)
  set +e
  python -m chrombpnet.CHROMBPNET pred_bw \
      "${FLAGS[@]}" \
      -cm  "${CHROMBPNET_PT}" \
      -cmb "${NOBIAS_PT}" \
      -bm  "${BIAS_PT}" \
      -r   "${REGIONS}" \
      -g   "${GENOME_PATH}" \
      -c   "${CHROM_PATH}" \
      -bw  "${OBSERVED_BW}" \
      -op  "${outdir}/pred" \
      -os  "${outdir}/stats" \
      -bs 256 -t 0 --device cuda \
      > "${outdir}/run.log" 2>&1
  rc=$?
  set -e
  elapsed=$(awk -v a="$(date +%s.%N)" -v b="${start}" 'BEGIN{printf "%.1f", a-b}')

  if [[ ${rc} -eq 0 ]]; then
    record pred_bw "${arm}" "$(printf '%.1f' "${elapsed}")" ok ""
    printf "  %-10s %7.1fs\n" "${arm}" "${elapsed}"
  elif grep -q "TritonRouteRefused" "${outdir}/run.log"; then
    # Expected on A100: no bitwise Triton route for sm_80.
    reason=$(grep -m1 -o "TritonRouteRefused.*" "${outdir}/run.log" | cut -c1-160)
    record pred_bw "${arm}" "" refused "${reason}"
    printf "  %-10s REFUSED  %s\n" "${arm}" "${reason:0:90}"
  else
    record pred_bw "${arm}" "" failed "see ${outdir}/run.log"
    printf "  %-10s FAILED   (exit %d)\n" "${arm}" "${rc}"
    echo "  ---- last 20 lines of ${outdir}/run.log ----"
    tail -20 "${outdir}/run.log" | sed 's/^/  | /'
    echo "  --------------------------------------------"
  fi
done

# Which artifacts the fused-kernel arm moved, relative to the default.
if [[ -d "${WORK}/pred_bw/default" ]]; then
  echo
  # Borrows the digest helpers from tests/regression/artifacts.py -- not the
  # harness itself -- so bigWigs, HDF5 and PNGs are compared by content
  # rather than raw bytes, and a difference is described rather than just
  # flagged.
  echo "### artifacts differing from the default arm"
  ( cd "${REPO}" && python - "${WORK}/pred_bw" <<'PYEOF'
import pathlib, sys
sys.path.insert(0, ".")
try:
    from tests.regression import artifacts
except Exception as exc:          # comparison is a bonus; timing is the point
    print(f"  (artifact comparison unavailable: {exc})")
    sys.exit(0)
root = pathlib.Path(sys.argv[1])
def manifest(d):
    return artifacts.manifest(d, ignore=("run.log",),
                              substitutions=((str(d), "<RUN>"),))
base_dir = root / "none"
if not any(base_dir.iterdir()):
    sys.exit(0)
base = manifest(base_dir)
for child in sorted(root.iterdir()):
    if child.name == "none" or not child.is_dir() or not any(child.iterdir()):
        continue
    other = manifest(child)
    differing = sorted(k for k in base if base[k] != other.get(k))
    print(f"  {child.name:8} {len(differing):2d}/{len(base)} differ"
          + (f": {', '.join(differing[:4])}" if differing else ""))
PYEOF
  ) | tee "${WORK}/artifact_diffs.txt"
fi

# ==================================================== 2. contribs_bw
# contribs_bw has no --predict-fk: deep_lift_shap hooks the ReLU modules of
# the raw module and needs an autograd graph, which a fused kernel has not
# got. So it is timed once rather than per arm.
#
# DeepLIFT is far slower per region than a forward, so this uses its own,
# smaller subsample.
CONTRIB_REGIONS="${WORK}/contrib_regions.bed"
head -n "${N_CONTRIB}" "${REGIONS}" > "${CONTRIB_REGIONS}"
echo
echo "### 2/3  contribs_bw ($(wc -l < "${CONTRIB_REGIONS}") regions)"
for arm in default; do
  outdir="${WORK}/contribs_bw/${arm}"
  rm -rf "${outdir}" 2>/dev/null || mv "${outdir}" "${outdir}.stale.$(date +%s)"
  mkdir -p "${outdir}"
  start=$(date +%s.%N)
  set +e
  python -m chrombpnet.CHROMBPNET contribs_bw \
      -m  "${NOBIAS_PT}" -r "${CONTRIB_REGIONS}" \
      -g  "${GENOME_PATH}" -c "${CHROM_PATH}" \
      -op "${outdir}/contrib" -pc profile \
      -t 0 --device cuda > "${outdir}/run.log" 2>&1
  rc=$?
  set -e
  elapsed=$(awk -v a="$(date +%s.%N)" -v b="${start}" 'BEGIN{printf "%.1f", a-b}')
  if [[ ${rc} -eq 0 ]]; then
    record contribs_bw "${arm}" "${elapsed}" ok ""
    printf "  %-10s %7.1fs\n" "${arm}" "${elapsed}"
  else
    record contribs_bw "${arm}" "" failed "see ${outdir}/run.log"
    printf "  %-10s FAILED   (exit %d)\n" "${arm}" "${rc}"
    tail -20 "${outdir}/run.log" | sed 's/^/  | /'
  fi
done

# ==================================================== 3. footprints
# Reaches the Triton forward through PredictionAdapter, so --predict-fk
# applies here as it does to pred_bw.
echo
echo "### 3/3  footprints"
for arm in "${ARMS[@]}"; do
  read -r -a FLAGS <<< "$(arm_flags "${arm}")"
  outdir="${WORK}/footprints/${arm}"
  rm -rf "${outdir}" 2>/dev/null || mv "${outdir}" "${outdir}.stale.$(date +%s)"
  mkdir -p "${outdir}"
  start=$(date +%s.%N)
  set +e
  python -m chrombpnet.CHROMBPNET footprints \
      "${FLAGS[@]}" \
      -m "${NOBIAS_PT}" -r "${NONPEAKS_BED}" \
      -g "${GENOME_PATH}" -fl "${FOLD_JSON}" \
      -op "${outdir}/fp" -pwm_f "${MOTIFS_TSV}" \
      -bs 256 --device cuda > "${outdir}/run.log" 2>&1
  rc=$?
  set -e
  elapsed=$(awk -v a="$(date +%s.%N)" -v b="${start}" 'BEGIN{printf "%.1f", a-b}')
  if [[ ${rc} -eq 0 ]]; then
    record footprints "${arm}" "${elapsed}" ok ""
    printf "  %-10s %7.1fs\n" "${arm}" "${elapsed}"
  elif grep -q "TritonRouteRefused" "${outdir}/run.log"; then
    reason=$(grep -m1 -o "TritonRouteRefused.*" "${outdir}/run.log" | cut -c1-160)
    record footprints "${arm}" "" refused "${reason}"
    printf "  %-10s REFUSED  %s\n" "${arm}" "${reason:0:90}"
  else
    record footprints "${arm}" "" failed "see ${outdir}/run.log"
    printf "  %-10s FAILED   (exit %d)\n" "${arm}" "${rc}"
    tail -20 "${outdir}/run.log" | sed 's/^/  | /'
  fi
done

echo "=================================================================="
echo " summary: ${SUMMARY}"
echo "=================================================================="
column -t -s $'\t' "${SUMMARY}"
