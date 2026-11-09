#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96GB
#SBATCH --job-name=chrombpnet_section4
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#
# Each of the blog's Section 4 changes measured on its own, old form against
# new, on one GPU node (benchmark_section4.py has the stages): every
# measurement in a fresh process, the forms in a different order each round,
# the file cache warmed first. ENCSR868FGK fold 0: the 20,000 peaks the pred_bw
# jobs predict, the fold's own models, its GC-matching inputs.
#
# From this folder, with site.env's cluster options and GPU constraints:
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_section4.sh slurm
#   sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_section4.sh slurm
#   python summarize_section4.py <OPTIMIZATION_OUT>   # also reads the kit-fused jobs' pred_bw totals
#
# Checkouts: CODE_MAIN (CHROMBPNET_REPO: chrombpnet-refactor) for the new forms,
# code-693d162 for the bigWig writer forms and the old predictions writer,
# the original package (ORIGINAL_REPO) for the old GC matching. CPU-only
# stages time the node's CPUs, so L40S and H200 nodes can differ there too.
#
# Usage: run_section4.sh [CLUSTER]
#   env: ROUNDS (3), ROUNDS_GC (2: the old matching takes ~10 min), STAGES
#        (gc,shared,bigwig,cast,h5), PT_ENV, CODE_MAIN, RUN_KIND (section4: the run tag's
#        suffix), RUN_TAG, OUT_ROOT (site.env's
#        OPTIMIZATION_OUT), ALLOW_BUSY_GPU
#   site.env: ORIGINAL_REPO, GENOME, CHROM_SIZES, REFERENCE_DIR, REFACTOR_MODELS,
#        OPTIMIZATION_OUT (unless OUT_ROOT is set); optional PT_ENV_GLIBC217 and the hook
#        pytorch_compiler (under Slurm)

set -eo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
HERE="${BLOG_SCRIPTS}/optimization"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"
resolve_cluster "${1:-}"
ROUNDS="${ROUNDS:-3}"
ROUNDS_GC="${ROUNDS_GC:-2}"
STAGES_ARG="${STAGES:-gc,shared,bigwig,cast,h5}"
FOLD_NUM=0
DEVICE=cuda
variants () {
  case "$1" in
    gc) echo "old new" ;;
    shared) echo "unshared shared" ;;
    bigwig) echo "list numpy fixed" ;;
    cast) echo "legacy batched" ;;
    h5) echo "gzip4 gzip9 gzip9threads gzip4threads" ;;
    *) echo "unknown stage '$1' (expected gc, shared, bigwig, cast, h5)" >&2; return 2 ;;
  esac
}
for stage in ${STAGES_ARG//,/ }; do variants "${stage}" > /dev/null; done

SITE_NEEDS=(ORIGINAL_REPO GENOME CHROM_SIZES REFERENCE_DIR REFACTOR_MODELS)
[[ -n "${OUT_ROOT:-}" ]] || SITE_NEEDS+=(OPTIMIZATION_OUT)
require_site "${SITE_NEEDS[@]}"
CODE_OLD="${HERE}/code-693d162"
CODE_MAIN="${CODE_MAIN:-${CHROMBPNET_REPO}}"
ORIGINAL="${ORIGINAL_REPO}"
FOLD="${REFACTOR_MODELS}/ATAC/ENCSR868FGK/fold_${FOLD_NUM}"
FOLD_JSON="${REFERENCE_DIR}/folds/fold_${FOLD_NUM}.json"
OUT_ROOT="${OUT_ROOT:-${OPTIMIZATION_OUT}}"
PT_ENV="$(default_pt_env)"
CANDIDATES="${FOLD}/data/output/ENCSR868FGK_auxiliary/candidates.bed"
FOREGROUND="${FOLD}/data/output/ENCSR868FGK_auxiliary/foreground.gc.bed"
for required in "${CODE_OLD}/chrombpnet/optimize.py" "${CODE_MAIN}/chrombpnet/CHROMBPNET.py" \
    "${ORIGINAL}/chrombpnet/helpers/make_gc_matched_negatives/get_gc_matched_negatives.py" \
    "${FOLD}/model/models/chrombpnet.pt" "${FOLD}/model/auxiliary/filtered.peaks.bed" \
    "${FOLD}/model/auxiliary/data_unstranded.bw" "${CANDIDATES}" "${FOREGROUND}" "${FOLD_JSON}" "${GENOME}"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 1; }
done

eval "$("$(conda_exe)" shell.bash hook)"
export LD_LIBRARY_PATH="$(ld_path_without_cuda)"
if [[ "${CLUSTER}" == slurm ]] && is_glibc217_env "${PT_ENV}"; then site_hook pytorch_compiler; fi
conda activate "${PT_ENV}"
export PYTHONNOUSERSITE=1

select_gpu
ARCH="cuda-${GPU_CC//./}"
GPU_SLUG="$(echo "${GPU_NAME}" | tr -cs '[:alnum:]' '-' | sed 's/^-*//; s/-*$//')"
RUN_TAG="${RUN_TAG:-${GPU_SLUG}_${ARCH}_${RUN_KIND:-section4}$(host_suffix)}"
start_results "${OUT_ROOT}/${RUN_TAG}/fold_${FOLD_NUM}"
mkdir -p "${WORK}/cwd"
RESULTS="${WORK}/results.jsonl"
: > "${RESULTS}"

( cd "${WORK}/cwd" && PYTHONPATH="${CODE_MAIN}" check_framework_gpu torch "${WORK}/gpu_check_pt.log" ) || exit 1

# The 20,000 peaks, drawn as the pred_bw jobs draw them.
awk -v n=20000 'BEGIN{srand(1234)} {if (NR<=n) r[NR]=$0;
     else {i=int(rand()*NR)+1; if (i<=n) r[i]=$0}}
     END{for (j=1;j<=n && j<=NR;j++) print r[j]}' "${FOLD}/model/auxiliary/filtered.peaks.bed" \
  | sort -k1,1 -k2,2n > "${WORK}/regions.bed"

echo "=================================================================="
echo " run tag : ${RUN_TAG}    host $(hostname) (${CLUSTER})    GPU ${GPU_ID}: ${GPU_NAME} (${ARCH})"
echo " load    : $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) CPUs    env ${PT_ENV}"
echo " stages  : ${STAGES_ARG}    rounds ${ROUNDS} (gc ${ROUNDS_GC})"
echo " output  : ${WORK}"
echo "=================================================================="

echo "### inputs: the fold's ChromBPNet predictions for the 20,000 peaks (one untimed pred_bw)"
( cd "${WORK}/cwd" && PYTHONPATH="${CODE_MAIN}" python -m chrombpnet.CHROMBPNET pred_bw \
    -cm "${FOLD}/model/models/chrombpnet.pt" -r "${WORK}/regions.bed" -g "${GENOME}" -c "${CHROM_SIZES}" \
    -bw "${FOLD}/model/auxiliary/data_unstranded.bw" -op "${WORK}/pred" -bs 256 -t 0 --device cuda ) \
  > "${WORK}/pred.log" 2>&1 || { tail -20 "${WORK}/pred.log"; exit 1; }
cat > "${WORK}/inputs.json" <<EOF
{"regions": "${WORK}/regions.bed", "genome": "${GENOME}", "chrom_sizes": "${CHROM_SIZES}",
 "observed_bigwig": "${FOLD}/model/auxiliary/data_unstranded.bw", "chrombpnet_pt": "${FOLD}/model/models/chrombpnet.pt",
 "preds_bed": "${WORK}/pred_chrombpnet_preds.bed", "predictions_h5": "${WORK}/pred_chrombpnet_predictions.h5",
 "candidates": "${CANDIDATES}", "foreground_gc": "${FOREGROUND}", "fold_json": "${FOLD_JSON}",
 "original_gc_script": "${ORIGINAL}/chrombpnet/helpers/make_gc_matched_negatives/get_gc_matched_negatives.py",
 "old_predictions_py": "${CODE_OLD}/chrombpnet/predictions.py"}
EOF
echo "### warm-up: file cache"
cat "${GENOME}" "${CANDIDATES}" "${FOREGROUND}" "${FOLD}/model/auxiliary/data_unstranded.bw" \
    "${WORK}/pred_chrombpnet_predictions.h5" > /dev/null

checkout () {  # stage variant -> the checkout to import chrombpnet from
  case "$1" in
    bigwig) echo "${CODE_OLD}" ;;
    *) echo "${CODE_MAIN}" ;;
  esac
}

measure () {  # stage variant round position
  local stage="$1" variant="$2" line rc
  # A fresh process per measurement, importing chrombpnet from the stage's checkout.
  line="$(cd "${WORK}/cwd" && PYTHONPATH="$(checkout "${stage}" "${variant}")" \
          python "${HERE}/benchmark_section4.py" "${stage}" "${variant}" "${WORK}" \
          2>> "${WORK}/${stage}_${variant}.log" | tail -1)" && rc=0 || rc=$?
  if [[ ${rc} -eq 0 && "${line}" == "{"* ]]; then
    echo "${line}" | python -c 'import json, sys; d = json.loads(sys.stdin.read()); d.update(round=int(sys.argv[1]), position=int(sys.argv[2]), run=sys.argv[3]); print(json.dumps(d))' \
      "$3" "$4" "${RUN_TAG}" >> "${RESULTS}"
    printf "  %-7s %-13s %s\n" "${stage}" "${variant}" "$(echo "${line}" | python -c 'import json,sys; d=json.loads(sys.stdin.read()); print("%(seconds)8.2f s  peak %(rss_peak_mb)7.0f MB  +%(rss_added_mb)6.0f MB  %(out_bytes)s B" % d)')"
  else
    printf "  %-7s %-13s FAILED (log %s)\n" "${stage}" "${variant}" "${WORK}/${stage}_${variant}.log"
    tail -5 "${WORK}/${stage}_${variant}.log" | sed 's/^/  | /'
  fi
}

for stage in ${STAGES_ARG//,/ }; do
  read -r -a FORMS <<< "$(variants "${stage}")"
  n=${#FORMS[@]}
  rounds=${ROUNDS}; [[ "${stage}" == gc ]] && rounds=${ROUNDS_GC}
  echo "### ${stage}: ${FORMS[*]}, ${rounds} rounds"
  for (( round = 0; round < rounds; round++ )); do
    for (( i = 0; i < n; i++ )); do
      measure "${stage}" "${FORMS[$(( (i + round) % n ))]}" "${round}" "${i}"
    done
    rm -rf "${WORK}/out"
  done
done
rm -rf "${WORK}/cwd"
echo "results: ${RESULTS}"
finish_results
