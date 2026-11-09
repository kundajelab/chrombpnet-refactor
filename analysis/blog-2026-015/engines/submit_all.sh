#!/bin/bash
# Submit the whole sweep: the shared nonpeaks, then one training array per
# engine. Or, with --contribution, one run_contribution.sh array per engine.
#
#   ./submit_all.sh slurm --container                   # submit the sweep
#   ./submit_all.sh slurm --container --dry-run         # print the sbatch lines only
#   ./submit_all.sh slurm --container --contribution   # rescore finished runs' subsamples
#   ./submit_all.sh slurm --original-retrain           # the original, retrained per row
#
# (slurm is the default; sherlock, the name these were submitted with, means
# the same.) Every job gets site.env's SBATCH_SITE_ARGS, and its logs go to
# site.env's LOG_DIR, or to this folder without one (_common.sh: site_sbatch).
#
#   --container      run every step inside the Ubuntu 22.04 image (_common.sh):
#                    required for JAX on a glibc older than 2.27, and it gives
#                    every engine the same system tools, so their runtimes
#                    compare. Without it, a submission that includes JAX from
#                    such a host is refused.
#   --contribution  submit only run_contribution.sh, one array per engine,
#                    with the same per-engine sbatch options as training
#   --deepshap-seeds submit only run_deepshap_seeds.sh: the original chrombpnet's
#                    DeepSHAP on each row's original model, one array per seed
#                    (SEEDS, default "1 2 3"; N_REGIONS; see that script), with
#                    the TensorFlow jobs' sbatch options. ENGINES does not apply.
#   --original-retrain submit only run_original_retrain.sh: the original
#                    chrombpnet trained on each row's sweep inputs and scored
#                    by its own DeepSHAP (SEEDS, default "1"; see that script),
#                    with the TensorFlow jobs' sbatch options. ENGINES does not
#                    apply; the nonpeaks and the PyTorch runs' subsamples must
#                    exist. Natively only: not with --container.
#
#   run_prep_nonpeaks.sh            array 0-19, CPU
#   run_train_full.sh <engine>      array 0-19, GPU, one array per engine
#   run_contribution.sh <engine>   array 0-19, GPU (--contribution only)
#   run_deepshap_seeds.sh           array 0-19, GPU (--deepshap-seeds only)
#   run_original_retrain.sh         array 0-19, GPU (--original-retrain only)
#
# Each training task waits only for the prep task of its own row
# (--dependency=aftercorr), and is cancelled if that prep task fails
# (--kill-on-invalid-dep). A prep task whose nonpeaks already exist exits at
# once, so re-running this script after a partial sweep is safe; so is
# resubmitting a training array, whose completed tasks skip themselves.
#
# Environment:
#   ENGINES="jax"      only these engines (default: pytorch tensorflow jax)
#   ARRAY=0-4          only these rows (default: every row; 0-based)
#   SKIP_PREP=1        do not submit the prep array (nonpeaks already exist)
#   GPU_CONSTRAINT=..  passed to every engine's training jobs as
#                      --constraint, e.g. one GPU model's feature name, so all
#                      engines run on one GPU type and their runtimes compare
#                      (your cluster lists its node features)
#   SBATCH_EXTRA=".."  more sbatch options for every GPU job, word-split, e.g.
#                      "--exclude=<nodes>"
#
# From site.env (an environment variable of the same name overrides it; an
# empty one drops it):
#   TF_GPU_CONSTRAINT  the TensorFlow jobs' constraint when GPU_CONSTRAINT is
#                      unset: GPUs of 16 GB or more. TensorFlow's DeepSHAP runs
#                      in a second process beside the one that trained, and the
#                      two do not fit 12 GB. (The sweep's: compute capability
#                      6.0, 7.0, 8.x and 9.0, leaving out 6.1, a 12 GB TITAN Xp,
#                      and 7.5, an 11 GB RTX 2080 Ti.)
#   JAX_GPU_CONSTRAINT the JAX jobs' constraint when GPU_CONSTRAINT is unset:
#                      compute capability 7.5 or newer, the GPUs the JAX env's
#                      cuDNN 9.27 has kernels for (not Pascal or Volta).
#   SBATCH_SHARED_GPU_ARGS  sbatch options that give the TensorFlow jobs a GPU
#                      in shared (Default) compute mode, where the cluster's
#                      GPUs default to Exclusive_Process, which refuses the
#                      TensorFlow engine's second (DeepSHAP) process

set -euo pipefail

# Read before _common.sh, which defines ENGINES as the list of all engines.
REQUESTED_ENGINES="${ENGINES:-}"

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"
cd "${HERE}"

parse_common_flags "$@"
CLUSTER="slurm"
DRY=""
CONTRIBUTIONS=""
DEEPSHAP_SEEDS=""
ORIGINAL_RETRAIN=""
for arg in ${ARGS[@]+"${ARGS[@]}"}; do
  case "${arg}" in
    --dry-run) DRY=1 ;;
    --contribution|--contributions) CONTRIBUTIONS=1 ;;
    --deepshap-seeds) DEEPSHAP_SEEDS=1 ;;
    --original-retrain) ORIGINAL_RETRAIN=1 ;;
    slurm|local) CLUSTER="${arg}" ;;
    sherlock) CLUSTER="slurm" ;;
    *) echo "unknown argument '${arg}' (usage: submit_all.sh [slurm|local] [--container] [--contribution|--deepshap-seeds|--original-retrain] [--dry-run])" >&2; exit 2 ;;
  esac
done
require_sweep REFERENCE_DIR GENOME CHROM_SIZES
[[ -n "${DEEPSHAP_SEEDS}${ORIGINAL_RETRAIN}" ]] && require_site ORIGINAL_REPO
[[ -n "${DEEPSHAP_SEEDS}" ]] && require_site ENCODE_MODELS
# The original runs with TensorFlow, whatever ENGINES says.
[[ -n "${DEEPSHAP_SEEDS}${ORIGINAL_RETRAIN}" ]] && REQUESTED_ENGINES="tensorflow"
CONTAINER_ARG=()
(( USE_CONTAINER )) && CONTAINER_ARG=(--container)

SELECTED=("${ENGINES[@]}")
if [[ -n "${REQUESTED_ENGINES}" ]]; then
  read -r -a SELECTED <<< "${REQUESTED_ENGINES}"
  for engine in "${SELECTED[@]}"; do
    is_engine "${engine}" || { echo "unknown engine '${engine}' (expected: ${ENGINES[*]})" >&2; exit 2; }
  done
fi
# Refuse up front rather than submit an array whose every task stops at once.
for engine in "${SELECTED[@]}"; do
  check_native "${engine}" "${CLUSTER}" || exit 2
done

ARRAY_OPT=()
[[ -n "${ARRAY:-}" ]] && ARRAY_OPT=(--array="${ARRAY}")
JAX_GPU_CONSTRAINT="${JAX_GPU_CONSTRAINT:-}"
TF_GPU_CONSTRAINT="${TF_GPU_CONSTRAINT:-}"
constraint_for() {
  local constraint="${GPU_CONSTRAINT:-}"
  [[ -z "${constraint}" && "$1" == "jax" ]] && constraint="${JAX_GPU_CONSTRAINT}"
  [[ -z "${constraint}" && "$1" == "tensorflow" ]] && constraint="${TF_GPU_CONSTRAINT}"
  [[ -n "${constraint}" ]] && echo "--constraint=${constraint}"
  return 0
}

# Sets ENGINE_OPT to the sbatch options one engine's GPU jobs need.
read -r -a SBATCH_EXTRA_OPT <<< "${SBATCH_EXTRA:-}"
engine_options() {
  local engine="$1" constraint
  ENGINE_OPT=(${SBATCH_EXTRA_OPT[@]+"${SBATCH_EXTRA_OPT[@]}"})
  constraint="$(constraint_for "${engine}")"
  [[ -n "${constraint}" ]] && ENGINE_OPT+=("${constraint}")
  # A GPU in Exclusive_Process compute mode refuses the TensorFlow engine's
  # second (DeepSHAP) process: site.env's SBATCH_SHARED_GPU_ARGS asks for one
  # in shared mode where the cluster's default is exclusive.
  local shared=()
  read -r -a shared <<< "${SBATCH_SHARED_GPU_ARGS:-}"
  if [[ "${engine}" == "tensorflow" && "${CLUSTER}" == "slurm" ]]; then ENGINE_OPT+=(${shared[@]+"${shared[@]}"}); fi
  return 0
}

# site.env's sbatch options and log directory, then these (_common.sh).
submit() { site_sbatch "$@"; }

if [[ -n "${DEEPSHAP_SEEDS}" ]]; then
  echo "=== the original chrombpnet's DeepSHAP on the original models, one array per seed" \
       "($( [[ -n "${N_REGIONS:-}" && "${N_REGIONS}" != 0 ]] && echo "first ${N_REGIONS} peaks" || echo "all 30K peaks"))"
  engine_options tensorflow
  read -r -a SEED_LIST <<< "${SEEDS:-1 2 3}"
  for seed in "${SEED_LIST[@]}"; do
    job="$(submit ${ARRAY_OPT[@]+"${ARRAY_OPT[@]}"} ${ENGINE_OPT[@]+"${ENGINE_OPT[@]}"} \
                  --export=ALL,SEEDS="${seed}" --job-name="engines_deepshap_seed${seed}" \
                  "${HERE}/run_deepshap_seeds.sh" "${CLUSTER}" ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
    printf "  seed %-5s job %s\n" "${seed}" "${job}"
  done
  echo
  echo "Then:  sbatch run_summarize.sh --profile-stopgrad off   # -> analysis/profile-stopgrad-off/"
  echo "       (--profile-stopgrad on: the pipeline's scores -> analysis/profile-stopgrad-on/)"
  echo "       ./plot_summarize.sh                                # both"
  exit 0
fi

if [[ -n "${ORIGINAL_RETRAIN}" ]]; then
  if (( USE_CONTAINER )); then
    echo "Error: --original-retrain runs natively, not with --container: the original's report PDF" \
         "needs the system's pango (site.env: report_libraries), which the image does not have." >&2
    exit 2
  fi
  echo "=== the original chrombpnet, retrained on each row's sweep inputs, then its DeepSHAP" \
       "(seeds ${SEEDS:-1})"
  engine_options tensorflow
  job="$(submit ${ARRAY_OPT[@]+"${ARRAY_OPT[@]}"} ${ENGINE_OPT[@]+"${ENGINE_OPT[@]}"} \
                --job-name="engines_original_retrain" \
                "${HERE}/run_original_retrain.sh" "${CLUSTER}" ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  printf "  original (new model) job %s\n" "${job}"
  echo
  echo "Resubmitting is safe: finished rows skip themselves, and a row that hit the time limit"
  echo "goes on from its last finished step."
  echo "Then:  sbatch run_summarize.sh --profile-stopgrad off   # -> analysis/profile-stopgrad-off/"
  echo "       (--profile-stopgrad on: the pipeline's scores -> analysis/profile-stopgrad-on/)"
  echo "       ./plot_summarize.sh                                # both"
  exit 0
fi

if [[ -n "${CONTRIBUTIONS}" ]]; then
  echo "=== contribution scores, one array per engine"
  for engine in "${SELECTED[@]}"; do
    engine_options "${engine}"
    job="$(submit ${ARRAY_OPT[@]+"${ARRAY_OPT[@]}"} ${ENGINE_OPT[@]+"${ENGINE_OPT[@]}"} \
                  --job-name="engines_contribution_${engine}" "${HERE}/run_contribution.sh" "${CLUSTER}" "${engine}" \
                  ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
    printf "  %-10s job %s\n" "${engine}" "${job}"
  done
  echo
  echo "Then:  ./run_summarize.sh && ./plot_summarize.sh"
  exit 0
fi

DEPENDENCY=()
if [[ "${SKIP_PREP:-0}" == "1" ]]; then
  echo "=== nonpeaks: not submitted (SKIP_PREP=1)"
else
  prep_job="$(submit ${ARRAY_OPT[@]+"${ARRAY_OPT[@]}"} "${HERE}/run_prep_nonpeaks.sh" "${CLUSTER}" \
                ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  echo "=== nonpeaks: job ${prep_job}"
  DEPENDENCY=(--dependency="aftercorr:${prep_job}" --kill-on-invalid-dep=yes)
fi

echo "=== training, one array per engine"
for engine in "${SELECTED[@]}"; do
  engine_options "${engine}"
  job="$(submit ${ARRAY_OPT[@]+"${ARRAY_OPT[@]}"} ${ENGINE_OPT[@]+"${ENGINE_OPT[@]}"} ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
                --job-name="engines_${engine}" "${HERE}/run_train_full.sh" "${CLUSTER}" "${engine}" \
                ${CONTAINER_ARG[@]+"${CONTAINER_ARG[@]}"})"
  printf "  %-10s job %s\n" "${engine}" "${job}"
done

echo
echo "Progress:  ./check_runs.sh        Results:  ./run_summarize.sh && ./plot_summarize.sh"
