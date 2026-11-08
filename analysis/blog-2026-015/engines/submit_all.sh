#!/bin/bash
# Submit the whole sweep: the shared nonpeaks, then one training array per
# engine. Or, with --contribution, one run_contribution.sh array per engine.
#
#   ./submit_all.sh sherlock --container                   # submit the sweep
#   ./submit_all.sh sherlock --container --dry-run         # print the sbatch lines only
#   ./submit_all.sh sherlock --container --contribution   # rescore finished runs' subsamples
#   ./submit_all.sh sherlock --original-retrain           # the original, retrained per row
#
#   --container      run every step inside the Ubuntu 22.04 image (_common.sh):
#                    required for JAX on sherlock, and it gives every engine
#                    the same system tools, so their runtimes compare.
#                    Without it, a submission that includes JAX on sherlock
#                    is refused.
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
#                      --constraint, e.g. "GPU_SKU:L40S", so all engines run
#                      on one GPU type and their runtimes compare
#                      (`sh_node_feat` lists the feature names on sherlock)
#   SBATCH_EXTRA=".."  more sbatch options for every GPU job, word-split, e.g.
#                      "--exclude=<node>,<node>"
#   TF_GPU_CONSTRAINT=..   the TensorFlow jobs' constraint when GPU_CONSTRAINT is
#                      unset. Default: GPUs of 16 GB or more -- compute
#                      capability 6.0 (P100), 7.0 (V100), 8.x and 9.0, leaving
#                      out 6.1 (12 GB TITAN Xp) and 7.5 (11 GB RTX 2080 Ti).
#                      TensorFlow's DeepSHAP runs in a second process beside
#                      the one that trained, and the two do not fit 12 GB.
#                      Set it empty to drop it.
#   JAX_GPU_CONSTRAINT=..  the JAX jobs' constraint when GPU_CONSTRAINT is
#                      unset. Default: compute capability 7.5 or newer, the
#                      GPUs the JAX env's cuDNN 9.27 has kernels for (not the
#                      Pascal P100/TITAN Xp or Volta V100 nodes). Set it
#                      empty to drop it.

set -euo pipefail

# Read before _common.sh, which defines ENGINES as the list of all engines.
REQUESTED_ENGINES="${ENGINES:-}"

HERE="$(dirname "$(readlink -f "$0")")"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

parse_common_flags "$@"
CLUSTER="sherlock"
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
    sherlock|local) CLUSTER="${arg}" ;;
    *) echo "unknown argument '${arg}' (usage: submit_all.sh [sherlock|local] [--container] [--contribution|--deepshap-seeds|--original-retrain] [--dry-run])" >&2; exit 2 ;;
  esac
done
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
JAX_GPU_CONSTRAINT="${JAX_GPU_CONSTRAINT-GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0}"
TF_GPU_CONSTRAINT="${TF_GPU_CONSTRAINT-GPU_CC:6.0|GPU_CC:7.0|GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0}"
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
  # sherlock's GPUs default to Exclusive_Process compute mode, which refuses
  # the TensorFlow engine's second (DeepSHAP) process; --gpu_cmode is
  # sherlock's own sbatch option for the job's GPUs.
  if [[ "${engine}" == "tensorflow" && "${CLUSTER}" == "sherlock" ]]; then ENGINE_OPT+=(--gpu_cmode=shared); fi
  return 0
}

submit() {
  if [[ -n "${DRY}" ]]; then
    echo "sbatch --parsable $*" >&2
    echo "<jobid>"
  else
    sbatch --parsable "$@"
  fi
}

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
         "needs the system's pango (on sherlock, a module), which the image does not have." >&2
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
