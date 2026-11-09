# Shared argument resolution for run_train_full.sh / run_retrain_full.sh / run_compare.sh.
# Sourced, not executed.
#
# The cluster argument is easy to forget, and forgetting it used to fail in a
# way that named the wrong argument: `sbatch run_retrain_full.sh right` put "right"
# into $1, left $2 unset, and the error read "line 39: 2: usage ...", pointing
# at the arm when the cluster was what went missing. Sixty array tasks died
# before printing anything.
#
# So the cluster is now optional and inferred, and all of these work:
#
#     run_X.sh slurm right
#     run_X.sh right                     cluster inferred
#     DATA_AUG=right sbatch run_X.sh     arm exported instead of passed
#
# Inference is safe because the two settings the cluster selects -- which
# conda env, and whether to run site.env's hooks -- follow from whether we are inside
# a Slurm allocation, which $SLURM_JOB_ID answers exactly.
#
# The caller must set VALID_ARMS before sourcing this, because run_compare.sh
# accepts a different set (it has nothing to compare `standard` against).

_is_valid_arm() {
  local needle="$1" arm
  for arm in "${VALID_ARMS[@]}"; do
    [[ "${arm}" == "${needle}" ]] && return 0
  done
  return 1
}

_usage() {
  # $0 is the script being executed; BASH_SOURCE inside this helper would
  # name the helper itself.
  local me; me="$(basename "$0")"
  {
    echo "usage: ${me} [cluster] <data-aug>"
    echo "   or: DATA_AUG=<arm> sbatch ${me} [cluster]"
    echo
    echo "  received : cluster=${1:-<unset>}  data-aug=${2:-<unset>}"
    echo "  valid arms: ${VALID_ARMS[*]}"
    echo "  cluster   : slurm | local  (inferred from \$SLURM_JOB_ID if omitted)"
  } >&2
  exit 2
}

# Sets CLUSTER and DATA_AUG, or exits 2 with a message naming what arrived.
resolve_args() {
  local first="${1:-}" second="${2:-}"

  if [[ -n "${first}" && -z "${second}" ]] && _is_valid_arm "${first}"; then
    # A single argument that names an arm: the cluster was omitted.
    DATA_AUG="${first}"
    CLUSTER=""
  else
    CLUSTER="${first}"
    DATA_AUG="${second:-${DATA_AUG:-}}"
  fi

  if [[ -z "${DATA_AUG}" ]]; then
    _usage "${first}" "${second}"
  fi
  if ! _is_valid_arm "${DATA_AUG}"; then
    echo "unknown data-aug '${DATA_AUG}'" >&2
    _usage "${first}" "${second}"
  fi

  if [[ -z "${CLUSTER}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then CLUSTER="slurm"; else CLUSTER="local"; fi
    echo "note: no cluster given; inferred '${CLUSTER}'"
  fi
  if [[ "${CLUSTER}" != "slurm" && "${CLUSTER}" != "local" ]]; then
    echo "unknown cluster '${CLUSTER}'" >&2
    _usage "${first}" "${second}"
  fi
}


# How many data rows the ids table actually has, excluding the header and any
# blank lines. Callers use this as the upper bound on SLURM_ARRAY_TASK_ID
# rather than a hardcoded number, so adding or removing a dataset from the
# table changes the valid range automatically instead of silently running the
# wrong row (or skipping a real one).
data_row_count() {
  local ids="${1:?data_row_count needs the ids file}"
  if [[ ! -f "${ids}" ]]; then
    echo "Error: ids file not found: ${ids}" >&2
    return 1
  fi
  local n
  n="$(awk 'NR > 1 && NF > 0 { rows++ } END { print rows + 0 }' "${ids}")"
  if (( n == 0 )); then
    echo "Error: ids file has no data rows: ${ids}" >&2
    return 1
  fi
  printf '%s\n' "${n}"
}


# The chrombpnet flags that reproduce one arm, into ARM_FLAGS.
#
# The arms are the old values of chrombpnet's --data-aug, and stay as labels:
# they name the run directories and the rows of results.tsv. The package
# has since split that flag in two -- --dnase-convention (where a DNase cut
# is recorded; DNASE only) and --data-aug (augment | forward-only |
# revcomp-only: which examples are reverse complemented) -- and every arm
# maps onto the pair exactly (the package CHANGELOG lists the mapping):
#
#   standard            --dnase-convention right-deprecated   (ATAC: no flags)
#   fr-only / rc-only   --data-aug forward-only / revcomp-only
#                       (+ --dnase-convention right-deprecated on DNASE)
#   right, left, both-half, both-full
#                       --dnase-convention <arm>   (DNASE only)
#
# Read ARM_FLAGS as ${ARM_FLAGS[@]+"${ARM_FLAGS[@]}"}: it is empty for
# standard on ATAC, and bash 4.2 (still on some clusters) calls an empty "${a[@]}" unbound.
#
#   arm_flags <arm> <assay>
arm_flags() {
  local arm="$1" assay="$2" convention=""
  ARM_FLAGS=()
  case "${arm}" in
    standard) convention="right-deprecated" ;;
    fr-only) ARM_FLAGS=(--data-aug forward-only); convention="right-deprecated" ;;
    rc-only) ARM_FLAGS=(--data-aug revcomp-only); convention="right-deprecated" ;;
    right|left|both-half|both-full)
      if [[ "${assay}" != "DNASE" ]]; then
        echo "Error: arm ${arm} is defined only for DNASE, not ${assay}" >&2
        return 1
      fi
      convention="${arm}" ;;
    *) echo "Error: unknown arm '${arm}'" >&2; return 1 ;;
  esac
  if [[ "${assay}" == "DNASE" ]]; then
    ARM_FLAGS+=(--dnase-convention "${convention}")
  fi
}
