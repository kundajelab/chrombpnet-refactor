# Shared by the engines sweep scripts. Sourced, not executed.
#
# The sweep trains the same ChromBPNet pipeline under each deep-learning
# engine -- pytorch, tensorflow, jax -- on 4 datasets x 5 folds. Each engine
# runs in its own conda environment, chrombpnet-<engine>, all three editable
# installs of the same checkout (CHROMBPNET_DIR below).
#
# Callers source ../_site.sh first, which reads this site's paths from
# site.env (../site.env.example). Everything here can be overridden from the
# environment:
#
#   CHROMBPNET_DIR   the checkout every env must import chrombpnet from
#                    (default: site.env's CHROMBPNET_REPO)
#   IDS_FILE         the dataset x fold table, one data row per array task
#                    (default: site.env's ENGINES_IDS)
#   ROOT             where runs are written (default: site.env's ENGINES_ROOT)
#   CONDA_ENV_PREFIX env names are ${CONDA_ENV_PREFIX}<engine>
#
# The rest is site.env's as it stands: REFERENCE_DIR (folds/fold_<k>.json and
# blacklist.bed.gz), GENOME, CHROM_SIZES, CONDA_ROOT, CONTAINER_IMAGE,
# CONTAINER_BINDS, TF28_LIBRARY_DIRS, and the hooks tf28_libraries and
# pytorch_compiler, which run under Slurm outside the container.

ENGINES=(pytorch tensorflow jax)

if ! declare -F require_site > /dev/null; then
  echo "Error: source ../_site.sh before $(basename "${BASH_SOURCE[0]}")" >&2
  exit 2
fi
CHROMBPNET_DIR="${CHROMBPNET_DIR:-${CHROMBPNET_REPO}}"
IDS_FILE="${IDS_FILE:-${ENGINES_IDS:-}}"
ROOT="${ROOT:-${ENGINES_ROOT:-}}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-chrombpnet-}"

GENOME_PATH="${GENOME:-}"
CHROM_PATH="${CHROM_SIZES:-}"
BLACKLIST_PATH="${REFERENCE_DIR:+${REFERENCE_DIR}/blacklist.bed.gz}"

# Stops, naming each site.env setting left empty, unless the sweep's run tree
# and ids table are known (ROOT and IDS_FILE, or site.env's ENGINES_ROOT and
# ENGINES_IDS) and every further site.env name given is set.
#   require_sweep [NAME...]
require_sweep() {
  local names=()
  [[ -n "${ROOT}" ]] || names+=(ENGINES_ROOT)
  [[ -n "${IDS_FILE}" ]] || names+=(ENGINES_IDS)
  require_site ${names[@]+"${names[@]}"} "$@"
}

# --container runs an engine's conda env inside a container: a plain Ubuntu
# 22.04 image (site.env's CONTAINER_IMAGE) -- the OS the envs were built and
# tested on -- running the same env from the host's filesystem (site.env's
# CONTAINER_BINDS), with the node's GPU driver bound in (--nv). Off by
# default, for every engine. Two reasons to turn it on:
#
#   jax         required on a host whose glibc is older than 2.27: jaxlib 0.11
#               and the CUDA 12 wheels it brings need 2.27 (Sherlock's nodes,
#               where the sweep ran, have 2.17). Without --container a JAX job
#               there stops at once and says so.
#   all three   the same OS userland for every engine. Making a bigwig pipes
#               the reads through the system's awk and sort; the host's
#               (CentOS 7's gawk and coreutils, in the host locale) took 2.4x
#               as long as the image's (mawk, coreutils 8.32, C.UTF-8) on the
#               same node -- engine-independent time in the runtime comparison.
#
# check_envs.sh --container pulls the image once, to CONTAINER_IMAGE (JAX_IMAGE,
# its old name, still works). USE_CONTAINER=1 in the environment means the same
# as --container.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${JAX_IMAGE:-}}"
CONTAINER_IMAGE_SOURCE="docker://ubuntu:22.04"
JAX_IMAGE="${CONTAINER_IMAGE}"   # its name when only JAX used it
case "${USE_CONTAINER:-0}" in
  1|always|yes|true) USE_CONTAINER=1 ;;
  *) USE_CONTAINER=0 ;;
esac
export USE_CONTAINER

# Takes --container out of the arguments (setting USE_CONTAINER=1) and leaves
# the rest, in order, in ARGS. Read ARGS as ${ARGS[@]+"${ARGS[@]}"}: an empty
# "${ARGS[@]}" is an error under set -u in bash 4.2 (CentOS 7's).
parse_common_flags() {
  ARGS=()
  local a
  for a in "$@"; do
    case "${a}" in
      --container) USE_CONTAINER=1 ;;
      *) ARGS+=("${a}") ;;
    esac
  done
  export USE_CONTAINER
}

in_container() { [[ -n "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; }

# Whether this script must re-run itself in the container: asked for, and not
# there yet. (Arguments are ignored; they named the engine and cluster when
# only JAX on the cluster used the container.)
needs_container() {
  in_container && return 1
  [[ "${USE_CONTAINER}" == "1" ]]
}

# This host's glibc version, e.g. 2.17 (empty when getconf cannot tell).
host_glibc() { getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}'; }

# Fails, saying what to do, for an engine that cannot run natively here: JAX
# on a glibc older than 2.27. (The cluster argument is no longer needed.)
check_native() {
  local engine="$1" glibc
  in_container && return 0
  [[ "${USE_CONTAINER}" == "1" ]] && return 0
  glibc="$(host_glibc)"
  if [[ "${engine}" == "jax" && -n "${glibc}" ]] && ! printf '2.27\n%s\n' "${glibc}" | sort -V -C; then
    echo "Error: the JAX engine needs --container here (jaxlib needs glibc >= 2.27;" \
         "this host has ${glibc})." >&2
    return 1
  fi
}

container_runtime() { command -v apptainer || command -v singularity; }

# Pulls the image if it is not there yet (check_envs.sh only: sixty training
# tasks pulling at once would be a mess).
ensure_image() {
  local runtime
  [[ -n "${CONTAINER_IMAGE}" ]] || require_site CONTAINER_IMAGE
  runtime="$(container_runtime)" || { echo "Error: neither apptainer nor singularity is on PATH" >&2; return 1; }
  [[ -f "${CONTAINER_IMAGE}" ]] && return 0
  echo "pulling ${CONTAINER_IMAGE_SOURCE} -> ${CONTAINER_IMAGE} (once)"
  mkdir -p "$(dirname "${CONTAINER_IMAGE}")"
  "${runtime}" pull "${CONTAINER_IMAGE}" "${CONTAINER_IMAGE_SOURCE}"
}

# The directories of a colon-separated list that exist, colon-separated.
existing_dirs() {
  local d dirs=() found=()
  IFS=: read -r -a dirs <<< "$1"
  for d in ${dirs[@]+"${dirs[@]}"}; do
    if [[ -d "${d}" ]]; then found+=("${d}"); fi
  done
  ( IFS=:; echo "${found[*]-}" )
}

# The TensorFlow engine's CUDA 11.2 and cuDNN 8.1 library directories on this
# host, colon-separated: site.env's TF28_LIBRARY_DIRS (those that exist) if it
# names any, else, under Slurm, the CUDA and cuDNN directories that site.env's
# hook tf28_libraries puts on LD_LIBRARY_PATH (module loads, say). Run on the
# host, because the container has neither `module` nor those directories;
# container_run binds them and hands them in as TF_CUDA_DIRS, which
# activate_engine puts on LD_LIBRARY_PATH.
tf_cuda_dirs() {
  local cluster="$1"
  if [[ -n "${TF28_LIBRARY_DIRS:-}" ]]; then
    existing_dirs "${TF28_LIBRARY_DIRS}"
  elif [[ "${cluster}" == "slurm" ]]; then
    # A subshell, so what the hook loads stays out of this shell.
    ( site_hook tf28_libraries > /dev/null 2>&1 || exit 1
      printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -iE 'cuda|cudnn' | paste -sd: - )
  fi
}

# Runs a command in the container: site.env's CONTAINER_BINDS (the host
# directories the envs, checkouts, data and outputs live in, those that exist
# here; Apptainer itself binds $HOME, /tmp and the working directory) and a
# TMPDIR outside /tmp bound, the GPU driver bound (--nv), and the C.UTF-8
# locale, which the image has, instead of whatever the host's LANG/LC_ALL name.
# The host environment -- the SLURM_* variables, USE_CONTAINER -- passes
# through; PATH does not (the image's replaces it).
#
#   container_run <engine> <cluster> <command...>
#
# For tensorflow the host's CUDA 11.2 / cuDNN 8.1 directories are bound too,
# and named in TF_CUDA_DIRS. (Where those are symlinks into a larger tree, as a
# module's may be, put that tree in CONTAINER_BINDS.)
container_run() {
  local engine="$1" cluster="$2" runtime binds=() d dirs=()
  shift 2
  [[ -n "${CONTAINER_IMAGE}" ]] || require_site CONTAINER_IMAGE
  runtime="$(container_runtime)" || { echo "Error: neither apptainer nor singularity is on PATH" >&2; return 1; }
  if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "Error: no container image at ${CONTAINER_IMAGE}. Run ./check_envs.sh --container once, which pulls it," >&2
    echo "       or: ${runtime##*/} pull ${CONTAINER_IMAGE} ${CONTAINER_IMAGE_SOURCE}" >&2
    return 1
  fi
  read -r -a dirs <<< "${CONTAINER_BINDS:-}"
  for d in ${dirs[@]+"${dirs[@]}"}; do
    if [[ -e "${d%%:*}" ]]; then binds+=(--bind "${d}"); fi
  done
  if [[ "${engine}" == "tensorflow" ]]; then
    TF_CUDA_DIRS="$(tf_cuda_dirs "${cluster}")" || TF_CUDA_DIRS=""
    if [[ -z "${TF_CUDA_DIRS}" ]]; then
      echo "Error: no CUDA 11.2 / cuDNN 8.1 found on this host to hand to TensorFlow in the container" \
           "(site.env: TF28_LIBRARY_DIRS, or under Slurm the hook tf28_libraries)" >&2
      return 1
    fi
    export TF_CUDA_DIRS
    IFS=: read -r -a dirs <<< "${TF_CUDA_DIRS}"
    for d in "${dirs[@]}"; do binds+=(--bind "${d}"); done
  fi
  if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" && "${TMPDIR}" != /tmp* ]]; then binds+=(--bind "${TMPDIR}"); fi
  echo "container: ${CONTAINER_IMAGE} (${engine})"
  LANG=C.UTF-8 LC_ALL= "${runtime}" exec --nv ${binds[@]+"${binds[@]}"} "${CONTAINER_IMAGE}" "$@"
}

# sbatch as the submit scripts call it (theirs and expand's): site.env's
# SBATCH_SITE_ARGS first (partition, nodes to avoid and the like; word-split),
# then, with site.env's LOG_DIR set, the job script's own #SBATCH --output and
# --error names under LOG_DIR -- without it the logs land where sbatch runs,
# and the submit scripts run it from their own folder -- then the arguments.
# With DRY set it prints the command and outputs <jobid> instead.
#   site_sbatch <sbatch options...> <job script> [its arguments...]
site_sbatch() {
  local a script="" opts=() name value
  read -r -a opts <<< "${SBATCH_SITE_ARGS:-}"
  for a in "$@"; do
    if [[ "${a}" != -* && -f "${a}" ]]; then script="${a}"; break; fi
  done
  if [[ -n "${LOG_DIR:-}" && -n "${script}" ]]; then
    for name in output error; do
      value="$(sed -n "s/^#SBATCH --${name}=//p" "${script}" | head -n 1)"
      if [[ -n "${value}" ]]; then opts+=("--${name}=${LOG_DIR}/${value}"); fi
    done
  fi
  if [[ -n "${DRY:-}" ]]; then
    echo "sbatch --parsable ${opts[*]+${opts[*]} }$*" >&2
    echo "<jobid>"
  else
    if [[ -n "${LOG_DIR:-}" ]]; then mkdir -p "${LOG_DIR}"; fi
    sbatch --parsable ${opts[@]+"${opts[@]}"} "$@"
  fi
}

# Checkpoint suffix each engine writes.
engine_suffix() {
  case "$1" in
    pytorch) echo ".pt" ;;
    tensorflow) echo ".h5" ;;
    jax) echo ".keras" ;;
    *) return 1 ;;
  esac
}

is_engine() {
  local needle="$1" engine
  for engine in "${ENGINES[@]}"; do
    [[ "${engine}" == "${needle}" ]] && return 0
  done
  return 1
}

_usage() {
  local me; me="$(basename "$0")"
  {
    echo "usage: ${me} [cluster] <engine>"
    echo "   or: ENGINE=<engine> sbatch ${me} [cluster]"
    echo
    echo "  received : cluster=${1:-<unset>}  engine=${2:-<unset>}"
    echo "  engines  : ${ENGINES[*]}"
    echo "  cluster  : slurm | local  (inferred from \$SLURM_JOB_ID if omitted; sherlock = slurm)"
  } >&2
  exit 2
}

# Sets CLUSTER and ENGINE, or exits 2 naming what arrived. The cluster is
# optional, as in dnase-rc/_args.sh: a lone argument that names an engine
# means the cluster was left out, and it is inferred from $SLURM_JOB_ID.
resolve_args() {
  local first="${1:-}" second="${2:-}"
  if [[ -n "${first}" && -z "${second}" ]] && is_engine "${first}"; then
    ENGINE="${first}"
    CLUSTER=""
  else
    CLUSTER="${first}"
    ENGINE="${second:-${ENGINE:-}}"
  fi
  [[ -n "${ENGINE}" ]] || _usage "${first}" "${second}"
  is_engine "${ENGINE}" || { echo "unknown engine '${ENGINE}'" >&2; _usage "${first}" "${second}"; }
  resolve_cluster "${CLUSTER}"
}

# Sets CLUSTER to slurm (a Slurm job) or local (a plain run on a GPU host);
# infers it when empty. sherlock, the name these runs were submitted with,
# means slurm.
resolve_cluster() {
  CLUSTER="${1:-}"
  if [[ -z "${CLUSTER}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then CLUSTER="slurm"; else CLUSTER="local"; fi
    echo "note: no cluster given; inferred '${CLUSTER}'"
  fi
  [[ "${CLUSTER}" == "sherlock" ]] && CLUSTER="slurm"
  if [[ "${CLUSTER}" != "slurm" && "${CLUSTER}" != "local" ]]; then
    echo "unknown cluster '${CLUSTER}' (expected slurm or local)" >&2
    exit 2
  fi
}

# How many data rows the ids table has, excluding the header and blank
# lines; the upper bound on SLURM_ARRAY_TASK_ID.
data_row_count() {
  local ids="${1:?data_row_count needs the ids file}"
  [[ -f "${ids}" ]] || { echo "Error: ids file not found: ${ids}" >&2; return 1; }
  local n
  n="$(awk 'NR > 1 && NF > 0 { rows++ } END { print rows + 0 }' "${ids}")"
  (( n > 0 )) || { echo "Error: ids file has no data rows: ${ids}" >&2; return 1; }
  printf '%s\n' "${n}"
}

# Reads data row $1 (1-based, after the header) of IDS_FILE into ASSAY,
# ENCID, BAM_PATH, PEAKS_BED_PATH, FOLD_NUM and TF_DIR, by header name.
read_row() {
  local row="$1" values
  values="$(awk -F'\t' -v row="${row}" '
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    NF > 0 { n++ }
    NF > 0 && n == row {
      printf "%s\t%s\t%s\t%s\t%s\t%s\n", $col["ASSAY"], $col["ENCID"], $col["BAM_PATH"],
             $col["PEAKS_BED_PATH"], $col["FOLD_NUM"], (col["TENSORFLOW_DIR"] ? $col["TENSORFLOW_DIR"] : "")
    }' "${IDS_FILE}" | tr -d '\r')"
  [[ -n "${values}" ]] || { echo "Error: no data row ${row} in ${IDS_FILE}" >&2; return 1; }
  IFS=$'\t' read -r ASSAY ENCID BAM_PATH PEAKS_BED_PATH FOLD_NUM TF_DIR <<< "${values}"
  FOLD_JSON_PATH="${REFERENCE_DIR}/folds/fold_${FOLD_NUM}.json"
  # The reference run script hardcodes 0.5; this table mixes both assays.
  if [[ "${ASSAY}" == "ATAC" ]]; then BIAS_THRESHOLD_FACTOR=0.5; else BIAS_THRESHOLD_FACTOR=0.8; fi
}

# Resolves the array task to a data row and reads it. Exits 0 with a SKIP
# line when the task lies past the table, so a wide --array is harmless.
select_row() {
  require_sweep REFERENCE_DIR GENOME CHROM_SIZES
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is unset. By hand: SLURM_ARRAY_TASK_ID=0 $(basename "$0") ..." >&2
    exit 2
  fi
  DATA_ROW=$((SLURM_ARRAY_TASK_ID + 1))
  local n
  n="$(data_row_count "${IDS_FILE}")" || exit 1
  if (( DATA_ROW < 1 || DATA_ROW > n )); then
    echo "SKIP: array task ${SLURM_ARRAY_TASK_ID} is data row ${DATA_ROW}, but ${IDS_FILE##*/} has ${n}."
    exit 0
  fi
  read_row "${DATA_ROW}" || exit 1
}

# The nonpeaks every engine trains on, written once per dataset and fold by
# run_prep_nonpeaks.sh, so the three engines see identical regions.
prep_dir() { echo "${ROOT}/preprocess/${ASSAY}/${ENCID}/fold_${FOLD_NUM}"; }

# Activates chrombpnet-<engine> and what that framework needs to reach a GPU,
# then checks the env imports chrombpnet from CHROMBPNET_DIR. Each framework
# brings or needs a different CUDA:
#
#   pytorch     torch 2.5.1+cu118 brings its own CUDA. Under Slurm, outside
#               the container, site.env's hook pytorch_compiler (the sweep
#               loaded gcc/14.2.0 there, as in the dnase-rc runs).
#   tensorflow  TF 2.8 needs CUDA 11.2 + cuDNN 8.1 from the system:
#               site.env's TF28_LIBRARY_DIRS if set, else under Slurm its hook
#               tf28_libraries (the modules the dnase-rc comparison jobs
#               load); in the container, those same host directories
#               (TF_CUDA_DIRS). Memory growth is on, because
#               DeepSHAP runs in a spawned child process that needs GPU
#               memory while the parent still holds what it trained with.
#               (The engine sets it itself since commit be83abc; set here
#               too, for checkouts before that.)
#   jax         jax[cuda12] brings CUDA 12 and cuDNN 9 as pip wheels, found
#               by RUNPATH -- which LD_LIBRARY_PATH outranks. A system CUDA 12
#               there (a login shell's ~/.bashrc may put one on it; a loaded
#               CUDA module would too) shadows the wheels, JAX fails its version
#               check and falls back to the CPU. So CUDA and cuDNN
#               directories are dropped from LD_LIBRARY_PATH for this job.
#               (Since commit 81d7170 the engine also loads the bundled
#               libraries first when it finds namesakes there.)
#
#   activate_engine <engine> <cluster> [<env> <checkout>]
#
# <env> and <checkout> name another env with the same framework needs and
# the checkout it must import chrombpnet from instead (run_original_retrain.sh:
# the original package's own env, chrombpnet, and its checkout).
activate_engine() {
  local engine="$1" cluster="$2" conda_exe
  local expect="${4:-${CHROMBPNET_DIR}}"
  CONDA_ENV="${3:-${CONDA_ENV_PREFIX}${engine}}"
  echo "conda environment: ${CONDA_ENV}"
  # conda by path, not by name: in the JAX container PATH is the image's
  # (apptainer does not pass the host's) and a `conda` shell function from
  # ~/.bashrc does not cross either. CONDA_EXE, which conda's shell hook
  # exports, does; failing that, the conda the envs live in (site.env's
  # CONDA_ROOT).
  conda_exe="${CONDA_EXE:-}"
  [[ -x "${conda_exe}" ]] || conda_exe="$(command -v conda 2>/dev/null || true)"
  if [[ ! -x "${conda_exe}" ]]; then
    [[ -n "${CONDA_ROOT:-}" ]] || require_site CONDA_ROOT
    conda_exe="${CONDA_ROOT}/bin/conda"
  fi
  eval "$("${conda_exe}" shell.bash hook)" || { echo "Error: no conda at ${conda_exe}" >&2; return 1; }
  conda activate "${CONDA_ENV}" || { echo "Error: cannot activate ${CONDA_ENV}" >&2; return 1; }
  case "${engine}" in
    pytorch)
      # The image's libstdc++ is new enough, and it has no `module`.
      if [[ "${cluster}" == "slurm" ]] && ! in_container; then site_hook pytorch_compiler || return 1; fi
      ;;
    tensorflow)
      local tf_dirs
      if in_container; then
        [[ -n "${TF_CUDA_DIRS:-}" ]] || { echo "Error: TF_CUDA_DIRS is empty: no CUDA 11.2 / cuDNN 8.1 was handed into the container" >&2; return 1; }
        export LD_LIBRARY_PATH="${TF_CUDA_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo "CUDA 11.2 / cuDNN 8.1 from the host: ${TF_CUDA_DIRS}"
      elif [[ -n "${TF28_LIBRARY_DIRS:-}" ]]; then
        tf_dirs="$(existing_dirs "${TF28_LIBRARY_DIRS}")"
        if [[ -n "${tf_dirs}" ]]; then export LD_LIBRARY_PATH="${tf_dirs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi
      elif [[ "${cluster}" == "slurm" ]]; then
        site_hook tf28_libraries || return 1
      fi
      export TF_FORCE_GPU_ALLOW_GROWTH=true
      export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
      ;;
    jax)
      LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" \
        | awk -v RS=: -v ORS=: 'NF && tolower($0) !~ /cuda|cudnn/' | sed 's/:$//')"
      export LD_LIBRARY_PATH
      echo "LD_LIBRARY_PATH (CUDA/cuDNN entries removed): ${LD_LIBRARY_PATH:-<empty>}"
      ;;
  esac
  local where
  where="$(python -c 'import chrombpnet, os; print(os.path.dirname(os.path.dirname(os.path.realpath(chrombpnet.__file__))))')" || return 1
  if [[ "${where}" != "$(realpath "${expect}")" ]]; then
    echo "Error: ${CONDA_ENV} imports chrombpnet from ${where}, not ${expect}" >&2
    return 1
  fi
  echo "chrombpnet: ${where} ($(CHROMBPNET_DIR="${expect}" chrombpnet_head))"
}

# Small GPUs: total memory under SMALL_GPU_MIB (default 16000 MiB), i.e.
# Sherlock's 12 GB TITAN Xp and 11 GB RTX 2080 Ti. On those the scripts
# lower the batch sizes that would not fit (run_train_full.sh,
# run_contribution.sh); on anything larger every run keeps the pipeline's
# defaults.
SMALL_GPU_MIB="${SMALL_GPU_MIB:-16000}"

# Total memory of the job's (first) GPU in MiB, or 0 when there is none.
gpu_total_mib() {
  local mib
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
  echo "${mib:-0}"
}

small_gpu() {
  local mib
  mib="$(gpu_total_mib)"
  (( mib > 0 && mib < SMALL_GPU_MIB ))
}

# Fails unless the engine's framework can run a convolution on a GPU. A
# framework that cannot reach CUDA falls back to the CPU without complaint
# and the job then runs into its time limit days later; a broken cuDNN only
# shows at the first convolution. ALLOW_CPU=1 skips the check.
gpu_preflight() {
  local engine="$1"
  if [[ "${ALLOW_CPU:-0}" == "1" ]]; then
    echo "gpu preflight: skipped (ALLOW_CPU=1)"
    return 0
  fi
  local code mode
  # TensorFlow's DeepSHAP runs in a spawned second process (the shap fork
  # needs TF1 graph mode) while the first still holds the GPU it trained on.
  # A GPU in Exclusive_Process compute mode -- some clusters' default,
  # Sherlock's among them -- refuses the second with
  # CUDA_ERROR_DEVICE_UNAVAILABLE, an hour into the run.
  if [[ "${engine}" == "tensorflow" ]]; then
    mode="$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null | head -1)"
    if [[ "${mode}" == Exclusive_Process* ]]; then
      echo "Error: the GPU is in ${mode} compute mode, which refuses the TensorFlow engine's" \
           "DeepSHAP process once training holds the GPU. Ask Slurm for a shared-mode GPU:" \
           "site.env's SBATCH_SHARED_GPU_ARGS, which submit_all.sh gives the TensorFlow jobs." >&2
      return 1
    fi
  fi
  case "${engine}" in
    pytorch) code='
import torch
assert torch.cuda.is_available(), "torch sees no GPU"
x = torch.randn(2, 4, 64, device="cuda")
torch.nn.functional.conv1d(x, torch.randn(8, 4, 5, device="cuda")).sum().item()
print("gpu preflight: torch", torch.__version__, "on", torch.cuda.get_device_name(0))' ;;
    tensorflow) code='
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
assert gpus, "TensorFlow sees no GPU (CUDA 11.2 / cuDNN 8.1 not found?)"
with tf.device("/GPU:0"):
    float(tf.reduce_sum(tf.nn.conv1d(tf.random.normal([2, 64, 4]), tf.random.normal([5, 4, 8]), 1, "VALID")))
print("gpu preflight: tensorflow", tf.__version__, "on", tf.config.experimental.get_device_details(gpus[0]).get("device_name"))' ;;
    jax) code='
import chrombpnet.engines.jax  # noqa: F401  (cuDNN preload, KERAS_BACKEND)
import jax, jax.numpy as jnp
gpus = [d for d in jax.devices() if d.platform == "gpu"]
assert gpus, "JAX sees no GPU: %s" % jax.devices()
x = jax.device_put(jnp.ones((2, 4, 64)), gpus[0])
float(jax.lax.conv_general_dilated(x, jnp.ones((8, 4, 5)), (1,), "VALID").sum())
print("gpu preflight: jax", jax.__version__, "on", gpus[0].device_kind)' ;;
  esac
  python -c "${code}" || {
    echo "Error: ${engine} cannot use a GPU here. See the lines above; set ALLOW_CPU=1 to run on the CPU anyway." >&2
    if [[ "${engine}" == "jax" ]]; then
      echo "       GPU compute capability: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)." \
           "The env's cuDNN 9.27 ships precompiled kernels for 7.5 (Turing) and newer only." >&2
    fi
    return 1
  }
}

# Fails unless everything the pipeline loads works here: each Python module
# with a compiled part, exercised (an HDF5 file, a bigwig, a numba
# compilation, a PNG, a PDF), and each command-line tool it runs. A GPU check
# alone loads none of these, and a library built for a newer glibc than the
# node's (Sherlock's: 2.17) fails only when it is first loaded -- hours into a
# run. Also catches a missing package.
runtime_check() {
  local engine="$1" tool out failed=0
  case "${engine}" in
    pytorch) RUNTIME_MODULES="torch tangermeme.deep_lift_shap memelite pybigtools hdf5plugin" ;;
    tensorflow) RUNTIME_MODULES="tensorflow shap deeplift" ;;
    jax) RUNTIME_MODULES="chrombpnet.engines.jax jax keras ml_dtypes hdf5plugin" ;;
  esac
  RUNTIME_MODULES="${RUNTIME_MODULES}" python - <<'PY' || failed=1
import importlib, os, sys, tempfile
common = ["numpy", "scipy", "pandas", "sklearn", "h5py", "pyBigWig", "pyfaidx", "numba",
          "matplotlib", "weasyprint", "chrombpnet.CHROMBPNET", "modiscolite.io", "modiscolite.report",
          "modiscolite.tfmodisco"]
bad = []
for name in common + os.environ["RUNTIME_MODULES"].split():
    try:
        importlib.import_module(name)
    except Exception as error:
        bad.append(f"import {name}: {type(error).__name__}: {error}")

def exercise(label, fn):
    try:
        fn()
    except Exception as error:
        bad.append(f"{label}: {type(error).__name__}: {error}")

tmp = tempfile.mkdtemp()
def hdf5():
    import h5py, numpy as np
    with h5py.File(os.path.join(tmp, "t.h5"), "w") as f:
        f.create_dataset("x", data=np.arange(10), compression="gzip")
def bigwig():
    import pyBigWig
    bw = pyBigWig.open(os.path.join(tmp, "t.bw"), "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries(["chr1"], [0], ends=[10], values=[1.0])
    bw.close()
def jit():
    from numba import njit
    assert njit(lambda x: x + 1)(1) == 2
def png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot([0, 1]); plt.savefig(os.path.join(tmp, "t.png")); plt.close()
def pdf():
    import weasyprint
    weasyprint.HTML(string="<p>check</p>").write_pdf(os.path.join(tmp, "t.pdf"))
for label, fn in (("hdf5 write", hdf5), ("bigwig write", bigwig), ("numba compile", jit),
                  ("png", png), ("pdf", pdf)):
    exercise(label, fn)
for line in bad:
    print("runtime check:", line, file=sys.stderr)
print(f"runtime check: python modules {'OK' if not bad else 'FAILED'}")
sys.exit(1 if bad else 0)
PY
  local status
  for tool in "bedtools --version" "samtools --version" "bgzip --version" "tomtom -version" "bedGraphToBigWig"; do
    # Status captured, not tested by set -e: bedGraphToBigWig without
    # arguments prints its usage and exits 255.
    out="$(${tool} 2>&1)" && status=0 || status=$?
    if (( status == 127 )) || grep -qE "GLIBC|cannot open shared object|not found" <<< "${out}"; then
      echo "runtime check: ${tool%% *}: $(head -n 2 <<< "${out}" | tr '\n' ' ')" >&2
      failed=1
    fi
  done
  (( failed == 0 )) && echo "runtime check: command-line tools OK"
  if (( failed )); then
    echo "Error: ${engine}'s environment cannot run the pipeline here (above)." >&2
    return 1
  fi
}

# "<commit> <branch>" of CHROMBPNET_DIR. git first; CentOS 7's system git
# (1.8) has no `-C` and predates worktrees, which that checkout is, so the
# fallback reads .git itself: the worktree's gitdir, its commondir, loose
# and packed refs. Needs a python on PATH (any activated env has one).
chrombpnet_head() {
  local out
  if out="$(git -C "${CHROMBPNET_DIR}" rev-parse HEAD --abbrev-ref HEAD 2>/dev/null)"; then
    echo ${out}
    return 0
  fi
  python - "${CHROMBPNET_DIR}" <<'PY' 2>/dev/null || echo "unknown unknown"
import os, sys
root = sys.argv[1]
gitdir = os.path.join(root, ".git")
if os.path.isfile(gitdir):
    gitdir = open(gitdir).read().split(":", 1)[1].strip()
    gitdir = gitdir if os.path.isabs(gitdir) else os.path.join(root, gitdir)
common = gitdir
if os.path.isfile(os.path.join(gitdir, "commondir")):
    common = os.path.normpath(os.path.join(gitdir, open(os.path.join(gitdir, "commondir")).read().strip()))
head = open(os.path.join(gitdir, "HEAD")).read().strip()
if not head.startswith("ref:"):
    print(head, "detached")
    sys.exit()
ref = head.split(None, 1)[1]
branch = ref.rsplit("refs/heads/", 1)[-1]
for base in (gitdir, common):
    if os.path.isfile(os.path.join(base, ref)):
        print(open(os.path.join(base, ref)).read().strip(), branch)
        sys.exit()
for line in open(os.path.join(common, "packed-refs")):
    if line.rstrip().endswith(" " + ref):
        print(line.split()[0], branch)
        sys.exit()
sys.exit(1)
PY
}

# Records what produced a run, beside it.
write_provenance() {
  local out="$1"
  {
    echo "date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "host: $(hostname)"
    echo "slurm: job ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}"
    echo "engine: ${ENGINE}"
    echo "conda_env: ${CONDA_ENV} (${CONDA_PREFIX:-?})"
    echo "container: $(in_container && echo "${CONTAINER_IMAGE}" || echo none)"
    echo "chrombpnet_dir: ${CHROMBPNET_DIR}"
    local commit branch modified
    read -r commit branch <<< "$(chrombpnet_head)"
    echo "chrombpnet_commit: ${commit}"
    echo "chrombpnet_branch: ${branch}"
    if modified="$(git -C "${CHROMBPNET_DIR}" status --porcelain --untracked-files=no 2>/dev/null)"; then
      echo "chrombpnet_dirty: $(printf '%s' "${modified}" | grep -c .) modified tracked files"
    else
      echo "chrombpnet_dirty: unknown (no git that reads this checkout)"
    fi
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "extra_args: ${EXTRA_ARGS:-}"
    echo "batch_size: ${BATCH_NOTE:-the pipeline defaults}"
  } > "${out}"
}
