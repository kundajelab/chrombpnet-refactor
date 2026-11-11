# Shared by the expand scripts. Sourced, not executed.
#
# Three opt-in PyTorch changes in a chrombpnet-refactor checkout of the expand
# branch (site.env's CHROMBPNET_EXPAND_REPO, at 22d90cb), compared against the
# engines sweep's default PyTorch runs:
#
#   new      bias pipeline --bias-profile-only, then pipeline
#            --model-selection pearsonr-jsd: a profile-only bias model, and
#            the epoch kept by validation counts r + 1 - median JSD
#   distill  `chrombpnet distill`: one bias-free student from a dataset's five
#            engines-sweep PyTorch fold models (the teachers)
#              lofo   one student per fold k, trained on fold k's training
#                     chromosomes only, scored on fold k's test chromosomes
#              final  one student per dataset, trained on every chromosome for
#                     the median epoch kept by its LOFO students
#
# Everything else -- container, env activation, GPU and runtime checks, the
# ids table, row selection, provenance -- is the engines sweep's
# (../engines/_common.sh, sourced below with this checkout and root).
#
# Callers source ../_site.sh first (site.env); each can be overridden from the
# environment:
#
#   CHROMBPNET_DIR   the checkout (default: site.env's CHROMBPNET_EXPAND_REPO)
#   ROOT             where this comparison writes (default: site.env's
#                    EXPAND_ROOT); ${ROOT}/preprocess links to the engines
#                    sweep's nonpeaks
#   ENGINES_ROOT     (site.env) the engines sweep, whose pytorch/ runs are the
#                    old models and the teachers; its ids table, ENGINES_IDS
#                    (or IDS_FILE), is this comparison's too
#   ENGINES_SCRIPTS  the engines sweep's scripts (default: ../engines)

if ! declare -F require_site > /dev/null; then
  echo "Error: source ../_site.sh before $(basename "${BASH_SOURCE[0]}")" >&2
  exit 2
fi
CHROMBPNET_DIR="${CHROMBPNET_DIR:-${CHROMBPNET_EXPAND_REPO:-}}"
ROOT="${ROOT:-${EXPAND_ROOT:-}}"
_expand_missing=()
[[ -n "${CHROMBPNET_DIR}" ]] || _expand_missing+=(CHROMBPNET_EXPAND_REPO)
[[ -n "${ROOT}" ]] || _expand_missing+=(EXPAND_ROOT)
[[ -n "${IDS_FILE:-}" ]] || _expand_missing+=(ENGINES_IDS)
require_site ${_expand_missing[@]+"${_expand_missing[@]}"} ENGINES_ROOT
unset _expand_missing
ENGINES_SCRIPTS="${ENGINES_SCRIPTS:-${BLOG_SCRIPTS}/engines}"
export CHROMBPNET_DIR ROOT
# chrombpnet-pytorch's editable install is the main checkout; this one comes
# first on PYTHONPATH (it passes into the container), and activate_engine's
# import check confirms it.
case ":${PYTHONPATH:-}:" in
  *":${CHROMBPNET_DIR}:"*) ;;
  *) export PYTHONPATH="${CHROMBPNET_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
# shellcheck source=../engines/_common.sh
source "${ENGINES_SCRIPTS}/_common.sh"
ENGINE=pytorch
FOLDS=(0 1 2 3 4)

# Which fold models the students learn from (TEACHER_SET, exported so it
# reaches the container re-exec):
#   default  the engines sweep's PyTorch runs, chrombpnet-refactor (default);
#            students in ${ROOT}/distill
#   updated  this comparison's ${ROOT}/new runs, chrombpnet-refactor (updated);
#            students in ${ROOT}/distill-updated
TEACHER_SET="${TEACHER_SET:-default}"
export TEACHER_SET
teacher_tree() {
  case "${1:-${TEACHER_SET}}" in
    default) echo "${ENGINES_ROOT}/pytorch" ;;
    updated) echo "${ROOT}/new" ;;
    *) echo "Error: TEACHER_SET must be default or updated, not '${1:-${TEACHER_SET}}'" >&2; return 1 ;;
  esac
}
distill_root() {
  case "${1:-${TEACHER_SET}}" in
    default) echo "${ROOT}/distill" ;;
    updated) echo "${ROOT}/distill-updated" ;;
    *) echo "Error: TEACHER_SET must be default or updated, not '${1:-${TEACHER_SET}}'" >&2; return 1 ;;
  esac
}

# Teacher k's run of this row's dataset (default: the row's fold), from the
# TEACHER_SET's tree: the fold model, and the run a student of fold k is
# scored against.
old_run_dir() { echo "$(teacher_tree)/${ASSAY}/${ENCID}/fold_${1:-${FOLD_NUM}}"; }

# The five teachers of this row's dataset, one per fold; fails unless each
# finished.
teacher_paths() {
  local k dir
  TEACHERS=()
  for k in "${FOLDS[@]}"; do
    dir="$(old_run_dir "${k}")"
    if [[ ! -f "${dir}/COMPLETED" || ! -f "${dir}/model/models/chrombpnet_nobias.pt" ]]; then
      echo "Error: teacher fold ${k} is not a finished run: ${dir}" >&2
      return 1
    fi
    TEACHERS+=("${dir}/model/models/chrombpnet_nobias.pt")
  done
}

# Datasets in the ids table's order, "ASSAY<TAB>ENCID" per line: the final
# students' array runs over these.
dataset_list() {
  awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
              NF > 0 { key = $col["ASSAY"] "\t" $col["ENCID"]; if (!(key in seen)) { seen[key] = 1; print key } }' \
      "${IDS_FILE}" | tr -d '\r'
}

# Reads dataset $1 (0-based) into ASSAY and ENCID, and the ids row of its
# first fold (for BAM_PATH, PEAKS_BED_PATH, TF_DIR). Exits 0 with a SKIP line
# past the end, as select_row does.
select_dataset() {
  local index="$1" line row
  require_sweep REFERENCE_DIR GENOME CHROM_SIZES
  line="$(dataset_list | sed -n "$((index + 1))p")"
  if [[ -z "${line}" ]]; then
    echo "SKIP: array task ${index} is past the $(dataset_list | wc -l) datasets of ${IDS_FILE##*/}."
    exit 0
  fi
  IFS=$'\t' read -r ASSAY ENCID <<< "${line}"
  row="$(awk -F'\t' -v a="${ASSAY}" -v e="${ENCID}" '
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    NF > 0 { n++ }
    NF > 0 && $col["ASSAY"] == a && $col["ENCID"] == e { print n; exit }' "${IDS_FILE}")"
  read_row "${row}" || exit 1
}

# Runs a step and appends its wall time to <dir>/timings.tsv.
#   timed <dir> <step> <command...>
timed() {
  local dir="$1" name="$2" start end
  shift 2
  [[ -f "${dir}/timings.tsv" ]] || printf 'step\tseconds\tstart\tend\n' > "${dir}/timings.tsv"
  start="$(date +%s)"
  echo "=== ${name}: $(date '+%Y-%m-%d %H:%M:%S')"
  "$@"
  end="$(date +%s)"
  printf '%s\t%s\t%s\t%s\n' "${name}" "$((end - start))" "${start}" "${end}" >> "${dir}/timings.tsv"
  echo "=== ${name}: done in $(( (end - start) / 60 )) min"
}

# The profile scores under the name run_summarize.py --profile-stopgrad off
# reads (chrombpnet_nobias.midpoint.profile_scores.h5), for scores that were
# computed with the midpoint weighting: chrombpnet's default, unless the
# interpret record says --contribs-profile-stopgrad.
#   link_midpoint_scores <interpret_subsample dir>
link_midpoint_scores() {
  local dir="$1" record="$1/chrombpnet_nobias.interpret.args.json"
  [[ -f "${dir}/chrombpnet_nobias.profile_scores.h5" ]] || { echo "Error: no profile scores in ${dir}" >&2; return 1; }
  if python -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("contribs_profile_stopgrad") else 1)' \
      "${record}"; then
    echo "Error: ${record} says --contribs-profile-stopgrad; these are not midpoint scores" >&2
    return 1
  fi
  ln -sfn chrombpnet_nobias.profile_scores.h5 "${dir}/chrombpnet_nobias.midpoint.profile_scores.h5"
}

# Scores a bias-free student against fold k as teacher k's run was scored:
#   join      student + teacher k's bias_model_scaled.pt -> models/chrombpnet.pt
#   predict   that model on teacher k's test peaks (filtered.peaks.bed, the
#             fold's test chromosomes) and bigwig, as the pipeline's own
#             evaluation -> evaluation/chrombpnet_{predictions.h5,metrics.json}
#   contributions  DeepSHAP of the student on teacher k's 30K subsample
#             (its interpreted_regions.bed; the same seed, so the same
#             shuffled references) -> auxiliary/interpret_subsample/
#   footprints     the student's marginal footprints at the bias motifs on
#             the shared nonpeaks -> evaluation/chrombpnet_nobias_max_bias_response.txt
#
#   evaluate_student <eval run dir> <fold k>
# The student is <eval run dir>/model/models/chrombpnet_nobias.pt.
evaluate_student() {
  local run="$1" k="$2" teacher model
  teacher="$(old_run_dir "${k}")"
  model="${run}/model"
  local fold_json="${REFERENCE_DIR}/folds/fold_${k}.json"
  local regions="${teacher}/model/auxiliary/interpret_subsample/chrombpnet_nobias.interpreted_regions.bed"
  for f in "${model}/models/chrombpnet_nobias.pt" "${teacher}/model/models/bias_model_scaled.pt" \
           "${teacher}/model/auxiliary/filtered.peaks.bed" "${teacher}/model/auxiliary/data_unstranded.bw" \
           "${teacher}/model/auxiliary/motif_to_pwm.tsv" "${regions}" "${fold_json}" \
           "${ROOT}/preprocess/${ASSAY}/${ENCID}/fold_${k}/negatives.bed.gz"; do
    [[ -f "${f}" ]] || { echo "Error: missing ${f}" >&2; return 1; }
  done
  mkdir -p "${model}/evaluation" "${model}/auxiliary/interpret_subsample"
  timed "${run}" join python "${CHROMBPNET_DIR}/legacy/helpers/reformat_chrombpnet_h5.py" \
    -cnb "${model}/models/chrombpnet_nobias.pt" -bm "${teacher}/model/models/bias_model_scaled.pt" \
    -o "${model}/models"
  mv "${model}/models/chrombpnet_recompiled.pt" "${model}/models/chrombpnet.pt"
  timed "${run}" predict chrombpnet predict -m "${model}/models/chrombpnet.pt" -g "${GENOME_PATH}" \
    -c "${CHROM_PATH}" -bw "${teacher}/model/auxiliary/data_unstranded.bw" \
    -r "${teacher}/model/auxiliary/filtered.peaks.bed" -n None -fl "${fold_json}" \
    -op "${model}/evaluation/chrombpnet" -d "${ASSAY}" --device cuda
  timed "${run}" contributions chrombpnet contribs_bw -m "${model}/models/chrombpnet_nobias.pt" \
    -r "${regions}" -g "${GENOME_PATH}" -c "${CHROM_PATH}" \
    -op "${model}/auxiliary/interpret_subsample/chrombpnet_nobias" -pc counts profile --device cuda
  link_midpoint_scores "${model}/auxiliary/interpret_subsample"
  timed "${run}" footprints chrombpnet footprints -m "${model}/models/chrombpnet_nobias.pt" \
    -r "${ROOT}/preprocess/${ASSAY}/${ENCID}/fold_${k}/negatives.bed.gz" -g "${GENOME_PATH}" -fl "${fold_json}" \
    -op "${model}/evaluation/chrombpnet_nobias" -pwm_f "${teacher}/model/auxiliary/motif_to_pwm.tsv" \
    --device cuda
  for f in "${model}/evaluation/chrombpnet_metrics.json" "${model}/evaluation/chrombpnet_predictions.h5" \
           "${model}/auxiliary/interpret_subsample/chrombpnet_nobias.counts_scores.h5" \
           "${model}/evaluation/chrombpnet_nobias_max_bias_response.txt"; do
    [[ -f "${f}" ]] || { echo "Error: ${f} was not written" >&2; return 1; }
  done
}

# The engines sweep's runtime check requires pybigtools for PyTorch, which
# this package does not import (and chrombpnet-pytorch, rebuilt since, does not
# have): the same check without it.
eval "$(declare -f runtime_check | sed 's/ pybigtools//')"

# Under Slurm, outside the container, the jobs run in site.env's
# PT_ENV_GLIBC217 when it names one: a build of the PyTorch env for glibc 2.17
# nodes, where chrombpnet-pytorch's PyTorch 2.14 wheel (it needs glibc 2.28)
# cannot run. The one these ran in on Sherlock (the checkout's
# envs/chrombpnet-pytorch-sherlock.yml) has PyTorch 2.5.1, the engines
# sweep's, and chrombpnet-pytorch's tangermeme, modisco and memelite.
# Elsewhere, and in the container (--container), chrombpnet-pytorch as before.
eval "$(declare -f activate_engine | sed '1s/^activate_engine /_engines_activate_engine /')"
activate_engine() {
  local env="${3:-}"
  if [[ -z "${env}" && "$2" == "slurm" && -n "${PT_ENV_GLIBC217:-}" ]] && ! in_container; then
    env="${PT_ENV_GLIBC217}"
  fi
  _engines_activate_engine "$1" "$2" "${env}" "${4:-}"
}
