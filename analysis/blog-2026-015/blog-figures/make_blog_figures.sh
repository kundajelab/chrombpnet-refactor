#!/bin/bash
# Rebuilds blog 2026-015's summary figures from the analyses' saved tables
# (no rescoring), with this repository's plotting scripts:
#   chrombpnet_engine_performance.png    engines sweep: test scores by engine
#   chrombpnet_engine_contribution.png   engines sweep: DeepLIFT/DeepSHAP vs the original
#   chrombpnet_refactor_default_updated.png
#                                        expand: test scores and DeepLIFT/DeepSHAP vs the original,
#                                        chrombpnet-refactor (default) and (updated) only
#   chrombpnet_distill_ensemble_student.png
#                                        expand: predictions and DeepLIFT/DeepSHAP vs the five-fold mean
# The original model has no ENCSR868FGK fold 2 scores; prepare_inputs.py fills them
# with the mean of its other four folds (NOTE.txt in each inputs directory says so).
#
#   bash make_blog_figures.sh
#
# From site.env (../site.env.example): BLOG_DIR (where the figures go), BLOG_FIGURES_WORK
# (inputs and intermediate plots), ENGINES_SUMMARY (engines/run_summarize.sh's output,
# profile-stopgrad-off; default from ENGINES_ROOT), EXPAND_ANALYSIS (expand's analysis
# directory; default from EXPAND_ROOT), CONDA_ROOT. Overrides: WORK, CONDA_ENV.

set -euo pipefail
if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"
HERE="${BLOG_SCRIPTS}/blog-figures"
BLOG_FIGURES_WORK="${WORK:-${BLOG_FIGURES_WORK:-}}"
ENGINES_SUMMARY="${ENGINES_SUMMARY:-${ENGINES_ROOT:+${ENGINES_ROOT}/analysis/profile-stopgrad-off}}"
EXPAND_ANALYSIS="${EXPAND_ANALYSIS:-${EXPAND_ROOT:+${EXPAND_ROOT}/analysis}}"
require_site BLOG_DIR BLOG_FIGURES_WORK ENGINES_SUMMARY EXPAND_ANALYSIS
WORK="${BLOG_FIGURES_WORK}"
if [[ -z "${CONDA_ENV:-}" ]]; then
  require_site CONDA_ROOT
  CONDA_ENV="${CONDA_ROOT}/envs/chrombpnet-pytorch"
fi
PY="${CONDA_ENV}/bin/python"
ENGINES="${ENGINES_SUMMARY}"
EXPAND="${EXPAND_ANALYSIS}"
PLOT="${CHROMBPNET_REPO}/analysis/engine/plot_summarize.py"

# The engines sweep's metrics predate base-2 JSD (e4f6460); the expand summary converted its own.
"${PY}" "${HERE}/prepare_inputs.py" "${ENGINES}" "${WORK}/engines" --fill-original ENCSR868FGK:2 \
  --jsd-base-e pytorch tensorflow jax original original_retrained
"${PY}" "${PLOT}" --dir "${WORK}/engines"
"${PY}" "${HERE}/prepare_inputs.py" "${EXPAND}/profile-stopgrad-off" "${WORK}/expand" \
  --drop student-lofo student-final --fill-original ENCSR868FGK:2
"${PY}" "${PLOT}" --dir "${WORK}/expand" --reference "chrombpnet-refactor (default)" --noun model \
  --colour "chrombpnet-refactor (default)=green" "chrombpnet-refactor (updated)=orange"

# The distillation plots, redrawn from compare_fold_means.py's tables.
mkdir -p "${WORK}/fold-mean-default"
cp "${EXPAND}"/fold-mean-default/fold_mean_{predictions,contributions}.tsv "${WORK}/fold-mean-default/"
"${PY}" "${HERE}/replot_fold_means.py" "${WORK}/fold-mean-default" "chrombpnet-refactor (default)"

cp "${WORK}/engines/summary_scores.png" "${BLOG_DIR}/chrombpnet_engine_performance.png"
cp "${WORK}/engines/summary_contributions_pearson.png" "${BLOG_DIR}/chrombpnet_engine_contribution.png"
"${PY}" "${HERE}/stack.py" "${BLOG_DIR}/chrombpnet_refactor_default_updated.png" \
  "${WORK}/expand/summary_scores.png" "${WORK}/expand/summary_contributions_pearson.png"
"${PY}" "${HERE}/stack.py" "${BLOG_DIR}/chrombpnet_distill_ensemble_student.png" \
  "${WORK}/fold-mean-default/fold_mean_predictions.png" "${WORK}/fold-mean-default/fold_mean_contributions_pearson.png"
echo "figures written to ${BLOG_DIR}"
