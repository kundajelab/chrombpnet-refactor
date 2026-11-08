#!/bin/bash
# Rebuilds blog 2026-015's summary figures from the analyses' saved tables
# (no rescoring), with the main checkout's plotting scripts:
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
#   env: CHROMBPNET_DIR (main checkout), BLOG_DIR, WORK (inputs and plots), CONDA_ENV

set -euo pipefail
OAK=/path/to/workdir
HERE="${OAK}/software/chrombpnet-refactor/analysis/blog-2026-015/blog-figures"
CHROMBPNET_DIR="${CHROMBPNET_DIR:-${OAK}/software/chrombpnet-refactor}"
BLOG_DIR="${BLOG_DIR:-${OAK}/blogs/genomicsxai.github.io/content/blogs/2026-015}"
WORK="${WORK:-${OAK}/chromatlas/test/blog-figures}"
CONDA_ENV="${CONDA_ENV:-${OAK}/software/miniconda3/envs/chrombpnet-pytorch}"
PY="${CONDA_ENV}/bin/python"
ENGINES="${OAK}/chromatlas/test/chrombpnet-refactor-engines/analysis/profile-stopgrad-off"
EXPAND="${OAK}/chromatlas/test/chrombpnet-refactor-expand/analysis"
PLOT="${CHROMBPNET_DIR}/analysis/engine/plot_summarize.py"

# The engines sweep's metrics predate base-2 JSD (eff8b48); the expand summary converted its own.
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
