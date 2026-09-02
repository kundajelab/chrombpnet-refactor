# Legacy archive

This directory contains historical scripts and analysis assets that are not
called by the supported `chrombpnet` CLI or its pipelines. They are retained
for provenance only, are not covered by the compatibility contract, and are
excluded from the installed `chrombpnet-refactor` package.

Contents:

- `workflows/`: pre-refactor TensorFlow/Keras training and tutorial workflows.
- `tests/`: obsolete shell integration tests for removed commands and `.h5`
  checkpoints.
- `modisco/`: the former standalone three-step motif workflow. Active motif
  discovery lives in `chrombpnet.evaluation.modisco.api`.
- `invivo_footprints/`: paper-analysis motif-hit calling helpers. These are
  unrelated to the supported `chrombpnet footprints` command.
- `figure_notebooks/`: historical paper and exploratory notebooks.
- `helpers/`: superseded standalone wrappers and checkpoint utilities.
- `preprocessing/`: hard-coded scripts used to generate documentation assets.

Files here may contain site-specific paths and are not expected to run without
manual adaptation. The supported workflow surface is documented by
`chrombpnet --help`.
