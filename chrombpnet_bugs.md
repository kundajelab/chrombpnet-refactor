# Bugs and issues in the TensorFlow reference repository

Audit baseline: `/path/to/workdir/software/chrombpnet` at
commit `09938fdb4397ec0006510e5251e48920a505d4de` (`v1.0.1-3-g09938fd`).

The issues below were discovered in the TensorFlow reference and initially
preserved for parity. They are now fixed in the PyTorch refactor. Cross-framework
tests continue to enforce numerical model parity while PyTorch-only regression
tests enforce the corrected behavior.

## Confirmed defects

### 1. Standalone QC commands cannot populate their required inputs

- Severity: high
- Reference: `chrombpnet/CHROMBPNET.py:25-29,40-44` and
  `chrombpnet/pipelines.py:194-196,406-410`
- Symptom: `qc` creates a new empty `auxiliary/` directory with
  `exist_ok=False`, then immediately reads `filtered.peaks.bed` and related
  files from that directory. Pre-populating them makes directory creation fail.
- Refactor status: **fixed**. QC consumes the user-provided peak/nonpeak files,
  and creates or reuses only output directories.

### 2. `pred_bw --debug-chr` is broken

- Severity: high
- Reference: `predict_to_bigwig.py:141-144,182-184,221-223` and
  `bigwig_helper.py:72-77`
- Symptom: `debug_chr` is a list but is compared to a chromosome string. The
  writer also reads `regions[i]` before assigning `i` on the first iteration.
- Refactor status: **fixed**. Filtering happens before sequence extraction and
  list-valued chromosome filters are handled consistently by the writer.

### 3. Single-region BigWig prediction loses its batch dimension

- Severity: high
- Reference: `predict_to_bigwig.py:151,191,230`
- Symptom: unrestricted `np.squeeze` converts a `(1, output_length)` profile to
  one dimension, but downstream softmax and writing code require two.
- Refactor status: **fixed**. Profile arrays are reshaped with an explicit batch
  dimension and single-region prediction is regression-tested.

### 4. Predicted total counts are reconstructed inconsistently

- Severity: medium
- Reference: `batchgen_generator.py:105-107`,
  `predict_to_bigwig.py:154,194,232`, and
  `marginal_footprinting.py:66-75`
- Symptom: the target is `log(1 + total_counts)`. Footprinting correctly uses
  `exp(logcount) - 1`, while prediction BigWigs use `exp(logcount)` and add one
  count per region.
- Refactor status: **fixed**. Every total-count reconstruction now uses
  `expm1(logcount)`.

### 5. Multi-model prediction statistics overwrite one another

- Severity: medium
- Reference: `predict_to_bigwig.py:154-160,194-200,232-238`
- Symptom: bias, combined, and no-bias predictions all receive the same
  `output_prefix_stats`; later models replace earlier statistics.
- Refactor status: **fixed**. Each model writes to a model-specific statistics
  filename.

### 6. `--seed` does not seed trainable ChromBPNet initialization

- Severity: medium
- Reference: `chrombpnet_with_bias_model.py:103-110`
- Symptom: the no-bias network is instantiated before TensorFlow, NumPy, and
  Python seeds are set. The requested seed therefore does not control its
  initial weights.
- Refactor status: **fixed**. Python, NumPy, CPU Torch, and CUDA Torch seeds are
  applied before model construction.

### 7. Softmax implementation can overflow

- Severity: medium
- Reference: `training/predict.py:63-65`,
  `predict_to_bigwig.py:111-113`, and
  `marginal_footprinting.py:50-52`
- Symptom: logits are centered by their mean instead of maximum before
  exponentiation. Large finite logits can produce infinities and NaNs.
- Refactor status: **fixed**. Output utilities subtract the row maximum before
  exponentiation and validate their input rank.

### 8. GC-negative selection can hang or fail when candidates are exhausted

- Severity: high
- Reference: `get_gc_matched_negatives.py:89-106,147-153`
- Symptom: `adjust_gc` has an unbounded random walk when every GC bucket is
  exhausted, and directly indexes a missing chromosome group.
- Refactor status: **fixed and redesigned**. Candidate GC values are integer
  NumPy bins, candidates are randomized once and sorted into contiguous pools,
  and `searchsorted` selects the nearest available bin. Sampling is without
  replacement and insufficient fold pools raise an immediate `ValueError`.

### 9. Negative subsampling can request more rows than exist

- Severity: medium
- Reference: `batchgen_generator.py:12-18`,
  `initializers.py:23-25`, and `find_chrombpnet_hyperparams.py:109-111`
- Symptom: sampling uses `replace=False` without bounding the requested number
  by the available nonpeaks, raising `ValueError` for sparse inputs or large
  ratios.
- Refactor status: **fixed**. Requested samples are bounded by available rows in
  training, validation, and hyperparameter selection.

### 10. External-command failures are ignored and paths are shell interpolated

- Severity: high
- Reference: `CHROMBPNET.py:93-96,122-146`, `pipelines.py:147-150,361-368`,
  `reads_to_bigwig.py:46-76`, and `auto_shift_detect.py:75-79`
- Symptom: return codes from bedtools, Modisco, and BigWig conversion are not
  consistently checked. Several commands use `shell=True` or `os.system` with
  unquoted user paths, causing confusing downstream failures and unsafe path
  interpretation.
- Refactor status: **fixed**. Active subprocesses use argument arrays, checked
  return codes, explicit pipelines, and Python-managed output files.

### 11. The packaged legacy workflows and tests do not match the package

- Severity: high
- Reference: `setup.py:18-25`, `tests/full_workflow.sh`,
  `tests/genomewide_gc_bin_test.sh`, and `workflows/*.sh`
- Symptom: scripts invoke console commands not registered by `setup.py`, refer
  to removed `src/` paths, use stale option names, and expect model filenames
  such as `chrombpnet_wo_bias.h5` while current code writes
  `chrombpnet_nobias.h5`.
- Refactor status: **fixed**. Legacy shell workflows are explicitly archived;
  the installed `chrombpnet` tree and maintained pytest/parity suites are the
  executable workflow and test surface.

### 12. Source installation does not declare all runtime requirements

- Severity: medium
- Reference: `requirements.txt`, `setup.py:10`, and imports in
  `reads_to_bigwig.py`, `auto_shift_detect.py`, and plotting/report modules
- Symptom: pip installation omits direct dependencies including `pyBigWig` and
  `termcolor`; `license_files` names nonexistent `LICENSE.txt` rather than
  `LICENSE`. The supplied tests additionally assume external bioinformatics
  binaries without checking them.
- Refactor status: **fixed**. Runtime, reporting, genomics-tool, and test
  dependencies are declared and installation is validated with `pip check`.

### 13. Empty or edge-filtered sequence sets crash one-hot conversion

- Severity: medium
- Reference: `one_hot.py:18-19`, `data_utils.py:8-18`, and
  `auto_shift_detect.py:120-136`
- Symptom: `dna_to_one_hot([])` indexes `seqs[0]`; an empty chromosome split,
  fully edge-filtered BED, or strand with no sampled reads therefore crashes
  with `IndexError` instead of a domain-specific error.
- Refactor status: **fixed**. Empty input produces a typed `(0, 0, 4)` array and
  shift estimation raises a domain-specific error when either strand is empty.

### 14. Prediction metrics contain an uncontrolled random component

- Severity: low
- Reference: `training/metrics.py:75-91`
- Symptom: the shuffled-label JSD baseline uses the global NumPy RNG without a
  local seed, so repeated prediction/QC runs can produce different plots.
- Refactor status: **fixed**. Metrics use a local, configurable NumPy generator.

### 15. Assertion-based input validation disappears under optimized Python

- Severity: medium
- Reference: assertions throughout `CHROMBPNET.py`, `training/train.py`,
  hyperparameter selection, model construction, and BigWig helpers
- Symptom: running Python with `-O` disables correctness and bounds checks.
- Refactor status: **fixed**. Runtime validation uses explicit exceptions or
  `argparse` errors throughout active package code and remains enabled under
  `python -O`.
