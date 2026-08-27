# Bugs and issues in the TensorFlow reference repository

Audit baseline: `/path/to/workdir/software/chrombpnet` at
commit `09938fdb4397ec0006510e5251e48920a505d4de` (`v1.0.1-3-g09938fd`).

The refactor intentionally preserves reachable reference behavior, including
the defects below, unless the issue is inherently tied to TensorFlow/Keras or to
obsolete packaging. Tests label preserved failures explicitly so they cannot be
mistaken for new regressions.

## Confirmed defects

### 1. Standalone QC commands cannot populate their required inputs

- Severity: high
- Reference: `chrombpnet/CHROMBPNET.py:25-29,40-44` and
  `chrombpnet/pipelines.py:194-196,406-410`
- Symptom: `qc` creates a new empty `auxiliary/` directory with
  `exist_ok=False`, then immediately reads `filtered.peaks.bed` and related
  files from that directory. Pre-populating them makes directory creation fail.
- Refactor status: preserved for strict command compatibility.

### 2. `pred_bw --debug-chr` is broken

- Severity: high
- Reference: `predict_to_bigwig.py:141-144,182-184,221-223` and
  `bigwig_helper.py:72-77`
- Symptom: `debug_chr` is a list but is compared to a chromosome string. The
  writer also reads `regions[i]` before assigning `i` on the first iteration.
- Refactor status: preserved.

### 3. Single-region BigWig prediction loses its batch dimension

- Severity: high
- Reference: `predict_to_bigwig.py:151,191,230`
- Symptom: unrestricted `np.squeeze` converts a `(1, output_length)` profile to
  one dimension, but downstream softmax and writing code require two.
- Refactor status: preserved.

### 4. Predicted total counts are reconstructed inconsistently

- Severity: medium
- Reference: `batchgen_generator.py:105-107`,
  `predict_to_bigwig.py:154,194,232`, and
  `marginal_footprinting.py:66-75`
- Symptom: the target is `log(1 + total_counts)`. Footprinting correctly uses
  `exp(logcount) - 1`, while prediction BigWigs use `exp(logcount)` and add one
  count per region.
- Refactor status: preserved.

### 5. Multi-model prediction statistics overwrite one another

- Severity: medium
- Reference: `predict_to_bigwig.py:154-160,194-200,232-238`
- Symptom: bias, combined, and no-bias predictions all receive the same
  `output_prefix_stats`; later models replace earlier statistics.
- Refactor status: preserved.

### 6. `--seed` does not seed trainable ChromBPNet initialization

- Severity: medium
- Reference: `chrombpnet_with_bias_model.py:103-110`
- Symptom: the no-bias network is instantiated before TensorFlow, NumPy, and
  Python seeds are set. The requested seed therefore does not control its
  initial weights.
- Refactor status: preserved by applying the training seed after combined-model
  construction. Bias-model initialization remains seeded as in the reference.

### 7. Softmax implementation can overflow

- Severity: medium
- Reference: `training/predict.py:63-65`,
  `predict_to_bigwig.py:111-113`, and
  `marginal_footprinting.py:50-52`
- Symptom: logits are centered by their mean instead of maximum before
  exponentiation. Large finite logits can produce infinities and NaNs.
- Refactor status: preserved in retained output utilities; model training uses
  PyTorch's stable `log_softmax` inside the loss.

### 8. GC-negative selection can hang or fail when candidates are exhausted

- Severity: high
- Reference: `get_gc_matched_negatives.py:89-106,147-153`
- Symptom: `adjust_gc` has an unbounded random walk when every GC bucket is
  exhausted, and directly indexes a missing chromosome group.
- Refactor status: preserved.

### 9. Negative subsampling can request more rows than exist

- Severity: medium
- Reference: `batchgen_generator.py:12-18`,
  `initializers.py:23-25`, and `find_chrombpnet_hyperparams.py:109-111`
- Symptom: sampling uses `replace=False` without bounding the requested number
  by the available nonpeaks, raising `ValueError` for sparse inputs or large
  ratios.
- Refactor status: preserved.

### 10. External-command failures are ignored and paths are shell interpolated

- Severity: high
- Reference: `CHROMBPNET.py:93-96,122-146`, `pipelines.py:147-150,361-368`,
  `reads_to_bigwig.py:46-76`, and `auto_shift_detect.py:75-79`
- Symptom: return codes from bedtools, Modisco, and BigWig conversion are not
  consistently checked. Several commands use `shell=True` or `os.system` with
  unquoted user paths, causing confusing downstream failures and unsafe path
  interpretation.
- Refactor status: preserved in copied preprocessing/workflow code.

### 11. The packaged legacy workflows and tests do not match the package

- Severity: high
- Reference: `setup.py:18-25`, `tests/full_workflow.sh`,
  `tests/genomewide_gc_bin_test.sh`, and `workflows/*.sh`
- Symptom: scripts invoke console commands not registered by `setup.py`, refer
  to removed `src/` paths, use stale option names, and expect model filenames
  such as `chrombpnet_wo_bias.h5` while current code writes
  `chrombpnet_nobias.h5`.
- Refactor status: documented compatibility artifacts are retained; the active
  `chrombpnet` command and new Python tests are the supported paths.

### 12. Source installation does not declare all runtime requirements

- Severity: medium
- Reference: `requirements.txt`, `setup.py:10`, and imports in
  `reads_to_bigwig.py`, `auto_shift_detect.py`, and plotting/report modules
- Symptom: pip installation omits direct dependencies including `pyBigWig` and
  `termcolor`; `license_files` names nonexistent `LICENSE.txt` rather than
  `LICENSE`. The supplied tests additionally assume external bioinformatics
  binaries without checking them.
- Refactor status: not preserved because the new conda specification and
  `pyproject.toml` must be installable.

### 13. Empty or edge-filtered sequence sets crash one-hot conversion

- Severity: medium
- Reference: `one_hot.py:18-19`, `data_utils.py:8-18`, and
  `auto_shift_detect.py:120-136`
- Symptom: `dna_to_one_hot([])` indexes `seqs[0]`; an empty chromosome split,
  fully edge-filtered BED, or strand with no sampled reads therefore crashes
  with `IndexError` instead of a domain-specific error.
- Refactor status: preserved.

### 14. Prediction metrics contain an uncontrolled random component

- Severity: low
- Reference: `training/metrics.py:75-91`
- Symptom: the shuffled-label JSD baseline uses the global NumPy RNG without a
  local seed, so repeated prediction/QC runs can produce different plots.
- Refactor status: preserved.

### 15. Assertion-based input validation disappears under optimized Python

- Severity: medium
- Reference: assertions throughout `CHROMBPNET.py`, `training/train.py`,
  hyperparameter selection, model construction, and BigWig helpers
- Symptom: running Python with `-O` disables correctness and bounds checks.
- Refactor status: retained framework-neutral assertions remain compatible;
  new PyTorch checkpoint/model boundaries use explicit exceptions.
