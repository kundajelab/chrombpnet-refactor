# Changelog

## `--engine` help describes the JAX engine

- The help still called `jax` "a placeholder, not implemented yet", and listed
  only `.pt` and `.h5` among the checkpoint suffixes that choose the engine
  when `--engine` is omitted. It now describes the JAX engine and lists
  `.keras`. A test checks the help against the engine registry: every engine
  and its checkpoint suffix. Help text only; no artifact changes.

## Smaller fixes, no output changes

- `density_scatter` (the counts scatter in the reports) crashed whenever
  predictions and observations had different NaN counts: the mask
  concatenated both `np.where` tuples. It now drops a point that is NaN in
  either.
- Matplotlib figures are closed once saved (counts scatter, histograms,
  marginal footprints, loss curves, the QC PWM), instead of accumulating for
  the life of the process. The saved images are byte-identical.
- `viz_sequence.plot_weights` returns the figure it draws.
- The Docker image sets `PYTHONNOUSERSITE=1` and empties `PYTHONPATH` and
  `PYTHONHOME`. Apptainer passes the host's `~/.local` and `PYTHONPATH` into
  the container, where they can shadow the image's packages. The image was
  not rebuilt to test this.
- All four follow kundajelab/chrombpnet#284. The PR closes the metrics
  plots only; the footprint, loss-curve and PWM figures are closed here too.

## TensorFlow and JAX engines: DeepSHAP honours the profile target's stop_gradient

- **Behaviour change: profile contribution scores, TensorFlow and JAX
  engines.** The profile target is `sum(softmax(stop_gradient(mn)) * mn)`
  over the mean-normalized logits `mn`. The stop_gradient makes the softmax
  fixed weights, so each logit of the input is weighted by `softmax(mn_x)`.
  The PyTorch engine computes exactly that, and is the reference. The
  kundajelab `shap` fork has its `StopGradient` handler commented out
  (`deep_tf.py:595`), so the original ChromBPNet, and with it the TensorFlow
  and JAX engines, treated the weights as a second varying input. The fork's
  two-input rule then weighted the logits by the midpoint
  `0.5 * (softmax(mn_x) + softmax(mn_r))` of the input's and the reference's
  softmax.
- TensorFlow: `attribution.honour_stop_gradient` registers the fork's
  `break_dependence` for `StopGradient` while the profile explainer is built
  and run, then restores the fork's table. JAX: `engine.attribute` passes
  `profile_weighting="softmax_x"`. `DeepLiftShap` keeps the PR's default,
  `"chrombpnet"`, which still reproduces the midpoint.
- Counts scores are unchanged. With the handler registered, the original's
  counts scores are bitwise identical.
- `tests/parity/tf_engine.py` registers the same handler on the reference
  side, and the engine still matches the original bitwise. Two checks are
  new. The handler leaves the original's counts scores bitwise unchanged.
  The engine's profile scores differ from the original's unpatched ones
  (relative L2 1.05 on the fixture). `tests/parity/jax_engine.py` compares
  against the patched original, and checks the same difference from the
  unpatched one.
- New `tests/parity/deepshap_engines.py` (opt-in
  `test_deepshap_engines.py`): one model with the same weights and
  references in all three engines. TensorFlow and JAX agree with PyTorch
  within 1.1e-6 relative L2 on the profile head and 1.3e-7 on counts. The
  midpoint weighting is 0.64 away.
- With the TensorFlow or JAX engine, the profile motif reports of
  `pipeline`, `qc`, `bias pipeline` and `bias qc` change with the scores.
  The counts reports do not. Nothing changes in the PyTorch engine.
- In the regression sweeps only profile-derived artifacts change: the
  profile scores, the profile MoDISco results, logos and reports, and the
  profile contribution bigwig. Models, predictions, counts scores and
  footprints are identical. The profile reports find different patterns,
  so the TensorFlow sweep now captures 149 artifacts (was 146) and the JAX
  sweep 149 (was 152).

## Motif-report settings on the command line

- **New `--interpret-subsample`, `--modisco-max-seqlets` and
  `--modisco-window`** on `pipeline`, `qc`, `bias pipeline` and `bias qc`.
  The motif reports at the end of these commands are built from contribution
  scores on 30 000 sampled peaks, with TF-MoDISco limited to 50 000 seqlets
  per pattern group and a 500 bp window, all fixed until now. The defaults are
  unchanged. `modisco_motifs` already has `--max-seqlets` and `--window`.
- The values are checked when parsed: positive integers, and an even window.
  In `pipeline` and `bias pipeline` a window wider than `--inputlen` is
  refused at startup rather than after training.
- The sample keeps its `auxiliary/30K_subsample_peaks.bed` name whatever its
  size.
- Off by default and dropped from the namespace when absent, so `args.json`
  is unchanged unless they are given. `tests/cli_surface.json` gains exactly
  their twelve entries. The names are kundajelab/chrombpnet#284's. Reusing
  `modisco_motifs`' short flags was not an option: `-n` is `--nonpeaks` in
  these commands. No artifact changes (regression harness).

## Bias report: negative patterns removed whole rows at a time

- **Behaviour change: one line of `bias qc`'s `overall_report.html`.** The
  bias report leaves the negative TF-MoDISco patterns out of its motif tables.
  It did so by dropping 15 lines from each `neg_` line, i.e. it assumed a
  13-column row. The logo paths in a row contain `neg_` as well and restart
  the count, so it also dropped the two lines after each negative row. With
  the negative rows last, as modiscolite writes them, that lost the table's
  `</tbody>`. Followed by a positive row, or with another row length (no motif
  database, other `top_n`), it would have cut rows in half.
- `modisco_table.drop_negative_patterns` now removes each row, `<tr>` through
  `</tr>`, whose first cell starts `neg_`, and leaves the rest of pandas'
  table as written. On the regression fixture the report gains its
  `  </tbody>` line and nothing else changes.
- The function's name and the whole-row approach are from
  kundajelab/chrombpnet#284. Its regex version puts `</tbody>` at the wrong
  indentation, so this one works line by line on pandas' layout.

## `--tomtom-lite`: motif reports without MEME

- **New `--tomtom-lite`** on `pipeline`, `qc`, `bias pipeline`, `bias qc` and
  `modisco_motifs` (not `train`/`bias train`, which never run reports).
  TF-MoDISco then matches motifs with TOMTOM-lite (memelite) instead of MEME's
  `tomtom`, so MEME is not needed. The report gives uncorrected p-values
  (`pval0..2`) instead of q-values, and its text says so.
- It needs TF-MoDISco 2.5 or later, so the TensorFlow environment
  (modisco-lite 2.3.2, the newest for Python 3.8) refuses it at startup.
- Off by default and dropped from the namespace when absent: `args.json`
  and every report are unchanged unless it is given (regression harness).
  `tests/cli_surface.json` gains exactly its five entries.
- From kundajelab/chrombpnet#284 (`modisco report-simple -l` there; here
  `report_motifs(ttl=True)`), including its report-wording helpers.

## A missing MEME `tomtom` fails at startup, not hours later

- `pipeline`, `qc`, `bias pipeline`, `bias qc` and `modisco_motifs` end with
  a TF-MoDISco motif report, and modiscolite looks up MEME's `tomtom` only
  then. The CLI now checks for it (`shutil.which("tomtom")`, the same
  lookup) before creating any directory, and exits with an error saying how
  to install it. `train` and `bias train`, which never run reports, are not
  checked.
- `modisco_motifs` on scores that yield no patterns used to finish without
  `tomtom`; it now requires it too.
- From kundajelab/chrombpnet#284, which checks the training commands but not
  `modisco_motifs`. No artifact changes.

## Clearer `reads_to_bigwig` failures

- A bedGraph with no reads left after filtering now stops with an error naming
  the input, the reference and the chrom sizes file, instead of reaching
  `bedGraphToBigWig` with an empty file. This changes behaviour only on that
  failure path.
- When `bedGraphToBigWig` fails, the error output says that every contig with
  reads must be in the chrom sizes file (`bedtools genomecov` only warns
  about them), then the error propagates as before.
- Uncompressed fragment files are read by `awk` directly instead of through
  `cat | awk`, with the same bytes. From kundajelab/chrombpnet#284. No
  artifact changes.

## gzip inputs with trailing garbage are accepted, with a warning

- **Behaviour change, input acceptance only.** A gzipped tagAlign or fragment
  file with data after its compressed stream makes gzip exit with status 2
  ("decompression OK, trailing garbage ignored") after writing the whole
  stream. chrombpnet 1.x accepted such files; this package raised. They are
  accepted again, with a `RuntimeWarning`, and give the same bigwig as the
  clean file. A truncated file (exit 1) still fails, and the error now names
  the command (`gzip -cd <file>`) rather than "input decompression".
- tagAlign inputs are decompressed with `gzip -cd` instead of `zcat`. The
  bytes are the same, since zcat runs `gzip -cd`, and on macOS/BSD `zcat`
  only reads `.Z` files.
- From kundajelab/chrombpnet#284; its tests are ported to
  `tests/test_preprocessing_streams.py`. No artifact changes (regression
  harness).

## The JAX engine: Keras 3 on JAX, from ChromBPNet 2.0

- **`--engine jax`** is no longer a placeholder. It is the Keras 3 / JAX port
  of the original ChromBPNet from
  [kundajelab/chrombpnet#284](https://github.com/kundajelab/chrombpnet/pull/284)
  ("ChromBPNet 2.0", NNFC-GMD/chrombpnet@14ca773). It brings the PR's
  architectures, loss, callbacks, `model.fit` call, loader for the original's
  `.h5` models, bias rescale and native JAX DeepSHAP behind the engine
  interface, on this package's shared preprocessing and downstream code. The
  PR's modules were imported verbatim first, and each module states what this
  package changes. Nothing else from the PR was taken.
- **Environment.** `envs/chrombpnet-jax.yml`: Python 3.12, NumPy 2, Keras
  3.15.1, `jax[cuda13]` 0.11.2 (NVIDIA driver >= 580; `jax[cuda12]` on older
  drivers). The shared code runs on NumPy 2 there; the one incompatibility
  was deepdish (see the next entry).
- **Checkpoints are `.keras`** (`<prefix>.keras`, `_nobias.keras`,
  `bias_model_scaled.keras`). The engine also reads the original's `.h5`
  models without TensorFlow (their bytecode count head is never run) and
  ChromBPNet 2.0's Keras 3 `.h5` files. Engine inference reads suffixes:
  `.keras` means JAX, an explicit `--engine` may name any engine that can read
  every input (`--engine jax -b original_bias.h5`), and inputs in both formats
  go to the engine that reads them all.
- **Agreement with the original**, checked by the new
  `tests/parity/jax_engine.py` on identical inputs, CPU, full float32. It
  passes all 67 checks:
  - predictions within 9.5e-7;
  - `test_on_batch` losses within 2.3e-7 relative;
  - a 20-step Adam trace within 2.6e-6 relative in loss and 1.9e-3 relative
    L2 in final weights;
  - the bias rescale within 4.8e-7;
  - DeepSHAP hypothetical scores within 8.8e-6 relative L2, with
    bitwise-identical references;
  - the original's 1.x models loaded within 6.0e-7;
  - a `.keras` round trip, exact;
  - `export --legacy-h5` files read by the TensorFlow engine and by the
    original within 9.5e-7.
  The tolerances are the PR's.
- **DeepSHAP references** are drawn as the TensorFlow engine draws them: one
  `RandomState(--seed)`, `--num-shuffles` single dinucleotide shuffles per
  sequence, in region order. The PR instead seeds each sequence from its own
  content. Each sequence's references are drawn exactly once, even across the
  explainer's prefetching and its out-of-memory retries. The profile head
  uses the `shap` fork's effective midpoint weighting, as the TensorFlow
  engine does. `--interpret-batch-size` counts (sequence, reference) pairs.
- **One `on_epoch_end` per pass.** Keras 3 also calls a dataset's
  `on_epoch_end` from `EpochIterator.reset()`, before the first epoch and
  before every validation pass, and it starts an extra pass to inspect the
  first batch. The PR's generator saw both. Here the adapter forwards
  `on_epoch_end` only after a pass begins, and the engine builds the model's
  loss and metrics before `fit`. The shared generator's augmentation (at
  `--benchmark max`, derived from `(seed, mode, epoch)`) therefore stays in
  step with the PyTorch and TensorFlow engines.
- **New options, JAX only** (refused for the other engines, and dropped from
  the namespace when absent, so `args.json` is unchanged): `--optimizer
  {adam,muon}`, `--muon-lr`, `--ema`, `--ema-momentum`, `--lr-schedule
  {constant,cosine}`, `--precision {default,highest,bf16}`. The defaults are
  the original recipe.
- **GPU:** on hosts whose login shell puts a system cuDNN on
  `LD_LIBRARY_PATH`, the pip-wheel cuDNN that jax[cuda13] brings would load
  the system cuDNN's engine libraries, and every GPU convolution would fail
  (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`). The engine preloads the
  bundled cuDNN in that case. Checked end to end with `--device cuda` on an
  L40S: `bias train`, `pred_bw` and `contribs_bw`.
- **`chrombpnet export --legacy-h5`** writes a JAX-engine model in the
  original's `.h5` layout, for the TensorFlow engine, chrombpnet 1.x,
  variant-scorer and bpnet-lite.
- `python -m tests.regression.harness ... --engine jax` runs the
  golden-artifact sweep through JAX: 152 artifacts across 10 scenarios,
  reproducible run to run. `.keras` archives are digested by content (save
  time and Keras's memory-address shared-object ids excluded).
- **No PyTorch or TensorFlow artifact moves**: the PyTorch sweep is 150/150
  identical to `main`, and the TensorFlow engine is still 42/42
  byte-identical to the original.

## `contribs_bw` reads contribution scores with h5py

- `importance_hdf5_to_bigwig` read `/projected_shap/seq` with
  `deepdish.io.load`, which no longer imports on NumPy 2. It now reads the
  same float16 array with h5py, with `hdf5plugin` for Blosc-compressed files
  written by the original package. Output is unchanged (regression harness).
- deepdish leaves `pyproject.toml`'s core dependencies; only `legacy/` uses it.

## TensorFlow engine: seeded DeepSHAP references

- **Behaviour change, TensorFlow engine only.** DeepSHAP's
  dinucleotide-shuffled references are now drawn from
  `np.random.RandomState(--seed)`, fresh for each head, so contribution
  scores are reproducible run to run. The original passed no generator to
  `deeplift.dinuc_shuffle`, which then drew from a new unseeded
  `RandomState` on every call, so two identical runs scored differently.
- `shap_utils.shuffle_several_times` gains an optional `rng`; with
  `rng=None` it behaves exactly as the original.
- `tests/parity/tf_engine.py` no longer patches the engine's shuffle. The
  engine seeds itself, and the reference worker is given the same per-head
  stream. A new check compares the CLI's spawned-subprocess path against the
  in-process scores, for both heads.

## TensorFlow engine: restore the best validation epoch when training runs out of epochs

- **Behaviour change, TensorFlow engine only.** The engine now saves the best
  validation epoch's weights in every case. The original saved the *last*
  epoch's whenever a run used all of `--epochs`:
  `EarlyStopping(restore_best_weights=True)` restores only when it actually
  stops training, and the final `model.save` then overwrote the best
  checkpoint that `ModelCheckpoint(save_best_only=True)` had written.
- After `fit`, the engine reloads that checkpoint's weights before the final
  save. When early stopping fired these are exactly the weights it had
  restored, so those runs are byte-identical to before; the optimizer state
  saved alongside is the final epoch's, as it always was. The no-bias model
  is now taken from the best epoch too.
- Gated by the `restore_best_checkpoint` benchmark control, which is on at
  every level the TensorFlow engine accepts. This is the behaviour the
  PyTorch engine already has, and what `save_best_only` and
  `restore_best_weights` together intend.
- `tests/parity/tf_engine.py` now checks this directly. Its out-of-epochs
  case is chosen so that the best epoch is not the last, and the engine's
  weights and no-bias model must equal the original's best-epoch checkpoint,
  which the reference worker snapshots before the original overwrites it.
  The early-stopping cases are still compared whole.

## TensorFlow engine: `--seed` now seeds the bias-factorized model's tower

- **Behaviour change, TensorFlow engine only.** The engine now seeds NumPy,
  TensorFlow and Python's `random` before building the model. The original
  implementation seeded only NumPy at that point: its bias-factorized builder
  seeds TensorFlow and `random` after loading the bias model and building the
  trainable tower. So `--seed` never reached that tower's initial weights,
  and two runs with the same seed started from different weights (defect 6 in
  `chrombpnet_bugs.md`, which the PyTorch engine already fixes).
- Only the tower's initial weights move. The builders still re-seed where
  they always did, so everything drawn after construction is unchanged,
  including Keras's batch shuffle. The bias model, which its builder seeds
  before building, is unaffected.
- `tests/parity/tf_engine.py` still passes all 38 checks, because it injects
  identical initial weights into both implementations. The new
  `tests/test_tensorflow_engine.py::test_seed_controls_the_bias_factorized_tower`
  pins the fix: it fails on the faithful port and passes now.

## The TensorFlow engine, and one environment per engine

- **`--engine tensorflow`**: the original Keras ChromBPNet
  (kundajelab/chrombpnet@09938fd), on this package's shared preprocessing,
  data generators and downstream code. Its model definitions, loss,
  callbacks, `model.fit` call, bias rescale and DeepSHAP code are the
  original's, imported verbatim and then adapted in a separate commit, so
  `git log -p chrombpnet/engines/tensorflow` shows every change. Checkpoints
  are Keras `.h5`, and models trained by the original package load and run
  through every command.
- **Byte-identical to the original**, given identical batches, initial
  weights and DeepSHAP references: `tests/parity/tf_engine.py` runs both
  implementations in the TensorFlow environment and requires exact equality
  of model configurations, predictions, losses, checkpoints (weights,
  optimizer state, Keras attributes), training logs, no-bias models, the
  bias rescale, both heads' contribution scores, and cross-loading. The
  original is checked against itself first. All 42 checks pass, in
  `chrombpnet-tensorflow` and in the original package's own environment.
  The engine departs from the original only in the three fixes listed above,
  each with its own entry.
- **Environments.** `envs/chrombpnet-pytorch.yml` (renamed from
  `environment.yml`, Python 3.11), `envs/chrombpnet-tensorflow.yml`
  (Python 3.8 with TF 2.8.0, Keras 2.8.0, TFP 0.15.0, protobuf 3.20.0 and
  numpy 1.23.4, exactly the original's), and `envs/chrombpnet-jax.yml` for
  the JAX placeholder. `pyproject.toml` moves torch/tangermeme to a
  `pytorch` extra, the TensorFlow stack to a `tensorflow` extra, and lowers
  `requires-python` to 3.8.
- **TF-MoDISco differs by environment.** Every TF-MoDISco release after
  modisco-lite 2.3.2 imports `memelite`, which needs Python >= 3.9, so the
  TensorFlow environment runs 2.3.2. The adapter code is shared, and
  `tests/test_modisco_compat.py` passes against both. For the same reason it
  runs WeasyPrint 61.2 (with pydyf 0.10.0). The original's 52.5 writes PDFs
  that differ from run to run.
- The shared code now runs on Python 3.8 with pandas 2.0. The one pandas
  2.1-only call (`DataFrame.map` in the report tables) falls back to
  `applymap`. `pytest` runs in every engine's environment and skips what that
  environment lacks: 206 passed and 14 skipped in `chrombpnet-tensorflow`.
- `python -m tests.regression.harness ... --engine tensorflow` runs the
  golden-artifact sweep through the TensorFlow engine: 146 artifacts across
  10 scenarios, reproducible run to run.
- **No PyTorch artifact moves.** Throughout the restructuring the PyTorch
  sweep reports all 150 artifacts identical to the previous `main`.

## `--engine`: choose the deep-learning framework

- **New `--engine {pytorch,tensorflow,jax}`** on every command that builds or
  loads a model: `pipeline`, `train`, `qc`, `bias pipeline|train|qc`,
  `predict`, `pred_bw`, `contribs_bw` and `footprints`. `prep` and
  `modisco_motifs` never touch a model and do not take it.
- **Omitted, it is inferred** from the input models' checkpoint suffixes
  (`.pt` PyTorch, `.h5`/`.hdf5` TensorFlow), else `pytorch`. An explicit
  `--engine` must agree with every input model, and input models from two
  engines are refused.
- `--predict-fk` and `--benchmark pytorch` are PyTorch-only and are refused
  for any other engine, whether it was typed or inferred.
- **No PyTorch artifact moves.** An omitted flag is removed from the argument
  namespace before any stage sees it, so `<prefix>.args.json` is
  byte-identical to before and records `engine` only when it was typed. The
  regression harness reports all 150 artifacts identical.
  `tests/cli_surface.json` gains exactly the ten `engine` entries.
- `jax` is a placeholder. It is accepted, and it fails before any output
  directory is created.
- The engine registry and interface live in `chrombpnet/engines/`; the
  PyTorch implementation moved to `chrombpnet/engines/pytorch/`, with the
  old module paths kept as aliases.

## `--data-aug` defaults to `both-half` on DNASE

- **Behaviour change.** An omitted `--data-aug` now selects `both-half` for
  `-d DNASE`, where it previously selected `standard`. ATAC is unaffected:
  an omitted flag still means `standard` there. Pass `--data-aug standard`
  explicitly for the old behaviour.
- The default is **assay-dependent**, which it has to be: `both-half`
  reshapes a bond, ATAC has not got one, and `conventions.resolve` raises for
  that combination. A single default name would therefore have made every
  unflagged ATAC run fail. So `--data-aug` now carries `default=None` and the
  resolution happens in `conventions.default_for`, where `--data-type` is
  known -- the same pattern `--benchmark` uses, and it keeps
  `<prefix>.args.json` a record of what was typed rather than of what it
  resolved to.
- `both-half` is preferred to the `right` shift for three reasons. A shift
  *corrects* the misregistration; splitting the cut across both bases
  flanking the bond *removes* the asymmetry that causes it, so there is
  nothing left to correct and no dependence on getting the sign right. It
  needs no extra label column, where a shift leaves one slot with no source.
  And the `0.5` scale preserves the track total exactly, so
  `counts_loss_weight` and the scale-invariant `-oth`/`-b` ratios carry over
  from `standard` unchanged. The cost is real and is the reason this is a
  choice rather than a fix: each cut is smeared over 2 bp, halving the
  target's effective resolution.
- **The regression harness cannot gate this.** Its fixture is ATAC-only, so
  every artifact digest is unchanged and a DNase default moving would be
  invisible to it. The proof is therefore direct: `tests/cli_surface.json`
  moves in exactly 7 entries and nothing else; new tests in
  `tests/test_preprocessing_conventions.py` run `reads_to_bigwig` with the
  flag omitted and assert the deposits land on both flanking bases, that the
  ATAC default is untouched, and that the track total still matches
  `standard`.
- `standard` is now opt-in rather than the default, so a test pins that it
  still means what it meant.

## `--ignore-bias-flag`: proceed past a failing bias-model check

- **New `--ignore-bias-flag`** on `pipeline` and `train`, downgrading
  `stages.check_bias_model_quality` from a `ValueError` to a warning.
- The check is worth keeping strict by default: a bias model whose predicted
  counts *anti*-correlate with observed peak counts is not describing
  sequence bias, and subtracting it makes the accessibility model worse.
- It is worth being able to override because the check cannot distinguish a
  bad model from bad data. Where peak and background counts overlap heavily
  -- primary tissue with a hot, diffuse background -- the count threshold
  falls below the nonpeak median, the bias model is fitted to a truncated
  left tail, and it anti-correlates for reasons no amount of retraining
  fixes. Proceeding is then a deliberate choice to study a sample the
  pipeline was not designed for.
- Opting out is audible, not silent: a 78-column banner on stdout plus a
  `RuntimeWarning`, and the message now carries the measured r and the
  threshold so the log records how bad it was.
- Only `pipeline` and `train` get the flag. `bias pipeline` trains a bias
  model but never gates on its peak-count correlation -- that check lives in
  the ChromBPNet pipeline, on the model it is handed -- so offering the flag
  there would imply a check that does not happen.

## Profile Pearson R in the evaluation JSON, always computed

- **New `metrics.profile_pearsonr`**: Pearson R across bases between the
  observed and predicted profile of each region. Computed unconditionally --
  there is no flag to enable it -- and recorded as `median_pearsonr` beside
  the existing JSD figures, for every subset, by both `predict` and
  `pred_bw`.
- It complements JSD rather than duplicating it. Both compare profile
  *shape*, but JSD is a divergence on normalized profiles while R is scale
  invariant and signed, so it distinguishes "wrong shape" from
  "anticorrelated shape". A constant, empty or non-finite profile has no
  defined R and reports `null` rather than 0.
- **The HTML and PDF tables are unchanged**, by design: the reports already
  carry counts Pearson R, and a test asserts that injecting
  `median_pearsonr` leaves all three rendered reports byte-identical. The
  figure is in the JSON for programmatic collation.
- Also pins the behaviour `--data-aug both-half` depends on: its 0.5-per-base
  deposits make counts non-integral, and the multinomial NLL's coefficient
  term is the only place integrality could matter. That term does not depend
  on the logits, so it shifts the reported loss without touching gradients,
  and the data term is linear in the counts -- `grad(half)` is exactly
  `0.5 * grad(full)`, same direction. `both-half` and `both-full` therefore
  optimize the same objective, which is what makes them each other's
  control. A test asserts the gradient identity and that every profile
  metric is scale invariant.

## `--data-aug`: the DNase reverse-complement convention, and `chrombpnet predict`

- **New `--data-aug`**, on `pipeline`, `train`, `qc`, their `bias`
  counterparts, `predict`, and `reads_to_bigwig`. The default, `standard`, is
  exactly today's behaviour and is byte-identical across the regression sweep.
- **The problem it exists to test.** DNase I cuts *between* two bases, so the
  0/+1 shift records a cut at the **bond**, indexed by the bond's right-hand
  base. Reverse-complement augmentation reverses the profile with no
  compensating offset, which carries the bond to the right place but leaves
  it named by the wrong base -- every reverse-complemented DNase training
  example is misregistered by one. ATAC's +4/-4 gives an odd-width footprint
  centred *on* a base and is unaffected.
- The geometry is not only derived (`chrombpnet/conventions.py`) but measured:
  the plus- and minus-strand cut PWMs shipped in
  `chrombpnet/data/{DNASE,ATAC}.ref.motifs.txt` align with each other's
  reverse complement at lag 0 for DNase and lag -1 for ATAC, across all nine
  references, and the required label shift is `lag + 1` in both cases. A test
  asserts it, so changing the deltas in `reads_to_bigwig` without changing
  the shift now fails loudly.
- The arms: `fr-only` / `rc-only` never / always reverse complement, in every
  split (validation and test previously never did); `right` adds the +1
  re-registration; `left` is its -1/0 mirror; `both-half` / `both-full`
  deposit on both flanking bases, which is symmetric about the bond and so
  needs no shift at all. `right`, `left` and `both-*` **raise** for ATAC
  rather than silently doing nothing -- a no-op there would return several
  identical numbers with nothing to say why.
- A shifted arm reads one extra base beyond the window so the leading slot
  holds real signal instead of a fabricated zero. Only `right` and `left` pay
  that; every other arm stays on the original `get_cts` code path.
- `both-half` conserves the track total exactly (`genomecov -scale 0.5`;
  bigWig values are float32 and 0.5 is binary-exact), and `both-full` doubles
  it. A test pins both.
- **New `chrombpnet predict`**: the prediction-and-metrics stage that `qc`
  runs first, without the attribution and MoDISco that follow it. Scoring a
  model against a given test split is what comparing two augmentation arms
  needs, and `training/predict.py` had no CLI of its own. Geometry is read
  off the checkpoint, as `qc` does.
- **The default is byte-identical.** Across the regression sweep 76 of 78
  artifacts are unchanged; the two that move are the `vars(args)` dumps
  gaining a `data_aug` key. Trained weights, bigWigs, predictions,
  contribution scores and footprints all reproduce exactly -- so the awk
  rewrite, the reallocation in `random_rev_comp`, the padded `get_cts` and
  the generator threading are all inert at `standard`.

## Bias models no longer reverse-complement their cached data in place

- **This changes trained bias models, and everything downstream of them.**
  `crop_revcomp_data`'s nonpeaks-only branch aliased the cached arrays rather
  than copying them, and the augmenter flips in place. That branch is the one
  every bias model takes, so each epoch reverse-complemented a fresh random
  half of an array that was *already* partly reverse-complemented. The
  intended 50/50 split became a random walk, and the `coords` strand column
  filled up with "r" and never returned to "f".
- Epoch 0 was always correct -- the cache is pristine the first time through
  -- so the divergence starts at epoch 1. The regression sweep shows exactly
  that: epoch 0's losses are identical, epoch 1's differ, and 46 of 78
  artifacts move, all of them downstream of the bias model. The two scenarios
  that do not depend on training (`prep splits`, `prep nonpeaks`) are
  untouched.
- It was survivable because plain reversal is an involution, so a
  double-flipped example is still valid data. That stops being true with the
  DNase reverse-complement shift, where a round trip destroys one bin, which
  is why this is fixed first and on its own.
- Retrain bias models to get the intended schedule. Existing ones are not
  invalid -- they saw real sequence throughout -- but their augmentation was
  not what the code claimed.

## `--optimize` removed; the fused-kernel forward becomes `--predict-fk`

- **`--optimize` is gone.** It existed so each optimization phase could be
  attributed during measurement; that measurement is finished and recorded
  below, so phases 1 and 3 are now simply how the package runs. The defaults
  do not change -- `--optimize 1,3` was already the default, so an ordinary
  run produces exactly the same bytes it did before.
- **`--predict-fk`** ("prediction using fused kernels") replaces
  `--optimize 2`, on `pred_bw`, `footprints`, `qc` and `bias qc`. It is the
  only optimization that was ever off by default, because it is the only one
  that changes the numbers: the Triton kernels are fp32-exact where the eager
  path runs TF32, so predictions move by ~1e-5. Not offered on `contribs_bw`,
  where DeepLIFT needs autograd and hooks that a fused kernel cannot provide.
- `chrombpnet/optimize.py` is replaced by `chrombpnet/fused_kernels.py`, which
  carries the single remaining choice. `$CHROMBPNET_OPTIMIZE` goes with it.
- The regression harness loses its `--optimize` passthrough.
- **The pre-optimization implementations are kept**, marked `Legacy:` and
  unreachable in normal operation: `one_hot._dna_to_one_hot_legacy`,
  `metrics._profile_metrics_looped`, and
  `model.PredictionAdapter._predict_legacy`. Each is pinned by a test
  asserting it still agrees with the shipping path, so "kept for reference"
  cannot quietly become "kept, and wrong".
- CLI surface: three `optimize` actions removed, four `predict_fk` added. No
  other action changes.
- **One recorded artifact moves**, and it is the flag's own record:
  `contribs_bw` dumps `vars(args)` into `<prefix>.interpret.args.json`, which
  gained an `optimize` key when the flag arrived and now loses it. Across the
  regression sweep that is the only difference -- 77 of 78 artifacts are
  byte-identical, which is the point: the removed switch's default was
  already what shipped.

## `--benchmark`: TensorFlow parity is the default, and `lite` is gone

- **Omitting `--benchmark` now means TensorFlow parity everywhere**, including
  from library callers. The level it selects was called `lite`; it is now
  called `tensorflow`, and `lite` is no longer accepted on the command line.
  Argparse rejects it and lists the valid names, so a stale script fails
  loudly rather than drifting.
- **`--benchmark none` is now `--benchmark pytorch`** -- PyTorch's own native
  initializations, Adam epsilon and loss aggregation, with no parity controls.
- `medium`, `max`, and the bare `--benchmark` (still `max`) are unchanged.
- **The one behavioural change** is for callers that reach the library with no
  `benchmark` attribute at all. That used to mean *no controls*, while the
  argparse default meant `lite` -- so `chrombpnet qc`, and any code building a
  namespace by hand, silently got PyTorch's native settings where a training
  run got parity ones. Absent and default are now the same thing. For `qc`
  this changes nothing observable: every generator-side control is `medium` or
  above, and `keras_batch_order_shuffle` is only read by `train.py`.
- **Two recorded artifacts move.** `<prefix>.args.json` dumps `vars(args)`, so
  an unflagged run now records `"benchmark": null` instead of `"benchmark":
  "lite"` -- the parser default is `None`, so the record says what was
  actually typed. Checkpoint provenance records the *normalized* level, so it
  reads `tensorflow` where it read `lite`. Nothing else about any output
  changes.
- Existing `.pt` files keep loading: the normalizer still reads `lite` as a
  deprecated alias for `tensorflow`, so old provenance is not a load error.

## Optimization phases, and fixed-step bigWigs by default

- Every optimization in the package is now selectable with `--optimize`, on
  `pred_bw`, `contribs_bw` and `footprints`. `none` runs the original code
  paths; `1` is the levers ported from the ChromBPNet optimization kit other
  than its Triton forward; `2` is that Triton forward; `3` is the
  optimizations that are not from the kit. They compose: `--optimize 1,3`,
  or `all`. The default is `1,3`.
- **This changes bigWig bytes.** Phase 3 includes a fixed-step writer: the
  same values in a different on-disk section type, 2.8x faster to write and
  1.5x smaller. `pyBigWig`, `deeptools` and the UCSC tools read identical
  values back; a byte-for-byte comparison against an older run will not
  match. Use `--optimize 1` for the previous encoding.
- Nothing else about the default output moves. Measured over 5,000 hg38
  regions with three models: `--optimize 1` reproduces all 18 artifacts of
  `--optimize none` exactly, and `--optimize 1,3` differs only in the three
  bigWigs.
- Phase 2 moves the numbers slightly, by design, and is off by default. Its
  kernels are fp32-exact where the eager path runs TF32 convolutions, so it
  is closer to an fp64 reference than the default is (1.0e-05 against
  1.1e-03), but predictions, metrics and plots all shift accordingly.
- `contribs_bw` records the setting: its `<prefix>.interpret.args.json`
  dumps `vars(args)`, so it gains an `optimize` key. That is the only change
  to that file, and it is worth having -- the record of a run should say
  which optimizations produced it.
- The regression harness takes `--optimize` too, so the whole scenario sweep
  can be replayed per phase: `python -m tests.regression.harness against
  --ref main --optimize none`.
- Wall clock on an L40S for that workload: 55.7s at `none`, 42.8s at `1`,
  38.2s at `3`, 29.6s at `1,3`. Phase 2 is 53.0s warm and pays a one-time
  JIT per machine on its first run.

## Test suite - libstdc++ preload no longer depends on collection order

- `chrombpnet/__init__.py` preloads the environment's `libstdc++.so.6` so that
  SciPy's compiled extensions can resolve `CXXABI_1.3.15` on clusters whose
  login shell puts an older system copy first. The preload only works if it
  runs before `torch`, which binds the system library on import.
- Nothing enforced that ordering, so it held only by luck of alphabetical
  collection. `pytest tests/test_models.py tests/test_metrics.py` imported
  torch first and died during collection with an `ImportError`. A new
  `tests/conftest.py` imports the package up front, which pytest loads before
  any test module.
- `tests/test_tensorflow.py`'s torch worker had the same latent ordering bug
  and is fixed the same way.

## PyTorch refactor - pred_bw metrics near chromosome edges
- `pred_bw --bigwig` passed the *unfiltered* region frame to
  `compare_with_observed`, while the predictions it was compared against
  covered only the regions that survived edge filtering. `get_seq` drops any
  region whose input window runs off the end of a chromosome, so as soon as
  one region sat near an edge the observed counts and the predictions
  disagreed in length. In practice this raised
  `RuntimeError: Invalid interval bounds!` from the observed-bigwig read
  rather than silently misaligning.
- The filtered frame is now used for both the `_preds.bed` output and the
  metrics comparison. Runs whose regions were all in bounds are unaffected;
  runs that previously crashed now produce metrics.

## PyTorch refactor - reports show four decimal places
- Scores in `overall_report.html` and `overall_report.pdf` are now formatted to
  four decimal places instead of two. Two was too coarse to compare runs: a
  pearsonr of 0.9012 and one of 0.9049 both printed as `0.9`.
- Values are formatted rather than merely rounded, so a score of 9 prints as
  `9.0000` and lines up with its neighbours. The underlying `*_metrics.json`
  is untouched and still carries full precision.
- Controlled by `report_shell.REPORT_DECIMALS`.

## Removed CITATION.cff
- The citation now lives directly in `README.md` rather than in a separate
  file that nothing pointed at except one line of prose.

## PyTorch refactor - the ChromBPNet report now rounds its metrics
- `make_html` computed `pd.json_normalize(...).round(2)` and then immediately
  overwrote the result with the unrounded frame on the next line, in all four
  of its metrics tables. The ChromBPNet report therefore displayed
  full-precision numbers (`0.488836`) where the bias report, which had a single
  call, displayed rounded ones (`0.49`).
- Both reports now round, via one shared helper. **This changes
  `overall_report.html` and `overall_report.pdf` for ChromBPNet runs**; the
  underlying `*_metrics.json` is unaffected and still carries full precision.

## PyTorch refactor - DNase footprinting and the footprint plot window
- The DNase marginal-footprinting table listed `dnase_2` with `dnase_1`'s
  sequence (`TTTACAAGTCCA`), so every DNase run footprinted the same motif
  twice and never footprinted the intended second one. The packaged
  `chrombpnet/data/motif_to_pwm.DNASE.tsv` has `TGTACTTACGAA` for `dnase_2`;
  the inserted table now matches it, and a test keeps the two in step.
  **This changes DNase footprint output.** ATAC is unaffected -- its five Tn5
  motifs already matched the packaged table exactly.
- Marginal footprint plots hardcoded a 200 bp window while building the x axis
  as an independent 200-element range, so any model with `outputlen < 200`
  failed with "x and y must have same first dimension" instead of plotting.
  The window is now clamped to the profile length. Models with
  `outputlen >= 200` plot exactly the same data as before.
- The footprint plots' x-axis label read "200bp arount motif insertion" and is
  now "<width>bp around motif insertion". This changes the footprint PNGs --
  measured as 1832 pixels per plot, all in the label -- but not the plotted
  curves or `*_footprints.h5`.

## PyTorch refactor - qc and bias qc are now reproducible
- `chrombpnet qc` and `chrombpnet bias qc` previously reported different
  metrics on every run for the same model and inputs. Neither command exposes
  `--seed` or `--benchmark`, so the batch generator's RNG fell back to the
  global `np.random` stream, which nothing on the qc code path ever seeds --
  `np.random.seed` is only called in `train.get_model`, which qc never reaches.
  `subsample_nonpeak_data` then drew a different 1:1 nonpeak subsample for the
  test set each time.
- Measured across two identical `bias qc` runs on a synthetic fixture:
  `coords/coords_center` in `bias_predictions.h5` differed by up to 32475 (an
  entirely different region set), `predictions/logcounts` by 0.05, and the
  counts pearsonr in `overall_report.html` moved from `-0.0` to `-0.21`.
- Test-mode generators now draw from a dedicated seeded stream at every
  benchmark level. Test-set nonpeak selection is data selection rather than
  augmentation, and the validation split already seeded the equivalent choice
  unconditionally. Training is unaffected: its generators keep the streams they
  had, so trained weights do not change.
- **This changes qc output.** The reported metrics are now a specific
  reproducible draw instead of a random one. Existing qc numbers are not
  comparable to new ones, though they were not comparable to each other either.

## PyTorch refactor - lite benchmark settings are now the default
- `--benchmark` now defaults to `lite` when the flag is omitted, so ordinary
  runs match the TensorFlow implementation this package is a port of:
  Glorot/Xavier Conv1d/Linear initialization with zero biases, Adam epsilon
  1e-7, Keras-style sample-weighted epoch losses, Keras-style shuffled batch
  visit order, a seeded count-threshold nonpeak draw, and restoration of the
  best-validation checkpoint. Shuffling and augmentation stay random, and no
  deterministic-kernel or TF32 constraints are applied, so throughput is
  unchanged.
- A bare `--benchmark` still means `max`, as before.
- Added `--benchmark none` to get PyTorch's own native defaults (Kaiming
  initialization, Adam epsilon 1e-8, per-batch loss averaging, sequential batch
  order). Use it to reproduce runs made before this change.

## PyTorch refactor - contribution-scoring batch size
- Added `-ibs/--interpret-batch-size` to every command that runs contribution
  scoring (`pipeline`, `train`, `qc`, `bias pipeline`, `bias train`,
  `bias qc`, `contribs_bw`). It defaults to the training batch size divided by
  four rather than reusing it, because DeepLIFT-SHAP needs roughly four times
  the GPU memory of a training step at the same batch size: examples and their
  references go through the model together, and Tangermeme's hooks cache a
  clone of the input and output of every non-linear module on top of the
  activations autograd already saves. Previously the interpret step inherited
  `-bs` directly, so the default 64 peaked at 18.3 GB for the 512-filter model
  and raised CUDA OOM on any GPU smaller than about 24 GB.

## PyTorch refactor - TF-MoDISco 2.6 migration
- Routed motif discovery and compact HTML reports through a shared Python API
  adapter, and ported legacy motif inspection and attribution-hit utilities.
- Removed imports of the deleted `modisco.visualization`,
  `tfmodisco_workflow`, and `densityadapted_hitscoring` modules.

##  Version - 1.5
- Fixed issue #150, regions_used not found while generating bigwigs from impotance h5s

## Version - 1.4
- (MAJOR) Bug in chrombpnet modisco_motifs command. seqlets was limited to 50000. If users wanted to change it to 1 million this did not happen.
- Filter peaks at edges for pred_bw command and bias pipleline. So bias evaluation now done on these filtered peaks.
- Preprocessing deafulted to use unix sort. Provided option to switch to bedtools sort.
- Provided option to use filter chromosomes option in preprocessing.

## Version - 1.3 - Inworks - 2022-12-11
- Changed pipelines to use modisco-lite, old modisco will soon be removed
- Added automatic shift scripts to repo and integrated with the pipeline
- modisco now outputs both html and pdf. pdf can be shared with anyone.
- Use ATAC, DNASE not ATAC_PE, DNASE_SE anymore.
- Simplyfing workflows to include only two main workflows, chrombpnet_train_tf_model, chrombpnet_train_bias_model
- Restructuring README, moving tutorial to additional documentation and introducing FAQ, and only two pipelines (chrombpnet_train_tf_model, chrombpnet_train_bias_model)

## [Unreleased] - 2022-02-28
- Typo fix - (PR#31-36)
- We can now specify ylimit in marginal footprinting plots (PR#27)
- PR#30 merged  to do fast genome wide gc binning and bug fix to ensure case-sensitve GC calculation. Adds a unit test to make checks.

## [Unreleased] - 2022-02-11
- Updated modisco version - (PR#25)

## [Unreleased] - 2022-02-07
- Typo fix - (PR#23-24)

## [Unreleased] - 2022-01-31
- Marginal footprinting no longer hardcoded to take only chr1, it now inputs the fold json. (PR#20)
- Invivo footprinting is made compatible with the new input repo changes. (PR#22)
- Pandas read bugs fixed for gc-matching scripts. (PR#15)
- Tutorial now exits when one part of the pipeline breaks instead of proceeding further. (PR#18)
- Added the fast binning scripts at src/helpers/make_gc_matched_negatives/get_genomewide_gc_buckets (PR#11)
- Added pseudocount in metrics.py (PR#13) - needs further review.

## [Unreleased] - 2022-01-24
- Seed setting fixed for training. Seed was set improperly - preventing shuffling for every epoch. (PR#19)
- return coordinates while training for debugging - returns status of the point such as revcomp,peak/nonpeak,coordinates etc

## [Unreleased] - 2022-01-15
- In `step6_train_chrombpnet_model` changed the file saving name bug  
- Changed scripts in `src/evaluatuion/make_bigwigs/` to be compatible with new dir structure

## [Unreleased] - 2022-01-13
- Added a note in README that users can continue to use the pre-trained bias model from chrombpnet-lite repo, if thats what they have been using till now
- Changed the default modisco crop setting in tutorial from 1000 to 500
- Fixed a bug in `src/evaluation/modisco/run_modisco.py` in directory creation
