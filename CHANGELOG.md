# Changelog

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
