# Changelog

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
