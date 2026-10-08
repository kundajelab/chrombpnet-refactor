# Golden-artifact regression harness

Replays the real `chrombpnet` command line against a synthetic fixture and
checks that every file it produces is byte-for-byte what it was before your
change. It exists because the rest of the suite asserts shapes, schemas and
closed-form identities but never pins an actual trained value — `test_end_to_end`
would pass through an arbitrary change in model output.

## Use

```bash
# 1. Record what a known-good revision produces.
git stash            # or check out the revision you trust
python -m tests.regression.harness capture --out /tmp/cbp-baseline

# 2. Make your changes, then check nothing moved.
python -m tests.regression.harness compare --baseline /tmp/cbp-baseline
```

Or in one step, which does the checkout for you in a temporary worktree:

```bash
python -m tests.regression.harness against --ref main
```

Useful flags: `--scenario 04_train` (repeatable) to narrow the sweep, and
`--reproducibility max` to give the training commands `--reproducibility-max`
(the deterministic-kernel level; `--reproducibility on` gives them
`--reproducibility`). Omitting it runs them with neither flag.

A successful compare prints `all N artifacts identical`. A failure names each
changed artifact and explains it in format-aware terms — which state-dict tensor
moved and by how much, which HDF5 dataset, how many bigwig positions.

## Engines

The sweep runs through any of the three deep-learning engines, from that
engine's environment:

```bash
# chrombpnet-pytorch (the default; argv exactly as written in scenarios.py)
python -m tests.regression.harness against --ref main

# chrombpnet-tensorflow
python -m tests.regression.harness capture --engine tensorflow --out /tmp/tf-base
python -m tests.regression.harness compare --engine tensorflow --baseline /tmp/tf-base

# chrombpnet-jax
python -m tests.regression.harness capture --engine jax --out /tmp/jax-base
python -m tests.regression.harness compare --engine jax --baseline /tmp/jax-base
```

For `--engine tensorflow` the harness swaps every `.pt` path for `.h5`,
appends `--engine tensorflow` to each command that takes it, and pins
TensorFlow's own thread pools (`TF_NUM_INTRAOP_THREADS`,
`TF_NUM_INTEROP_THREADS`) and `SOURCE_DATE_EPOCH`, which fontTools stamps into
the font subsets WeasyPrint 61.2 embeds in every PDF. PyTorch runs get none of
that: not even an explicit
`--engine pytorch`, which `<prefix>.args.json` would record. A baseline
records its engine in `engine.txt`, and `compare` refuses a mismatch.

A healthy TensorFlow sweep captures **149 artifacts across 10 scenarios**
(PyTorch: 150). The difference is all in the qc scenarios' motif logos, whose
number depends on which patterns TF-MoDISco finds for each engine's model --
and the TensorFlow environment runs TF-MoDISco 2.3.2, the newest release that
installs on Python 3.8, where the PyTorch one runs the git checkout.

Keras serializes the bias-factorized model's logsumexp `Lambda` as marshalled
bytecode that embeds its defining file's absolute path, so
`artifacts.hdf5_digest` digests that one attribute by the bytecode's
instructions, constants and names. Every other HDF5 attribute digests exactly
as before.

For `--engine jax` the `.pt` paths become `.keras`, and JAX runs on the CPU
platform only (`JAX_PLATFORMS=cpu`, so no CUDA context even where a GPU is
visible) with XLA's CPU thread pool at one thread. A healthy JAX sweep
captures **149 artifacts across 10 scenarios**, reproducibly; again only the
motif-logo count differs. `.keras` archives are digested by content
(`artifacts.keras_archive_digest`): Keras stamps the time of saving into each
one and marks shared objects with a memory address, so their bytes differ
between two saves of the same model.

The TensorFlow engine's *byte-identity with the original Keras ChromBPNet*, and
the JAX engine's agreement with it, are different questions from "did my
change move the output", and they have their own harnesses:
`tests/parity/tf_engine.py` and `tests/parity/jax_engine.py`.

**Checkouts other than the installed one.** `capture` and `compare` run the
CLI as `python -m chrombpnet.CHROMBPNET` from the workspace, so it imports
whichever `chrombpnet` the environment resolves -- with an editable install,
the checkout it was installed from. To test another checkout (a worktree, a
detached copy), put it on `PYTHONPATH` and set
`CHROMBPNET_EXPECT_PACKAGE_ROOT` to it, so the harness verifies the CLI really
imports it rather than silently comparing the installed code with itself.

## Why baselines are not committed

The artifacts depend on the installed PyTorch build, so a committed digest would
be a per-environment constant that goes stale on every dependency bump. The
question this harness answers — "did *my change* move the output" — only ever
compares two runs in one environment, so baselines are captured locally.

## What makes the comparison valid

Every artifact the CLI writes is byte-identical across runs, including `.pt`
checkpoints, matplotlib PNGs, WeasyPrint PDFs, bigwigs and the HTML reports.
That holds only because three things are pinned:

- **The output path.** Several artifacts embed absolute paths: the PDFs
  reference their images by absolute path and `<prefix>.args.json` is a dump of
  the argparse namespace. Every run therefore executes in one fixed directory
  (`RUN_DIR`), and a captured baseline keeps a *copy* of that tree so mismatches
  can still be explained. This is why the harness does not use pytest's
  `tmp_path`. Override with `CHROMBPNET_REGRESSION_RUN_DIR` to run two
  environments side by side -- the worktree and baseline that `against --ref`
  creates are derived from it, so one override isolates a whole invocation — but use the same value for a capture and the
  compare that reads it.
- **CPU, single-threaded.** `--device cpu` throughout, with `OMP_NUM_THREADS=1`
  and friends, because threaded CPU reductions do not have a fixed summation
  order. At `tensorflow` no deterministic-kernel constraint is applied at all,
  so GPU runs are not reproducible.
- **`MPLBACKEND=Agg`**, so plot rendering does not depend on the display or the
  font cache.

## Both reproducibility levels matter

Assert at `--reproducibility max` *and* with neither flag. `max` catches
numeric changes. No flag is the stricter test: it is the only level at which the
augmentation RNG is the *global* `np.random` stream that `train.py` also draws
its batch-order permutation from, so a change in the **number** of global draws
is observable in the trained weights there and invisible at `max`.

This is verified rather than assumed. Injecting one extra
`np.random.random_sample()` into `augment.random_crop` moves 29 artifacts,
including every `tf_model.*` tensor in both checkpoints and 8000 bigwig
positions.

## Fixture constraints

`fixture.py` is small but three of its constants are pinned by hardcoded values
elsewhere in the package, and `test_regression.py` guards each one:

| Constraint | Why |
|---|---|
| `OUTPUTLEN >= 200` | `marginal_footprinting.py:145` plots `footprint[outputlen//2-100 : outputlen//2+100]` against `range(200)` |
| `INPUTLEN >= 500` | the qc scenarios run motif discovery with the default `--modisco-window`, 500 (`api.run_motifs`), and `centered_slice` rejects any contribution track shorter than the window |
| under 20 000 reads | `auto_shift_detect.sample_reads` pipes through `shuf` with no `--random-source`; above `2 * num_samples` it really subsamples and the shift estimate stops being reproducible |

The genome is long (200 kb/chromosome) but reads and training regions occupy
only the first 40 kb. `prep nonpeaks` slops and excludes every peak, so it needs
a lot of unexcluded sequence to draw GC-matched candidates from, while the read
ceiling above caps how much of it can be covered by reads. GC matching reads
sequence only, so confining the reads satisfies both.

The Tn5 consensus is planted at each read anchor because `compute_shift`
requires all reference motifs to agree on one standard shift, which random
sequence does not produce. `fixture._validate_anchor_spacing` enforces that no
two plantings overlap; if they do, the averaged PWM stops matching the reference
and preprocessing fails with "Input shift is non-standard".

## Coverage

Ten scenarios: `prep splits`, `prep nonpeaks`, `bias train`, `train`,
`bias qc`, `qc`, `pred_bw`, `contribs_bw`, `footprints`, `modisco_motifs`.
Together they exercise the modules that had no regression coverage at all,
which are also the ones with the worst duplication: both report generators,
both hyperparameter finders, `reads_to_bigwig`, `auto_shift_detect`,
`build_pwm_from_bigwig`, `importance_hdf5_to_bigwig` and all three model
branches of `predict_to_bigwig`.

The three scenarios that run motif discovery need `modiscolite`, which is only
the optional `[reports]` extra; they skip where it is absent.

## `qc` and `bias qc` are compared

They used not to be. Neither command seeded the global `np.random` stream that
draws the test set's 1:1 nonpeak subsample, so two identical runs reported
different metrics (counts pearsonr `-0.0` in one, `-0.21` in the other).
98bd410 made them reproducible. Since then `05_bias_qc` and `06_qc` are
compared byte for byte like the other scenarios, and the `nondeterministic=`
marker in `scenarios.py` is unused.


## Comparing optimization phases

`--optimize` reaches `pred_bw`, `contribs_bw` and `footprints`, so the whole
sweep can be replayed per phase:

```bash
python -m tests.regression.harness capture --out /tmp/base --optimize none
python -m tests.regression.harness compare --baseline /tmp/base --optimize 1
```

Capture one phase, compare the other **against the same tree**. Do not reach
for `against --ref` here: it passes the flag to the ref's CLI as well, so any
ref older than the flag fails outright.

Phase 1 reproduces `--optimize none` across all ten scenarios, with one
expected exception: `08_contribs_bw/contrib.interpret.args.json` dumps
`vars(args)` and therefore records the flag's own value.

Phase 2 cannot be exercised here at all. The fixture is `-fil 4`, and the
Triton route refuses any width outside `{128, 512}` by design -- the counts
kernel would otherwise produce zeros silently. `tests/test_triton_route.py`
carries that route's verification instead.
