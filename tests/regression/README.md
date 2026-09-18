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
`--benchmark max` to run the deterministic-kernel level. Omitting `--benchmark`
uses the CLI default, `lite`.

A successful compare prints `all N artifacts identical`. A failure names each
changed artifact and explains it in format-aware terms — which state-dict tensor
moved and by how much, which HDF5 dataset, how many bigwig positions.

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
  environments side by side — but use the same value for a capture and the
  compare that reads it.
- **CPU, single-threaded.** `--device cpu` throughout, with `OMP_NUM_THREADS=1`
  and friends, because threaded CPU reductions do not have a fixed summation
  order. At `lite` no deterministic-kernel constraint is applied at all, so GPU
  runs are not reproducible.
- **`MPLBACKEND=Agg`**, so plot rendering does not depend on the display or the
  font cache.

## Both benchmark levels matter

Assert at `--benchmark max` *and* at the default `lite`. `max` catches numeric
changes. `lite` is the stricter test: it is the only level at which the
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
| `INPUTLEN >= 500` | the qc pipelines run motif discovery through `pipelines._run_modisco_motifs`, whose `window` defaults to 500 and is not reachable from the CLI |
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
