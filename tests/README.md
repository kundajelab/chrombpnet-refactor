# Testing

```bash
pytest                                   # unit and CLI tests, offline

# Check that a change did not move any CLI output. Replays ten real
# `chrombpnet` invocations against a synthetic fixture and compares every
# produced file -- checkpoints, HDF5, bigwigs, PNGs, PDFs, HTML, logs --
# against a baseline captured from another revision.
python -m tests.regression.harness against --ref main

# Check the CLI surface: every argument declaration against a committed
# snapshot, plus the full --help text of every subcommand.
pytest tests/test_cli_surface.py
tests/dump_cli_help.sh > /tmp/help.txt

# Cross-framework parity, opt-in: needs a second environment alongside this
# one. Compares 13 arrays and a predictions HDF5 at atol=1e-6, rtol=1e-5.
CHROMBPNET_TF_PARITY=1 pytest tests/test_tensorflow.py

# TensorFlow engine, from the chrombpnet-tensorflow environment: agreement
# with the original Keras implementation, run in its own chrombpnet
# environment (opt-in, several minutes) ...
CHROMBPNET_TF_ENGINE_PARITY=1 pytest tests/parity/test_tf_engine.py
# ... and the golden-artifact sweep run through --engine tensorflow.
python -m tests.regression.harness capture --engine tensorflow --out DIR

# JAX engine: agreement with the original (needs chrombpnet, the original's
# environment, chrombpnet-tensorflow and chrombpnet-jax side by side; opt-in,
# several minutes) ...
CHROMBPNET_JAX_ENGINE_PARITY=1 pytest tests/parity/test_jax_engine.py
# ... and the golden-artifact sweep, from the chrombpnet-jax environment.
python -m tests.regression.harness capture --engine jax --out DIR

# DeepSHAP across engines: the TensorFlow and JAX engines against PyTorch,
# whose semantics are the reference (needs all three environments; opt-in).
CHROMBPNET_DEEPSHAP_ENGINES=1 pytest tests/parity/test_deepshap_engines.py
```

`pytest` runs in every engine's environment; tests that need a framework the
environment lacks are skipped.

Run the regression harness with the environment's `bin/` on `PATH` (the
scenarios call `bedtools` and `bedGraphToBigWig`). Run from any checkout
other than the one the environment's editable install points at, set
`PYTHONPATH=<checkout>` and `CHROMBPNET_EXPECT_PACKAGE_ROOT=<checkout>`, or
the scenarios import the installed copy instead of the code under test.

See [`regression/README.md`](regression/README.md) for what makes the
artifact comparison valid and which fixture constants are pinned by
hardcoded values elsewhere in the package.
