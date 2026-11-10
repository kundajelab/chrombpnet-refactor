# Engines and their environments

ChromBPNet runs on one of four deep-learning engines, two of them on
TensorFlow: `tensorflow` (the JAX engine's Keras 3 code) and `tensorflow-old`
(the original's Keras 2 code). Everything else --
preprocessing, GC-matched negatives, the data generators, metrics, bigwigs,
footprints, motif discovery and the reports -- is shared, framework-neutral
code; DeepLIFT/DeepSHAP is the only downstream step that differs between
engines.

## Environments

Each engine has its own conda environment. All three run Python 3.12 and
NumPy 2, but their frameworks cannot share one: each brings its own CUDA and
cuDNN wheels, at releases the others do not accept. `chrombpnet-tensorflow`
serves both TensorFlow engines: Keras 3 for `tensorflow`, and tf-keras (Keras
2) for `tensorflow-old`. Create them from the repository root:

| Engine | Environment file | Python and framework | GPU libraries |
|---|---|---|---|
| PyTorch (default) | `chrombpnet-pytorch.yml` | Python 3.12, PyTorch 2.14.1 | the PyTorch wheel's CUDA 12.6 build; falls back to the CPU |
| TensorFlow (`tensorflow`, `tensorflow-old`) | `chrombpnet-tensorflow.yml` | Python 3.12, TensorFlow 2.21 with tf-keras 2.21 and Keras 3.15.1 | `tensorflow[and-cuda]`: CUDA 12.5 and cuDNN 9.3 as pip wheels |
| JAX | `chrombpnet-jax.yml` | Python 3.12, Keras 3.15, JAX 0.11.2 | `jax[cuda12]`: CUDA 12 and cuDNN 9 as pip wheels |

Each file installs the package in editable mode with its engine's extra
(`-e ..[pytorch]` and so on) and applies the supplemental pip requirements
in `requirements-<engine>.txt`. Every environment also supplies the
command-line tools preprocessing shells out to (`bedtools`, `samtools`,
`bedGraphToBigWig`), MEME's `tomtom` for the motif reports, and the
Pango/Cairo stack WeasyPrint needs for the PDF reports.

Notes on GPUs:

- **PyTorch**: the CUDA 12.6 build runs on any CUDA 12 driver (525 or newer)
  and has kernels for compute capability 5.0 to 9.0, Maxwell to Hopper. For
  Blackwell GPUs (10.0, 12.0) install the `cu130` build instead (driver 580
  or newer), or `torch==2.13.0` from `https://download.pytorch.org/whl/cu129`
  on an older driver; `requirements-pytorch.txt` says where.
- **TensorFlow**: the wheels need any CUDA 12 driver (525 or newer).
  `requirements-tensorflow.txt` pins cuDNN at 9.3, the release TensorFlow
  2.21 is built against: newer cuDNN 9 releases have precompiled kernels
  only for compute capability 7.5 and up, and convolutions then fail on
  Pascal and Volta GPUs (P100, TITAN Xp, V100). For the same reason the
  engine loads its bundled CUDA libraries first where a system CUDA or
  cuDNN on `LD_LIBRARY_PATH` has the same names, as the JAX engine does.
  Memory growth is on (`TF_FORCE_GPU_ALLOW_GROWTH=true`),
  because `tensorflow-old`'s contribution scoring runs in a second process
  on the same GPU while the first still holds what it trained with. For the same reason
  the GPU must not be in Exclusive_Process compute mode.
- **JAX**: the `jax[cuda12]` wheels need any CUDA 12 driver (525 or newer);
  `jax[cuda13]==0.11.2` also works, with driver 580 or newer. In practice
  the engine needs a GPU of compute capability 7.5 (Turing) or newer: the
  cuDNN 9 build the wheels bring has precompiled kernels only for those,
  so convolutions fail on Pascal and Volta GPUs (P100, TITAN Xp, V100).
  A system CUDA or cuDNN on `LD_LIBRARY_PATH` would shadow the libraries the
  wheels bring, so the engine loads the bundled ones first when it finds
  namesakes there.
- **glibc**: the PyTorch 2.14 wheels need glibc 2.28 or newer, TensorFlow
  2.21's and jaxlib 0.11's 2.27. On an older system (CentOS 7 has 2.17) run
  the environments in a container with a newer base, such as Ubuntu 22.04,
  or, for PyTorch, create `chrombpnet-pytorch-glibc.yml`: PyTorch 2.5.1,
  the last release built for glibc 2.17, on Python 3.11 and NumPy 1.26; for
  both TensorFlow engines, `chrombpnet-tensorflow-glibc.yml`: TensorFlow
  2.20, the last release built for glibc 2.17, with tf-keras 2.20 and Keras
  3.15.1, on Python 3.11 and NumPy 1.26. JAX has no glibc 2.17 build of the
  release this package uses (0.11; the last is 0.7), so on such hosts Keras 3
  runs natively as `--engine tensorflow`, or the JAX environment in a
  container.
- Every command accepts `--device cpu`, `--device cuda`, or a specific
  device such as `--device cuda:1`.

## Choosing an engine

```bash
chrombpnet pipeline --engine pytorch ...         # the default
chrombpnet pipeline --engine tensorflow ...      # Keras 3, from the chrombpnet-tensorflow env
chrombpnet pipeline --engine jax ...             # from the chrombpnet-jax env
chrombpnet pipeline --engine tensorflow-old ...  # chrombpnet 1.x's Keras 2 code, from the chrombpnet-tensorflow env
chrombpnet pred_bw -cm models/chrombpnet.h5 ...  # engine inferred: tensorflow-old
```

`--engine` is accepted by every command that builds or loads a model. When it
is omitted the input models' checkpoint suffixes decide (`.pt` PyTorch, `.h5`
tensorflow-old, `.keras` the installed Keras 3 engine: JAX where it is
installed, else `tensorflow`), else PyTorch. An explicit `--engine` must be
able to read every input model: `--engine jax`, `--engine tensorflow` and
`--engine pytorch` also read the original's `.h5` models (bias, full and
no-bias), so `chrombpnet pipeline --engine pytorch -b original_bias.h5 ...` or
`chrombpnet pred_bw --engine pytorch -cm model.chrombpnet.fold_0.<ENCID>.h5 ...`
works, and inputs in two formats (`-bm bias.h5 -cm chrombpnet.keras`, or `-bm
bias.h5 -cm chrombpnet.pt`) go to an installed engine that reads both. PyTorch
converts the weights on load (`chrombpnet/engines/pytorch/keras_h5.py`: the same
arrays as bpnet-lite's `BPNet.from_chrombpnet`, the lengths read from the file);
in Python, `load_model("model.chrombpnet.fold_0.<ENCID>.h5")` returns the
`ChromBPNet` and `save_model` keeps it as a `.pt`. The two Keras 3 engines read
each other's `.keras` files. `prep` and `modisco_motifs` never touch a
model, take no `--engine`, and run in any of the environments.

| | PyTorch | tensorflow | JAX | tensorflow-old |
|---|---|---|---|---|
| environment | `chrombpnet-pytorch` (Python 3.12, PyTorch 2.14) | `chrombpnet-tensorflow` (Python 3.12, TF 2.21, Keras 3.15) | `chrombpnet-jax` (Python 3.12, Keras 3.15, JAX 0.11) | `chrombpnet-tensorflow` (Python 3.12, TF 2.21, tf-keras) |
| checkpoints | `.pt`; also reads `.h5` | `.keras`, the JAX engine's format; also reads `.h5` | `.keras` (Keras 3); also reads `.h5` | `.h5` (Keras 2, readable by chrombpnet 1.x) |
| DeepLIFT/DeepSHAP | Tangermeme | the JAX engine's rules, ported to TensorFlow | native JAX (ChromBPNet 2.0) | the kundajelab `shap` fork's, vendored |
| TF-MoDISco | git checkout (2.6 API) | git checkout | git checkout | git checkout |
| status | default | within float32 tolerances of the original | within float32 tolerances of the original | within float tolerance of the original |
| reproducibility gate | `tests/regression` (golden artifacts) | `tests/parity/jax_engine.py --engine tensorflow` | `tests/parity/jax_engine.py` | `tests/parity/tf_engine.py` |

## How the engines plug in

The engine is a narrow interface (`chrombpnet/engines/base.py`): build and
train a model, load one for prediction, rescale the bias model's count head,
and compute hypothetical contribution scores. The shared code reaches a model
only through it, and the predictor it hands back has the Keras-shaped surface
the shared code always used (NLC one-hot in, `(profile_logits, logcounts)`
out).

**tensorflow-old** is the original implementation (kundajelab/chrombpnet@09938fd)
behind that interface: its model definitions, loss, callbacks, `model.fit`
call, bias rescale and DeepSHAP code, each module stating what it changes.
It runs them on TensorFlow 2.21 with tf-keras, the Keras 2 API (the engine
sets `TF_USE_LEGACY_KERAS=1` before TensorFlow is imported, and refuses to
run on Keras 3), where the original ran TensorFlow 2.8 on Python 3.8. The
two releases' kernels round differently, so given identical batches, initial
weights and shuffle references the engine reproduces the original's
checkpoints, logs, predictions and contribution scores within float
tolerance rather than byte for byte: `tests/parity/tf_engine.py` runs the
original in its own environment and compares every output. Porting it to
TensorFlow 2.21 took four changes, each stated in its module:

- DeepSHAP is the kundajelab `shap` fork's `TFDeepExplainer`, copied into
  `chrombpnet/engines/tensorflow_old/deep_tf.py`, because the fork no longer
  imports on Python 3.12 with NumPy 2.
- The bias-factorized model's count head is a `Lambda` of a named function:
  1.x saved its `lambda` as Python 3.8 bytecode, which Python 3.11 and newer
  cannot unmarshal. The engine recognises that head in the original's models
  without running it, and writes every `.h5` it saves in the form chrombpnet
  1.x's Keras 2.8 reads: the head as 1.x's bytecode, and without two config
  fields tf-keras added since.
- Training uses tf-keras's legacy `Adam`, the optimizer of TensorFlow 2.8.
  The newer default rounds differently, and saves an optimizer
  configuration Keras 2.8 cannot rebuild, so chrombpnet 1.x could not load
  the models compiled (as `chrombpnet train -b` loads a bias model).
- `tf.keras.utils.set_random_seed` seeds training, so that tf-keras draws
  its initializers' seeds from a generator of its own: drawn from Python's
  `random`, as tf-keras otherwise does, they would move the batch order
  `fit` shuffles with it away from the original's.

It loads models trained by the original package, so every downstream
command works on existing `.h5` models. Initial weights for a given `--seed`
are not 1.x's, because tf-keras draws them differently. Beyond the port,
it differs from the original only where this package fixes a defect, each in
its own commit and CHANGELOG entry: `--seed` now seeds the bias-factorized model's trainable
tower; the best validation checkpoint is restored even when training runs out
of epochs; and DeepSHAP's dinucleotide shuffles are seeded by `--seed`.
Attribution runs in a spawned
subprocess, because the shap fork needs TF1 graph mode, which is
process-global; `--interpret-batch-size` has no effect there.

**JAX** is the Keras 3 / JAX port of the original from
[kundajelab/chrombpnet#284](https://github.com/kundajelab/chrombpnet/pull/284)
("ChromBPNet 2.0"): its architectures (the original ones, with a registered
`LogSumExp` count head instead of a bytecode `Lambda`), loss, callbacks,
`model.fit`, loader for the original's `.h5` models (no TensorFlow needed, and
their bytecode is never run), bias rescale, and native JAX DeepSHAP -- the
`shap` fork's multipliers, with both heads from one jitted pass per chunk.
Each module states what this package changes. The defaults are the original training
recipe. It cannot be byte-identical to TensorFlow 2.8, so
`tests/parity/jax_engine.py` checks agreement with the original on identical
inputs: predictions within 1e-4, a 20-step Adam trace within 1e-3, DeepSHAP
within 1e-4 relative L2 in both profile weightings (below) with
bitwise-identical references (drawn exactly as tensorflow-old draws
them), and the original's models loaded within 1e-4. Models are
`.keras` files; `chrombpnet export --legacy-h5 -m model.keras -o model.h5`
writes a copy in the original's `.h5` layout, which tensorflow-old,
chrombpnet 1.x, variant-scorer and bpnet-lite read. `--interpret-batch-size`
counts (sequence, reference) pairs, as for PyTorch.

The JAX engine also has ChromBPNet 2.0's opt-in training options, which the
tensorflow engine shares and PyTorch and tensorflow-old refuse: `--optimizer muon` (Muon on the dilated convolutions,
experimental), `--muon-lr`, `--ema`/`--ema-momentum`, `--lr-schedule cosine`
and `--precision {default,highest,bf16}` (bf16 is faster but loses accuracy).

**tensorflow** runs the JAX engine's Keras 3 code on the TensorFlow
backend (`KERAS_BACKEND=tensorflow`, set when the engine is chosen): the same
architectures, loss, optimizers and options (`--optimizer`, `--ema`,
`--lr-schedule`, `--precision`), callbacks, `model.fit` call, `.keras`
checkpoints and model loader, so a model either engine trains loads in the
other, and `chrombpnet export --engine tensorflow` writes the 1.x `.h5`
layout. Two parts are backend-specific: device and precision settings
(`--precision highest` turns TensorFlow's TF32 off, `--reproducibility-max`
turns its op determinism on) and DeepSHAP, whose rules
(`chrombpnet/engines/tensorflow/deeplift_tf.py`) are the JAX engine's,
op for op, on a `tf.GradientTape`. It runs in `chrombpnet-tensorflow`, beside
`tensorflow-old`, and is the Keras 3 engine for glibc 2.17 hosts.
`tests/parity/jax_engine.py --engine tensorflow` holds it to the JAX engine's
checks against the original (predictions, losses, a 20-step Adam trace, the
bias rescale, `export`, DeepSHAP), and `tests/parity/deepshap_engines.py`
compares its DeepSHAP with the other three engines', tensorflow-old's (the
original's DeepSHAP code) included. Until 2026-11, `--engine tensorflow` was
the Keras 2 engine now called `tensorflow-old`.

**The profile DeepSHAP target** is `sum(softmax(stop_gradient(mn)) * mn)` over
the mean-normalized profile logits `mn`; the stop-gradient means the softmax
to act as fixed weights. chrombpnet 1.x and bpnet compute it with the
kundajelab `shap` fork, which has its `StopGradient` handler commented out
(`deep_tf.py:595`), so the weights count as a second varying input and each
logit of an (input, reference) pair is weighted by the midpoint
`0.5 * (softmax(mn_x) + softmax(mn_r))`. All four engines do the same by
default. `--contribs-profile-stopgrad` (on `pipeline`, `qc`, `bias pipeline`,
`bias qc` and `contribs_bw`) honours the stop-gradient instead, weighting each
logit by the input's own softmax. The counts head is the same either way.
`tests/parity/deepshap_engines.py` checks that the four engines agree within
1e-4 in both modes; on a real model the two profile weightings correlate at
about 0.97.

`--predict-fused-kernel-exact` / `--predict-fused-kernel-fast` (the fused Triton
forward) are PyTorch-only and refused for the other engines. `--reproducibility` and `--reproducibility-max` work on all
four.
