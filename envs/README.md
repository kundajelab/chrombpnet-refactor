# Engines and their environments

ChromBPNet runs on one of three deep-learning engines. Everything else --
preprocessing, GC-matched negatives, the data generators, metrics, bigwigs,
footprints, motif discovery and the reports -- is shared, framework-neutral
code; DeepLIFT/DeepSHAP is the only downstream step that differs between
engines.

## Environments

Each engine has its own conda environment. All three run Python 3.12 and
NumPy 2, but their frameworks cannot share one: each brings its own CUDA and
cuDNN wheels, at releases the others do not accept, and the TensorFlow
engine needs Keras 2 (tf-keras) where the JAX engine needs Keras 3. Create
them from the repository root:

| Engine | Environment file | Python and framework | GPU libraries |
|---|---|---|---|
| PyTorch (default) | `chrombpnet-pytorch.yml` | Python 3.12, PyTorch 2.14.1 | the PyTorch wheel's CUDA 12.6 build; falls back to the CPU |
| TensorFlow | `chrombpnet-tensorflow.yml` | Python 3.12, TensorFlow 2.21 with tf-keras 2.21 | `tensorflow[and-cuda]`: CUDA 12.5 and cuDNN 9.3 as pip wheels |
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
  because contribution scoring runs in a second process on the same GPU
  while the first still holds what it trained with. For the same reason
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
  the environments in a container with a newer base, such as Ubuntu 22.04.
- Every command accepts `--device cpu`, `--device cuda`, or a specific
  device such as `--device cuda:1`.

## Choosing an engine

```bash
chrombpnet pipeline --engine pytorch ...      # the default
chrombpnet pipeline --engine tensorflow ...   # from the chrombpnet-tensorflow env
chrombpnet pipeline --engine jax ...          # from the chrombpnet-jax env
chrombpnet pred_bw -cm models/chrombpnet.h5 ...   # engine inferred: TensorFlow
```

`--engine` is accepted by every command that builds or loads a model. When it
is omitted the input models' checkpoint suffixes decide (`.pt` PyTorch, `.h5`
TensorFlow, `.keras` JAX), else PyTorch. An explicit `--engine` must be able to
read every input model: `--engine jax` also reads the original's `.h5` models,
so `chrombpnet pipeline --engine jax -b original_bias.h5 ...` works, and inputs
in both formats (`-bm bias.h5 -cm chrombpnet.keras`) go to JAX. `prep` and
`modisco_motifs` never touch a model, take no `--engine`, and run in any of the
three environments.

| | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| environment | `chrombpnet-pytorch` (Python 3.12, PyTorch 2.14) | `chrombpnet-tensorflow` (Python 3.12, TF 2.21, tf-keras) | `chrombpnet-jax` (Python 3.12, Keras 3.15, JAX 0.11) |
| checkpoints | `.pt` | `.h5` (Keras 2, readable by chrombpnet 1.x) | `.keras` (Keras 3); also reads `.h5` |
| DeepLIFT/DeepSHAP | Tangermeme | the kundajelab `shap` fork's, vendored | native JAX (ChromBPNet 2.0) |
| TF-MoDISco | git checkout (2.6 API) | git checkout | git checkout |
| status | default | within float tolerance of the original | within float32 tolerances of the original |
| reproducibility gate | `tests/regression` (golden artifacts) | `tests/parity/tf_engine.py` | `tests/parity/jax_engine.py` |

## How the engines plug in

The engine is a narrow interface (`chrombpnet/engines/base.py`): build and
train a model, load one for prediction, rescale the bias model's count head,
and compute hypothetical contribution scores. The shared code reaches a model
only through it, and the predictor it hands back has the Keras-shaped surface
the shared code always used (NLC one-hot in, `(profile_logits, logcounts)`
out).

**TensorFlow** is the original implementation (kundajelab/chrombpnet@09938fd)
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
  `chrombpnet/engines/tensorflow/deep_tf.py`, because the fork no longer
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
bitwise-identical references (drawn exactly as the TensorFlow engine draws
them), and the original's models loaded within 1e-4. Models are
`.keras` files; `chrombpnet export --legacy-h5 -m model.keras -o model.h5`
writes a copy in the original's `.h5` layout, which the TensorFlow engine,
chrombpnet 1.x, variant-scorer and bpnet-lite read. `--interpret-batch-size`
counts (sequence, reference) pairs, as for PyTorch.

The JAX engine also has ChromBPNet 2.0's opt-in training options, refused for
the other engines: `--optimizer muon` (Muon on the dilated convolutions,
experimental), `--muon-lr`, `--ema`/`--ema-momentum`, `--lr-schedule cosine`
and `--precision {default,highest,bf16}` (bf16 is faster but loses accuracy).

**The profile DeepSHAP target** is `sum(softmax(stop_gradient(mn)) * mn)` over
the mean-normalized profile logits `mn`; the stop-gradient means the softmax
to act as fixed weights. chrombpnet 1.x and bpnet compute it with the
kundajelab `shap` fork, which has its `StopGradient` handler commented out
(`deep_tf.py:595`), so the weights count as a second varying input and each
logit of an (input, reference) pair is weighted by the midpoint
`0.5 * (softmax(mn_x) + softmax(mn_r))`. All three engines do the same by
default. `--contribs-profile-stopgrad` (on `pipeline`, `qc`, `bias pipeline`,
`bias qc` and `contribs_bw`) honours the stop-gradient instead, weighting each
logit by the input's own softmax. The counts head is the same either way.
`tests/parity/deepshap_engines.py` checks that the three engines agree within
1e-4 in both modes; on a real model the two profile weightings correlate at
about 0.97.

`--predict-fused-kernel` (the fused Triton forward) is PyTorch-only and refused for the
other engines. `--reproducibility` and `--reproducibility-max` work on all
three.
