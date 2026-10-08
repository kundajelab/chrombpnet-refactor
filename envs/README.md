# Engines and their environments

ChromBPNet runs on one of three deep-learning engines. Everything else --
preprocessing, GC-matched negatives, the data generators, metrics, bigwigs,
footprints, motif discovery and the reports -- is shared, framework-neutral
code; DeepLIFT/DeepSHAP is the only downstream step that differs between
engines.

## Environments

Each engine has its own conda environment, because their frameworks cannot
share one: TensorFlow 2.8 needs Python <= 3.10, the PyTorch stack is on 3.11,
and JAX 0.11 needs Python >= 3.12 and NumPy 2. Create them from the
repository root:

| Engine | Environment file | Python and framework | GPU libraries |
|---|---|---|---|
| PyTorch (default) | `chrombpnet-pytorch.yml` | Python 3.11, PyTorch 2.5.1 | CUDA 11.8 build from the PyTorch wheel; falls back to the CPU |
| TensorFlow | `chrombpnet-tensorflow.yml` | Python 3.8, TensorFlow 2.8.0 | CUDA 11.2 and cuDNN 8.1 from the system (modules, or `/usr/local`) |
| JAX | `chrombpnet-jax.yml` | Python 3.12, Keras 3.15, JAX 0.11.2 | `jax[cuda12]`: CUDA 12 and cuDNN 9 as pip wheels |

Each file installs the package in editable mode with its engine's extra
(`-e ..[pytorch]` and so on) and applies the supplemental pip requirements
in `requirements-<engine>.txt`. Every environment also supplies the
command-line tools preprocessing shells out to (`bedtools`, `samtools`,
`bedGraphToBigWig`), MEME's `tomtom` for the motif reports, and the
Pango/Cairo stack WeasyPrint needs for the PDF reports.

Notes on GPUs:

- **JAX**: the `jax[cuda12]` wheels need any CUDA 12 driver (525 or newer);
  `jax[cuda13]==0.11.2` also works, with driver 580 or newer. In practice
  the engine needs a GPU of compute capability 7.5 (Turing) or newer: the
  cuDNN 9 build the wheels bring has precompiled kernels only for those,
  so convolutions fail on Pascal and Volta GPUs (P100, TITAN Xp, V100).
  A system CUDA or cuDNN on `LD_LIBRARY_PATH` would shadow the libraries the
  wheels bring, so the engine loads the bundled ones first when it finds
  namesakes there.
- **TensorFlow**: memory growth is on (`TF_FORCE_GPU_ALLOW_GROWTH=true`),
  because contribution scoring runs in a second process on the same GPU
  while the first still holds what it trained with. For the same reason
  the GPU must not be in Exclusive_Process compute mode.
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
| environment | `chrombpnet-pytorch` (Python 3.11) | `chrombpnet-tensorflow` (Python 3.8, TF 2.8.0) | `chrombpnet-jax` (Python 3.12, Keras 3.15, JAX 0.11) |
| checkpoints | `.pt` | `.h5` (Keras 2) | `.keras` (Keras 3); also reads `.h5` |
| DeepLIFT/DeepSHAP | Tangermeme | kundajelab `shap` fork | native JAX (ChromBPNet 2.0) |
| TF-MoDISco | git checkout (2.6 API) | modisco-lite 2.3.2 (newest for Python 3.8) | git checkout |
| status | default | byte-identical to the original | within float32 tolerances of the original |
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
Given identical batches, initial weights and shuffle references it reproduces
the original's checkpoints, logs, predictions and contribution scores byte for
byte, which `tests/parity/tf_engine.py` checks against a checkout of the
original (for the profile scores, with the original's `StopGradient` handler
restored, below). It loads models trained by the original package, so every
downstream command works on existing `.h5` models. It differs from the
original only where this package fixes a defect, each in its own commit and
CHANGELOG entry: `--seed` now seeds the bias-factorized model's trainable
tower; the best validation checkpoint is restored even when training runs out
of epochs; DeepSHAP's dinucleotide shuffles are seeded by `--seed`; and the
profile DeepSHAP target's `stop_gradient` is honoured, so each logit is
weighted by the input's own softmax, as in the PyTorch engine. The `shap` fork
has that handler commented out, so the original weighted it by the midpoint of
the input's and the reference's softmax. Attribution runs in a spawned
subprocess, because the shap fork needs TF1 graph mode, which is
process-global; `--interpret-batch-size` has no effect there.

**JAX** is the Keras 3 / JAX port of the original from
[kundajelab/chrombpnet#284](https://github.com/kundajelab/chrombpnet/pull/284)
("ChromBPNet 2.0"): its architectures (the original ones, with a registered
`LogSumExp` count head instead of a bytecode `Lambda`), loss, callbacks,
`model.fit`, loader for the original's `.h5` models (no TensorFlow needed, and
their bytecode is never run), bias rescale, and native JAX DeepSHAP -- the
`shap` fork's multipliers, with both heads from one jitted pass per chunk.
The profile target's `stop_gradient` is honoured, as in the other two
engines, where the PR reproduces the fork's midpoint weighting. Each module
states what this package changes. The defaults are the original training
recipe. It cannot be byte-identical to TensorFlow 2.8, so
`tests/parity/jax_engine.py` checks agreement with the original on identical
inputs: predictions within 1e-4, a 20-step Adam trace within 1e-3, DeepSHAP
within 1e-4 relative L2 (against the original with its `StopGradient` handler
restored) with bitwise-identical references (drawn exactly as the TensorFlow
engine draws them), and the original's models loaded within 1e-4. Models are
`.keras` files; `chrombpnet export --legacy-h5 -m model.keras -o model.h5`
writes a copy in the original's `.h5` layout, which the TensorFlow engine,
chrombpnet 1.x, variant-scorer and bpnet-lite read. `--interpret-batch-size`
counts (sequence, reference) pairs, as for PyTorch.

The JAX engine also has ChromBPNet 2.0's opt-in training options, refused for
the other engines: `--optimizer muon` (Muon on the dilated convolutions,
experimental), `--muon-lr`, `--ema`/`--ema-momentum`, `--lr-schedule cosine`
and `--precision {default,highest,bf16}` (bf16 is faster but loses accuracy).

`--predict-fk` (the fused Triton forward) and `--benchmark pytorch` are
PyTorch-only and are refused for other engines.
