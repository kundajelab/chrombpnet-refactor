# ChromBPNet

Bias-factorized, base-resolution deep learning models of chromatin
accessibility. ChromBPNet learns to predict base-resolution ATAC-seq or
DNase-seq coverage from DNA sequence, while explicitly factoring out the
sequence bias of the assay enzyme (Tn5 or DNase I). The bias-free model can
then be interpreted to recover cis-regulatory sequence syntax, transcription
factor footprints, and the effect of regulatory variants.

The deep-learning engine is a choice: **PyTorch** (the default),
**TensorFlow** (the original Keras implementation, byte-identical to it given
the same inputs) or **JAX** (Keras 3 on JAX, from ChromBPNet 2.0; within
float32 tolerances of the original). Everything else -- preprocessing,
GC-matched negatives, the data generators, metrics, bigwigs, footprints, motif
discovery and the reports -- is shared, framework-neutral code; DeepLIFT/DeepSHAP
is the only downstream step that differs between engines. See
[Engines](#engines).

## Table of contents

- [Installation](#installation)
- [QuickStart](#quickstart)
  - [Training a bias-factorized ChromBPNet model](#training-a-bias-factorized-chrombpnet-model)
  - [Training a bias model](#training-a-bias-model)
- [Engines](#engines)
- [Command line](#command-line)
- [Reproducibility](#reproducibility)
- [Python API](#python-api)
- [Testing](#testing)
- [Citation and license](#citation-and-license)

## Installation

Each engine has its own conda environment, because their frameworks cannot
share one (TensorFlow 2.8 needs Python <= 3.10, the PyTorch stack is on 3.11,
and JAX 0.11 needs Python >= 3.12 and NumPy 2):

```bash
conda env create -f envs/chrombpnet-pytorch.yml      # default engine
conda activate chrombpnet-pytorch

conda env create -f envs/chrombpnet-tensorflow.yml   # --engine tensorflow
conda env create -f envs/chrombpnet-jax.yml          # --engine jax
```

Run these from the repository root: each file installs the package in
editable mode with its engine's extra (`-e ..[pytorch]` and so on) and applies
the supplemental pip requirements in `envs/requirements-<engine>.txt`.

The PyTorch environment installs a CUDA-enabled build that falls back to CPU
automatically; the JAX environment installs `jax[cuda12]` (CUDA 12 as pip
wheels: any CUDA 12 driver, >= 525, and also the Pascal and Volta GPUs that
CUDA 13 dropped; `jax[cuda13]==0.11.2` works on Turing or newer with driver
>= 580). A system CUDA or cuDNN on `LD_LIBRARY_PATH` would shadow the
libraries those wheels bring, so the JAX engine loads the bundled ones first
when it finds namesakes there. Every command accepts `--device cpu`, `--device cuda`, or a
specific device such as `--device cuda:1`.

Conda also supplies the command-line tools the preprocessing step shells out to
(`bedtools`, `samtools`, `bedGraphToBigWig`) and the native Pango/Cairo stack
WeasyPrint needs to render the PDF reports.

Motif discovery and the motif reports use TF-MoDISco, installed from the
checkout named in `envs/requirements-pytorch.txt` and in the `[reports]` extra of
`pyproject.toml`. It is optional: everything except `modisco_motifs` and the
motif section of the `qc` reports works without it.

The motif reports match each motif against a motif database with MEME's
`tomtom`, which every environment installs from conda (`meme`). The commands
that end with a report check for it before doing any work. With
`--tomtom-lite` they use TOMTOM-lite (memelite) instead, so MEME is not
needed, and the report lists uncorrected p-values instead of q-values.
TOMTOM-lite needs TF-MoDISco 2.5 or later, so the TensorFlow environment,
whose newest TF-MoDISco is 2.3.2, refuses the flag.

## QuickStart

### Training a bias-factorized ChromBPNet model

With a pre-trained bias model in hand:

```bash
# -ibam takes a BAM; use -ifrag for a fragment file or -itag for a tagAlign.
chrombpnet pipeline \
  -ibam /path/to/input.bam \
  -d ATAC \
  -g /path/to/hg38.fa \
  -c /path/to/hg38.chrom.sizes \
  -p /path/to/peaks.bed \
  -n /path/to/nonpeaks.bed \
  -fl /path/to/fold_0.json \
  -b /path/to/bias.pt \
  -o /path/to/output/dir/
```

Use `chrombpnet train` instead of `pipeline` to stop after training and its
report, skipping evaluation, contribution scoring and motif discovery.

The main `pipeline` evaluation also writes
`profile_metrics.peaks.median_pearsonr` in
`evaluation/chrombpnet_metrics.json` (with the optional filename prefix).
This is the median Pearson R across test peaks, calculated across bases within
each peak using observed counts and predicted profile probabilities, without
smoothing or log transforms. Constant or non-finite profiles have undefined R
and are excluded; the value is `null` if no peaks have a defined correlation.
The same metric is included under each evaluated subset in the JSON outputs of
standalone `qc` and `predict`, bias evaluation (including the input bias check),
and `pred_bw` with ground truth. The outputs are:

- ChromBPNet `pipeline`/`qc`: `evaluation/[prefix_]chrombpnet_metrics.json`.
- Bias `pipeline`/`qc`: `evaluation/[prefix_]bias_metrics.json`; the main
  pipeline's input bias check uses unprefixed `evaluation/bias_metrics.json`.
- Standalone `predict`: `<output-prefix>_metrics.json`.
- `pred_bw` with `--bigwig`: `<output-prefix>_<model>_metrics.json`, where
  `<model>` is `chrombpnet`, `chrombpnet_nobias`, or `bias`, for supplied models.
  This uses `profile_metrics.regions.median_pearsonr` across evaluated regions.

HTML/PDF reports continue to show their existing metrics without profile
Pearson R.

#### Input format

- `-ibam` / `-ifrag` / `-itag`: filtered reads, as BAM, fragment file or
  tagAlign. Exactly one is required. Example files:
  [bam](https://storage.googleapis.com/chrombpnet_data/input_files/ENCSR868FGK_merged.bam),
  [fragment](https://storage.googleapis.com/chrombpnet_data/input_files/example.fragments.tsv),
  [tagalign](https://storage.googleapis.com/chrombpnet_data/input_files/example.tagAlign).
  The enzyme shift is estimated from the reads; you do not supply it.
- `-d`: assay type, `ATAC` or `DNASE`.
- `-g`: reference genome FASTA, e.g.
  [hg38.fa](https://storage.googleapis.com/chrombpnet_data/input_files/hg38.genome.fa).
- `-c`: two-column chromosome-sizes file, e.g.
  [hg38.chrom.sizes](https://storage.googleapis.com/chrombpnet_data/input_files/hg38.chrom.sizes).
- `-p`: peaks, in 10-column narrowPeak format. Only `chr`, `start`, `end` and
  the summit offset (column 10) are used; every region is centred on
  `start + summit`. Example:
  [peaks.bed](https://storage.googleapis.com/chrombpnet_data/input_files/ENCSR868FGK_relaxed_peaks_no_blacklist.bed).
- `-n`: non-peak background regions, same 10-column format. Generate your own
  with `chrombpnet prep nonpeaks` (below). Example:
  [nonpeaks.bed](https://storage.googleapis.com/chrombpnet_data/input_files/ENCSR868FGK_nonpeaks_no_blacklist.bed).
- `-fl`: JSON naming the train/valid/test chromosome split. Generate one with
  `chrombpnet prep splits`, or use the published
  [five-fold splits](https://zenodo.org/records/7443683/files/folds.zip?download=1).
- `-b`: a bias model `.pt` checkpoint. Bias models generally transfer between
  samples of the same assay; train one with `chrombpnet bias pipeline` below.
- `-o`: output directory.

Optional flags worth knowing: `-fp` prefixes every output filename, `-bs` sets
the training batch size, `-ibs` the contribution-scoring batch size (defaults
to `-bs / 4`, because attribution needs about four times the GPU memory of a
training step), and `--benchmark` selects the reproducibility level described
under [Reproducibility](#reproducibility).

For guidance on preparing reads and peaks, see the
[preprocessing notes](https://github.com/kundajelab/chrombpnet/wiki/Preprocessing).

#### Output format

`chrombpnet train` writes:

```text
models/
    chrombpnet.pt                  combined model
    chrombpnet_nobias.pt           bias-corrected accessibility model
    bias_model_scaled.pt           input bias model, rescaled to this sample
logs/
    chrombpnet.log                 loss per epoch
    chrombpnet.log.batch           loss per batch per epoch
    chrombpnet.args.json           every argument the run was given
    chrombpnet_model_params.tsv
    chrombpnet_data_params.tsv
auxiliary/
    data_unstranded.bw             shifted coverage built from the reads
    filtered.peaks.bed
    filtered.nonpeaks.bed
evaluation/
    overall_report.html
    overall_report.pdf
    bw_shift_qc.png                enzyme-shift check; should look like the motif
    epoch_loss.png
    bias_metrics.json              sanity check that the bias model alone
    bias_predictions.h5            does not explain the peaks
    bias_only_peaks.counts_pearsonr.png
    bias_only_peaks.profile_jsd.png
```

`chrombpnet pipeline` runs all of the above and then adds:

```text
auxiliary/
    motif_to_pwm.tsv               the motifs that were footprinted
    30K_subsample_peaks.bed        peaks used for contribution scoring
    chrombpnet_nobias_footprints.h5
    interpret_subsample/
        chrombpnet_nobias.interpreted_regions.bed   the regions scored, in row order
        chrombpnet_nobias.counts_scores.h5          DeepSHAP scores, counts head
        chrombpnet_nobias.profile_scores.h5         DeepSHAP scores, profile head
        modisco_results_profile_scores.h5
evaluation/
    chrombpnet_metrics.json        the headline count and profile metrics
    chrombpnet_predictions.h5
    chrombpnet_only_peaks.counts_pearsonr.png
    chrombpnet_only_peaks.profile_jsd.png
    chrombpnet_nobias.<motif>.footprint.png    one per motif in the table
    chrombpnet_nobias_max_bias_response.txt
    modisco_profile/               motif report (HTML, PNG logos, TSV)
    chrombpnet_nobias_profile.pdf
```

Read `evaluation/overall_report.html` first: it collects the shift QC image,
the loss curve, the count and profile metrics, the marginal footprints and the
discovered motifs into one page.

### Training a bias model

A bias model is trained on background regions only, so it can learn the
enzyme's sequence preference without learning biology:

```bash
chrombpnet bias pipeline \
  -ibam /path/to/input.bam \
  -d ATAC \
  -g /path/to/hg38.fa \
  -c /path/to/hg38.chrom.sizes \
  -p /path/to/peaks.bed \
  -n /path/to/nonpeaks.bed \
  -fl /path/to/fold_0.json \
  -b 0.5 \
  -o /path/to/bias/dir/
```

Here `-b` is the bias threshold factor, not a model path: non-peak regions
whose counts exceed `-b x` the 1st-percentile peak count are dropped, so the
model never sees background that looks like signal. Start at `0.5` for ATAC
and `0.8` for DNase. The trained model lands at `models/bias.pt` and is what
you pass to `chrombpnet pipeline -b`.

## Engines

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
restored, below). It loads models trained by the original package, so every downstream
command works on existing `.h5` models. It differs from the original only where
this package fixes a defect, each in its own commit and CHANGELOG entry:
`--seed` now seeds the bias-factorized model's trainable tower; the best
validation checkpoint is restored even when training runs out of epochs;
DeepSHAP's dinucleotide shuffles are seeded by `--seed`; and the profile
DeepSHAP target's `stop_gradient` is honoured, so each logit is weighted by the
input's own softmax, as in the PyTorch engine. The `shap` fork has that
handler commented out, so the original weighted it by the midpoint of the
input's and the reference's softmax. Attribution runs in a
spawned subprocess, because the shap fork needs TF1 graph mode, which is
process-global; `--interpret-batch-size` has no effect there.

**JAX** is the Keras 3 / JAX port of the original from
[kundajelab/chrombpnet#284](https://github.com/kundajelab/chrombpnet/pull/284)
("ChromBPNet 2.0"): its architectures (the original ones, with a registered
`LogSumExp` count head instead of a bytecode `Lambda`), loss, callbacks,
`model.fit`, loader for the original's `.h5` models (no TensorFlow needed, and
their bytecode is never run), bias rescale, and native JAX DeepSHAP -- the
`shap` fork's multipliers, with both heads from one jitted pass per chunk.
The profile target's `stop_gradient` is honoured, as in the other two
engines, where the PR reproduces the fork's midpoint weighting. Each module states what this package
changes. The defaults are the original training recipe. It cannot be byte-identical
to TensorFlow 2.8, so `tests/parity/jax_engine.py` checks agreement with the
original on identical inputs: predictions within 1e-4, a 20-step Adam trace
within 1e-3, DeepSHAP within 1e-4 relative L2 (against the original with
its `StopGradient` handler restored) with bitwise-identical references (drawn
exactly as the TensorFlow engine draws them), and the
original's models loaded within 1e-4. Models are `.keras` files;
`chrombpnet export --legacy-h5 -m model.keras -o model.h5` writes a copy in the
original's `.h5` layout, which the TensorFlow engine, chrombpnet 1.x,
variant-scorer and bpnet-lite read. `--interpret-batch-size` counts
(sequence, reference) pairs, as for PyTorch.

The JAX engine also has ChromBPNet 2.0's opt-in training options, refused for
the other engines: `--optimizer muon` (Muon on the dilated convolutions,
experimental), `--muon-lr`, `--ema`/`--ema-momentum`, `--lr-schedule cosine`
and `--precision {default,highest,bf16}` (bf16 is faster but loses accuracy).

`--predict-fk` (the fused Triton forward) and `--benchmark pytorch` are
PyTorch-only and are refused for other engines.

## Command line

```text
chrombpnet pipeline          train a bias-factorized model, then evaluate it
chrombpnet train             train only, then write the training report
chrombpnet qc                evaluate an already-trained model
chrombpnet bias pipeline     train a bias model, then evaluate it
chrombpnet bias train        train a bias model only
chrombpnet bias qc           evaluate an already-trained bias model
chrombpnet prep nonpeaks     GC-matched background regions from peaks
chrombpnet prep splits       chromosome train/valid/test split JSON
chrombpnet pred_bw           prediction bigwigs (and metrics, given -bw)
chrombpnet contribs_bw       contribution-score bigwigs
chrombpnet modisco_motifs    motif discovery from contribution scores
chrombpnet footprints        marginal footprints for given motifs
chrombpnet export            a JAX-engine model in the original's .h5 layout
```

Run `chrombpnet <command> --help` for the full argument list.

`pipeline`, `qc`, `bias pipeline` and `bias qc` end with TF-MoDISco motif
reports, built from contribution scores on a fixed-seed sample of the peaks.
Four options control them, all off by default:

| option | default |
|---|---|
| `--interpret-subsample N` | 30 000 peaks scored (all of them if there are fewer) |
| `--modisco-max-seqlets N` | 50 000 seqlets per positive/negative pattern group |
| `--modisco-window BP` | 500 bp centred window; even, at most the input length |
| `--tomtom-lite` | match motifs with TOMTOM-lite instead of MEME's `tomtom` (also on `modisco_motifs`) |

`modisco_motifs` takes its own `--max-seqlets` and `--window`.

## Optimizations

`pred_bw` runs about **47 % faster** than the code this package started from,
on 5,000 hg38 regions with three models on an L40S (55.7s -> 29.6s). Those
optimizations are always on and byte-identical to what they replaced, with one
exception: bigWigs are written as fixed-step blocks, which stores the same
values in a different on-disk section type (2.8x faster to write, ~1.5x
smaller). They came in two groups -- levers ported from the
[ChromBPNet optimization kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/main/chrombpnet)
(Apache-2.0), and optimizations written here -- and the CHANGELOG records what
each was worth.

The one optimization that is **not** on by default is the fused Triton
forward, because it changes the numbers:

```bash
chrombpnet pred_bw --predict-fk      # prediction using fused kernels
```

Available on `pred_bw`, `footprints`, `qc` and `bias qc`. Not on
`contribs_bw`: DeepLIFT needs autograd and hooks on the `nn.ReLU` modules,
which a fused kernel cannot provide.

It runs the forward on the kit's vendored `chrombpnet_k1` kernels, which are
fp32-exact where the eager path runs TF32 convolutions. Against an fp64
reference it errs 1.0e-05 where the default errs 1.1e-03 -- so it is *more*
accurate than the shipped path, while still moving the profile head by ~1e-5.
That is why it is opt-in. It needs an H100 or H200 and refuses by name on
anything else rather than falling back silently; on an L40S it buys accuracy
rather than speed.

## Cut conventions and reverse complementation

Two flags. `--dnase-convention` (DNASE only) selects how a DNase cut is
recorded and what reverse complementation does about it; omitting it
selects **`both-half`**. `--data-aug` selects which examples are reverse
complemented, the same for either assay; omitting it selects **`augment`**
(half of the training examples). An explicit `--dnase-convention` on an ATAC
run is an error rather than a silent no-op. Pass `--dnase-convention
right-deprecated` for the DNase behaviour this package shipped before the
flags existed (deprecated: kept to reproduce earlier runs only).

They exist because the two assays record different things. DNase I is a
nuclease: it cuts the backbone *between* two bases, so the 0/+1 shift records
the event at a **bond**, named by that bond's right-hand base — the `+1` on
minus reads is exactly what makes both strands name the same bond the same
way. Tn5 leaves a 9 bp stagger, and 9 being odd, ATAC records a **base**.

Reversing a profile maps a base to its mirror correctly. It does *not* do the
same for a bond: the bond lands in the right place but is left named by the
wrong base, so a reverse-complemented DNase profile is misregistered by one.
The correction is to reverse and then shift one position — `i → W-i` rather
than `i → W-1-i`.

| `--dnase-convention` | preprocessing | RC shift |
|---|---|---|
| `right` | DNase 0/+1 | **+1** |
| `left` | DNase −1/0 | **−1** |
| **`both-half`** *(default)* | 0.5 on both bases flanking the cut | 0 |
| `both-full` | 1.0 on both bases flanking the cut | 0 |
| `right-deprecated` | DNase 0/+1 | 0 — misregistered by one; deprecated |

| `--data-aug` | splits flipped |
|---|---|
| **`augment`** *(default)* | train, 50 % |
| `forward-only` | none |
| `revcomp-only` | train, valid and test, 100 % |

ATAC (+4/−4) records a base, so plain reversal is already right and it takes
no `--dnase-convention`; `--data-aug` applies to it unchanged.

The flags replace the earlier single `--data-aug` with seven values. Its old
values map exactly onto them:

| old `--data-aug` | now |
|---|---|
| `standard` | `--dnase-convention right-deprecated` on DNASE; nothing on ATAC |
| `fr-only` | `--data-aug forward-only` (on DNASE also `--dnase-convention right-deprecated`) |
| `rc-only` | `--data-aug revcomp-only` (on DNASE also `--dnase-convention right-deprecated`) |
| `right`, `left`, `both-half`, `both-full` | `--dnase-convention` of the same name |

`both-half` is the DNase default in preference to the `right` shift. A shift
corrects the misregistration; splitting the cut across both bases flanking
the bond removes the asymmetry that causes it, so there is nothing left to
correct and no dependence on getting the sign right. It also leaves no label
slot without a source, so the label window is not widened. The cost is that
each cut is smeared over 2 bp; the `0.5` scale keeps the track total exactly
what one deposit per cut gives, so `counts_loss_weight` and the `-oth`/`-b`
ratios carry over unchanged.

Both flags are on `pipeline`, `train`, `qc`, their `bias` counterparts and
`predict`; `reads_to_bigwig` takes `--dnase-convention` (the bigwig does not
depend on `--data-aug`).

The geometry is measured, not just derived: the plus- and minus-strand cut
PWMs in `chrombpnet/data/{DNASE,ATAC}.ref.motifs.txt` align with each other's
reverse complement at lag 0 for DNase and lag −1 for ATAC, and the required
label shift is `lag + 1` in both cases. `tests/test_conventions.py` asserts
it, so changing the shift deltas without changing the correction fails.

## Scoring one model on one split

```bash
chrombpnet predict -m model.pt -d DNASE -g hg38.fa -c hg38.chrom.sizes \
    -r peaks.bed -n None -fl fold_0.json -bw observed.bw -op out
```

Writes `out_predictions.h5` and `out_metrics.json`. This is the stage `qc`
runs first, without the attribution and MoDISco that follow it — useful when
what you want is a metric for a model against a given test split. Geometry is
read off the checkpoint.

## Reproducibility

`--benchmark` controls how much of the training run is held fixed. **Omit it**
for the default, `tensorflow`.

| Level | Effect |
|---|---|
| `pytorch` | PyTorch's own defaults: Kaiming initialization, Adam epsilon 1e-8, per-batch loss averaging, sequential batch order |
| *(omitted)* = `tensorflow` | Glorot/Xavier initialization with zero biases, Adam epsilon 1e-7, sample-weighted epoch losses, shuffled batch visit order, a seeded count-threshold draw, and restoration of the best-validation checkpoint |
| `medium` | `tensorflow` plus a pinned example schedule: jitter, reverse complements and negative subsamples come from a seeded per-epoch stream instead of the global RNG (`generator_rng`), epoch-start shuffling is off (`fixed_example_order`), and the validation set stops being re-cropped each epoch (`frozen_validation`) |
| `max` | `medium` plus fixed initial weight arrays, a replayable augmentation schedule, deterministic kernels, TF32 disabled and scalar Adam dispatch |

A bare `--benchmark` with no value means `max`.

The default is `tensorflow` because this package is a port: matching the
implementation it was ported from is the ordinary case, not a special mode.

Two runs at the same level with the same `--seed` produce identical
checkpoints, logs and metrics. This holds on GPU as well as CPU: the data
pipeline is seeded before the model is built, and cuDNN autotuning is left at
its default (off), so kernel selection does not vary between runs.

`max` goes further and *requires* determinism rather than relying on it:
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic`, and TF32
off for both cuDNN and cuBLAS matmuls. That costs throughput, and on CUDA
PyTorch additionally demands `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the
environment before it will allow deterministic cuBLAS reductions -- the
package does not set this for you.

One step sits outside this: the enzyme shift is estimated from 20 000 reads
sampled with an unseeded `shuf`. The estimate is an integer chosen by an argmax
over an averaged position weight matrix and is stable in practice, but it is
not seeded.

## Python API

```python
import torch
from chrombpnet import BiasModel, ChromBPNet, ModelConfig, load_model, save_model

config = ModelConfig(
    input_length=2114,
    output_length=1000,
    filters=512,
    n_dilation_layers=8,
    counts_loss_weight=1.0,
)
bias = BiasModel(ModelConfig(2114, 1000, filters=128, n_dilation_layers=4))
model = ChromBPNet(config, bias)

# Native tensor layout is NCL: batch, A/C/G/T channels, sequence length.
profile_logits, logcounts = model(torch.zeros(2, 4, 2114))

save_model(model, "chrombpnet.pt")
model = load_model("chrombpnet.pt")
```

The model has two heads: `profile_logits` over `output_length` positions, and a
single `logcounts` value per region, which is `log(1 + total counts)`. The
bias branch is frozen and stays in eval mode.

Historical scripts and paper-analysis assets that the CLI does not reach are
isolated under [`legacy/`](legacy/README.md). They are kept for provenance and
are not installed with the package.

## Testing

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

# TensorFlow engine, from the chrombpnet-tensorflow environment: byte-identity
# with the original Keras implementation (opt-in, several minutes) ...
CHROMBPNET_TF_ENGINE_PARITY=1 pytest tests/parity/test_tf_engine.py
# ... and the golden-artifact sweep run through --engine tensorflow.
python -m tests.regression.harness capture --engine tensorflow --out DIR

# JAX engine: agreement with the original (needs chrombpnet-tensorflow and
# chrombpnet-jax side by side; opt-in, several minutes) ...
CHROMBPNET_JAX_ENGINE_PARITY=1 pytest tests/parity/test_jax_engine.py
# ... and the golden-artifact sweep, from the chrombpnet-jax environment.
python -m tests.regression.harness capture --engine jax --out DIR

# DeepSHAP across engines: the TensorFlow and JAX engines against PyTorch,
# whose semantics are the reference (needs all three environments; opt-in).
CHROMBPNET_DEEPSHAP_ENGINES=1 pytest tests/parity/test_deepshap_engines.py
```

`pytest` runs in every engine's environment; tests that need a framework the
environment lacks are skipped.

See [`tests/regression/README.md`](tests/regression/README.md) for what makes
the artifact comparison valid and which fixture constants are pinned by
hardcoded values elsewhere in the package.

## Citation and license

If you use this software, please cite:

> Pampari A, Shcherbina A, Nair S, Schreiber J, Patel A, Wang A, Kundu S,
> Shrikumar A, Kundaje A. *Bias factorized, base-resolution deep learning
> models of chromatin accessibility reveal cis-regulatory sequence syntax,
> transcription factor footprints and regulatory variants.*
> <https://github.com/kundajelab/chrombpnet>, doi:10.5281/zenodo.7567627

Released under the MIT license; see `LICENSE`.
