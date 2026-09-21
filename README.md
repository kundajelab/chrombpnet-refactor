# ChromBPNet

Bias-factorized, base-resolution deep learning models of chromatin
accessibility. ChromBPNet learns to predict base-resolution ATAC-seq or
DNase-seq coverage from DNA sequence, while explicitly factoring out the
sequence bias of the assay enzyme (Tn5 or DNase I). The bias-free model can
then be interpreted to recover cis-regulatory sequence syntax, transcription
factor footprints, and the effect of regulatory variants.

This is a PyTorch implementation. Models are `torch.nn.Module`s and
checkpoints are `.pt` files carrying a format version, the architecture
configuration and a standard `state_dict`.

## Table of contents

- [Installation](#installation)
- [QuickStart](#quickstart)
  - [Training a bias-factorized ChromBPNet model](#training-a-bias-factorized-chrombpnet-model)
  - [Training a bias model](#training-a-bias-model)
- [Command line](#command-line)
- [Reproducibility](#reproducibility)
- [Python API](#python-api)
- [Testing](#testing)
- [Citation and license](#citation-and-license)

## Installation

```bash
conda env create -f environment.yml
conda activate chrombpnet-refactor
```

Run this from the repository root: `environment.yml` installs the package in
editable mode and applies the supplemental pip requirements in
`requirements.txt`.

The environment installs a CUDA-enabled PyTorch build that falls back to CPU
automatically. Every command accepts `--device cpu`, `--device cuda`, or a
specific device such as `--device cuda:1`.

Conda also supplies the command-line tools the preprocessing step shells out to
(`bedtools`, `samtools`, `bedGraphToBigWig`) and the native Pango/Cairo stack
WeasyPrint needs to render the PDF reports.

Motif discovery and the motif reports use TF-MoDISco, installed from the
checkout named in `requirements.txt` and in the `[reports]` extra of
`pyproject.toml`. It is optional: everything except `modisco_motifs` and the
motif section of the `qc` reports works without it.

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
        chrombpnet_nobias.profile_scores.h5
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
```

Run `chrombpnet <command> --help` for the full argument list.

## Optimizations

Every optimization is selectable, so a run can be compared against the
unoptimized code in the same working tree:

```bash
chrombpnet pred_bw --optimize none    # original code paths
chrombpnet pred_bw --optimize 1       # Anthropic-kit levers, minus the Triton forward
chrombpnet pred_bw --optimize 2       # the kit's Triton forward
chrombpnet pred_bw --optimize 3       # optimizations not from the kit
chrombpnet pred_bw --optimize 1,3     # any union; 'all' is shorthand (default: 1,3)
```

Available on `pred_bw`, `contribs_bw` and `footprints`.

| Phase | Contents | Output |
|---|---|---|
| `1` | Memory-mapped windows and a one-hot lookup table, the finish-stage pre-import, the vectorized profile JSD, the bigWig write and metrics in a worker process, and skipping teardown once outputs are durable. All ported from the [ChromBPNet optimization kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/main/chrombpnet) (Apache-2.0). | Byte-identical |
| `2` | The kit's `chrombpnet_k1` Triton forward, vendored byte-identical. Needs an L40S, H100 or H200; refuses by name elsewhere rather than falling back. | Moves ~1e-5. It is fp32-exact where the eager path uses TF32, so it is *more* accurate than the default. |
| `3` | Sharing inputs across `pred_bw`'s models, numpy writer arguments, lazy stats accumulation, the batched cast, and fixed-step bigWigs. Not from the kit. | Byte-identical except the bigWigs, which hold the same values in a different section type |

Measured on 5,000 hg38 regions with three models on an L40S: 55.7s at
`none`, 42.8s at `1`, 38.2s at `3`, **29.6s at `1,3`**. Phase 2 is roughly
neutral on that card -- the kit's own table routes L40S production traffic
to the eager graph -- and is a win on H100/H200.

## Reproducibility

`--benchmark` controls how much of the training run is held fixed. It defaults
to `lite`.

| Level | Effect |
|---|---|
| `none` | PyTorch's own defaults: Kaiming initialization, Adam epsilon 1e-8, per-batch loss averaging, sequential batch order |
| `lite` | Glorot/Xavier initialization with zero biases, Adam epsilon 1e-7, sample-weighted epoch losses, shuffled batch visit order, a seeded count-threshold draw, and restoration of the best-validation checkpoint |
| `medium` | `lite` plus a pinned example schedule: jitter, reverse complements and negative subsamples come from a seeded per-epoch stream instead of the global RNG (`generator_rng`), epoch-start shuffling is off (`fixed_example_order`), and the validation set stops being re-cropped each epoch (`frozen_validation`) |
| `max` | `medium` plus fixed initial weight arrays, a replayable augmentation schedule, deterministic kernels, TF32 disabled and scalar Adam dispatch |

A bare `--benchmark` with no value means `max`.

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
```

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
