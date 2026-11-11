# `pred_bw` optimization benchmark

What each set of optimizations is worth in wall clock: `chrombpnet pred_bw`
on the same regions, observed bigWig and three models (bias, ChromBPNet,
no-bias), run several ways ("arms"):

| arm | what runs |
|---|---|
| `original` | chrombpnet 1.x (TensorFlow 2.8) on the original's `.h5` models |
| `none` | chrombpnet-refactor at commit `4962484`, `--optimize none`: no optimizations |
| `kit` | `4962484`, `--optimize 1`: the [ChromBPNet optimization kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/main/chrombpnet)'s changes, without its Triton forward |
| `ours` | `4962484`, `--optimize 3`: this package's own changes |
| `both` | `4962484`, `--optimize 1,3`: the default at that commit |
| `kitfk` | `4962484`, `--optimize 1,2`: the kit's changes with its Triton forward |
| `fkonly` | `4962484`, `--optimize 2`: the Triton forward alone |
| `main` | the current package, default (should match `both`) |
| `fk` | the current package, `--predict-fused-kernel`: the kit's fused Triton kernels (refused without a CUDA GPU) |

`--optimize` was removed once this measurement was made, so the phase
arms run commit `4962484`. A first sweep always ran the unoptimized arm first,
on a cold file cache, and overstated it; `benchmark_pred_bw.py` removes that:
the genome, bigWig and models are read once before anything is timed, every
arm gets an untimed warm-up run, each round runs the arms in a different
order, and every run is a fresh process with only the `pred_bw` command
timed.

| file | |
|---|---|
| `benchmark_pred_bw.py` | runs the arms on one GPU and writes `timings.tsv` |
| `summarize.py` | medians per arm, speedups between arms, and medians by position in the round |

## Inputs

- **Models**: the three `.pt` files of a trained fold (`models/chrombpnet.pt`,
  `chrombpnet_nobias.pt`, `bias_model_scaled.pt` from `chrombpnet pipeline` or
  `train`), and for `original` the same three as `.h5` from chrombpnet 1.x.
  Its `auxiliary/filtered.peaks.bed` and `auxiliary/data_unstranded.bw` make
  good `--peaks` and `--observed-bigwig`.
- **Code**: this repository for `main`/`fk`; commit `4962484` for the phase arms,
  without touching your checkout:

  ```bash
  mkdir code-4962484 && git -C /path/to/chrombpnet-refactor archive 4962484 | tar -x -C code-4962484
  ```

  and for `original` a checkout of [kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet)
  with an environment that runs it.
- **Interpreters**: any environment with this package's PyTorch dependencies
  serves every refactor arm (each arm imports its own checkout through
  `PYTHONPATH`). Nothing is activated for you: pass each arm's `python`, and
  anything its framework needs with `--env-original KEY=VALUE` /
  `--env-refactor KEY=VALUE` (TensorFlow 2.8 needs CUDA 11.2 and cuDNN 8 on
  `LD_LIBRARY_PATH`, for example). The published numbers used one torch 2.5.1
  environment (NumPy 1.26) for every refactor arm; commit `4962484` predates
  this package's move to torch 2.14 and NumPy 2 and has not been run there.

## Run

One fold on one GPU (about 1–2 hours at 20,000 regions and 3 rounds):

```bash
FOLD=/path/to/refactor_run/fold_0/model
ORIG=/path/to/original_run/fold_0/model
python benchmark_pred_bw.py \
    --out-dir results/L40S/fold_0 --label fold_0 \
    --genome hg38.fa --chrom-sizes hg38.chrom.sizes \
    --peaks $FOLD/auxiliary/filtered.peaks.bed --n-regions 20000 \
    --observed-bigwig $FOLD/auxiliary/data_unstranded.bw \
    --chrombpnet-pt $FOLD/models/chrombpnet.pt \
    --nobias-pt $FOLD/models/chrombpnet_nobias.pt \
    --bias-pt $FOLD/models/bias_model_scaled.pt \
    --refactor-repo /path/to/chrombpnet-refactor \
    --refactor-python /path/to/envs/torch-env/bin/python \
    --phase-repo code-4962484 \
    --chrombpnet-h5 $ORIG/models/chrombpnet.h5 \
    --nobias-h5 $ORIG/models/chrombpnet_nobias.h5 \
    --bias-h5 $ORIG/models/bias_model_scaled.h5 \
    --original-repo /path/to/chrombpnet \
    --original-python /path/to/envs/chrombpnet/bin/python \
    --env-original LD_LIBRARY_PATH=/path/to/cuda-11.2/lib64:/path/to/cudnn-8/lib
```

Leave out an arm's inputs to skip it, or pick arms with `--arms none,kit,ours,both`.
`--rounds` (3), `--warmup-regions` (2000), `--batch-size` (256) and `--device`
(`cuda`) set the rest; `--regions` predicts a BED as is instead of
subsampling `--peaks`. For several folds or GPUs, run it once per fold and GPU
with its own `--out-dir` (one job each on a cluster), then:

```bash
python summarize.py results/
```

`summarize.py` groups by GPU (the `gpu` column, from `nvidia-smi`), so every
fold of a GPU is pooled into one median per arm.
