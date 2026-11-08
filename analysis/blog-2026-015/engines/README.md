# ChromBPNet across engines: PyTorch, TensorFlow, JAX

Does the same ChromBPNet pipeline give the same models under each
deep-learning engine? The package (`software/chrombpnet-refactor`) runs one
framework-neutral pipeline -- preprocessing, data generators, metrics,
footprints, TF-MoDISco, reports -- with the model behind an engine:

| engine | environment | what it is |
|---|---|---|
| `pytorch` | `chrombpnet-pytorch` (Python 3.12, torch 2.14.1+cu126; Python 3.11 and torch 2.5.1+cu118 for the runs below) | this package's PyTorch implementation, the default |
| `tensorflow` | `chrombpnet-tensorflow` (Python 3.12, TF 2.21 with tf-keras; Python 3.8 and TF 2.8.0 for the runs below) | the original Keras ChromBPNet, byte-identical to it given the same inputs |
| `jax` | `chrombpnet-jax` (Python 3.12, Keras 3.15.1, JAX 0.11.2) | the Keras 3 / JAX port from kundajelab/chrombpnet#284 |

The sweep is **4 datasets x 5 folds x 3 engines = 60 runs**, each a full
`chrombpnet bias pipeline` + `chrombpnet pipeline`, like
`dnase-rc/run_train_full.sh` with the engine in place of the `--data-aug`
arm.

| ENCID | assay | biosample | array tasks |
|---|---|---|---|
| ENCSR868FGK | ATAC | K562 | 0-4 |
| ENCSR000EOT | DNase | K562 | 5-9 |
| ENCSR163PKT | DNase | stomach M54 | 10-14 |
| ENCSR337UIU | ATAC | stomach F53 | 15-19 |

(the dnase-rc ids table, `ids_chrombpnet_refactor_benchmark_data_aug.txt`;
array task `t` is data row `t+1`, fold = its `FOLD_NUM`.)

## Running it

```bash
# 1. On one GPU node: each env activates, imports the checkout, runs a
#    convolution on the GPU, and loads and exercises everything the pipeline
#    uses -- inside the container, as the jobs will run (see "The container").
#    The first run pulls the image. The -C asks for a GPU the JAX env can use
#    (compute capability 7.5+).
srun -p <partitions> --gpus=1 --time=0:30:00 --gpu_cmode=shared \
     -C "GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0" ./check_envs.sh sherlock --container

# 2. Submit: 20 prep tasks (CPU), then 3 x 20 training tasks (GPU).
./submit_all.sh sherlock --container --dry-run    # see the sbatch lines
./submit_all.sh sherlock --container

# 3. Watch, and resubmit what failed (completed tasks skip themselves).
#    check_runs.sh asks squeue which tasks are still running or pending.
#    Cancel an engine's old array before resubmitting it (scancel <jobid>):
#    two tasks of one engine on one dataset and fold would share a directory.
./check_runs.sh
ENGINES=jax SKIP_PREP=1 ./submit_all.sh sherlock --container

# 4. Score the 30K peak subsample's missing heads (runs finished before the
#    pipeline kept the counts head; heads already scored are skipped).
#    JAX needs the container on sherlock; PyTorch and TensorFlow run natively.
ENGINES="pytorch tensorflow" ./submit_all.sh sherlock --contribution
ENGINES=jax ./submit_all.sh sherlock --container --contribution

# 5. Collate (a CPU job: the first run reads the original contribution
#    scores, ~11 GB per dataset and fold) and plot.
sbatch run_summarize.sh
./plot_summarize.sh
```

`--container` (off by default) runs every step's conda env inside an Ubuntu
22.04 image; JAX on sherlock needs it, and a submission including JAX there
without it is refused. Leave it out to run PyTorch and TensorFlow natively,
as the first sweep did.

`submit_all.sh` arguments: `[sherlock|local] [--container] [--contribution]
[--dry-run]`. Options (environment): `ENGINES="jax tensorflow"`,
`ARRAY=0-4`, `SKIP_PREP=1`, `GPU_CONSTRAINT="GPU_SKU:L40S"` (passed to every
engine's training jobs as `--constraint`; `sh_node_feat` lists sherlock's
features) and `JAX_GPU_CONSTRAINT` (the JAX jobs' own, by default compute
capability 7.5 or newer; empty drops it). The `GPU_CC:*` names are my
reading of sherlock's feature list: if `sbatch` rejects them, check
`sh_node_feat` and set `JAX_GPU_CONSTRAINT` (and the `-C` above) to match.

One task by hand, on a machine with a GPU:

```bash
SLURM_ARRAY_TASK_ID=15 ./run_prep_nonpeaks.sh local
SLURM_ARRAY_TASK_ID=15 ./run_train_full.sh local jax [--container]
DRY_RUN=1 SLURM_ARRAY_TASK_ID=15 ./run_train_full.sh local jax   # resolve only
SLURM_ARRAY_TASK_ID=15 ./run_contribution.sh local jax
```

`run_train_full.sh` also takes `FORCE=1` (retrain a completed run),
`IGNORE_BIAS=0`, `EXTRA_ARGS="..."` (appended to both pipelines),
`BATCH_SIZE=N` and `ALLOW_CPU=1`; `run_contribution.sh` takes `HEADS="counts"`,
`FORCE=1`, `INTERPRET_BATCH_SIZE=N` and `DRY_RUN=1`. See their headers.

On GPUs under 16 GB (`SMALL_GPU_MIB`; sherlock's 12 GB TITAN Xp, 11 GB RTX
2080 Ti) `run_contribution.sh` scores PyTorch and JAX at an interpret batch
size of 8 (TensorFlow explains one example at a time regardless). TensorFlow
training needs a GPU of 16 GB or more, which `submit_all.sh` requests (see
"The first sweep").

## What is held equal

Everything except the engine:

- **The same nonpeaks.** `run_prep_nonpeaks.sh` writes them once per dataset
  and fold, under `preprocess/`, and every engine trains on that file; each
  training task waits for its own prep task (`--dependency=aftercorr`).
- **The same preprocessing and test regions.** The bigwig, the peak
  filtering and the test split are framework-neutral code, so the three
  engines' runs of one dataset and fold are scored on the same regions.
  `run_summarize.sh` verifies it (`inputs.tsv`, and `--check-bigwigs`).
- **The same settings**: each command's defaults (`--seed 1234`,
  `--dnase-convention both-half` on DNase, `--data-aug augment`, 50 epochs,
  early stopping),
  the bias threshold factor (0.5 ATAC, 0.8 DNase, as in dnase-rc) and
  `--ignore-bias-flag` (as in dnase-rc: the stomach samples' own bias models
  fail the quality check in every configuration; `IGNORE_BIAS=0` enforces
  it).

What cannot be equal is the arithmetic: three frameworks initialize,
schedule and round differently, so the models differ by more than float
noise even at the same seed. The question is whether they are equally
good, and how far apart they are.

## The comparison

`run_summarize.sh` writes five tables to a directory under
`/path/to/workdir/chromatlas/test/chrombpnet-refactor-engines/analysis/`
and prints a headline (`run_summarize.py`'s docstring has the details). The
directory says which of the engines' profile scores were read (`OUT_DIR`
overrides it; a relative one is taken inside `analysis/`):

- `profile-stopgrad-off/`: `--profile-stopgrad off`, the default: the
  engines' profile scores rescored with the original's weighting
  (`chrombpnet_nobias.midpoint.profile_scores.h5`, below);
- `profile-stopgrad-on/`: `--profile-stopgrad on`, the pipeline's own profile
  scores, which in this sweep honour the stop-gradient.

`plot_summarize.sh` draws each of them in place (or one, `--dir
profile-stopgrad-off`).

| table | what |
|---|---|
| `results.tsv` | test metrics: each engine's ChromBPNet model (`own`) and bias model (`bias`), and the old TensorFlow ChromBPNet from the atlas (`original`, the ids table's `TENSORFLOW_DIR`) |
| `agreement.tsv` | for each engine, dataset and fold: its test predictions against the original TensorFlow model's (`TENSORFLOW_DIR/evaluation/chrombpnet_predictions.h5`) -- Pearson r of the predicted log counts and median JS distance of the predicted profiles, on the regions both scored |
| `contributions.tsv` | for each engine, dataset, fold, head (counts, profile) and kind (hypothetical, projected): its DeepSHAP scores on its 30K peak subsample against the original model's genome-wide scores (the ids table's `DEEPLIFT_COUNTS_DIR` and `DEEPLIFT_PROFILE_DIR`, or `DEEPLIFT_DIR` for both), regions matched by summit and kept only with identical sequence; per region Pearson r and cosine similarity over the 2114 bp input (hypothetical: the 4 x 2114 scores; projected: the observed base's score per position), and their median and mean over regions |
| `inputs.tsv` | sha256 of each run's filtered peaks and nonpeaks (and bigwig with `--check-bigwigs`): are the engines' test regions identical? |
| `timings.tsv` | wall time of each step of each run, with its GPU |

The contribution comparison reads ~11 GB of original scores per dataset and
fold (all peaks; ~3 min each on the lab GPU host, 4 at a time), so run it as a job
(`sbatch run_summarize.sh`); its results are cached in
`.contributions_cache/`, keyed by the input files' sizes and times, so a
rerun only redoes what changed. `--no-contributions` skips it. A run's
counts scores exist once `run_contribution.sh` has scored them; until then
only the profile rows appear.

### Profile scores with the original's weighting

The profile DeepSHAP target weights the profile logits by their softmax under
a stop-gradient. chrombpnet 1.x, whose genome-wide scores are the `original`
here, leaves the stop-gradient off (its shap fork has the handler commented
out), so each logit is weighted by the midpoint of the input's and the
reference's softmax. Every run of this sweep used code from after 2026-10-02
(`provenance.txt`: 352b1d7, 43701aa, 168ac35), which honoured the
stop-gradient in all three engines, so the pipeline's
`chrombpnet_nobias.profile_scores.h5` are stop-gradient scores. Since
2026-10-10 chrombpnet's default is the original's midpoint weighting
(`--contribs-profile-stopgrad` for the other). The counts head does not
depend on it.

To compare like with like, rescore the profile head with the midpoint
weighting into files of its own, and summarize those into a directory of
their own:

```bash
cd /path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/engines
# 1. chrombpnet_nobias.midpoint.profile_scores.h5 beside each run's own scores
PROFILE_WEIGHTING=midpoint ENGINES="pytorch tensorflow" ./submit_all.sh sherlock --contribution
PROFILE_WEIGHTING=midpoint ENGINES=jax ./submit_all.sh sherlock --container --contribution
# 2. once those have finished: tables and plots under analysis/profile-stopgrad-off/
sbatch run_summarize.sh --profile-stopgrad off   # -> analysis/profile-stopgrad-off/
./plot_summarize.sh --dir profile-stopgrad-off
```

`run_contribution.py` refuses to run if the installed chrombpnet predates
`--contribs-profile-stopgrad`, which would score with its old default. It
writes the midpoint scores under their own names; the pipeline's stop-gradient
files are left alone, and `PROFILE_WEIGHTING=stopgrad` reproduces them exactly
(checked bitwise on the regression fixture). In `profile-stopgrad-off/contributions.tsv` the
last column, `profile_scores`, says which profile scores were compared; the
plot's subtitle says so too.

### How far DeepSHAP moves on the same model

DeepSHAP averages over 20 dinucleotide-shuffled references, and two runs with
different references disagree even on one model. To read the engines'
agreement with the original on that scale, `run_deepshap_seeds.sh` runs the
original chrombpnet's DeepSHAP -- kundajelab/chrombpnet at 09938fd
(`ORIGINAL_REPO`), its own `interpret.main`, in the TensorFlow env with the
kundajelab shap fork -- on each row's original model, once per seed. The only
change to the original is that its references are drawn from
`RandomState(seed)` instead of unseeded. The model is the ENCODE release in
`chromatlas/<ASSAY>/<ENCID>/models/fold_N/`, the one whose genome-wide scores
are the stored originals; the regions are the engines' whole 30K subsample
(`N_REGIONS=N` for the first N only).

```bash
cd /path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/engines
SEEDS="1 2 3" ./submit_all.sh sherlock --deepshap-seeds     # one 20-row GPU array per seed, ~4 h a task
# once they have finished (with the engines' midpoint profile scores, the original's weighting):
sbatch run_summarize.sh --profile-stopgrad off   # -> analysis/profile-stopgrad-off/
./plot_summarize.sh --dir profile-stopgrad-off
```

`run_summarize.py` adds them to `contributions.tsv` as more rows: engine
`original:seed<s>` against reference `original` (each seeded run against the
stored scores) and against `original:seed<t>` (two seeded runs against each
other: the ceiling no comparison of DeepSHAP scores is expected to clear).
`plot_summarize.sh` draws them in `summary_contributions_{pearson,cosine}`:
the seeded runs against the stored scores as a solid grey bar per fold,
"original (same model; diff seed)" -- the same model and code, only the references differ:
the ceiling -- just after the retrained original's and before the engines';
and, with two or more seeds, the runs against one another as a solid line per
fold. Outputs:
`${ROOT}/original/<ASSAY>/<ENCID>/fold_N/` (`regions.bed`, `seed_<s>/`).

### The original, retrained

The stored scores come from one particular training run of the original
code; every engine is another training run, so part of the engines' distance
from those scores is the distance between two trained models, whatever the
code. `run_original_retrain.sh` measures that part: it retrains the original
chrombpnet itself on each row's sweep inputs -- the BAM, peaks, nonpeaks,
fold and bias threshold factor every engine used -- and scores the result
with its own DeepSHAP exactly as `run_deepshap_seeds.sh` scores the ENCODE
model (seed 1, the engines' 30K subsample).

- Code and env: kundajelab/chrombpnet at 09938fd (`ORIGINAL_REPO`) in its own
  conda env, `chrombpnet` (`ORIGINAL_ENV`): an editable install of that
  checkout with the original's pinned requirements, including tfmodisco
  0.5.16.0, which its read-shift estimate imports and the TensorFlow engine's
  env does not have. For DeepSHAP it has the same TensorFlow 2.8.0, shap fork,
  deeplift, numpy and h5py as `chrombpnet-tensorflow`.
- Natively only (`--container` is refused). The original's `train` and
  `bias train` end by writing their report PDF with weasyprint, which loads
  pango and cairo from the system -- its Dockerfile installs them with apt --
  and the env does not bring them. On sherlock the job loads them as modules,
  as the chrombpnet runs do (`module load system`, `pango`, `system cairo`);
  the lab GPU host and the lab CPU host have them in the OS; the container image has neither. The
  first submission (job 47622917), without the modules, stopped on all 20 rows
  at the runtime check for this reason.
- Steps (`run_original_retrain.py` runs each from the original's checkout):
  the original's `chrombpnet bias train`, its `chrombpnet train` on that bias
  model, the test-set predictions its `pipeline` makes next (`train` stops
  before them), then `run_deepshap_seeds.py` on the no-bias model. `train`
  rather than `pipeline` skips footprints, MoDISco and reports, which nothing
  here reads; the model is the same.
- One change to the original: its bias-model check is an assert, and the
  stomach samples' bias models fail it (as they fail the engines' same check,
  which the sweep ran as a warning with `--ignore-bias-flag`). `IGNORE_BIAS=1`,
  the default, makes that one assert the same warning.
- What stays the original's own: its read shifts (on DNase the 0/+1 cut
  convention, not both-half), its bias model, its training.
- A finished step is not run again, so a task that hits the 2-day limit is
  resubmitted and goes on from its next step.

```bash
./submit_all.sh sherlock --original-retrain                  # one 20-row GPU array, natively (not --container)
# once it has finished:
sbatch run_summarize.sh --profile-stopgrad off   # -> analysis/profile-stopgrad-off/
./plot_summarize.sh --dir profile-stopgrad-off
```

`run_summarize.py` adds engine `original_retrained` to `results.tsv` (its
test scores), `agreement.tsv` and `inputs.tsv`, and two kinds of row to
`contributions.tsv`: `original_retrained:seed1` against reference `original`
(the stored scores: the row each engine is to be read against) and against
`original:seed1` (the ENCODE model's run with the same references, so the two
models alone differ). The figures give it a panel in `summary_scores` and
`summary_agreement`, and a hollow bar, "original (new model)", first in each
fold of `summary_contributions_*` (then "original (same model; diff seed)", then the
engines'). Outputs:
`${ROOT}/original_retrained/<ASSAY>/<ENCID>/fold_N/` (`bias_model/`,
`model/`, `regions.bed`, `seed_1/`).

`plot_summarize.sh` draws, as `.png`, `.pdf` and a `.tsv` table view, with
the datasets in the same order as dnase-rc's plots (ENCSR000EOT,
ENCSR868FGK, ENCSR163PKT, ENCSR337UIU):

- `summary_scores`: counts Pearson r and 1 - profile JSD on peaks, one
  panel per engine (and `original`, and `original (new model)`), each mean
  marked with its difference from pytorch (`--reference` changes that).
- `summary_agreement`: the agreement table, one panel per engine (engine vs
  original; and the retrained original vs original).
- `summary_contributions_pearson`, `summary_contributions_cosine`: the
  contributions table's per-region medians, one panel per head and kind,
  one bar per engine, after a hollow one for the retrained original and a
  grey one for the original's model rerun with a new seed.
- `summary_runtime`: hours per run and step, one bar per engine. Runtimes
  compare only on the same GPU type; the GPU is written over each bar when
  types differ (use `GPU_CONSTRAINT` for a clean runtime comparison).

The original is a separately trained model, so neither agreement is
expected to reach 1: what matters is how the engines compare with one
another. Two quirks of the originals: some atlas runs' prediction files
store `coords_center` shifted by half the input window (1057 bp) from the
summit, so the agreement uses whichever offset matches the most regions and
records it (`center_offset`); and two rows have no usable original
predictions (ENCSR868FGK fold 2: no file; ENCSR337UIU fold 0: the file is
truncated, 8 MB of 300 MB), so they have no agreement rows. On DNase the original also used the original 0/+1 cut convention,
now `--dnase-convention right-deprecated`; the engines the package default
(both-half).

## Environments

All three are editable installs of `/path/to/workdir/
software/chrombpnet-refactor` (they pointed at the `chrombpnet-engines`
worktree until 2026-10-05, when its branch was merged into main), so **a job
runs whatever that checkout holds when it starts**: do not switch its branch
or edit it while jobs run. Each run records the commit, and whether the tree
was modified, in `provenance.txt`. The old `chrombpnet-refactor` env is
gone; `chrombpnet-pytorch-sherlock` still points at the same checkout.

How each reaches a GPU (`activate_engine` in `_common.sh`):

- **pytorch**: torch 2.5.1+cu118 brings its CUDA; `module load gcc/14.2.0`
  on sherlock, as in the dnase-rc runs. Same stack as
  `chrombpnet-pytorch-sherlock`.
- **tensorflow**: TF 2.8 needs CUDA 11.2 + cuDNN 8.1 from the system:
  `module load cuda/11.2.0 cudnn/8.1.1.33` on sherlock (what
  `dnase-rc/run_compare.sh` loads for the original package), the
  `/usr/local` copies elsewhere. `TF_FORCE_GPU_ALLOW_GROWTH=true`, because
  the engine's DeepSHAP runs in a spawned child process that needs GPU
  memory while the parent still holds what it trained with (the engine sets
  it itself since commit 636394d; the script keeps it). On GPUs newer
  than TF 2.8 knows (L40S, H100) it JIT-compiles from PTX at start-up: slow
  to start, then fine. For the same second process the GPU must not be in
  Exclusive_Process compute mode, sherlock's default: the first sweep's
  TensorFlow tasks all died at the bias model's DeepSHAP with
  `CUDA_ERROR_DEVICE_UNAVAILABLE` (worker exit code -6). `submit_all.sh`
  gives the TensorFlow jobs `--gpu_cmode=shared` (sherlock's own sbatch
  option), and the TensorFlow preflight refuses an exclusive-mode GPU at
  the start of a task.
- **jax**: `jax[cuda12]==0.11.2`, CUDA 12.9 and cuDNN 9 as pip wheels --
  the repository's default since commit 352b1d7 (it was `jax[cuda13]`).
  CUDA 13 dropped Pascal and Volta GPUs (the dnase-rc jobs landed on TITAN
  Xp nodes, compute capability 6.1) and needs driver >= 580, while CUDA 12
  supports both and needs only a CUDA 12 driver (>= 525). The env's
  previous package list is kept in `envs/chrombpnet-jax.before-cuda12.txt`.
  The wheels are found by RUNPATH, which LD_LIBRARY_PATH outranks: on
  the lab GPU host, ~/.bashrc puts CUDA 12.4 there, which made JAX fall back to the
  CPU. The job drops CUDA/cuDNN entries from LD_LIBRARY_PATH, and since
  commit e460daf the engine also loads its bundled libraries first when it
  finds namesakes there.

Every training task first runs a convolution on the GPU in its framework
(`gpu_preflight`) and stops if it cannot, rather than training on the CPU
into the two-day limit. The pipelines also get `--device cuda`, which
raises instead of falling back. Then `runtime_check` imports and exercises
everything else the pipeline loads -- h5py, pyBigWig, numba (a compilation),
TF-MoDISco, matplotlib (a PNG), WeasyPrint (a PDF) -- and runs each
command-line tool, so a library that cannot load on the node stops the job
in its first minute instead of hours in.

### Sherlock's glibc

**Since 2026-10-19 (chrombpnet-refactor commits 04ef088 and 8e95d39)** the
envs are newer than this section describes, and none runs natively on a
glibc-2.17 node. `chrombpnet-pytorch` (PyTorch 2.14, numba 0.68) needs glibc
2.28: `envs/glibc_scan.py` finds GLIBC_2.28 in torch and igraph, and 2.27 in
numba, llvmlite, hdf5plugin, triton and the nvidia wheels.
`chrombpnet-tensorflow` (TF 2.21) and `chrombpnet-jax` need 2.27. Run them
with `--container` on such nodes. The scripts in `run/`, `runtime/` and
`dnase-rc/` default to `chrombpnet-pytorch`, stop at once on a node with an
older glibc, and take `CONDA_ENV=chrombpnet-pytorch-sherlock` (torch 2.5.1,
built on sherlock) to reproduce earlier runs. What follows describes the
envs these runs used.

Sherlock's nodes run glibc 2.17 (the first `check_envs.sh` on sherlock
failed on JAX with ``GLIBC_2.23' not found``). The three envs were built on
the lab GPU host (glibc 2.35), where pip and conda pick the newest builds, some of
which need glibc 2.27-2.34. A scan of every compiled file in each env
(`readelf -V`; `python3 envs/glibc_scan.py <env prefix>`) found them;
`chrombpnet-pytorch-sherlock`, built on sherlock, has none:

- **chrombpnet-pytorch**: numba/llvmlite (TF-MoDISco, tangermeme),
  pybigtools and hdf5plugin. Replaced with glibc-2.17 builds: the
  manylinux2014 wheels of numba 0.67.0 and llvmlite 0.49.0, and pybigtools
  0.2.4 and hdf5plugin 7.1.0 as `chrombpnet-pytorch-sherlock` built them
  from source on sherlock (no glibc-2.17 wheel exists), repacked into wheels.
  All four are in `envs/wheels-glibc217/`; the previous package list is
  `envs/chrombpnet-pytorch.before-glibc217.txt`. Same versions as
  `chrombpnet-pytorch-sherlock`; tests and the cross-engine DeepSHAP
  harness give the same results.
- **chrombpnet-tensorflow**: its libcurl (a glibc-2.28 build) is linked by
  pyBigWig, HDF5 and htslib, so the TF pipeline would have failed at its
  first bigwig or HDF5 file. Rebuilt from `envs/chrombpnet-tensorflow.yml`
  with `CONDA_OVERRIDE_GLIBC=2.17`, so conda picked glibc-2.17 builds;
  every pip package and every version that matters (TF, Keras, numpy,
  shap, deeplift, modisco-lite, h5py, pyBigWig, WeasyPrint) is unchanged,
  and the byte-identity harness passes. Left: `cryptography` and the
  TensorBoard data server, which the pipeline never loads. The previous
  package lists are `envs/chrombpnet-tensorflow.before-glibc217.*`.
- **chrombpnet-jax**: jaxlib 0.11, its CUDA plugin, the NVIDIA wheels,
  ml_dtypes, Pillow, cairo -- no recent-enough JAX GPU stack exists for
  glibc 2.17 (the newest manylinux2014 jaxlib, 0.7.0, needs cuDNN >= 9.8,
  whose wheels need glibc 2.27). So on sherlock the JAX jobs run in a
  container (next section).

### The container (`--container`)

With `--container`, `run_train_full.sh`, `run_prep_nonpeaks.sh`,
`run_contribution.sh` and `check_envs.sh` run themselves again inside
`apptainer exec --nv` (or `singularity`), with an Ubuntu 22.04 image -- the
OS the envs were built and tested on (the lab GPU host) -- that runs the same conda env
from /oak, with the node's GPU driver bound in. /oak, /scratch and a TMPDIR
outside /tmp are bound; the host's environment variables (the SLURM ones,
CONDA_EXE) pass through, except PATH, which is the image's -- so conda is
called by its path (`CONDA_EXE`, else `$CONDA_ROOT/bin/conda`), never by
name; the locale is C.UTF-8. It is off by default, for every engine.

- **jax** needs it on sherlock (glibc, above); without it a JAX task there
  stops at once and says so, and `submit_all.sh` refuses to submit.
- **tensorflow**: the image has no CUDA 11.2 / cuDNN 8.1 and no `module`, so
  the host's directories are found first (the modules' on sherlock, loaded
  in a subshell; `/usr/local` elsewhere), bound into the container with
  `/share/software`, and put on LD_LIBRARY_PATH inside (`TF_CUDA_DIRS`).
- **pytorch**: no `module load gcc/14.2.0` inside; the image's libstdc++ is
  new enough (the lab GPU host runs the same env natively on the same OS).
- Why use it for all three: the same system tools. Making a bigwig pipes
  the reads through the system's `awk` and `sort`, and the first sweep made
  its bigwigs 2.2-2.4x faster in the JAX container than PyTorch and
  TensorFlow did on the host, on the same node and data -- engine-independent
  time in the runtime comparison (see "The first sweep").
- The image is `/path/to/workdir/software/containers/ubuntu-22.04.sif`
  (`CONTAINER_IMAGE`, or the old `JAX_IMAGE`, overrides it). `check_envs.sh
  --container` pulls it the first time (`apptainer pull ...
  docker://ubuntu:22.04`); jobs never pull. `USE_CONTAINER=1` in the
  environment is the same as `--container`. `provenance.txt` records the
  image a run used.
- The JAX env's cuDNN 9.27 ships precompiled kernels for compute capability
  7.5 (Turing) and newer only -- not the P100/TITAN Xp (Pascal) or V100
  (Volta) nodes -- so the JAX jobs get `JAX_GPU_CONSTRAINT` (see above). If
  a JAX job lands on an older GPU anyway, its GPU preflight fails in seconds
  and says why.
- the lab GPU host has no Apptainer, so the wrapper logic is tested here with a
  stand-in `apptainer` that gives commands the environment the real one
  does (image PATH, no shell functions): pull, exec, the check, and a
  training task into preprocessing. On sherlock (2026-10-03) the real one
  pulled the image and ran the check with the GPU visible and glibc 2.35
  inside; that first run stopped at `conda: command not found` (the PATH
  above), fixed since.

### Provenance without git

Sherlock's system git (1.8) has no `git -C` and predates worktrees, which
the engines checkout is. `provenance.txt` therefore falls back to reading
the commit and branch from the `.git` files directly; whether the tree was
modified is then recorded as unknown. `run_deepshap_seeds.py` does the same
for the original's commit in `run.json` (the runs before 2026-10-13 recorded
it as null there).

## Smoke test (the lab GPU host, 2026-10-02)

Before handing over, the whole chain ran on the lab GPU host for one row -- task 15,
ENCSR337UIU (ATAC, stomach, the smallest BAM) fold 0 -- per engine, shortened
with `EXTRA_ARGS="--epochs 1 --interpret-subsample 2000"`, all three engines
sharing one L40S at once:

```bash
SLURM_ARRAY_TASK_ID=15 ./run_prep_nonpeaks.sh local
EXTRA_ARGS="--epochs 1 --interpret-subsample 2000" SLURM_ARRAY_TASK_ID=15 ./run_train_full.sh local <engine>
```

| engine | bias pipeline | chrombpnet pipeline | counts r (peaks) | median JSD (peaks) |
|---|---|---|---|---|
| pytorch | 82 min | 81 min | 0.721 | 0.551 |
| tensorflow | 64 min | 137 min | 0.728 | 0.544 |
| jax | 77 min | 117 min | 0.703 | 0.552 |

One epoch, so the scores say nothing about the engines; what the run shows
is that every step works under each engine on a GPU (`--device cuda`), the
checkpoints are `.pt` / `.h5` / `.keras`, the three engines' filtered peaks,
nonpeaks and bigwigs are byte-identical (`summarize.sh --check-bigwigs`:
3 of 3), and `check_runs.sh`, `summarize.sh` and `plot_summarize.sh` read
the result. Their output and the logs are in `analysis/smoke_test/`; the runs
themselves were deleted on 2026-10-13. Runtimes there are for a GPU shared
three ways and are not representative.

The smoke test ran before the glibc fixes below. On sherlock, the first
`check_envs.sh` (2026-10-03, P100 node) passed PyTorch and TensorFlow's GPU
checks and failed JAX on glibc; that led to the fixes in "Sherlock's glibc"
and the container. After them, `check_envs.sh local` passes for all three
engines on the lab GPU host (JAX also through a stand-in container), the scripts pass
their dry runs under bash 4.2 (CentOS 7's; empty arrays under `set -u`
needed the `${a[@]+"${a[@]}"}` form), and the PyTorch and TensorFlow envs
pass their test suites, the TF byte-identity harness (44/44) and the
cross-engine DeepSHAP harness. Run `check_envs.sh` on sherlock again
before submitting.

## The first sweep (2026-10-03 to 10-05): failures and runtimes

Arrays 46432885 (PyTorch), 46456253 (TensorFlow, after the
`--gpu_cmode=shared` resubmit) and 46432887 (JAX), all native except JAX.
JAX finished 20/20, PyTorch 18/20 (tasks 5 and 10 still running on slow
TITAN Xp nodes on 10-05), TensorFlow 13/20.

**The seven TensorFlow failures all ran on 12 GB TITAN Xp GPUs**; every
finished TensorFlow run had 16 GB or more (P100, V100, L40S).

- Tasks 1, 6, 7, 9: out of GPU memory in the ChromBPNet stage's DeepSHAP
  worker ("TensorFlow DeepSHAP worker failed with exit code 1"; the
  worker's own stderr shows the OOM). The parent process has trained,
  predicted and footprinted with TF by then, and with memory growth its
  pool only grows and is never returned; the worker is a second process on
  the same GPU (`engines/tensorflow/attribution.py`). On 12 GB nothing is
  left. The bias stage survives because its model is smaller.
- Tasks 10, 11, 17: `CUDA_ERROR_ILLEGAL_ADDRESS`, then `F
  device_event_mgr.cc:221 Unexpected Event status: 1`, in plain bias-model
  training -- a GPU or node fault (<node>, <node>), not engine
  code: other TensorFlow tasks trained on TITAN Xp nodes, and the user's
  PyTorch dnase-rc jobs hit the same illegal-address error there.
- **`-bs 32` was not a fix.** Measured on the lab GPU host (2026-10-06) with an L40S
  held down to 12 GB free, the real ChromBPNet model and the real DeepSHAP
  worker:

  | `-bs` | training process holds | DeepSHAP worker |
  |---|---|---|
  | 64 | 9,670 MiB | out of memory at 2,316 MiB (the sweep's failure) |
  | 64, process capped at 7.5 GB | 8,852 MiB | out of memory |
  | 32 | 5,574 MiB | fits, peak 5,510 MiB (11,084 MiB in all) |

  That emulation trained and predicted but did not run the rest of the
  pipeline, and on the real reruns (2026-10-06/07, TITAN Xp) the worker ran
  out of memory at `-bs 32` too (tasks 6 and 9), and once in the *bias*
  stage at 64 (task 17). The cause is TensorFlow-only: its DeepSHAP has to
  run in a second process (the shap fork needs TF1 graph mode, which is
  process-global), and the first keeps every byte of GPU memory it grew to;
  PyTorch and JAX score contributions in the same process, reusing the
  memory training freed. So TensorFlow simply needs a GPU of 16 GB or more:
  `submit_all.sh` now gives the TensorFlow jobs `TF_GPU_CONSTRAINT` (compute
  capability 6.0, 7.0, 8.x, 9.0), and `run_train_full.sh` no longer lowers
  the batch size on small GPUs. The reruns were relaunched on L40S at
  `-bs 64` (array 46951206, tasks 1, 6, 9, 17).
- Tasks 10, 11 and 17 crashed on <node> (twice) and <node>;
  `SBATCH_EXTRA="--exclude=<node>,<node>"` keeps the rerun off them.
- Rerun, natively (TensorFlow gets a 16 GB+ GPU by default now):
  `SBATCH_EXTRA="--exclude=<node>,<node>" ENGINES=tensorflow
  ARRAY=... SKIP_PREP=1 ./submit_all.sh sherlock`. The reruns' pipeline keeps
  both heads' contribution scores itself.

**Why JAX ran fastest, then TensorFlow, then PyTorch** (median hours per
run: JAX 10.5, TF 19.7, PyTorch 25.8) is mostly not the engines:

| engine | median, all runs | on L40S/A100 | on P100/TITAN Xp/V100 |
|---|---|---|---|
| pytorch | 25.8 h (n=18) | 13.0 h (n=5) | 30.7 h (n=12) |
| tensorflow | 19.7 h (n=13) | 13.1 h (n=7) | 36.8 h (n=6) |
| jax | 10.5 h (n=20) | 9.8 h (n=11) | none (constrained to CC >= 7.5) |

- **GPU and node assignment** (about 85% of the PyTorch-JAX gap): JAX was
  constrained to compute capability >= 7.5, so it never ran on the old sh02
  nodes, whose CPUs also make MoDISco and the bigwig slower. TensorFlow
  looks faster than PyTorch only because it failed on every TITAN Xp, so
  more of its survivors ran on L40S.
- **The bigwig** (engine-independent, built twice per run) was 2.2-2.4x
  faster in the JAX container than on the host, same node and data:
  CentOS 7's gawk/coreutils and the host locale vs the image's mawk,
  coreutils 8.32 and C.UTF-8. `--container` for every engine removes it.
- **MoDISco** is about half of every L40S run, CPU-bound and roughly
  engine-independent (the TF env's modisco-lite 2.3.2 is 10-30% slower than
  2.6.0).
- **The engine itself**, same L40S and dataset: DeepSHAP on the ChromBPNet
  profile head ~24 seq/s JAX, ~8 PyTorch (batch of 16 under-fills the GPU),
  ~6 TensorFlow (one example at a time); training ~1.75x faster per epoch in
  JAX than TensorFlow; early stopping (11-27 epochs) adds +-50% noise. On
  equal hardware the order is JAX < PyTorch < TensorFlow.

For a fair runtime comparison: one GPU constraint for every engine
(`GPU_CONSTRAINT`), `--container` for every engine, and compare the
engine-specific steps (seconds per epoch, DeepSHAP rate) rather than whole
runs.

## Outputs

```
/path/to/workdir/chromatlas/test/chrombpnet-refactor-engines/
  preprocess/<ASSAY>/<ENCID>/fold_N/negatives.bed.gz      shared nonpeaks
  <engine>/<ASSAY>/<ENCID>/fold_N/
    bias_model/  model/                                    the two pipelines
    provenance.txt  timings.tsv  COMPLETED
  original/<ASSAY>/<ENCID>/fold_N/seed_<s>/                the original's DeepSHAP, seeded
  original_retrained/<ASSAY>/<ENCID>/fold_N/               the original, retrained
    bias_model/  model/  seed_<s>/  provenance.txt  timings.tsv  COMPLETED
  analysis/profile-stopgrad-on/                            run_summarize.sh's tables and
                                                           plot_summarize.sh's figures, from
                                                           the pipeline's profile scores
  analysis/profile-stopgrad-off/                           the same, --profile-stopgrad off
  analysis/smoke_test/                                     the 2026-10-02 smoke test
```

About 5 GB per run, so ~300 GB for the sweep. `run_contribution.sh` adds
`model/auxiliary/interpret_subsample/chrombpnet_nobias.counts_scores.h5`
(~0.7 GB) and `chrombpnet_nobias.contribution.args.json` to a run. Logs:
`sherlock/logs/engines_{prep,train,contribution,deepshap_seeds,original_retrain}.<job>_<task>.{out,err}` and
`engines_summarize.<job>.{out,err}`.
