# Optimization benchmark

What `--predict-fk` -- the fused Triton forward -- is worth, measured on the
five folds trained by
`../run/run_chrombpnet_refactor_benchmark_260911.sh` -- their own `.pt`
checkpoints and their own peak regions. Real models, real data, nothing
synthetic.

## Site configuration

Every path, environment location and cluster option comes from the blog
folder's `site.env` (`../site.env`, from `../site.env.example`; or
`BLOG_SITE_ENV=<file>`); each script stops and names any value it needs that
is left empty. A value already in the environment wins over the file's, so
`OUT_ROOT=<dir> bash run_sections34.sh local` points one run elsewhere. The
scripts here read:

| variable | used for |
|---|---|
| `CHROMBPNET_REPO` | chrombpnet-refactor (default: this checkout) |
| `ORIGINAL_REPO` | chrombpnet 1.x, the `original` arms and the old GC matching |
| `KIT_ROOT`, `KIT_HOME` | the optimization kit's checkout, and its weights, data and JIT cache |
| `GENOME`, `CHROM_SIZES`, `REFERENCE_DIR` | hg38, and `folds/fold_<k>.json` |
| `CONDA_ROOT` | the conda installation holding the envs (`bin/conda`, `envs/`) |
| `CONTAINER_IMAGE`, `CONTAINER_BINDS` | cherimoya's Ubuntu 22.04 container, and the directories bound into it |
| `REFACTOR_MODELS` | chrombpnet-refactor's ENCSR868FGK models: `ATAC/ENCSR868FGK/fold_<k>/model/{models,auxiliary}/` |
| `ORIGINAL_MODELS` | chrombpnet 1.x's models of the same folds, laid out alike (`.h5`) |
| `CHERIMOYA_MODELS` | CATv1's models: `ATAC/ENCSR868FGK/fold_<k>/cherimoya.torch` (and its kernel cache, `.cache/`) |
| `BENCHMARK_RUNS` | where the September benchmark left `chrombpnet-refactor-<level>-cleanup/` |
| `OPTIMIZATION_OUT` | the blog's run tree (`OUT_ROOT` of the Sections 3-4, kit, head-to-head and cherimoya jobs) |
| `OPTIMIZATION_ANALYSIS` | the September sweep's run tree (`run_chrombpnet_optimization_*.sh`, `summarize.sh`) |
| `PT_ENV_GLIBC217` | optional: a build of the PyTorch env for glibc 2.17 nodes, the default under Slurm |
| `CONTAINER_UNDER_SLURM` | optional: `1` runs cherimoya in the container under Slurm |
| `SBATCH_SITE_ARGS`, `SBATCH_L40S`, `SBATCH_H200`, `SBATCH_H100`, `LOG_DIR` | sbatch options for the cluster and for each GPU, and where the submit scripts send logs |

Under Slurm (`CLUSTER=slurm`) the jobs call site.env's hooks where the cluster
provides a library or compiler: `site_tf28_libraries` (CUDA 11.2 and cuDNN 8.1
for TensorFlow 2.8), `site_report_libraries` (1.x's PDF report),
`site_kit_compiler` (a C11 compiler for the kit's Triton launchers) and
`site_pytorch_compiler` (a newer gcc and its libstdc++, before the glibc 2.17
PyTorch env).
A site that needs none defines none. The `sbatch` lines below are run from this
folder, after `source ../site.env` (or with the options written out);
`--partition`, node exclusions and the like go in `SBATCH_SITE_ARGS`.

## The September benchmark

```bash
sbatch ${SBATCH_SITE_ARGS} --constraint=<A100 feature> run_chrombpnet_optimization_benchmark.sh slurm
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H100} run_chrombpnet_optimization_benchmark.sh slurm
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_chrombpnet_optimization_benchmark.sh slurm
./summarize.sh "${OPTIMIZATION_ANALYSIS}"   # once the array finishes
```

The script names no GPU itself: the card is the sbatch line's `--constraint`.
`--array=0-4` runs one fold per task.

**Find the feature name first — do not guess it:**

```bash
./list_gpu_skus.sh <partitions>              # the GPU nodes' features in those partitions
./list_gpu_skus.sh <partitions> '<pattern>'  # only the features matching a pattern
```

A cluster's names are often more specific than the bare model (something like
`A100_SXM4`, not `A100`), and they can differ between partitions.
A constraint matching no node does **not** fail the submission: Slurm accepts
the job and leaves it pending indefinitely with reason `BadConstraints`, which
reads like an ordinary queue wait. After submitting, check:

```bash
squeue -u $USER      # NODELIST(REASON) showing (BadConstraints) means no match
```

`run_chrombpnet_optimization_benchmark.sh CLUSTER [FOLD_NUM] [N_REGIONS] [PHASES]`

| argument | default | notes |
|---|---|---|
| `CLUSTER` | `slurm` | `local` skips the site hook |
| `FOLD_NUM` | `$SLURM_ARRAY_TASK_ID`, else 0 | the array supplies it on the cluster |
| `N_REGIONS` | `20000` | deterministically subsampled from the fold's peaks; `all` uses every peak (~270k) |
| `ARMS` | `default,fk` | `default` is how the package now runs; `fk` adds `--predict-fk` |

`N_CONTRIB` (env, default 2000) sets the smaller subsample for `contribs_bw`,
which is far slower per region because it runs DeepLIFT rather than a forward.

**Environment.** `CONDA_ENV` defaults to `chrombpnet-pytorch` (torch
2.14.1+cu126), which needs glibc 2.28 or newer. Some clusters' GPU nodes (the
blog's L40S ones) run glibc 2.17; there the script stops before activating the
env and says so. Resubmit with `--exclude=<node>`, or set `CONDA_ENV` to a
build for glibc 2.17 (site.env's `PT_ENV_GLIBC217`; the earlier runs': torch
2.5.1+cu118). `GLIBC_FLOOR` (default 2.28) moves the check. The same applies
to `PT_ENV` in the pred_bw re-run scripts below.

## What it runs

A preflight check, then three commands:

| stage | command | arms |
|---|---|---|
| 1 | `pred_bw` | per arm -- reaches the fused forward through `PredictionAdapter` |
| 2 | `contribs_bw` | **once only.** No `--predict-fk`: `deep_lift_shap` hooks the ReLU modules of the raw model and needs an autograd graph, which the fused kernels have not got |
| 3 | `footprints` | per arm -- also reaches the fused forward |

> **Historical note.** This script used to sweep a six-way `--optimize`
> matrix (`none,1,2,3,1|3,all`) to attribute each optimization phase. That
> measurement is complete and recorded in the repository CHANGELOG (55.7s ->
> 29.6s on `pred_bw`), and the flag has since been removed: phases 1 and 3
> are simply how the package runs. To re-run the original sweep, check out
> commit `693d162`. The text below that still says "phase" refers to that
> historical run.

`footprints` uses `data_utils.get_seq`, which
deliberately keeps the per-region pyfaidx loop.

`summarize.sh` prints this reachability under each table so a flat number is
not misread.

## The one thing to expect

**The `fk` arm is refused on A100, by design.** The Triton kernels carry a
per-architecture table, and it records that sm_80 has no bitwise route --
K1's fp32 kernels do not reproduce that card's log-counts at every batch
shape. The script records the refusal reason and carries on, so an A100 job
still yields the `default` arm. Fused-kernel timings come from H100/H200.

## Output

```
${OPTIMIZATION_ANALYSIS}/<run tag>/fold_N/      (OUT_ROOT)
    timings.tsv          fold, gpu, arch, stage, arm, seconds, status, detail
    artifact_diffs.txt   which artifacts the fk arm moved vs default
    regions.bed          the subsample actually predicted
    contrib_regions.bed  the smaller subsample used for contribs_bw
    pred_bw/<arm>/       that arm's outputs and run.log
    contribs_bw/default/
    footprints/<arm>/
```

`summarize.sh` collates the folds into one table with mean, standard
deviation and speedup against `default`, and names any arm that was refused.

Artifacts are compared with the digest helpers from
`tests/regression/artifacts.py`, so bigWigs, HDF5 and PNGs are compared by
content rather than raw bytes. That is the only thing borrowed from the
repository's regression harness; the harness itself is not run here -- it
exercises a synthetic fixture and its own freshly trained models, which
answers "did anything change" rather than "how much faster on my data".

## Notes

- The Triton cache is shared at `.triton-cache/` beside `OUT_ROOT` (or
  `TRITON_CACHE_DIR`) rather than `$HOME`, so only
  the first job of a sweep pays the JIT and array tasks do not race. The kit's
  precompiled caches target torch 2.4.1 / triton 3.0.0 and will not match this
  stack, so the first `fk` run on a new card compiles its kernels. Expect
  that run to be markedly slower than the warm ones.
- `pred_bw` is given `-bw`, so the metrics stage runs and the finish-stage
  overlap is actually exercised. The observed bigWig is ~1.2 GB per fold.
- bigWigs are written as fixed-step blocks: the same values in a different
  on-disk section type from what the package shipped before this work.
- A failing run prints the tail of its log inline, so the job output is
  self-contained even if the results directory is later cleaned.

## If the preflight says CUDA is unusable

`torch.cuda.is_available()` returns False for several unrelated reasons and
reports none of them. `./diagnose_cuda.sh` separates them: whether
`libcuda.so.1` loads, whether `/dev/nvidia*` are present,
`CUDA_VISIBLE_DEVICES`, the GPU's compute mode, and the real exception from
`torch.cuda.init()`.

One cause is a torch built for a newer CUDA than the driver: the
`chrombpnet-pytorch` environment is `torch 2.14.1+cu126` and needs a driver for
CUDA 12.6 or newer, so it reports no devices on an older one (the earlier
runs' glibc 2.17 build, `torch 2.5.1+cu118`, needs only CUDA 11.8).
The preflight prints both CUDA versions for that reason.

What is *not* a cause: the GPU being an H100 or H200. Both builds ship `sm_90`
kernels, so Hopper is supported.

## Re-run for the blog's speedup tables (`run_pred_bw_rerun.sh`)

The phase sweep above always ran the unoptimized arm first, on a cold file
cache, which overstated it. `run_pred_bw_rerun.sh` re-measures `pred_bw` with
the genome, bigWig and models read once before anything is timed, an untimed
warm-up run of every arm, and the arms in a different order each round
(`ROUNDS`, default 3). Every arm predicts the same 20,000 peaks of the fold
against the same observed bigWig:

| arm | what runs |
|---|---|
| `original` | chrombpnet 1.x (TF 2.8, env `chrombpnet`) on the fold's TF models |
| `none`, `kit`, `ours`, `both` | `code-693d162/` (`git archive 693d162`) with `--optimize none`, `1`, `3`, `1,3` |
| `main`, `fk` | the current checkout, default and `--predict-fk` |

```bash
./list_gpu_skus.sh <partitions>
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_pred_bw_rerun.sh slurm
PT_ENV="${PT_ENV_GLIBC217}" \
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_pred_bw_rerun.sh slurm   # glibc 2.17
python summarize_rerun.py "${OPTIMIZATION_OUT}"   # medians per arm, speedups, and medians by position
```

Results: `${OPTIMIZATION_OUT}/<gpu>/fold_N/timings.tsv` (`OUT_ROOT`).
The refactor arms run in `PT_ENV`: `chrombpnet-pytorch` (torch 2.14.1) by
default, or the glibc 2.17 build (`PT_ENV_GLIBC217`; the blog's: torch 2.5.1)
on the glibc 2.17 L40S nodes, where the default stops with that hint. Timings
from before 2026-10-21 used the glibc 2.17 build on every card. No container
is needed.

### Kit changes with the fused kernels (`run_pred_bw_kit_fused.sh`)

A copy of `run_pred_bw_rerun.sh` with two more arms from `code-693d162/`:
`kitfk` (`--optimize 1,2`: the kit's changes with its Triton forward, without
this package's own) and `fkonly` (`--optimize 2`: the Triton forward alone),
timed with `none`, `kit`, `both` and `fk` in the same job. It writes under its
own run tag (`<gpu>_<arch>_kit-fused`), so it never clears the other job's
results; `summarize_rerun.py` reports each run tag separately.

```bash
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_pred_bw_kit_fused.sh slurm
ARMS=original,none,kit,ours,both,main,kitfk,fkonly,fk \
  sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_pred_bw_kit_fused.sh slurm
```

Under Slurm `PT_ENV` defaults to site.env's `PT_ENV_GLIBC217` when it names
one (the blog's: torch 2.5.1): the blog's L40S nodes and at least some of its
H200 nodes run glibc 2.17, where `chrombpnet-pytorch` cannot. If the H200
nodes sit in a partition of their own, put it in `SBATCH_H200` (a constraint
matching no node in the requested partitions leaves the job in
`BadConstraints`).

Both scripts drop every CUDA and cuDNN directory the job inherits from the
submitting shell before running their site hooks. On 2026-10-21 the kit env's
torch reported cuDNN 9.4 on an H200 node, not its own 9.1: cuDNN 9 finds its
sub-libraries on `LD_LIBRARY_PATH` before its own directory, so another
cuDNN 9 there gets mixed in (reproduced on a workstation). The kit's `exact`
mode then died loading cuDNN. `gpu_check.py` now refuses that mix before
anything is timed, and writes the loaded libraries and `LD_LIBRARY_PATH` to
`gpu_check_*.log`. `run_kit_modes.sh` also runs the `site_kit_compiler` hook (a
newer gcc): `exact` mode's kernels are not in the kit's shipped Triton cache,
and Triton compiles their launcher with the node's compiler, which on a gcc 4.8
node has no `stdatomic.h`.

### The optimization kit itself on stock chrombpnet (`run_kit_modes.sh`)

For the blog's "in original `chrombpnet`" column: the kit's own modes on its
pinned stock chrombpnet 1.0.1 (`stock` = `--mode off`, `stockdet` = `off
--det 1`, `exact`, `fast`), five rounds in rotating order, after a warm-up of
each, on the kit's model (ENCODE ENCFF142IOR, GM12878 ATAC fold 0) and the
same 20,000 regions the refactor runs predict. The kit has no switch for a
single optimization, so only its modes are measured. On an L40S `fast` runs
stock's TensorFlow graph inside the kit's pipeline (its changes without the
fused kernels) and `exact` the fp32 fused kernels; on an H200 both use them.

```bash
bash setup_kit_env.sh                       # once: env chrombpnet-kit, weights, data
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_kit_modes.sh slurm
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_kit_modes.sh slurm
```

### On a GPU host instead of a Slurm cluster

`run_pred_bw_kit_fused.sh` and `run_kit_modes.sh` also run as plain bash
scripts (`_common.sh` holds what differs). Without a Slurm job they take
`local` themselves, and:

- run on the first GPU in `CUDA_VISIBLE_DEVICES` (0 if unset), made the only
  visible one, and refuse it if anything already runs there
  (`ALLOW_BUSY_GPU=1` overrides);
- run the five folds one after another (`run_pred_bw_kit_fused.sh local 2`, or
  `0,3`, for some);
- give PyTorch its own CUDA wheels by dropping any CUDA / cuDNN directory a
  login shell put on `LD_LIBRARY_PATH`, and TensorFlow 2.8 `/usr/local`'s CUDA
  11.2 and cuDNN 8.1 (`TF_CUDA_DIRS=<dir>:<dir>` names others);
- check first that each env's framework reaches the GPU, because TensorFlow 2.8
  without its CUDA libraries runs on the CPU without an error;
- add the host to the run tag (`<gpu>_<arch>_kit-fused_<host>`), so a local run
  never replaces a Slurm job's results, and keep a Triton cache per host.

```bash
cd analysis/blog-2026-015/optimization       # in this checkout
nvidia-smi                                   # pick an idle GPU
CUDA_VISIBLE_DEVICES=<idle GPU> nohup bash -c \
  'ARMS=original,none,kit,ours,both,main,kitfk,fkonly,fk bash run_pred_bw_kit_fused.sh; bash run_kit_modes.sh' \
  > <log dir>/optimization.$(hostname -s).log 2>&1 &
python summarize_rerun.py <OPTIMIZATION_OUT> # once both have finished
```

The two scripts run one after the other, not side by side: a shared host's CPUs
and disk are shared, and `pred_bw` uses both. Each header records the load
average when it started.

### Per-lever effects in original chrombpnet (`run_kit_modes.sh` arm `stages`)

The kit offers no switch for a single lever (by design: a mode is the whole
composition), so each lever is measured as the stage it replaces: stock's
time for that stage against the kit's. The `stages` arm runs stock
chrombpnet 1.0.1's `pred_bw`, unchanged, under `stock_stage_timer.py`, which
times featurisation, the forward, the bigWig write, the metrics, the HDF5
write, the finish-stage imports and (with the job's clock) the teardown. The
kit arms keep their `opt_manifest.json`, whose run record has the kit's own
stage timings. `summarize_kit_levers.py` pairs the two per GPU:

```bash
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_kit_modes.sh slurm
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_kit_modes.sh slurm
python summarize_kit_levers.py "${OPTIMIZATION_OUT}"
```

The kit runs several stages beside the forward, so its time is what stays on
the critical path (e.g. the wait for the bigWig writer at close).

### Section 4 changes, one at a time (`run_section4.sh`)

`benchmark_section4.py` times one change in one process, old form against new;
`run_section4.sh` runs every measurement in a fresh process, the forms in a
rotating order, on ENCSR868FGK fold 0 (the 20,000 pred_bw peaks, the fold's
models and GC-matching inputs). Stages: `gc` (prep nonpeaks' GC matching, the
original package's against this one's), `shared` (pred_bw's per-model inputs
for three models, per model against once), `bigwig` (list, NumPy-argument and
fixed-step writes, at 693d162), `cast` (the legacy whole-input cast against the
batched one), `h5` (gzip 4, gzip 9 + shuffle, the same on threads).
`summarize_section4.py` prints the blog cells, with the kit-fused jobs'
pred_bw totals.

```bash
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_section4.sh slurm
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_section4.sh slurm
python summarize_section4.py "${OPTIMIZATION_OUT}"
```

### Cherimoya (`run_cherimoya.sh`)

Cherimoya's own commands on the comparison table's rows: `evaluate` (the
20,000 peaks, metrics against the observed bigWig), `attribute` (profile head,
2,000 peaks, 2,114 bp, 20 shuffles), `marginalize` (the fold's 5 motifs over
its test non-peaks), with CATv1's ENCSR868FGK models (same folds as
ChromBPNet's), inputs from `cherimoya_inputs.py`. The `cherimoya` env (torch
2.13, Triton 3.7, as cherimoya's uv.lock) needs glibc >= 2.28, which no build
of torch >= 2.9 avoids, so on a cluster of glibc 2.17 nodes
(`CONTAINER_UNDER_SLURM=1`) every command runs in the Ubuntu 22.04 container
(`CONTAINER_IMAGE`, with `CONTAINER_BINDS` bound into it). Each task compiles into a
kernel cache of its own (per run tag and fold) and warms the compiled commands
(`evaluate`, `marginalize`) on the full input: with one cache shared by the
array's tasks, 5 of 45 timed runs on the L40S took 10-15x longer while other
tasks were compiling into it (the medians held: 15 s, 161 s, 66 s).

```bash
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_cherimoya.sh slurm
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_H200} run_cherimoya.sh slurm
python summarize_rerun.py "${OPTIMIZATION_OUT}"   # run tags *_cherimoya
```

### Head-to-head with cherimoya (`run_headtohead.sh`)

cherimoya writes no predicted bigWigs or HDF5 files, so the blog compares it
only on the work it also does. `headtohead.py` runs either package's own CLI
command with those writers replaced by no-ops (and refuses if a writer it was
meant to skip never ran): `predict` = `pred_bw` with the full model only, the
20,000 peaks scored against the observed bigWig (cherimoya `evaluate`);
`attribute` = `contribs_bw`'s DeepSHAP on the profile head, 2,000 peaks, 20
shuffled references (cherimoya `attribute`). The job also times both packages'
full `contribs_bw`.

```bash
sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_headtohead.sh slurm
python summarize_rerun.py "${OPTIMIZATION_OUT}"   # run tag *_headtohead: orig_/ref_ predict, attr, contribs
```

### The blog's last values (`submit_blog_remaining.sh`)

As of 2026-10-30 two jobs fill everything still open in blog 2026-015:
`bash submit_blog_remaining.sh` submits `run_section4.sh` on an H200 node (the
six "H200: pending" parts of Section 4) and `run_headtohead.sh` on an L40S (the
comparison table's four "Pending" cells and new `contribs_bw` totals);
`h200` or `l40s` as the argument submits one (with site.env's `SBATCH_H200` /
`SBATCH_L40S`, `SBATCH_SITE_ARGS`, and logs in `LOG_DIR`). Then
`python summarize_section4.py <OPTIMIZATION_OUT>` and
`python summarize_rerun.py <OPTIMIZATION_OUT>` print the numbers.
