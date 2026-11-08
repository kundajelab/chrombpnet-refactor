# Optimization benchmark

What `--predict-fk` -- the fused Triton forward -- is worth, measured on the
five folds trained by
`../run/run_chrombpnet_refactor_benchmark_260911.sh` -- their own `.pt`
checkpoints and their own peak regions. Real models, real data, nothing
synthetic.

```bash
sbatch --constraint="GPU_SKU:A100" run_chrombpnet_optimization_benchmark.sh sherlock
sbatch --constraint="GPU_SKU:H100" run_chrombpnet_optimization_benchmark.sh sherlock
sbatch --constraint="GPU_SKU:H200" run_chrombpnet_optimization_benchmark.sh sherlock
./summarize.sh            # once the array finishes
```

A `--constraint` on the sbatch line overrides the directive in the script.
`--array=0-4` runs one fold per task.

**Find the feature name first — do not guess it:**

```bash
./list_gpu_skus.sh                 # GPU_SKU values present in <partitions>
./list_gpu_skus.sh gpu,owners      # or any other partition list
```

Sherlock's names are more specific than the bare model (expect something like
`GPU_SKU:A100_SXM4`, not `GPU_SKU:A100`), and they differ between partitions.
A constraint matching no node does **not** fail the submission: Slurm accepts
the job and leaves it pending indefinitely with reason `BadConstraints`, which
reads like an ordinary queue wait. After submitting, check:

```bash
squeue -u $USER      # NODELIST(REASON) showing (BadConstraints) means no match
```

`run_chrombpnet_optimization_benchmark.sh CLUSTER [FOLD_NUM] [N_REGIONS] [PHASES]`

| argument | default | notes |
|---|---|---|
| `CLUSTER` | `sherlock` | `local` skips the module load |
| `FOLD_NUM` | `$SLURM_ARRAY_TASK_ID`, else 0 | the array supplies it on the cluster |
| `N_REGIONS` | `20000` | deterministically subsampled from the fold's peaks; `all` uses every peak (~270k) |
| `ARMS` | `default,fk` | `default` is how the package now runs; `fk` adds `--predict-fk` |

`N_CONTRIB` (env, default 2000) sets the smaller subsample for `contribs_bw`,
which is far slower per region because it runs DeepLIFT rather than a forward.

**Environment.** `CONDA_ENV` defaults to `chrombpnet-pytorch` (torch
2.14.1+cu126), which needs glibc 2.28 or newer. Some of Sherlock's GPU nodes
(the L40S ones) run glibc 2.17; there the script stops before activating the
env and says so. Resubmit with `--exclude=<node>`, or set
`CONDA_ENV=chrombpnet-pytorch-sherlock` (torch 2.5.1+cu118, built on
Sherlock, the env of the earlier runs). `GLIBC_FLOOR` (default 2.28) moves the
check. The same applies to `PT_ENV` in the pred_bw re-run scripts below.

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
chromatlas/test/chrombpnet-refactor-optimization/analysis/<run tag>/fold_N/
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

- The Triton cache is shared at `.triton-cache/` rather than `$HOME`, so only
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
CUDA 12.6 or newer, so it reports no devices on an older one
(`chrombpnet-pytorch-sherlock`, `torch 2.5.1+cu118`, needs only CUDA 11.8).
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
./list_gpu_skus.sh
sbatch --constraint="GPU_SKU:<H200 name>" run_pred_bw_rerun.sh sherlock
PT_ENV=chrombpnet-pytorch-sherlock \
  sbatch --constraint="GPU_SKU:<L40S name>" run_pred_bw_rerun.sh sherlock   # glibc 2.17
python summarize_rerun.py      # medians per arm, speedups, and medians by position
```

Results: `chromatlas/test/chrombpnet-refactor-optimization/analysis-rerun/<gpu>/fold_N/timings.tsv`.
The refactor arms run in `PT_ENV`: `chrombpnet-pytorch` (torch 2.14.1) by
default, or `chrombpnet-pytorch-sherlock` (torch 2.5.1, built on Sherlock) on
the glibc 2.17 L40S nodes, where the default stops with that hint. Timings
from before 2026-10-21 used `chrombpnet-pytorch-sherlock` on every card. No
container is needed.

### Kit changes with the fused kernels (`run_pred_bw_kit_fused.sh`)

A copy of `run_pred_bw_rerun.sh` with two more arms from `code-693d162/`:
`kitfk` (`--optimize 1,2`: the kit's changes with its Triton forward, without
this package's own) and `fkonly` (`--optimize 2`: the Triton forward alone),
timed with `none`, `kit`, `both` and `fk` in the same job. It writes under its
own run tag (`<gpu>_<arch>_kit-fused`), so it never clears the other job's
results; `summarize_rerun.py` reports each run tag separately.

```bash
sbatch --constraint="GPU_SKU:L40S" run_pred_bw_kit_fused.sh sherlock
ARMS=original,none,kit,ours,both,main,kitfk,fkonly,fk \
  sbatch -p owners --constraint="GPU_SKU:H200_SXM5" run_pred_bw_kit_fused.sh sherlock
```

On Sherlock `PT_ENV` defaults to `chrombpnet-pytorch-sherlock` (torch 2.5.1):
the L40S nodes and at least some H200_SXM5 nodes (<node>, <node>) run
glibc 2.17, where `chrombpnet-pytorch` cannot. The H200_SXM5 nodes are in
`owners` only; with the scripts' `<partitions>` the job sits in
`BadConstraints`, so pass `-p owners`.

Both scripts drop every CUDA and cuDNN directory the job inherits from the
submitting shell before loading their modules. On 2026-10-21 the kit env's
torch reported cuDNN 9.4 on an H200 node, not its own 9.1: cuDNN 9 finds its
sub-libraries on `LD_LIBRARY_PATH` before its own directory, so another
cuDNN 9 there gets mixed in (reproduced on the lab CPU host). The kit's `exact` mode
then died loading cuDNN. `gpu_check.py` now refuses that mix before anything
is timed, and writes the loaded libraries and `LD_LIBRARY_PATH` to
`gpu_check_*.log`. `run_kit_modes.sh` also loads `gcc/14.2.0`: `exact` mode's
kernels are not in the kit's shipped Triton cache, and Triton compiles their
launcher with the node's gcc 4.8, which has no `stdatomic.h`.

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
sbatch --constraint="GPU_SKU:L40S" run_kit_modes.sh sherlock
sbatch -p owners --constraint="GPU_SKU:H200_SXM5" run_kit_modes.sh sherlock
```

### On a lab GPU host instead of Sherlock

`run_pred_bw_kit_fused.sh` and `run_kit_modes.sh` also run as plain bash
scripts (`_common.sh` holds what differs). Without a Slurm job they take
`local` themselves, and:

- run on the first GPU in `CUDA_VISIBLE_DEVICES` (0 if unset), made the only
  visible one, and refuse it if anything already runs there
  (`ALLOW_BUSY_GPU=1` overrides);
- run the five folds one after another (`run_pred_bw_kit_fused.sh local 2`, or
  `0,3`, for some);
- give PyTorch its own CUDA wheels by dropping `~/.bashrc`'s CUDA 12 / cuDNN 9
  from `LD_LIBRARY_PATH`, and TensorFlow 2.8 `/usr/local`'s CUDA 11.2 and
  cuDNN 8.1 (`TF_CUDA_DIRS=<dir>:<dir>` names others);
- check first that each env's framework reaches the GPU, because TensorFlow 2.8
  without its CUDA libraries runs on the CPU without an error;
- add the host to the run tag (`<gpu>_<arch>_kit-fused_<host>`), so a local run
  never replaces a Sherlock job's results, and keep a Triton cache per host.

```bash
cd /path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/optimization
nvidia-smi                                   # pick an idle GPU
CUDA_VISIBLE_DEVICES=<idle GPU> nohup bash -c \
  'ARMS=original,none,kit,ours,both,main,kitfk,fkonly,fk bash run_pred_bw_kit_fused.sh; bash run_kit_modes.sh' \
  > /path/to/workdir/sherlock/logs/optimization.$(hostname -s).log 2>&1 &
python summarize_rerun.py                    # once both have finished
```

The two scripts run one after the other, not side by side: a lab host's CPUs
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
sbatch --constraint="GPU_SKU:L40S" run_kit_modes.sh sherlock
sbatch -p owners --constraint="GPU_SKU:H200_SXM5" run_kit_modes.sh sherlock
python summarize_kit_levers.py
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
sbatch --constraint="GPU_SKU:L40S" run_section4.sh sherlock
sbatch -p owners --constraint="GPU_SKU:H200_SXM5" run_section4.sh sherlock
python summarize_section4.py
```

### Cherimoya (`run_cherimoya.sh`)

Cherimoya's own commands on the comparison table's rows: `evaluate` (the
20,000 peaks, metrics against the observed bigWig), `attribute` (profile head,
2,000 peaks, 2,114 bp, 20 shuffles), `marginalize` (the fold's 5 motifs over
its test non-peaks), with CATv1's ENCSR868FGK models (same folds as
ChromBPNet's), inputs from `cherimoya_inputs.py`. The `cherimoya` env (torch
2.13, Triton 3.7, as cherimoya's uv.lock) needs glibc >= 2.28, which no build
of torch >= 2.9 avoids, so on Sherlock every command runs in the Ubuntu 22.04
container (`software/containers/ubuntu-22.04.sif`). Each task compiles into a
kernel cache of its own (per run tag and fold) and warms the compiled commands
(`evaluate`, `marginalize`) on the full input: with one cache shared by the
array's tasks, 5 of 45 timed runs on the L40S took 10-15x longer while other
tasks were compiling into it (the medians held: 15 s, 161 s, 66 s).

```bash
sbatch --constraint="GPU_SKU:L40S" run_cherimoya.sh sherlock
sbatch -p owners --constraint="GPU_SKU:H200_SXM5" run_cherimoya.sh sherlock
python summarize_rerun.py        # run tags *_cherimoya
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
sbatch --constraint="GPU_SKU:L40S" run_headtohead.sh sherlock
python summarize_rerun.py        # run tag *_headtohead: orig_/ref_ predict, attr, contribs
```

### The blog's last values (`submit_blog_remaining.sh`)

As of 2026-10-30 two jobs fill everything still open in blog 2026-015:
`bash submit_blog_remaining.sh` submits `run_section4.sh` on an H200 node (the
six "H200: pending" parts of Section 4) and `run_headtohead.sh` on an L40S (the
comparison table's four "Pending" cells and new `contribs_bw` totals);
`h200` or `l40s` as the argument submits one. Then `python summarize_section4.py`
and `python summarize_rerun.py` print the numbers.
