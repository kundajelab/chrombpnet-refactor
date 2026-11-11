# expand: profile-only bias, Pearson/JSD selection, distillation

Slurm wrappers for a chrombpnet-refactor checkout of branch `expand` at
22d90cb (the last commit with `--model-selection pearsonr-jsd`), compared with
the engines sweep's PyTorch runs. What is compared, and how to read it: that
checkout's `analysis/expand/README.md`.

Paths come from `../site.env` (`../site.env.example`; `BLOG_SITE_ENV=<file>`
points elsewhere): `CHROMBPNET_EXPAND_REPO` (that checkout), `EXPAND_ROOT`
(this comparison's run tree), `ENGINES_ROOT` and `ENGINES_IDS` (the engines
sweep's tree and ids table), and as in `../engines/` `REFERENCE_DIR`, `GENOME`,
`CHROM_SIZES`, `CONDA_ROOT`, `CONTAINER_IMAGE`, `CONTAINER_BINDS`. Each script
stops and names a setting it needs that is empty; an environment variable
wins over site.env's (`CHROMBPNET_DIR=`, `ROOT=` and `IDS_FILE=` replace the
checkout, the run tree and the ids table for one run). The submit scripts
give every job site.env's `SBATCH_SITE_ARGS` and send its logs to `LOG_DIR`
(without one, to this folder); submit from this folder. The cluster argument
is `slurm` or `local` (`sherlock`, which these runs were submitted with,
means `slurm`); `<NAME>` below stands for site.env's value.

| script | does |
|---|---|
| `_common.sh` | sets the checkout (`PYTHONPATH`), `ROOT` and the engines tree, then sources `../engines/_common.sh`; picks the conda env |
| `run_train_new.sh` | array 0-19: `bias pipeline --bias-profile-only`, `pipeline --model-selection pearsonr-jsd` -> `${ROOT}/new/` |
| `run_distill.sh lofo` | array 0-19: student k from the five old fold models, then scored on fold k -> `${ROOT}/distill/lofo/` |
| `run_distill.sh final` | array 0-3 (datasets): one student on every chromosome -> `${ROOT}/distill/final/<ASSAY>/<ENCID>/student/` |
| `run_distill.sh final-score` | array 0-19: that student scored on each fold k -> `${ROOT}/distill/final/<ASSAY>/<ENCID>/fold_k/` |
| `run_summarize.sh` | `run_summarize.py`, `plot_summarize.py`, `plot_expand.py` -> `${ROOT}/analysis/profile-stopgrad-off/` |
| `submit_all.sh` | submits the four arrays (final after lofo, final-score after final) |
| `run_shared_scores.sh` | array 0-43 (4 datasets x 11 models): DeepSHAP and predictions of one model on its dataset's shared peaks -> `${ROOT}/shared/${TEACHER_SET}/` |
| `run_compare_shared.sh` | CPU: the students vs their teachers' five-fold mean on those peaks -> `${ROOT}/analysis/fold-mean-${TEACHER_SET}/` |
| `submit_shared.sh` | ensemble vs distillation for one teacher set: distill (lofo, final; `SCORE=0`) -> shared scores -> comparison |

`TEACHER_SET` (`_common.sh`) picks whose students: `default` (the engines
sweep's fold models, students in `${ROOT}/distill`, the original runs above)
or `updated` (`${ROOT}/new`, students in `${ROOT}/distill-updated`).

```bash
./submit_shared.sh slurm                                       # updated: distill, score, compare
TEACHER_SET=default ONLY="scores compare" ./submit_shared.sh slurm   # the control
```

Under Slurm the GPU scripts run natively, in site.env's `PT_ENV_GLIBC217`
when it is set -- these ran in `chrombpnet-pytorch-sherlock`: PyTorch 2.5.1,
the engines sweep's, since `chrombpnet-pytorch`'s 2.14 needs glibc 2.28 and
Sherlock's nodes have 2.17; its tangermeme, modisco and memelite are
`chrombpnet-pytorch`'s (`envs/chrombpnet-pytorch-sherlock.yml` in the
checkout; its glibc-2.17 wheels are in `../engines/envs/wheels-glibc217/`) --
else in `chrombpnet-pytorch`. `--container`
runs them in the Ubuntu image with `chrombpnet-pytorch` instead, as the
`new/` and `distill/` runs below were. The CPU scripts (`run_summarize.sh`,
`run_compare_shared.sh`) import no torch and run natively in
`chrombpnet-pytorch` either way.

The runs in `new/` and `distill/`:

```bash
./submit_all.sh slurm --container --dry-run   # check
./submit_all.sh slurm --container
sbatch <SBATCH_SITE_ARGS> run_summarize.sh    # when they finish
```

`ROOT` defaults to site.env's `EXPAND_ROOT`, whose `preprocess/` links to the
engines sweep's nonpeaks (`ln -s <ENGINES_ROOT>/preprocess <EXPAND_ROOT>/preprocess`;
`submit_all.sh` says so if it is missing). A quick check on any GPU
host: `ROOT=<that>/smoke_test EXTRA_ARGS="--epochs 1 --interpret-subsample
2000" SLURM_ARRAY_TASK_ID=15 ./run_train_new.sh local --container`, and
likewise `run_distill.sh local lofo` (`EXTRA_ARGS="--epochs 1"`) and
`run_distill.sh local final` (`EPOCHS=1 SCORE_FOLDS=0`).

No job asks for particular GPUs (`GPU_CONSTRAINT`, `DISTILL_GPU_CONSTRAINT`
add a `--constraint`). Resubmitting resumes: a trained student is scored, not
retrained.
