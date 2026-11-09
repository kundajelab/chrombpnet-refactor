# Blog 2026-015: the scripts behind every figure and table

The jobs, benchmarks and plotting steps that produced the figures and tables in the
genomics x AI blog post *ChromBPNet-refactor: An optimized, reproducible implementation in
PyTorch/TensorFlow/JAX with bug fixes* (2026-015), as they were run. The analysis code
they call lives beside this folder: `analysis/dnase-cutsite/`, `analysis/engine/` and
`analysis/expand/` (each with its own README). `engines/` and `dnase-rc/` link to those
copies, so the jobs always run the current analysis code.

| Folder | What it ran |
|---|---|
| `dnase-rc/` | the DNase-seq cut-site convention sweep (Section 2) |
| `optimization/` | `pred_bw` benchmarks: the optimization kit, chrombpnet-refactor, the original, cherimoya (Sections 3-4, the comparison table) |
| `engines/` | the PyTorch / TensorFlow / JAX training and DeepLIFT/DeepSHAP sweep (Section 5) |
| `bias-counts/` | the bias models' counts across ENCODE models (Section 6, profile-only bias) |
| `expand/` | profile-only bias + Pearson r/JSD checkpoint selection, and distillation (Section 6) |
| `blog-figures/` | rebuilds the blog's summary figures from the saved tables |

## Running them

Every path, environment and cluster setting comes from one file, `site.env`:

```bash
cp site.env.example site.env   # then fill in what the scripts you run need
```

- **Settings.** `site.env.example` lists each value and the folders that read it; a script
  stops and names any value it needs that is left empty. A value set in the environment wins
  over the file's (`OUT_ROOT=<dir> ./run_section4.sh local`), and `BLOG_SITE_ENV=<file>`
  points at a site.env kept elsewhere. Python scripts take their paths as arguments.
- **Slurm.** Job scripts take `slurm` or `local` first and are submitted from their own
  folder. Their `#SBATCH` lines hold only resources: the partition, excluded nodes and GPU
  model come from `SBATCH_SITE_ARGS` and `SBATCH_L40S` / `SBATCH_H200`, which the submit
  scripts add (with a bare sbatch, pass them: `sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S}
  run_sections34.sh slurm`). Module loads and library paths go in site.env's optional hooks.
- **Environments.** chrombpnet-refactor's `envs/` (`chrombpnet-pytorch`, and the glibc 2.17
  builds `chrombpnet-pytorch-glibc` / `-tensorflow-glibc` for older cluster nodes); the
  original package in a `chrombpnet` env (TensorFlow 2.8; CUDA 11.2 / cuDNN 8.1); the
  optimization kit in `chrombpnet-kit` (`optimization/setup_kit_env.sh`); cherimoya 0.3.1
  in an Ubuntu 22.04 container (`CONTAINER_IMAGE`; `optimization/run_cherimoya.sh`).
- **Code snapshots.** Two benchmark arms run an older chrombpnet-refactor from an export of
  one commit; create them first:
  ```bash
  git archive 693d162 | tar -x -C optimization/code-693d162   # run_pred_bw_kit_fused.sh (the --optimize phases)
  git archive d407ca8 | tar -x -C optimization/code-d407ca8   # run_sections34.sh refactor4_prev (before streaming)
  ```
- **The "updated" arm** (Section 6) was trained with `--model-selection pearsonr-jsd`, removed
  in 21fef5b; `expand/` runs the checkout of `3ddb713` that `CHROMBPNET_EXPAND_REPO` names.

The blog's runs used a Slurm cluster's L40S and H200 nodes and two lab GPU hosts.

## Figure and table map

Each step's outputs are the next step's inputs; the last command of each row writes what the
blog shows.

| Blog item | Run (GPU / cluster) | Summarize and plot (CPU, from saved results) |
|---|---|---|
| Section 2: `chrombpnet_dnase.png` | `dnase-rc/submit_all.sh` (`run_retrain_full.sh` per convention arm over all 20 rows, training through `run_train_full.sh`), then `run_compare.sh` (cross-evaluation) | `dnase-rc/summarize.sh` (writes `<run tree>/analysis/results.tsv`), then `dnase-rc/plot_summarize.sh --results <run tree>/analysis/results.tsv --out <dir>/summary_scores --fill-original ENCSR868FGK:2`: `summary_scores.png` |
| Section 2: `chrombpnet_dnase_cross_original_test.png` | as above | `summary_scores_aug_on_standard_tests_original.png` |
| Section 2: `chrombpnet_dnase_cross_new_test.png` | as above | `summary_scores_standard_on_aug_tests_original.png` |
| Section 3 table | `optimization/submit_l40s_kit_fast_k1.sh` (`run_sections34.sh`, arms stock, stages, fast, fast_k1; `kit_hook_l40s_k1fast/` runs the kit's Triton TF32 forward on the L40S, which the kit's own table does not); `malloc_env` row: `ARMS=stock,stockmalloc RUN_KIND=sections34-fast-rerun sbatch ${SBATCH_SITE_ARGS} ${SBATCH_L40S} run_sections34.sh slurm` | `optimization/summarize_sections34.py` (per-stage pairing: `summarize_kit_levers.py`, `stock_stage_timer.py`) |
| Section 4 table, rows 1-3, 5, 7, 8 | `optimization/run_section4.sh` (`benchmark_section4.py`) | `optimization/summarize_section4.py` |
| Section 4 table, rows 4, 6, Total | `run_sections34.sh` with its default arms (`RUN_KIND=sections34-fast`), and `ARMS=stock,stockmalloc,fast,refactor4,refactor4_fast RUN_KIND=sections34-fast-rerun` (both ran the gzip-4 arms through `refactor_gzip4.py`; `optimization/submit_gzip4_headtohead.sh` re-times stock against them through `--h5-gzip-level 4 --no-h5-shuffle`) | `optimization/summarize_sections34.py` |
| Comparison table: `pred_bw` row | the Section 4 Total runs | `optimization/summarize_sections34.py` |
| Comparison table: head-to-head and `contribs_bw` rows | `optimization/run_headtohead.sh` (`headtohead.py` switches the writers off; `submit_headtohead_fast.sh` adds `--predict-fused-kernel-fast`), `run_cherimoya.sh` (`cherimoya_inputs.py`) | `optimization/summarize_rerun.py` |
| Section 5: `chrombpnet_engine_performance.png`, `chrombpnet_engine_contribution.png` | `engines/submit_all.sh` (`run_prep_nonpeaks.sh`, `run_train_full.sh` per engine), `run_contribution.sh`, `run_deepshap_seeds.sh`, `run_original_retrain.sh` | `engines/run_summarize.sh`, then `blog-figures/make_blog_figures.sh` |
| Section 6: `chrombpnet_bias_counts.png` | – (reads ENCODE models and bigWigs) | `bias-counts/`, in the order its README gives; `make_figure.py DATA_DIR OUT.png` |
| Section 6: `chrombpnet_refactor_default_updated.png` | `expand/submit_all.sh` (`run_train_new.sh`) | `expand/run_summarize.sh`, then `blog-figures/make_blog_figures.sh` |
| Section 6: `chrombpnet_distill_ensemble_student.png` | `expand/submit_all.sh` (`run_distill.sh`), `submit_shared.sh` (`run_shared_scores.sh`) | `expand/run_compare_shared.sh`, then `blog-figures/make_blog_figures.sh` |

Not produced here: the schematics (`atac.png`, `dnase.png`), Anthropic's release figure
(`fable_mythos.png`), and the two figures motivating Section 2 (`chrombpnet_performance.png`,
`AG_performance.png`), which come from earlier analyses.

## What has been checked

From the saved results, with a site.env holding the original runs' values (2026-11-09):
`summarize_sections34.py`, `summarize_rerun.py`, `summarize_section4.py` and
`summarize_kit_levers.py` print exactly what the original copies print (the tables'
numbers); `blog-figures/make_blog_figures.sh`, `bias-counts/make_figure.py` and
`dnase-rc/plot_summarize.sh` rebuild the blog's figures byte for byte on the same host
(matplotlib output depends on the host's fonts). Every shell script passes `bash -n`, every
Python file parses, each job script stops naming the settings an empty site.env leaves out,
and dry runs resolve the same paths and commands as the original scripts. The GPU and
training jobs were not re-run for this check.
