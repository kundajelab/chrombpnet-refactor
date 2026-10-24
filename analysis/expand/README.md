# Profile-only bias, Pearson/JSD model selection and distillation

Three opt-in PyTorch changes (CHANGELOG, top entry), evaluated against the
default models of the engines sweep (`../engine/`), on its 4 datasets x 5
folds:

| run | what it is |
|---|---|
| `old` | the engines sweep's `pytorch` runs: the default pipeline |
| `new` | `bias pipeline --bias-profile-only`, then `pipeline --model-selection pearsonr-jsd` |
| `student-lofo` | `distill` of a dataset's five `old` fold models, one student per fold k, trained on fold k's training chromosomes |
| `student-final` | `distill` of the same five, one student per dataset, on every chromosome |

`new` changes (1) and (2) together and is compared with `old` as a whole;
no run isolates either change. Each student is scored on fold k as teacher k
was: joined with teacher k's `bias_model_scaled.pt`, predicted on teacher
k's test peaks, DeepSHAP on teacher k's 30K subsample with the same seed,
and marginal footprints on the same nonpeaks.

## Reading the results

- **Noise.** `old` and `new` also differ in PyTorch (2.5.1 then, 2.14 now),
  GPU types and the machines' load. For scale: across the engines sweep's
  three engines -- the same pipeline, three frameworks -- the mean counts r
  spans 0.7649-0.7666 and the mean 1 - JSD 0.4436-0.4440 in base 2
  (0.5368-0.5371 as that sweep reported it, in base e). A difference smaller
  than that is not evidence either way.
- **JSD units.** The old runs' metrics (written at 352b1d7), the original's
  and the retrained original's report median JSD in base e; the current code
  reports base 2 (eff8b48). Base-2 JSD = base-e JSD / sqrt(ln 2), medians
  included, so `run_summarize.py --jsd-base-e old original
  original_retrained` puts every run on base 2. Without it, 1 - JSD looks
  ~0.09 worse for `new` and the students than for `old`. The two originals'
  DNase JSD stays off even then: they predict chrombpnet 1.x's 0/+1 cut
  convention, every other run `both-half`, a smoother target.
- **One input differs: `new`'s ChromBPNet training negatives.** The old runs'
  code (352b1d7) drew them from the outlier-filtered nonpeaks with a fixed
  `random_state=1`; since 24dee61 the draw takes `--seed` (1234), so `new`
  draws another subset of the same pool (`inputs.tsv` flags
  `filtered.nonpeaks.bed` for every row; in the smoke test 94,333 of
  152,052 lines are shared, the test-chromosome negatives among them).
  Everything else old and new draw uses 1234 in both, and their bigwigs,
  filtered peaks and bias-model regions are identical. `--seed 1` would
  match the negatives but move every other draw, so the runs keep 1234.
  The engines' spread above was measured on identical negatives.
- **Bias-free counts.** With a profile-only bias the accessibility model's
  counts head explains all the counts, including what the bias model's used
  to. So `new`'s `chrombpnet_nobias` counts are not bias-corrected;
  `expand_bias_response` measures whether bias leaks into the profile.
- **LOFO students are mildly optimistic.** Student k never sees fold k's
  test chromosomes, but four of its five teachers trained on them, and their
  targets elsewhere carry what they learned there.
- **The final student has seen every test chromosome**, so its test scores
  are not held out. Its rows show what one model trained on everything
  gives on each fold's test set, not how it generalizes.
- **Inference cost.** A student has the teachers' architecture, so it costs
  what one fold model costs, a fifth of predicting with all five.

## Steps

On Sherlock, with the wrappers in
`sherlock/runs/chromatlas/chrombpnet-refactor/expand/` (they set
`PYTHONPATH` to this checkout and run in the Ubuntu container):

```bash
./submit_all.sh sherlock --container   # new (20), student-lofo (20), student-final (4, after lofo),
                                       # student-final scored per fold (20, after final)
sbatch run_summarize.sh                # after all of them finish
```

What they run, per dataset and fold (`<run>` is the output directory):

```bash
# new: the engines sweep's commands (../engine/README.md, step 2) with
chrombpnet bias pipeline ${COMMON} -b <0.5|0.8> -fp <ENCID> --bias-profile-only -o <run>/bias_model
chrombpnet pipeline ${COMMON} -b <run>/bias_model/models/<ENCID>_bias.pt \
    --model-selection pearsonr-jsd --ignore-bias-flag -o <run>/model

# student-lofo, fold k (T_k = the old run of fold k):
chrombpnet distill -t <T_0..T_4>/model/models/chrombpnet_nobias.pt -g <genome.fa> \
    -bw T_k/model/auxiliary/data_unstranded.bw -p T_k/model/auxiliary/filtered.peaks.bed \
    -n T_k/model/auxiliary/filtered.nonpeaks.bed -fl <fold_k.json> -d <ASSAY> \
    -op <run>/model/models/chrombpnet_nobias
# student-final: the same on fold 0's inputs, with --all-chromosomes and
# --epochs = the median over the LOFO students of the epoch kept, plus one

# scoring each student on fold k:
python legacy/helpers/reformat_chrombpnet_h5.py -cnb <student>.pt \
    -bm T_k/model/models/bias_model_scaled.pt -o <run>/model/models   # -> chrombpnet.pt
chrombpnet predict -m <run>/model/models/chrombpnet.pt -r T_k/model/auxiliary/filtered.peaks.bed \
    -n None -bw T_k/model/auxiliary/data_unstranded.bw -fl <fold_k.json> ... \
    -op <run>/model/evaluation/chrombpnet
chrombpnet contribs_bw -m <student>.pt -pc counts profile \
    -r T_k/model/auxiliary/interpret_subsample/chrombpnet_nobias.interpreted_regions.bed ... \
    -op <run>/model/auxiliary/interpret_subsample/chrombpnet_nobias
chrombpnet footprints -m <student>.pt -r <the fold's nonpeaks> -fl <fold_k.json> \
    -pwm_f T_k/model/auxiliary/motif_to_pwm.tsv ... -op <run>/model/evaluation/chrombpnet_nobias
```

Collating and plotting:

```bash
python ../engine/run_summarize.py --root <engines root> --ids <ids.tsv> \
    --biosamples ../engine/biosamples.tsv --out-dir <out> --profile-stopgrad off \
    --runs old=<engines root>/pytorch new=<root>/new \
           student-lofo=<root>/distill/lofo student-final=<root>/distill/final \
    --versus student-lofo=old student-final=old \
    --jsd-base-e old original original_retrained
python ../engine/plot_summarize.py --dir <out> --reference old --noun model
python plot_expand.py --dir <out>
```

`plot_summarize.py` draws the engines sweep's figures for these runs:
`summary_scores`, `summary_agreement` (vs the original, and each student vs
its teacher), `summary_contributions_{pearson,cosine}` and `summary_runtime`.
`plot_expand.py` adds:

| figure | shows |
|---|---|
| `expand_epochs` | epochs trained, and the epoch kept |
| `expand_bias` | the bias model's profile 1 - JSD on nonpeaks and peaks; its counts r where it has a counts head |
| `expand_bias_response` | the bias-free model's largest marginal-footprint response to the bias motifs |
| `expand_contributions_versus_{pearson,cosine}` | each student's DeepSHAP against its teacher's, on the same regions and references |
