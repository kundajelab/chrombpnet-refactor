# ChromBPNet across engines: PyTorch, TensorFlow, JAX

Does the same pipeline give equally good models under each deep-learning
engine, and how far are they from the original ChromBPNet? Every engine
trains a bias model and a ChromBPNet model on the same inputs, and each is
compared with the original (kundajelab/chrombpnet) on:

- **test performance**: counts Pearson r and 1 - profile JSD on test peaks;
- **agreement**: each model's test predictions against the original
  model's;
- **contribution scores**: DeepSHAP on a 30K peak subsample against the
  original model's genome-wide scores, read against two references: the
  original's DeepSHAP rerun on the same model with another seed (the
  ceiling), and the original retrained from scratch on the same inputs
  (another model from the same code).

The sweep behind the published figures was 4 datasets x 5 folds x 3 engines
(ENCSR868FGK ATAC K562, ENCSR000EOT DNase K562, ENCSR163PKT DNase stomach,
ENCSR337UIU ATAC stomach), with PyTorch 2.5.1 and the TensorFlow engine on
TF 2.8; the current environments (`envs/`) are newer.

## Scripts

| script | what it does |
|---|---|
| `run_contribution.py` | rescore a finished run's 30K subsample with DeepSHAP, e.g. the profile head with the original's weighting |
| `run_deepshap_seeds.py` | the original's own DeepSHAP on an original model, once per seed (original package env) |
| `run_original_retrain.py` | one step of the original package (bias train, train, predict), from its checkout (original package env) |
| `run_summarize.py` | collate the run tree into `results.tsv`, `agreement.tsv`, `contributions.tsv`, `inputs.tsv`, `timings.tsv`, `epochs.tsv`, `bias_response.tsv`, `runs.tsv` |
| `plot_summarize.py` | draw `summary_scores`, `summary_agreement`, `summary_contributions_{pearson,cosine}`, `summary_runtime` (.png, .pdf, .tsv) |

Both default to the engines (`<root>/pytorch`, `tensorflow`, `jax`).
`run_summarize.py --runs LABEL=DIR ...` collates any labelled set of run
trees instead, and `--versus LABEL=REFERENCE` also compares two of them
directly; `plot_summarize.py` then draws those runs (`../expand/`).

Each takes `--help`; their docstrings describe the outputs in detail.

## Inputs

- Per dataset: a BAM and a peak file; per fold: the fold JSON. Plus the
  genome, chrom sizes and a blacklist (hg38 here).
- The original models of each dataset and fold, as released by ENCODE: the
  no-bias model (`.h5`), its test-set outputs (`chrombpnet_metrics.json`,
  `chrombpnet_predictions.h5`, in `TENSORFLOW_DIR`) and its genome-wide
  DeepSHAP scores (`<ENCID>.counts_scores.h5`, `<ENCID>.profile_scores.h5`,
  `<ENCID>.interpreted_regions.bed.gz`, in `DEEPLIFT_COUNTS_DIR` /
  `DEEPLIFT_PROFILE_DIR`, or one `DEEPLIFT_DIR`).
- A dataset x fold table, one row per dataset and fold (`ids.example.tsv`):
  `ASSAY` (ATAC or DNASE), `ENCID`, `BAM_PATH`, `PEAKS_BED_PATH`, `FOLD_NUM`,
  `TENSORFLOW_DIR`, `DEEPLIFT_COUNTS_DIR`, `DEEPLIFT_PROFILE_DIR`.
- `biosamples.tsv`: optional labels for the plots (ENCID -> biosample).
- A checkout of the original, kundajelab/chrombpnet at 09938fd, installed in
  its own environment (Python 3.8, TensorFlow 2.8, with its pinned
  requirements), for the original's DeepSHAP and the retrained original.

## Steps

Everything lands in one run tree, `<root>`, which `run_summarize.py` reads.
Below, `<run>` is `<root>/<engine>/<ASSAY>/<ENCID>/fold_N`, `-b` is the bias
threshold factor (0.5 for ATAC, 0.8 for DNase) and `<sfx>` the engine's
checkpoint suffix (`.pt`, `.h5`, `.keras`).

**1. Nonpeaks**, once per dataset and fold, shared by every engine (any env):

```bash
chrombpnet prep nonpeaks -g <genome.fa> -c <chrom.sizes> -p <peaks> -fl <fold_N.json> \
    -br <blacklist.bed.gz> -o <root>/preprocess/<ASSAY>/<ENCID>/fold_N/nonpeaks
```

**2. Each engine** (`--engine pytorch|tensorflow|jax`, in the environment from
`envs/chrombpnet-<engine>.yml`, on a GPU). Everything else is each command's
default -- `--seed 1234`, `--dnase-convention both-half` on DNase,
`--data-aug augment`:

```bash
NONPEAKS=<root>/preprocess/<ASSAY>/<ENCID>/fold_N/nonpeaks_negatives.bed
COMMON="-ibam <bam> -d <ASSAY> -g <genome.fa> -c <chrom.sizes> -p <peaks> -n ${NONPEAKS} \
        -fl <fold_N.json> --engine <engine> --device cuda"
chrombpnet bias pipeline ${COMMON} -b <0.5|0.8> -fp <ENCID> -o <run>/bias_model
chrombpnet pipeline ${COMMON} -b <run>/bias_model/models/<ENCID>_bias<sfx> \
    --ignore-bias-flag -o <run>/model
```

`--ignore-bias-flag` lets runs whose bias model fails the quality check go on
(the stomach samples' do, under every engine). For `summary_runtime`, record
each run's wall times in `<run>/timings.tsv` (`step`, `seconds`, `start`,
`end`, one row per pipeline), its GPU in `<run>/provenance.txt` (a line
`gpu: <name>`), and write `<run>/COMPLETED` when both pipelines are done;
the other tables do not need them.

**3. Profile scores with the original's weighting.** The original's DeepSHAP
weights the profile logits by the midpoint of the input's and the
reference's softmax (its shap fork leaves the stop-gradient handler
commented out). `run_summarize.py`'s default, `--profile-stopgrad off`, reads
scores computed that way, under names of their own:

```bash
python run_contribution.py --run-dir <run> --heads profile --profile-weighting midpoint --device cuda
```

(in that engine's environment). chrombpnet's own default is now the
midpoint weighting, so for new runs these should equal the pipeline's
`chrombpnet_nobias.profile_scores.h5`.

**4. The original's DeepSHAP across seeds** (the ceiling), in the original's
environment, on the same 30K regions the engines scored:

```bash
python run_deepshap_seeds.py --original-repo <chrombpnet@09938fd> \
    --model <the original model.chrombpnet_nobias.fold_N.<ENCID>.h5> \
    --regions-source <root>/pytorch/<ASSAY>/<ENCID>/fold_N/model/auxiliary/interpret_subsample/chrombpnet_nobias.interpreted_regions.bed \
    --genome <genome.fa> --out <root>/original/<ASSAY>/<ENCID>/fold_N --seeds 1 2 3
```

**5. The original, retrained** on the same inputs, in the original's
environment (its `train` writes a PDF report through WeasyPrint, which needs
the system's Pango and Cairo). `--ignore-bias-flag` turns the original's
bias-model assert into the same warning as step 2:

```bash
OUT=<root>/original_retrained/<ASSAY>/<ENCID>/fold_N
R="python run_original_retrain.py --original-repo <chrombpnet@09938fd>"
ORIG="-ibam <bam> -d <ASSAY> -g <genome.fa> -c <chrom.sizes> -p <peaks> -n ${NONPEAKS} -fl <fold_N.json>"
${R} chrombpnet bias train ${ORIG} -b <0.5|0.8> -fp <ENCID> -o ${OUT}/bias_model
${R} --ignore-bias-flag chrombpnet train ${ORIG} -b ${OUT}/bias_model/models/<ENCID>_bias.h5 -o ${OUT}/model
${R} predict -g <genome.fa> -b ${OUT}/model/auxiliary/data_unstranded.bw \
    -p ${OUT}/model/auxiliary/filtered.peaks.bed -n None -fl <fold_N.json> \
    -m ${OUT}/model/models/chrombpnet.h5 -o ${OUT}/model/evaluation/chrombpnet -bs 64
python run_deepshap_seeds.py --original-repo <chrombpnet@09938fd> \
    --model ${OUT}/model/models/chrombpnet_nobias.h5 --regions-source <the same 30K regions> \
    --genome <genome.fa> --out ${OUT} --seeds 1
```

Seed 1 matches step 4's first seed, so the retrained model and the released
one are scored against the same references.

**6. Collate and plot** (any environment with numpy, scipy, h5py, pandas and
matplotlib):

```bash
python run_summarize.py --root <root> --ids <ids.tsv> --biosamples biosamples.tsv \
    --out-dir <root>/analysis/profile-stopgrad-off
python plot_summarize.py --dir <root>/analysis/profile-stopgrad-off
```

Reading the original scores costs ~11 GB per dataset and fold, so the first
`run_summarize.py` is slow; each comparison is cached in
`<out-dir>/.contributions_cache` and later runs redo only what changed.
`--profile-stopgrad on` reads the pipeline's own profile scores instead;
`--no-contributions` skips the DeepSHAP comparison.

## Reading the results

The original is a separately trained model, so no agreement reaches 1;
what matters is how the engines compare with one another and with the
retrained original. On DNase the original used the 0/+1 cut convention
(`--dnase-convention right-deprecated`) and the engines `both-half`, so their
DNase profile scores are on different targets. Two quirks of the released
originals: some prediction files store the centre shifted by half the input
window (`run_summarize.py` uses whichever offset matches the most regions and
records it as `center_offset`), and a missing or truncated prediction file
leaves that dataset and fold without agreement rows.
