# Bias model counts: blog 2026-015 figure

The figure under "Profile-only bias model" in blog 2026-015
(`content/blogs/2026-015/chrombpnet_bias_counts.png`). Data live in
`/path/to/workdir/chromatlas/test/bias-counts/`. Run everything with the
`chrombpnet-pytorch` env's python (h5py, pyBigWig, pandas, matplotlib).

| Step | Script | Output (in the data dir) |
|---|---|---|
| Which bias model each model and fold uses | `identify_bias_models.py bias_models.tsv` | `bias_models.tsv` |
| Panel 1: the main bias models' fold-1 test-set counts | `bias_test_counts.py fig1` | `fig1/*_fold1_test_counts.tsv` |
| Panel 2: each model's total predicted counts, from bigWig headers | `bigwig_totals.py prediction_totals.tsv` | `prediction_totals.tsv` |
| Panel 2: totals over shared positions where headers can't be compared | `matched_totals.py prediction_totals.tsv matched_totals.tsv stream` | `matched_totals.tsv` |
| The figure | `make_figure.py . OUT.png` | the PNG; prints the numbers it shows |
| Check: header totals against sums over every base | `check_exact.py FILE.bigWig [URL]` | – |

## How the bias models were identified

The training logs name the bias model's directory inconsistently (several users' paths, a
`chrombppnet` typo in folds 1-4, `chrombpnet_model_feb15` batches), so models are grouped by
weights. A scaled bias model differs from the pretrained one it came from only in the counts
head's constant (checked: convolutions, profile head and counts kernel are identical), so
hashing everything except the counts head groups models by source. Sources matched in
`chromatin-atlas-2022/reference/` and `chrombpnet/folds/*/bias_model/bias.h5`:

| Bias model | Assay | Models (all five folds) | Source |
|---|---|---|---|
| ENCSR880CUB (prefrontal cortex) | DNase | 389 | `reference/ENCSR880CUB_bias_0.9.h5`, `reference/ENCSR880CUB/fold_N/bias.h5` |
| ENCSR283TME (naive CD8+ T cells) | DNase | 348 | `reference/ENCSR283TME_bias_0.9.h5`, `reference/ENCSR283TME/fold_N/bias.h5` |
| K562 ATAC (merged; = ENCSR868FGK's bigWig) | ATAC | 299 + others in some folds (315 in fold 1) | `reference/atac_bias_models/fold_N/bias.h5` (folds 1-4; fold 0's source is not on Oak) |
| HepG2 DNase PE | DNase | 99 | `chrombpnet/folds/DNASE/HEPG2/*_fold_N_*` (folds 1-4) |
| ENCSR146KFX (astrocyte) | DNase | 28 | `reference/ENCSR146KFX_bias_0.9.h5` (fold 0) |

About 290 other models each have their own bias model.

## Panel 1

Each bias model's predictions on its own training experiment's fold-1 test peaks and non-peaks
were saved when that experiment's ChromBPNet model was trained (`bias_predictions.h5` in
`chrombppnet_model_encsr880cub_bias_fold_1`, `chrombppnet_model_encsr283tme_bias_fold_1`,
`ATAC/ENCSR868FGK/chrombpnet_model_feb15_fold_1`). Observed counts are the experiment's bigWig
over the same 1,000 bp, as log(1 + counts); Pearson r reproduces each run's `bias_metrics.json`
to four decimals. The pretraining runs themselves (on Sherlock scratch) are gone, and those on
Oak saved metrics but no predictions.

## Panel 2

The portal's "predicted signal profile" (with bias) and "bias-corrected predicted signal
profile" (without) bigWigs, the fold means of each model's predictions over its selected peak
regions (`model_enffs_download.tsv`). A bigWig's header holds a total summary computed from
the full-resolution data. It matches a sum over every base to about 1e-13, locally and over
HTTP, so most totals take one 512-byte range request. Two kinds of pairs can't be compared
from their headers, and `matched_totals.py` downloads both files, sums the positions where
both hold a finite value, and deletes them:
- 184 models whose with-bias file covers fewer positions than the no-bias one (median 94%,
  lowest 73%, spread across every chromosome);
- 9 models whose files hold NaN values (NaN header totals).

The 9 models with NaN values are left out: their files also hold overflowed values, so their
totals over finite positions reach 1e40 to 1e42, against 1e6 to 1e9 for every other model
(ENCSR062DXC, ENCSR148MFM, ENCSR167SYF, ENCSR268QCH, ENCSR383RDZ, ENCSR493VDS, ENCSR502NDK,
ENCSR735BIM, ENCSR861NUH). In the 184 coverage-mismatched models, the with-bias positions are
always a subset of the no-bias ones; matching raises their bias share by about 2 points.

Result (1,502 models): with - without = 6.9 ± 7.2 million counts (mean ± SD), 13.5 ± 8.5% of
the with-bias total, median 11.5% (DNase 10.6%, ATAC 13.7%).

ENCSR842KCP is left out of the statistics. Its with-bias total (9.3M) is below its no-bias
total (106.5M) over identical positions, which can't happen: the full model's counts are the
no-bias and bias models' counts added, region by region. The portal labels match the submitted
filenames, so if the files are swapped, that happened when they were generated.
