# The DNase reverse-complement experiment

Does reverse-complement augmentation hurt DNase models, and if so which
correction fixes it?

## The claim being tested

DNase I cuts the backbone **between** two bases, so the event is a bond. The
0/+1 shift records it at the bond's right-hand base — that is exactly what
the `+1` on minus reads buys, and it is what makes the profile array
bond-indexed rather than base-indexed.

Reverse-complement augmentation reverses the profile with no compensating
offset. That carries the bond to the right place but leaves it named by the
wrong base, so **every reverse-complemented DNase example is misregistered by
one base**. Half of each training set teaches the model a profile shifted one
base from the other half.

Tn5 leaves a 9 bp stagger. 9 is odd, so ATAC's event is centred **on** a
base, whose mirror under reversal is itself. ATAC is unaffected — which is
what makes it the control here.

`DNase_RC_ChromBPNet_technical_note.pdf` measures the cost on K562
ENCSR000EOT fold 0:

| model | per-window r | pooled r |
|---|---|---|
| current code (RC, no shift) | 0.3560 | 0.5416 |
| **with the +1 fix** | **0.4751** | **0.6395** |
| no RC augmentation at all | 0.4694 | 0.6009 |

Rows 1 and 3 are the striking pair: as things stand, RC augmentation on DNase
is slightly **worse than not augmenting at all**. That result is from one
track, which is why this sweep covers four datasets.

## The geometry is confirmed before any model is trained

Three independent ways, none of which needs a GPU.

1. **The algebra.** `chrombpnet/conventions.py` derives `i -> W-i`.
2. **The shipped reference PWMs.** All nine references in
   `chrombpnet/data/{DNASE,ATAC}.ref.motifs.txt` align with their own reverse
   complement at lag 0 (DNase) and lag −1 (ATAC), each decisive by >0.3 in
   correlation. Pinned by
   `tests/test_conventions.py::test_the_reference_pwms_agree_with_the_resolved_shift`.
3. **Real reads**, via `./strand_pwm_check.py --ids <ids table> --genome <hg38.fa>
   --out <run tree>/analysis/strand_pwm_check` — see that `strand_pwm_check.png`:

   | ENCID | assay | best lag | corr | runner-up | label shift |
   |---|---|---|---|---|---|
   | ENCSR000EOT | DNASE | **+0** | 0.986 | 0.161 | **+1** |
   | ENCSR868FGK | ATAC | **−1** | 0.999 | 0.408 | **+0** |

In the figure, the left panel is the plus-strand cut motif and the middle
panel is the reverse complement of the minus-strand motif; the dashed line
marks the bond immediately left of the deposit base. For DNase the two
motifs sit at the **same** position — both strands name the same bond, so
the array is bond-registered and the label needs +1. For ATAC they are
**offset by one** — the deposit is a base, whose mirror is itself, so no
shift is needed. The right panel is the correlation against lag; in both
cases the winner beats every alternative by a wide margin.

The relation is `label_shift = best_lag + 1` for both assays.

**One trap**, if you re-run this yourself: `auto_shift_detect.get_pwms`
centres its PWMs on the raw 5′ ends (`start` for plus, `end` for minus).
Those are the deposit positions for DNase 0/+1, but not for ATAC +4/−4,
whose deposits are `start+4` and `end−5`. `strand_pwm_check.py` re-registers
onto deposit bases first, using the deltas from
`conventions.shift_delta_pairs`. Skipping that step makes ATAC report lag 0
with correlation 0.999 — confident, and wrong.

## The arms

The arms are named after the values of chrombpnet's `--data-aug` when this
experiment ran, and the names still label the run directories and results.
The package has since split that flag into `--dnase-convention` (where a
DNase cut is recorded; DNASE only) and `--data-aug` (augment | forward-only |
revcomp-only). The scripts translate each arm into the pair
(`arm_flags` in `_args.sh`), exactly:

| arm | chrombpnet flags now |
|---|---|
| `standard` | `--dnase-convention right-deprecated` (ATAC: none) |
| `fr-only` | `--data-aug forward-only` (+ `--dnase-convention right-deprecated` on DNASE) |
| `rc-only` | `--data-aug revcomp-only` (+ `--dnase-convention right-deprecated` on DNASE) |
| `right`, `left`, `both-half`, `both-full` | `--dnase-convention <arm>` |

`right-deprecated` is the old `standard` geometry (0/+1 without the +1
re-registration), kept in the package to reproduce these runs only.

| arm | preprocessing | RC shift | which splits are flipped |
|---|---|---|---|
| `standard` | DNase 0/+1 | 0 | train, 50 % |
| `fr-only` | unchanged | — | none |
| `rc-only` | unchanged | 0 | train, valid **and** test, 100 % |
| `right` | unchanged | **+1** | train, 50 % |
| `left` | DNase −1/0 | **−1** | train, 50 % |
| `both-half` | 0.5 on both flanking bases | 0 | train, 50 % |
| `both-full` | 1.0 on both flanking bases | 0 | train, 50 % |

`standard`, `fr-only` and `rc-only` run on **both assays**. The other four
reshape a bond, which ATAC has not got, so the CLI refuses them for ATAC and
the scripts skip those rows.

### What each arm should show

- **`right` is the fix.** Expected to recover the technical note's numbers.
- **`left` should tie with `right`.** It is the same geometry mirrored. If it
  does not tie, the argument has a hole.
- **`rc-only` should look like `fr-only`, not like `standard`.** Flip
  everything and the misregistration becomes self-consistent — the model just
  learns the mirrored convention. The damage in `standard` comes from
  *mixing* two conventions inside one training set. If `rc-only` instead
  resembles `standard`, something beyond the bond convention is also wrong.
- **`both-half` vs `both-full`** is a free internal consistency check: they
  differ only by a factor of two in total counts, which `counts_loss_weight`
  re-derives per run, so they should agree. Note `both-*` trades the
  registration problem for a 2 bp smear — it is not a pure convention change.
- **ATAC should be flat** across `standard` / `fr-only` / `rc-only`. If it is
  not, the explanation above is wrong.

## The arms verified on real reads

`./check_arms_on_real_reads.sh [ENCID] [CHROM]` builds one chromosome's
bigwig under each arm and checks the relationships between them hold at
genome scale, which the synthetic unit tests cannot show. On 31,250,652 chr1
reads from ENCSR000EOT:

| arm | total | nonzero positions | vs standard |
|---|---:|---:|---:|
| standard | 31,250,652 | 15,029,686 | 1.000000 |
| right | 31,250,652 | 15,029,686 | 1.000000 |
| left | 31,250,652 | 15,029,686 | 1.000000 |
| both-half | 31,250,652 | 27,106,482 | **1.000000** |
| both-full | 62,501,304 | 27,106,482 | **2.000000** |

Total deposits equal the read count exactly. `right` is byte-identical to
`standard` (it changes augmentation, not preprocessing), `left` is
`standard` shifted one base, `both-half` conserves the total to 0.00e+00
relative difference and equals `(left + right) / 2` exactly, and `both-full`
doubles it.

Note the nonzero count for `both-*`: 15.0M to 27.1M. That is the 2 bp smear
made visible, and it is the cost those arms pay for removing the
registration problem structurally.

Outputs land in `arm_check/<ENCID>/` (~800 MB, mostly a regenerable
tagAlign). The script skips any bigwig it has already built, so re-running
is cheap.

## Scope: all four datasets, the stomach rows past the bias guard

The sweep runs **data rows 1-20**: `ENCSR868FGK` (ATAC, K562) and
`ENCSR000EOT` (DNase, K562) in rows 1-10, and the two primary **stomach
tissue** datasets in rows 11-20. The stomach bias models fail ChromBPNet's
own guard in every arm:

```
Bias model pearsonr performance in peaks is: -0.6
ValueError: Bias model has strongly negative peak count correlation
```

The split is exactly cell line vs primary tissue:

| ENCID | assay | biosample | rows | outcome |
|---|---|---|---|---|
| ENCSR868FGK | ATAC | K562 | 1-5 | trains |
| ENCSR000EOT | DNase | K562 | 6-10 | trains |
| ENCSR163PKT | DNase | stomach M54 | 11-15 | trains past the bias guard (`--ignore-bias-flag`) |
| ENCSR337UIU | ATAC | stomach F53 | 16-20 | trains past the bias guard (`--ignore-bias-flag`) |

A bias model whose predicted counts *anti*-correlate with observed peak
counts at r ~ -0.5 is genuinely broken, and it fails before the ChromBPNet
model trains at all — so this is a data/threshold question, not an
augmentation one. `-b/--bias-threshold-factor` (0.5 ATAC / 0.8 DNase here) is
the first thing to try. The runs here instead pass `--ignore-bias-flag`
(`run_train_full.sh`'s default, `IGNORE_BIAS=1`), which turns the guard into a
warning: every row trains its own bias model, the stomach rows included, and
the stomach results are best read with that caveat (below).

`SLURM_ARRAY_TASK_ID` is **0-based**: task `t` runs data row `t+1`, so
`--array=0-9` is rows 1-10. Any task past the ids table's last row exits 0
with a message rather than running, so a wider array range is harmless.

### Why the stomach datasets cannot train a bias model

Investigated; the failure is real and is not a bug, a wrong file, or a
tunable parameter.

**What the guard checks.** `find_bias_hyperparams` sets
`counts_threshold = quantile(peak_counts, 0.01) * bias_threshold_factor`
(0.8 for DNase) and trains the bias model on nonpeaks *below* that threshold.
The bias model must see only sequence bias, never accessibility, so the
threshold has to sit under the weakest peaks. `stages.py:104` then checks
that the resulting model's predicted counts correlate positively with
observed counts in peaks, and refuses at r <= -0.5.

**What is actually different.** Counts over 1 kb windows:

| | K562 | stomach |
|---|---|---|
| peak median | 343 | 236 |
| nonpeak median | 42 | **120** |
| median peak / median nonpeak | **8.2x** | **2.0x** |
| summit / flank enrichment | **7.5x** | **3.0x** |
| peak 1st-%ile / nonpeak median | 1.93x | **0.71x** |
| nonpeaks below the threshold | 76% | **25%** |

The last two rows are the whole story. In stomach the weakest 1% of peaks
have *fewer* reads than a median background region, so the threshold (68)
lands below the nonpeak median (120) and only the bottom quarter of nonpeaks
qualifies. That is selection on the outcome variable: within that truncated
tail, sequence features that predict high counts are systematically absent,
because regions carrying them were excluded for being too high. The bias
model learns an inverted sequence-to-count relation and anti-correlates on
peaks at r ~ -0.5. The guard is doing its job.

The cause is sample quality, not preprocessing: peak signal is 2.2x weaker
and background 1.4x hotter than K562, giving 2.5x worse enrichment. The BAMs
are correct -- ENCODE `output_type: alignments` is the *filtered* product
(`unfiltered alignments` is the raw one), GRCh38, and the two K562 datasets
trained fine from the same file flavour.

**Stringency does not rescue it.** Keeping only the strongest peaks by the
peak caller's own `signalValue`:

| peaks kept | K562 thr/nonpeak med | stomach thr/nonpeak med | stomach retained |
|---|---|---|---|
| 100% | 1.58x | 0.57x | 25% |
| 50% | 2.42x | 0.63x | 28% |
| 25% | 4.60x | 0.79x | 38% |
| 10% | 7.90x | 1.15x | 58% |

Even the top 10% of stomach peaks does not reach what K562 achieves with its
*entire* peak set. Note also how weakly `signalValue` tracks measured counts
there -- going 100% -> 10% moves the count 1st-%ile only 85 -> 171 (2x),
against 81 -> 405 (5x) for K562, which is itself a symptom of a noisy
experiment. The shipped `100K.ranked.subsample.overlap.bed` (top 50%) reaches
only 0.75x.

(Beware the tempting shortcut of ranking peaks by their counts *in the same
bigwig* and then measuring the count percentile of the survivors: that is
circular and manufactures a healthy-looking threshold out of nothing.)

**Conclusion.** No threshold exists that both excludes accessibility and
retains a representative nonpeak set, because the two distributions overlap.
Raising `-b` past ~2.0 would admit nonpeaks more accessible than many peaks,
which stops being a bias model. These two experiments are unsuitable for the
bias-factorized approach as designed; the reverse-complement question is
answerable from the K562 DNase dataset with the K562 ATAC control.

### An alternative not used for the results: a borrowed bias model

The blog's stomach runs trained their own bias models, as above;
`submit_all.sh` does not run this script. `run_train_chrombpnet.sh` covers
data rows 11-20 (array tasks 10-19) and
skips bias training entirely, taking the fold-0 bias model from the K562
dataset of the same assay and the same arm:

```
ENCSR163PKT (DNASE, stomach)  <-  ENCSR000EOT (DNASE, K562) fold 0
ENCSR337UIU (ATAC,  stomach)  <-  ENCSR868FGK (ATAC,  K562) fold 0
```

```bash
sbatch run_train_chrombpnet.sh slurm right
DRY_RUN=1 SLURM_ARRAY_TASK_ID=10 ./run_train_chrombpnet.sh right   # check wiring only
```

A bias model captures enzyme sequence preference, a property of the assay
rather than the biosample, and `chrombpnet pipeline` rescales it to the
target read depth (`bias_model_scaled.pt`). Matching the *arm* matters as
much as matching the assay, because `left` and `both-*` change the bigwig
itself. The donor must therefore already exist for that arm; if it does not,
the script says which array task produces it rather than failing obscurely.

**It does clear the guard.** `chrombpnet pipeline` still runs
`check_bias_model_quality` on the borrowed model against *this* dataset's
peaks, so borrowing bypasses the bias *training* problem, not the check.
Measured directly with `chrombpnet predict`, the K562 fold-0 DNase bias model
scored on ENCSR163PKT peaks gives **counts Pearson r = +0.52**, against a
rejection threshold of -0.5 and the stomach's own bias model at -0.5 to -0.6.

**But read that number carefully.** It is *higher* than the same K562 bias
model on K562 peaks (0.25), which is not a sign of quality -- it is the
low signal-to-noise showing up again. Where peaks are weakly accessible,
more of their count variation is explainable by generic sequence and
mappability effects, which is exactly what a bias model captures. So the
bias-corrected model has proportionally less genuine accessibility left to
learn, and these runs are best read as "does the reverse-complement effect
reproduce in a harder sample", not as a second independent confirmation.

**Two further caveats, both recorded in each run's
`borrowed_bias_model.txt`:**

- Folds 1-4 borrow a **fold-0** bias model, which saw their test
  chromosomes while training. For a bias model (nonpeaks only, no
  locus-specific accessibility) the leakage is mild, and sharing one bias
  model across folds is common practice -- but it is not zero.
- These models are consequently not strictly comparable with the K562 runs,
  which each trained their own bias model.

## Checking and resuming

```bash
./check_runs.sh            # what finished, what failed, and why
./check_runs.sh -v         # every run, not just the incomplete ones
./check_runs.sh -a right   # one arm

sbatch run_retrain_full.sh slurm right       # re-run only what is missing
```

`check_runs.sh` cross-checks the artifacts on disk against the Slurm logs
(`run_train_full.sh`'s and `run_retrain_full.sh`'s). A run is complete when its
`models/chrombpnet.pt` was written to the end and its
`evaluation/chrombpnet_metrics.json` is whole (`_complete.sh`): the pipeline
writes the metrics after the model and its test-set predictions, and nothing
after them feeds `summarize.sh`. A run whose log never printed `Completed run`
is still complete, and is listed (`-v`) as such: later steps (footprints,
contribution scores, TF-MoDISco, the report) may be missing. In the blog's run
tree, 15 runs are like that: 14 finished every step but their script was edited
while they ran, so bash executed a fragment of the new version instead of the
last line; one hit the time limit during contribution scoring.

`run_retrain_full.sh` applies the same test (`_complete.sh`) per array task
and exits 0 when the run is already complete, so the whole array can be
resubmitted after a partial sweep without deleting a finished fold. Training
itself is delegated to `run_train_full.sh`, so the two cannot drift apart.

**The cluster argument is optional.** All three runners share `_args.sh`, and
each of these works:

```bash
sbatch run_retrain_full.sh slurm right
sbatch run_retrain_full.sh right                # cluster inferred from $SLURM_JOB_ID
DATA_AUG=right sbatch run_retrain_full.sh       # arm exported instead of passed
```

It was made optional after omitting it wasted 60 array tasks: `sbatch
run_retrain_full.sh right` put `right` into `$1` and left `$2` unset, so the error
read `line 39: 2: usage ...` — naming argument 2 when argument 1 was the one
missing. A bad or absent arm now prints what it actually received alongside
the valid names, and the job stops before touching any data.

`biosamples.tsv` maps ENCID to a short biosample label for `summarize.sh`'s
`biosample` column. It is a cache, so summarizing needs no network;
regenerate with `./refresh_biosamples.sh` when the ids file changes.

## Site configuration

The scripts take their paths from the blog folder's site configuration (`../site.env`, or
the file `BLOG_SITE_ENV` names; `../site.env.example` lists every value) and stop, naming
the values, when one they need is empty:

| Variable | What it is | Read by |
|---|---|---|
| `DNASE_RC_ROOT` | the run tree: `<arm>/<ASSAY>/<ENCID>/fold_<N>/`, `comparisons/`, `standard_tensorflow/`, `analysis/` | the runners, `summarize.sh`, `check_runs.sh`, `resave_tf_models.sh` |
| `DNASE_RC_IDS` | the ids table: `ASSAY ENCID BAM_PATH PEAKS_BED_PATH FOLD_NUM TENSORFLOW_DIR ...`, one row per dataset and fold | the same, `refresh_biosamples.sh`, `check_arms_on_real_reads.sh` |
| `ATLAS_DIR` | the chromatin atlas: `<ASSAY>/<ENCID>/preprocessing/{downloads,bigWigs}/` | `00_prepare_bams.sh`, `run_compare.sh`, `resave_tf_models.sh` |
| `RAW_BAM_DIR` | holds `ENCSR868FGK/merged.bam`, linked rather than downloaded | `00_prepare_bams.sh` |
| `GENOME`, `CHROM_SIZES`, `REFERENCE_DIR` | hg38, its sizes, and `folds/fold_<N>.json` + `blacklist.bed.gz` | the runners, `check_arms_on_real_reads.sh` |
| `ORIGINAL_REPO` | chrombpnet 1.x, which scores the TensorFlow models | `run_compare.sh`, `resave_tf_models.sh` |
| `CONDA_ROOT` | the conda installation holding `chrombpnet-pytorch`, `chrombpnet` and `bpnet` | all |
| `PT_ENV_GLIBC217` | optional: the glibc 2.17 environment's name (default `chrombpnet-pytorch-glibc`) | the runners |
| `LOG_DIR`, `SBATCH_SITE_ARGS` | where Slurm logs go (and are read back from); the cluster's own sbatch options | `submit_all.sh`, `check_runs.sh`, `run_retrain_full.sh` |

Two optional site.env functions run where they are needed: `site_pytorch_compiler` before
`PT_ENV_GLIBC217` is activated on a Slurm node, and `site_tf28_libraries` before TensorFlow
2.8 runs (CUDA 11.2 / cuDNN 8.1: module loads, or `LD_LIBRARY_PATH`). A value in the
environment wins over site.env's, and the scripts' own overrides (`ROOT`, `OUT`,
`IDS_FILE`, `CONDA_ENV`, `TF_CHROMBPNET_DIR`, `TF_BIGWIG`, `LOGS`, `DNASE_RC_DIR`) still
apply. The cluster argument is `slurm` or `local`. Submit jobs from this folder: under
Slurm a job finds it through `SLURM_SUBMIT_DIR`. The `#SBATCH --output` paths are relative,
so a plain `sbatch` writes its logs here; `submit_all.sh` sends them to `LOG_DIR`.

## Running it

```bash
./00_prepare_bams.sh                     # ~27.5 GB, once
./submit_all.sh slurm                    # every arm, all 20 rows
sbatch run_compare.sh slurm right        # arm model vs standard model
./summarize.sh | column -t -s $'\t'
```

`submit_all.sh` submits `run_retrain_full.sh` once per arm as a 20-way array
(4 datasets × 5 folds), which trains through `run_train_full.sh` whatever is
not already complete; `sbatch --array=0-19 run_retrain_full.sh slurm <arm>` does
one arm. `run_compare.sh` takes the arm as an argument and runs the same
20-way array. `standard` must be trained before any comparison, since it is
the reference.

## Reading the comparisons

`run_compare.sh <arm>` scores **both** the arm's model and the `standard`
model on the arm's own test data, and prints the delta.

- **`fr-only`, `rc-only`, `right`** leave the bigwig untouched, so both
  models see identical targets. Clean model-vs-model comparison; the
  difference is attributable to the augmentation alone.
- **`left`, `both-half`, `both-full`** *change* the bigwig. The standard
  model is then being scored against a shifted or re-weighted target it was
  never trained for, and will lose for reasons unrelated to reverse
  complementation. Read these as evidence about their own convention, not as
  head-to-head tests. The script prints which case applies by comparing the
  two bigwigs.

## Plotting the scores

```bash
./plot_summarize.sh --results <run tree>/analysis/results.tsv --out <dir>/summary_scores
                                    # -> <dir>/summary_scores.png / .pdf / .tsv
./plot_summarize.sh --results ... --out ... --arm right --arm standard
./plot_summarize.sh --results ... --out ... --source headtohead
```

One panel per arm; x nests biosample > ENCID (assay) > fold, with two bars
per fold and a horizontal rule at that dataset's mean over folds.

Both measures share one y-axis on purpose. Profile JSD is a divergence, so
lower is better; plotting **1 - JSD** flips it to higher-is-better and onto
the same 0-1 scale as a correlation. Two measures of different scale would
need two charts, never two y-axes.

The layout is computed once over all arms rather than per panel, so a column
means the same dataset and fold in every arm and a missing run reads as a
gap. Colours are the validated red/blue pair `#e34948` / `#2a78d6` (OKLab
Delta E 21.6 under protanopia, 32.3 under normal vision, both clear of the
>= 8 and >= 15 floors); a legend is always present, so identity is never
colour alone, and `summary_scores.tsv` is written beside the figure as a
table view.

## Data

| ENCID | assay | source |
|---|---|---|
| ENCSR000EOT | DNASE | ENCFF425WDA + ENCFF205FNC, merged |
| ENCSR163PKT | DNASE | ENCFF620MUD |
| ENCSR337UIU | ATAC | ENCFF601PGM |
| ENCSR868FGK | ATAC | already local; symlinked, not copied |

Each ends up as one `alignments_merged.bam` (+ `.bai`) in that ENCID's
`${ATLAS_DIR}/<ASSAY>/<ENCID>/preprocessing/downloads/`.
`00_prepare_bams.sh` is idempotent — an ENCID whose final BAM already passes
`samtools quickcheck` is skipped, and partial downloads resume.

## A bug fixed on the way here

`crop_revcomp_data`'s nonpeaks-only branch — the one **every bias model**
takes — aliased its cached arrays instead of copying them, and the augmenter
flips in place. Each epoch therefore reverse-complemented a fresh random half
of an already partly flipped array: the intended 50/50 split drifted into a
random walk, and the strand column filled with `"r"` and never reset.

It was survivable only because plain reversal is an involution. Adding the
shift breaks that — a round trip then destroys one bin — so it is fixed
first, in its own commit, before any of these arms exist. Bias models trained
before that commit saw real sequence throughout, but not the augmentation
schedule the code claimed.
