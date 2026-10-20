# DNase cut-site convention and reverse-complement augmentation

The analysis behind the blog post's DNase figures: test-set scores of
ChromBPNet models trained under each cut-site convention and augmentation
arm, and the cross-evaluation of those models against the original models'
test sets and vice versa.

## What it tests

DNase I cuts the backbone *between* two bases, so a DNase event is a bond;
Tn5's 9 bp stagger centres an ATAC event *on* a base. chrombpnet 1.x recorded
a DNase cut at the bond's right-hand base (0/+1) and then reverse-complemented
half of the training examples by plain reversal, which names each mirrored
bond by the wrong base: every reverse-complemented DNase example was
misregistered by one base, so a model learned a mixture of two conventions.
ATAC is unaffected and serves as the control. `chrombpnet/conventions.py`
derives the geometry; `--dnase-convention` and `--data-aug` select it.

## Arms

| arm | chrombpnet flags | assays |
|---|---|---|
| `standard` | `--dnase-convention right-deprecated` (ATAC: none) | both |
| `fr-only` | `--data-aug forward-only` (+ `--dnase-convention right-deprecated` on DNASE) | both |
| `rc-only` | `--data-aug revcomp-only` (+ `--dnase-convention right-deprecated` on DNASE) | both |
| `right`, `left`, `both-half`, `both-full` | `--dnase-convention <arm>` | DNASE only |

On the figures `standard` is labelled `original-refactor`, and
`standard_tensorflow` (the original chrombpnet 1.x models) `original`.
`right-deprecated` reproduces 1.x's geometry and exists only for this.

## Inputs

Per dataset: a BAM (or fragments / tagAlign), a peak set, hg38 FASTA and
chrom sizes, a blacklist, and chrombpnet's five fold JSONs. The published
figures use four ENCODE datasets (`biosamples.tsv`):

| ENCID | assay | biosample | reads |
|---|---|---|---|
| ENCSR868FGK | ATAC | K562 | ENCODE alignments |
| ENCSR000EOT | DNase | K562 | ENCFF425WDA + ENCFF205FNC, merged |
| ENCSR163PKT | DNase | stomach (M54) | ENCFF620MUD |
| ENCSR337UIU | ATAC | stomach (F53) | ENCFF601PGM |

For the `original` rows, an original chrombpnet 1.x run per dataset and fold
(its `chrombpnet.h5`, `filtered.peaks.bed`, `chrombpnet_metrics.json`, and the
bigWig it trained on), listed in an ids table (below).

## 1. Train every arm, every fold

Lay the runs out as `<root>/<arm>/<ASSAY>/<ENCID>/fold_<N>/`, which is what
`summarize.py` reads. With `<run> = <root>/<arm>/<ASSAY>/<ENCID>/fold_<N>` and
`<flags>` the arm's flags above:

```bash
mkdir -p <run>/data
chrombpnet prep nonpeaks -g <genome.fa> -p <peaks.bed.gz> -c <chrom.sizes> \
    -fl <fold_N.json> -br <blacklist.bed.gz> -o <run>/data/<ENCID>
bgzip -c <run>/data/<ENCID>_negatives.bed > <run>/data/negatives.bed.gz

chrombpnet bias pipeline -ibam <reads.bam> -d <ASSAY> -g <genome.fa> -c <chrom.sizes> \
    -p <peaks.bed.gz> -n <run>/data/negatives.bed.gz -fl <fold_N.json> \
    -b <0.5 for ATAC, 0.8 for DNASE> -o <run>/bias_model -fp <ENCID> <flags>

chrombpnet pipeline -ibam <reads.bam> -d <ASSAY> -g <genome.fa> -c <chrom.sizes> \
    -p <peaks.bed.gz> -n <run>/data/negatives.bed.gz -fl <fold_N.json> \
    -b <run>/bias_model/models/<ENCID>_bias.pt -o <run>/model --ignore-bias-flag <flags>
```

The stomach datasets cannot train a usable bias model of their own (their
peak and background count distributions overlap), so their `chrombpnet
pipeline` takes `-b` from the K562 dataset of the same assay **and the same
arm**, fold 0, instead of running `bias pipeline`.

## 2. Cross-evaluate

For every arm except `standard`, on each fold, score four (model, test set)
pairs on the fold's test chromosomes, peaks only. A test set is a run's
`model/auxiliary/filtered.peaks.bed` and `model/auxiliary/data_unstranded.bw`
(for `standard_tensorflow`: the original run's peaks and the bigWig it trained
on). With `<cmp> = <root>/comparisons/<arm>/<ASSAY>/<ENCID>/fold_<N>`:

```bash
# the standard model on the arm's test set (with the arm's flags)
chrombpnet predict -m <root>/standard/.../model/models/chrombpnet.pt -d <ASSAY> \
    -g <genome.fa> -c <chrom.sizes> -fl <fold_N.json> -n None \
    -r <arm peaks> -bw <arm bigwig> -op <cmp>/standard <arm flags>
# the arm's model on standard's test set (with standard's flags)
chrombpnet predict -m <arm model> ... -r <standard peaks> -bw <standard bigwig> \
    -op <cmp>/on_standard/arm <standard flags>
# the original model on the arm's test set, with chrombpnet 1.x itself
PYTHONPATH=<chrombpnet-1.x checkout> python -m chrombpnet.training.predict \
    -m <original chrombpnet.h5> -g <genome.fa> -b <arm bigwig> -p <arm peaks> \
    -n None -fl <fold_N.json> -o <cmp>/standard_tensorflow -bs 256
# the arm's model on the original's test set
chrombpnet predict -m <arm model> ... -r <original peaks> -bw <original bigwig> \
    -op <cmp>/on_standard_tensorflow/arm <standard flags>
```

Only `rc-only`'s flags change what `predict` does (it reverse-complements
every test example). For `fr-only`, `rc-only` and `right` the bigWig is
identical to `standard`'s, so those are clean model-vs-model comparisons;
`left` and `both-*` change the target itself.

chrombpnet 1.x models saved under Python 3.7 do not load in 1.x's Python 3.8
environment (their count-head `Lambda` is marshalled bytecode); re-save them
under 3.8 first, or score them with this package's TensorFlow engine instead.

## 3. Summarize and plot

```bash
python summarize.py --root <root> --out <results.tsv> \
    --ids <ids.tsv> --biosamples biosamples.tsv
python plot_summarize.py --results <results.tsv> --out <figures>/summary_scores \
    --fill-original ENCSR868FGK:2
```

`--ids` is a TSV with columns `ASSAY`, `ENCID`, `FOLD_NUM` and
`TENSORFLOW_DIR` (the original run's directory for that fold); without it
the `original` rows are left out. `--fill-original` draws a fold with no
original run at the mean of that dataset's other folds (ENCSR868FGK fold 2
in the published figures); drawing only, the tables keep measured values.

`plot_summarize.py` writes, each as `.png`, `.pdf` and `.tsv`:

| figure | shows |
|---|---|
| `summary_scores` | each model on its own test set (the blog's `chrombpnet_dnase.png`) |
| `summary_scores_aug_on_standard_tests` (+ `_original`, `_refactor`) | each arm's model on the original models' test sets (`chrombpnet_dnase_cross_original_test.png`) |
| `summary_scores_standard_on_aug_tests` (+ `_original`, `_refactor`) | the original models on each arm's test set (`chrombpnet_dnase_cross_new_test.png`) |

Both measures share one axis: counts Pearson r and 1 - profile JSD, higher
is better for both.
