# GC-matched negative regions

Use the maintained command:

```bash
chrombpnet prep nonpeaks \
  --genome genome.fa \
  --chrom-sizes genome.chrom.sizes \
  --peaks peaks.narrowPeak \
  --chr-fold-path fold.json \
  --output-prefix output/background
```

Foreground and candidate BED files use chromosome, start, end, and GC fraction
columns; foreground narrowPeak input additionally uses the summit in column 10.
Train and validation folds use `--neg-to-pos-ratio-train` negatives per positive,
while the test fold remains 1:1.

GC fractions are converted to integer bins from 0 through 100. Candidate rows
are stored in NumPy arrays, randomized once, sorted by fold and GC bin, and
sampled without replacement. When the exact bin is empty, binary search chooses
the nearest bin with remaining candidates. Equal-distance ties are resolved by
the seeded NumPy generator. If a fold does not have enough candidates, the
command exits immediately with a capacity error.

Outputs are `<prefix>_negatives.bed`, intermediate files beneath
`<prefix>_auxiliary/`, and a foreground-versus-negative GC histogram.
