# Preprocessing scripts for ChromBPNet

The scripts in this folder are pre-processing steps to convert input reads to Bigwig format for trainining ChromBPNet models.

## Requirements

To run these scripts you will need `bedtools`, `samtools` and `bedGraphToBigWig` (from ucsc) tools. These scripts are tested for bulk and single-cell ATAC-seq, and for bulk DNase-seq.

## BAM/fragment file/tagAlign file to Bigwig

We convert input BAMS to appropriately shifted Bigwigs consistent with our training pipeline: +4/-4 for ATAC, and for DNase the cut is recorded as `--dnase-convention` says (default `both-half`; see [Accessibility signal conventions](#accessibility-signal-conventions)).

```
usage: reads_to_bigwig.py [-h] -g GENOME
                          (-ibam INPUT_BAM_FILE | -ifrag INPUT_FRAGMENT_FILE | -itag INPUT_TAGALIGN_FILE)
                          -c CHROM_SIZES -op OUTPUT_PREFIX -d {ATAC,DNASE} [--bsort] [--no-st]
                          [--tmpdir TMPDIR] [-ps PLUS_SHIFT] [-ms MINUS_SHIFT]
                          [--ATAC-ref-path ATAC_REF_PATH] [--DNASE-ref-path DNASE_REF_PATH]
                          [--num-samples NUM_SAMPLES]
                          [--dnase-convention {right,left,both-half,both-full,right-deprecated}]

Convert input BAM/fragment/tagAlign file to appropriately shifted unstranded Bigwig

options:
  -h, --help            show this help message and exit
  -g GENOME, --genome GENOME
                        reference genome fasta file
  -ibam INPUT_BAM_FILE, --input-bam-file INPUT_BAM_FILE
                        Input BAM file
  -ifrag INPUT_FRAGMENT_FILE, --input-fragment-file INPUT_FRAGMENT_FILE
                        Input fragment file
  -itag INPUT_TAGALIGN_FILE, --input-tagalign-file INPUT_TAGALIGN_FILE
                        Input tagAlign file
  -c CHROM_SIZES, --chrom-sizes CHROM_SIZES
                        Chrom sizes file
  -op OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
                        Output prefix (path/to/prefix)
  -d {ATAC,DNASE}, --data-type {ATAC,DNASE}
                        assay type
  --bsort               use bedtools sort (default is unix sort)
  --no-st               No streaming in preprocessing
  --tmpdir TMPDIR       tmp dir path for unix sort command
  -ps PLUS_SHIFT, --plus-shift PLUS_SHIFT
                        Plus strand shift applied to reads. Estimated if not specified
  -ms MINUS_SHIFT, --minus-shift MINUS_SHIFT
                        Minus strand shift applied to reads. Estimated if not specified
  --ATAC-ref-path ATAC_REF_PATH
                        Path to ATAC reference motifs (chrombpnet/data/ATAC.ref.motifs.txt used by
                        default)
  --DNASE-ref-path DNASE_REF_PATH
                        Path to DNASE reference motifs (chrombpnet/data/DNASE.ref.motifs.txt used
                        by default)
  --num-samples NUM_SAMPLES
                        Number of reads to sample from BAM/fragment/tagAlign file for shift
                        estimation
  --dnase-convention {right,left,both-half,both-full,right-deprecated}
                        DNASE only: where a cut is recorded; see chrombpnet.conventions. Default:
                        both-half. right-deprecated is the convention before this flag, kept for
                        reproducibility only.
```

Please supply one of BAM(`-ibam`)/fragment file(`-ifrag`)/tagAlign file(`-itag`) as input. The script generates an unstranded Bigwig- forward and reverse strands are combined with appropriate shifting (+4/-4 for ATAC; for DNase, the 0/+1 shift recorded as `--dnase-convention` says). Output is stored at `{OUTPUT_PREFIX}_unstranded.bw`. The directory in the prefix, if applicable, must already exist.

If supplying a fragment file, it should minimally have 3 columns for chr, start and end. Each line must represent a fragment with Tn5 transposition events at both ends.

If supplying a tagAlign file, it should minimally contain 3 columns for chr, start and end, and a 6th column with the strand. 

Both gzipped and plain text files are allowed. Sorting is not expected.

The `CHROM_SIZES` file should be a tab-separated file with two columns. First column is the chromosome and second column is the chromsome length. Make sure the input BAM/fragment/tagAlign file are consistent with the chromosomes in `CHROM_SIZES`.

`chrombpnet prep bigwig` runs this script from the package's command line,
with the same options. Its output goes to the training commands with `-ibw`.

### Example usage

```bash
python -m chrombpnet.helpers.preprocessing.reads_to_bigwig \
    -ifrag my_sample.frag.tsv.gz \
    -g hg38.fa \
    -c hg38.chrom.sizes \
    -op /output/directory/my_sample \
    -d ATAC
```

Example usage when we have an input ATAC-seq fragment file `my_sample.frag.tsv.gz`. Output is stored to `/output/directory/my_sample_unstranded.bw`, please ensure the directory `/output/directory/` exists.

### Automatic shift detection

Most ATAC-seq (single-cell and bulk) pipelines shift Tn5 reads by +4/-5 by default. However when combining analyses with other tools, the effective shift can be different than +4/-5 due to off-by-one errors. Our script handles such cases by default by automatically detecting the enzyme shift (for ATAC and DNase) and correcting it appropriately (+4/-4 for ATAC and 0/+1 for DNase) for consistency with the training pipeline.

In rare cases, you may see an error such as "Input file shifts inconsistent" or "Input shift is non-standard". In such cases, if you know the actual shift for your input file (typically +0/+0 for BAMs, and +4/-5 for ATAC fragment/tagAligns) you can supply them using the `--plus-shift` and `--minus-shift` flags (`-ps`, `-ms`). However, if you are uncertain, please reach out to us by submitting an [Issue](https://github.com/kundajelab/chrombpnet/issues).


## Accessibility signal conventions

Where a cut is recorded, and what reverse-complement augmentation does about
it, are set by two flags. They are defined in `chrombpnet/conventions.py`
and used both here (the bigwig) and in training (the data generators).

- `--dnase-convention` (DNASE only) selects how a DNase cut is recorded and
  what reverse complementation does about it; omitting it selects
  **`both-half`**. An explicit `--dnase-convention` on an ATAC run is an
  error rather than a silent no-op. `--dnase-convention right-deprecated` is
  the DNase behaviour this package shipped before the flags existed
  (deprecated: kept to reproduce earlier runs only).
- `--data-aug` selects which examples are reverse complemented, the same for
  either assay; omitting it selects **`augment`** (half of the training
  examples).

Both flags are on `chrombpnet pipeline`, `train`, `qc`, their `bias`
counterparts and `predict`. `reads_to_bigwig` takes `--dnase-convention`
only: the bigwig does not depend on `--data-aug`.

They exist because the two assays record different things. DNase I is a
nuclease: it cuts the backbone *between* two bases, so the 0/+1 shift records
the event at a **bond**, named by that bond's right-hand base -- the `+1` on
minus reads is exactly what makes both strands name the same bond the same
way. Tn5 leaves a 9 bp stagger, and 9 being odd, ATAC records a **base**.

Reversing a profile maps a base to its mirror correctly. It does *not* do the
same for a bond: the bond lands in the right place but is left named by the
wrong base, so a reverse-complemented DNase profile is misregistered by one.
The correction is to reverse and then shift one position: `i -> W-i` rather
than `i -> W-1-i`.

| `--dnase-convention` | preprocessing | RC shift |
|---|---|---|
| `right` | DNase 0/+1 | **+1** |
| `left` | DNase -1/0 | **-1** |
| **`both-half`** *(default)* | 0.5 on both bases flanking the cut | 0 |
| `both-full` | 1.0 on both bases flanking the cut | 0 |
| `right-deprecated` | DNase 0/+1 | 0 (misregistered by one; deprecated) |

| `--data-aug` | splits flipped |
|---|---|
| **`augment`** *(default)* | train, 50 % |
| `forward-only` | none |
| `revcomp-only` | train, valid and test, 100 % |

ATAC (+4/-4) records a base, so plain reversal is already right and it takes
no `--dnase-convention`; `--data-aug` applies to it unchanged.

`both-half` is the DNase default in preference to the `right` shift. A shift
corrects the misregistration; splitting the cut across both bases flanking
the bond removes the asymmetry that causes it, so there is nothing left to
correct and no dependence on getting the sign right. It also leaves no label
slot without a source, so the label window is not widened. The cost is that
each cut is smeared over 2 bp; the `0.5` scale keeps the track total exactly
what one deposit per cut gives, so `counts_loss_weight` and the `-oth`/`-b`
ratios carry over unchanged.

The flags replace an earlier single `--data-aug` with seven values. Its old
values map exactly onto them:

| old `--data-aug` | now |
|---|---|
| `standard` | `--dnase-convention right-deprecated` on DNASE; nothing on ATAC |
| `fr-only` | `--data-aug forward-only` (on DNASE also `--dnase-convention right-deprecated`) |
| `rc-only` | `--data-aug revcomp-only` (on DNASE also `--dnase-convention right-deprecated`) |
| `right`, `left`, `both-half`, `both-full` | `--dnase-convention` of the same name |

The geometry is measured, not just derived: the plus- and minus-strand cut
PWMs in [`chrombpnet/data/{DNASE,ATAC}.ref.motifs.txt`](../../data/) align
with each other's reverse complement at lag 0 for DNase and lag -1 for ATAC,
and the required label shift is `lag + 1` in both cases.
[`tests/test_conventions.py`](../../../tests/test_conventions.py) asserts it,
so changing the shift deltas without changing the correction fails.


## Discussion


### Suggestions for preparing input data

Please ensure your input files are filtered for quality metrics and duplicates are removed appropriately. For single-end data, we recommend not removing duplicates. Fragment files obtained from Chromap, cellranger and similar tools are suitable with this pipeline. For single-cell analyses, you may wish to aggregate fragments by cluster instead of sample to train cluster-specific ChromBPNet models.


### Wondering why we are using the  +4/-4 ATAC shift (instead of the popular +4/-5 ATAC shift) and 0/+1 DNASE shift (instead of the popular no shift)? 

Here we show you what a PWM built from unstranded bigwigs look like when considering different shifts. To convert bigwigs to PWM use the scripts provided in the `analysis/` directory. From the below images we see that the signal on forward and revere strand reinforce themselves when we consider +4/-4 shift giving us the Tn5/DNASE-I bias motif PWM that we know of in literature ([Figure 1 in HINT-ATAC][url3] paper).


#### PWM from unstranded ATAC bigwig with no shift

![Image](images/atac_no_shift.png)

#### PWM from  unstranded GM12878 ATAC bigwig with +4/-4 shift on forward/reverse strand respectively 

![Image](images/atac_44_shift.png)

#### PWM from unstranded GM12878 ATAC bigwig with +4/-5 shift on forward/reverse strand respectively

![Image](images/atac_45_shift.png)


#### PWM from unstranded GM12878 DNASE bigwig with no shift 

![Image](images/dnase_no_shift.png)


#### PWM from unstranded GM12878 DNASE bigwig with 0/+1 shift on forward/reverse strand respectively

![Image](images/dnase_01_shift.png)

The images can be reproduced by running the scripts in the `images/directoy`

### What is the intuition behind performing the different filtering step on PE and SE?

When considering paired-end data, duplicates can be marked correctly because we have paired-end reads and hence we can remove them. This is not possible when considering single-end reads and hence we do not remove duplicates (here if we attempt to do this we might be throwing away actual signal information)

[url1]: https://github.com/ENCODE-DCC/atac-seq-pipeline
[url2]: https://broadinstitute.github.io/picard/explain-flags.html
[url3]: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1642-2



