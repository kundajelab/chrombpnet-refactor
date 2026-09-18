## Generate chrombpnet predictions on given regions and output to a bigwig file. 

`chrombpnet pred_bw` generates bigwig tracks from model predictions on a set of input regions. 
Please provide absolute paths to this script. 

```
usage: CHROMBPNET.py pred_bw [-h] [-bm BIAS_MODEL] [-cm CHROMBPNET_MODEL]
                             [-cmb CHROMBPNET_MODEL_NB] -r REGIONS -g GENOME
                             -c CHROM_SIZES -op OUTPUT_PREFIX
                             [-os OUTPUT_PREFIX_STATS] [-bs BATCH_SIZE]
                             [-t TQDM] [-d DEBUG_CHR [DEBUG_CHR ...]]
                             [-bw BIGWIG] [--device DEVICE]

required arguments:
  -bm BIAS_MODEL, --bias-model BIAS_MODEL
                        Path to bias model .pt checkpoint (at least one model
                        is required)
  -cm CHROMBPNET_MODEL, --chrombpnet-model CHROMBPNET_MODEL
                        Path to combined ChromBPNet .pt checkpoint
  -cmb CHROMBPNET_MODEL_NB, --chrombpnet-model-nb CHROMBPNET_MODEL_NB
                        Path to no-bias ChromBPNet .pt checkpoint
  -r REGIONS, --regions REGIONS
                        10 column bed file of regions for prediction
  -g GENOME, --genome GENOME
                        Genome fasta
  -c CHROM_SIZES, --chrom-sizes CHROM_SIZES
                        Chromosome sizes 2 column tab-separated file
  -op OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
                        Output prefix for bigwig files

optional arguments:
  -os OUTPUT_PREFIX_STATS, --output-prefix-stats OUTPUT_PREFIX_STATS
                        Output stats on bigwig
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size to use for prediction
  -t TQDM, --tqdm TQDM  Use tqdm. If yes then you need to have it installed.
  -d DEBUG_CHR [DEBUG_CHR ...], --debug-chr DEBUG_CHR [DEBUG_CHR ...]
                        Run for specific chromosomes only (e.g. chr1 chr2) for
                        debugging
  -bw BIGWIG, --bigwig BIGWIG
                        If provided .h5 with predictions are output along with
                        calculated metrics considering bigwig as groundtruth.
  --device DEVICE       PyTorch device (default: CUDA when available)
```

