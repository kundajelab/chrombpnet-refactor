#!/bin/bash
# Blog 2026-015 comparison table, the "Best comparison" pred_bw row with
# chrombpnet-refactor on the optimization kit's Triton TF32 kernels: the original
# package and chrombpnet-refactor each predict with the full model over the 20,000
# peaks and score them against the observed bigWig, with no bigWig or predictions
# HDF5 written (headtohead.py), as cherimoya evaluate does. Arms, all in one job
# per fold (array 0-4 = folds 0-4), 3 rounds after an untimed warm-up:
#   orig_predict      chrombpnet 1.x (TensorFlow 2.8)
#   ref_predict       chrombpnet-refactor, standard PyTorch (TF32 convolutions in cuDNN)
#   ref_predict_fast  chrombpnet-refactor with --predict-fused-kernel-fast
# Run tag *_headtohead-fast, so the earlier *_headtohead results (the attribution
# rows) are kept.
#
#   bash submit_headtohead_fast.sh
#   python summarize_rerun.py      # the NVIDIA-L40S_cuda-89_headtohead-fast section
set -eo pipefail
cd /path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/optimization
ARMS=orig_predict,ref_predict,ref_predict_fast RUN_KIND=headtohead-fast \
  sbatch --constraint="GPU_SKU:L40S" --time=03:00:00 --job-name=chrombpnet_headtohead_fast run_headtohead.sh sherlock
