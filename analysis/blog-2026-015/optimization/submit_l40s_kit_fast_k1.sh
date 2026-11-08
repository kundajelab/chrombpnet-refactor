#!/bin/bash
# Blog 2026-015 Section 3, L40S column with the kit's Triton TF32 forward: stock, its
# stage timings (the per-lever cells' stock side), the kit's `fast` as shipped (TensorFlow's
# forward on an L40S) and `fast_k1` (the same mode with the Triton TF32 forward, via
# kit_hook_l40s_k1fast/), all in one job, 5 rounds after an untimed warm-up.
#
#   bash submit_l40s_kit_fast_k1.sh
#   python summarize_sections34.py      # the NVIDIA-L40S_cuda-89_sections34-fastk1 section
set -eo pipefail
cd /path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/optimization
ARMS=stock,stages,fast,fast_k1 RUN_KIND=sections34-fastk1 \
  sbatch --constraint="GPU_SKU:L40S" --time=02:00:00 --job-name=chrombpnet_kit_fastk1 run_sections34.sh sherlock
