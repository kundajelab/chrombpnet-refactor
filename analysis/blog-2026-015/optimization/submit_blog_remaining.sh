#!/bin/bash
# Submits the jobs that fill the last values in blog 2026-015:
#
#   h200  run_section4.sh on an H200 node (one job, ~20 min): the six
#         "H200: pending" parts of Section 4 (shared inputs, fixed-step
#         bigWigs, NumPy writer arguments, batched cast, HDF5 on threads,
#         GC matching).
#   l40s  run_headtohead.sh on an L40S (array 0-4, one task per fold): the four
#         "Pending" cells of the comparison table, and new contribs_bw totals
#         for both packages (the table's 385 s and 256 s predate gzip 9).
#
#   bash submit_blog_remaining.sh            # both
#   bash submit_blog_remaining.sh h200       # Section 4 on H200 only
#   bash submit_blog_remaining.sh l40s       # head-to-head only
#
# Both import the main checkout (software/chrombpnet-refactor) when they run,
# so don't change it while they are queued or running.
# When they finish:
#   python summarize_section4.py     # Section 4 cells, L40S and H200 side by side
#   python summarize_rerun.py        # head-to-head medians (run tag *_headtohead)

set -eo pipefail
cd /path/to/workdir/software/chrombpnet-refactor/analysis/blog-2026-015/optimization

which="${1:-both}"
case "${which}" in
  both|h200|l40s) ;;
  *) echo "usage: $0 [both|h200|l40s]" >&2; exit 2 ;;
esac
if [[ "${which}" != l40s ]]; then
  echo -n "Section 4, H200: "
  sbatch -p owners --constraint="GPU_SKU:H200_SXM5" run_section4.sh sherlock
fi
if [[ "${which}" != h200 ]]; then
  echo -n "head-to-head, L40S: "
  sbatch --constraint="GPU_SKU:L40S" run_headtohead.sh sherlock
fi
