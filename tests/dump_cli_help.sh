#!/bin/bash
# Dump `--help` for the whole command tree, for before/after comparison.
#
# The snapshot in cli_surface.json pins each argument's flags, default,
# required flag, choices, nargs and type, but not the order they print in or
# their help text. When restructuring parsers.py, capture this before and after
# and diff it:
#
#   tests/dump_cli_help.sh > /tmp/help_before.txt
#   # ... make changes ...
#   tests/dump_cli_help.sh > /tmp/help_after.txt
#   diff /tmp/help_before.txt /tmp/help_after.txt
#
# COLUMNS is pinned so argparse wraps identically regardless of terminal width.
for cmd in "" "pipeline" "train" "qc" "bias" "bias pipeline" "bias train" "bias qc" \
           "prep" "prep nonpeaks" "prep splits" "pred_bw" "contribs_bw" \
           "modisco_motifs" "footprints"; do
  echo "########## chrombpnet $cmd --help"
  COLUMNS=100 python -m chrombpnet.CHROMBPNET $cmd --help 2>&1
done
