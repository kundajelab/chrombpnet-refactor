#!/usr/bin/env python3
"""Inputs for run_cherimoya.sh, from one fold's chrombpnet-refactor files:

    python cherimoya_inputs.py OUT_DIR PEAKS_BED NONPEAKS_BED FOLD_JSON MOTIFS_TSV N_REGIONS N_CONTRIB

writes into OUT_DIR
    regions.bed          N_REGIONS peaks, the same sample run_pred_bw_kit_fused.sh
                         draws (reservoir sampling, awk srand(1234)), sorted
    contrib_regions.bed  its first N_CONTRIB lines (what contribs_bw was given)
    test_nonpeaks.bed    the non-peaks on the fold's test chromosomes (what
                         footprints marginalizes over)
    *.centered.bed       each of those as summit-1..summit+1: cherimoya centres
                         a window on the interval's midpoint, chrombpnet on
                         start + summit, so these give it the same 2,114 bp
    motifs.meme          motif_to_pwm.tsv's consensus strings as one-hot PWMs
and prints the three counts. The sample is drawn by the same awk program, so
on the same host type it is byte-identical to the regions the refactor and
original arms predicted (run_cherimoya.sh compares it with that job's
regions.bed when one exists).
"""

import json
import os
import subprocess
import sys

AWK = ('BEGIN{srand(1234)} {if (NR<=n) r[NR]=$0; else {i=int(rand()*NR)+1; if (i<=n) r[i]=$0}} '
       'END{for (j=1;j<=n && j<=NR;j++) print r[j]}')


def main():
    out, peaks, nonpeaks, fold_json, motifs_tsv, n_regions, n_contrib = sys.argv[1:8]
    os.makedirs(out, exist_ok=True)
    sample = subprocess.run(["awk", "-v", "n=%d" % int(n_regions), AWK, peaks],
                            check=True, capture_output=True, text=True).stdout
    # The job's own awk and sort, in its locale, as run_pred_bw_kit_fused.sh ran
    # them; so this runs on the host, never in the container (whose mawk draws
    # a different rand() stream than the nodes' gawk).
    ordered = subprocess.run(["sort", "-k1,1", "-k2,2n"], input=sample, check=True,
                             capture_output=True, text=True).stdout
    lines = ordered.splitlines(keepends=True)
    test = set(json.load(open(fold_json))["test"])
    with open(nonpeaks) as source:
        test_lines = [line for line in source if line.split("\t", 1)[0] in test]
    for name, rows in (("regions", lines), ("contrib_regions", lines[:int(n_contrib)]),
                       ("test_nonpeaks", test_lines)):
        with open(os.path.join(out, name + ".bed"), "w") as handle:
            handle.writelines(rows)
        with open(os.path.join(out, name + ".centered.bed"), "w") as handle:
            for row in rows:
                fields = row.rstrip("\n").split("\t")
                centre = int(fields[1]) + int(fields[9])
                handle.write("%s\t%d\t%d\n" % (fields[0], centre - 1, centre + 1))
    n_test = len(test_lines)
    with open(motifs_tsv) as source, open(os.path.join(out, "motifs.meme"), "w") as handle:
        handle.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                     "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for row in source:
            if not row.strip():
                continue
            name, consensus = row.split()[:2]
            handle.write("MOTIF %s\nletter-probability matrix: alength= 4 w= %d nsites= 1 E= 0\n"
                         % (name, len(consensus)))
            for base in consensus.upper():
                handle.write(" ".join("1.0" if base == b else "0.0" for b in "ACGT") + "\n")
            handle.write("\n")
    print(len(lines), min(len(lines), int(n_contrib)), n_test)


if __name__ == "__main__":
    main()
