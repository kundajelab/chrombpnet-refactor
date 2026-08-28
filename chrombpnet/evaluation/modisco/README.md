# TF-MoDISco integration

ChromBPNet uses the TF-MoDISco 2.6 checkout at
`/path/to/workdir/software/tfmodisco`. Version 2.6 renamed
the import package from `modiscolite` to `modisco` and replaced the old
workflow-object HDF5 hierarchy with:

```text
pos_patterns/pattern_N/{sequence,contrib_scores,hypothetical_contribs,seqlets}
neg_patterns/pattern_N/{sequence,contrib_scores,hypothetical_contribs,seqlets}
```

The active pipeline and `chrombpnet modisco_motifs` use
`chrombpnet.evaluation.modisco.api`, not the external `modisco` executable.
This is intentional: it gives every ChromBPNet entry point identical input
cropping and output behavior and remains usable while developing an editable
TF-MoDISco checkout.

For the retained three-step helper workflow:

1. `run_modisco.py` discovers motifs and writes new-format HDF5 plus seqlet
   FASTA and untrimmed logos.
2. `fetch_tomtom.py` reads the new pattern groups and writes TomTom matches.
3. `visualize_motif_matches.py` reads the same groups and builds the legacy
   shareable motif table.

The standard pipelines instead call TF-MoDISco's current compact report API,
which continues to produce `motifs.html` and therefore remains compatible with
ChromBPNet's combined HTML and PDF reports.
