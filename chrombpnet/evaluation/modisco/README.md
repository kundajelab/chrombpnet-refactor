# TF-MoDISco integration

ChromBPNet installs TF-MoDISco from the checkout named in `envs/requirements-pytorch.txt`
and in the `[reports]` extra of `pyproject.toml`. Note
that the *distribution* is named `modisco` -- that is the name
`importlib.metadata.version` answers to -- while the *import package* is still
`modiscolite`, which is what `api.py` imports. The pinned version replaced the
old workflow-object HDF5 hierarchy with:

```text
pos_patterns/pattern_N/{sequence,contrib_scores,hypothetical_contribs,seqlets}
neg_patterns/pattern_N/{sequence,contrib_scores,hypothetical_contribs,seqlets}
```

The active pipeline and `chrombpnet modisco_motifs` use
`chrombpnet.evaluation.modisco.api`, not the external `modisco` executable.
This is intentional: it gives every ChromBPNet entry point identical input
cropping and output behavior and remains usable while developing an editable
TF-MoDISco checkout.

The standard pipelines instead call TF-MoDISco's current compact report API,
which continues to produce `motifs.html` and therefore remains compatible with
ChromBPNet's combined HTML and PDF reports.

The unsupported standalone three-step runner and visualizer were moved to
`legacy/modisco/` so they cannot be mistaken for active pipeline components.
