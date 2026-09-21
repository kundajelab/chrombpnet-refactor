# chrombpnet_k1 — vendored Triton kernels

`kernels.py` and `arch_tiles.json` are copied **byte-identical** from the
ChromBPNet optimization kit, so `sha256sum` can compare them against upstream
indefinitely. No modifications were made to either file; the local wrapper
lives outside this directory, in `chrombpnet/triton_forward.py`.

| | |
|---|---|
| Upstream | https://github.com/anthropics/uplifting-biomolecular-modeling |
| Path | `chrombpnet/opt/kit/torch/chrombpnet_k1/` |
| Commit | `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` (2026-09-17) |
| Licence | Apache-2.0, Copyright 2026 Anthropic, PBC — see `LICENSE`, `NOTICE` |

```
kernels.py       sha256 7026b40a233544fbbc49d464731e42e084cd53097ca7f60f606e8cf145dac915
arch_tiles.json  sha256 9a886921533527199b9fcb3c2a46ee19995bd08eb2aa7eb1610bd184a8d7cbfa
```

The `kernels.py` digest is worth noting: its first 16 hex characters,
`7026b40a233544fb`, are exactly the `kernels_sha256_16` the kit records in
its shipped `JIT_IDENTITY.json`, so this is the file its numerics
certificates were measured against. A test pins both digests.

## What was deliberately not vendored

* `forward.py` — its `apply()` loads Keras `.h5` weights through bpnet-lite,
  which this package has no use for (checkpoints here are `.pt`); its
  `featurize` and `predict` duplicate `chrombpnet.sequences` and
  `PredictionAdapter`; and roughly 250 of its lines manage a shipped Triton
  cache that cannot apply here, because the caches were built for
  torch 2.4.1 / triton 3.0.0 and both local environments are newer. The ~60
  lines that are genuinely needed — tile selection, the counts form, the
  batch bound, the two-submodel composition — are reimplemented in
  `chrombpnet/triton_forward.py` under this package's conventions (explicit
  `raise`, no `os.environ` mutation, no unconditional `print`).
* `triton_cache_sm90/`, `triton_cache_sm89_l40s/` — hundreds of megabytes of
  precompiled kernels for a torch/triton pair neither environment has. The
  kernels JIT-compile on first use instead, once per machine.
* `kernels_tc.py` — the TF32 tensor-core variant. It is the A100 route; the
  fp32 kernels are the route for the cards here.
* `_te_guard.py`, `_oom.py`, `cache_identity.py`, the `chrombpnet_opt` CLI,
  the TensorFlow-side kit, and the vendored bpnet-lite / tangermeme trees.
