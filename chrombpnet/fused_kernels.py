"""Whether prediction runs on the fused Triton kernels (``--predict-fk``).

Every other optimization in this package is unconditional: they were measured
phase by phase behind a ``--optimize`` switch, the measurement is recorded in
the CHANGELOG, and they are byte-identical to the code they replaced. The
Triton forward is the one that is not -- it is fp32-exact where the eager path
runs TF32, so it moves the profile head by ~1e-5 and needs an H100-class GPU.
That makes it a user's choice rather than a default, and this module is the
one bit of state that choice needs.

The setting is process-global for the same reason the old one was:
``PredictionAdapter`` is constructed several frames below the CLI, in code
shared with training, so threading a flag down would mean changing signatures
that have nothing else to do with it. :func:`configure_from_args` is called
once, from ``CHROMBPNET.main``, before any work starts.
"""

from __future__ import annotations

#: Set by `configure`. False unless a command was given ``--predict-fk``.
_ENABLED = False


def configure(enabled):
    """Set the process-wide choice and return it."""
    global _ENABLED
    _ENABLED = bool(enabled)
    return _ENABLED


def enabled():
    """True when prediction should use the fused Triton kernels."""
    return _ENABLED


def configure_from_args(args):
    """Apply ``args.predict_fk``.

    Commands without the flag -- the pipelines, ``train``, ``prep``,
    ``contribs_bw`` -- leave it off. ``contribs_bw`` deliberately has no flag:
    DeepLIFT needs autograd and hooks on the ``nn.ReLU`` modules, which a
    fused kernel cannot provide.
    """
    return configure(getattr(args, "predict_fk", False))
