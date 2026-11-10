"""Whether prediction runs on the fused Triton kernels, and in which precision.

``--predict-fused-kernel-exact`` runs the optimization kit's fp32 kernels
(its ``exact`` mode's forward); ``--predict-fused-kernel-fast`` runs its TF32
tensor-core kernels (its ``fast`` mode's forward). Neither is the default.

Every other optimization in this package is unconditional: they were measured
phase by phase behind a ``--optimize`` switch, the measurement is recorded in
the CHANGELOG, and they are byte-identical to the code they replaced. The
Triton forward is the one that is not: the exact kernels are fp32 where the
eager path runs TF32, so they move the profile head by ~1e-5; the fast kernels
are TF32 like the eager path but sum in a different order, so they move it by
about TF32's own error; and both need a CUDA GPU. That makes it a user's
choice rather than a default, and this module is the one bit of state that
choice needs.

The setting is process-global for the same reason the old one was:
``PredictionAdapter`` is constructed several frames below the CLI, in code
shared with training, so threading a flag down would mean changing signatures
that have nothing else to do with it. :func:`configure_from_args` is called
once, from ``CHROMBPNET.main``, before any work starts.

When the kernels cannot run -- no CUDA device, no usable Triton, a kernel that
does not compile or disagrees with the eager forward on this GPU, or a model
they cannot serve -- prediction falls back to the eager forward, with a
warning (:func:`fall_back`): the standard PyTorch forward, TF32 convolutions,
as without the flag. A failure of the host turns the mode off for the rest of
the process, so later models do not try again.
"""

from __future__ import annotations

import sys
import warnings

#: The modes, and the Triton dot precision each runs its convolutions in.
PRECISIONS = {"exact": "ieee", "fast": "tf32"}

#: Set by `configure`: None (the eager forward), "exact" or "fast".
_MODE = None

#: Every fallback so far, as (flag, reason, whole process?), for whoever
#: needs to know the fused route did not engage (benchmarks, tests).
FALLBACKS = []

#: In every fallback warning, so a log can be searched for it.
FALLBACK_MARKER = "fused Triton kernels did not engage"


def configure(mode):
    """Set the process-wide mode and return it.

    ``mode`` is ``"exact"``, ``"fast"``, or anything false for the eager
    forward. ``True`` means ``"exact"``: the single flag's meaning before
    ``fast`` existed.
    """
    global _MODE
    if mode is True:
        mode = "exact"
    if mode and mode not in PRECISIONS:
        raise ValueError(f"unknown fused-kernel mode {mode!r}; expected one of {sorted(PRECISIONS)}")
    _MODE = mode or None
    return _MODE


def enabled():
    """True when prediction should use the fused Triton kernels."""
    return _MODE is not None


def mode():
    """``"exact"``, ``"fast"`` or None."""
    return _MODE


def precision():
    """The Triton dot precision of the current mode: ``"ieee"`` or ``"tf32"``."""
    if _MODE is None:
        raise RuntimeError("the fused kernels are not enabled")
    return PRECISIONS[_MODE]


def flag(mode_name=None):
    """The command-line flag of a mode (the current one by default)."""
    return f"--predict-fused-kernel-{mode_name or _MODE}"


def fall_back(reason, whole_process=True):
    """The fused kernels cannot serve this prediction: say so, loudly, and
    let the caller use the eager forward.

    ``whole_process``: the reason is the host (device, Triton, the kernels on
    this GPU), so the mode is turned off for every later model too; otherwise
    it is this model alone (``TritonModelRefused``) and later models try
    again.
    """
    global _MODE
    current = flag()
    FALLBACKS.append((current, str(reason), whole_process))
    message = f"{current}: the {FALLBACK_MARKER}: {reason}"
    banner = "!" * 78
    scope = "every model of this run" if whole_process else "this model"
    print(f"\n{banner}\n!! WARNING: {message}\n"
          f"!! Predicting {scope} with the standard PyTorch forward instead\n"
          "!! (TF32 convolutions, as without the flag).\n"
          f"{banner}\n", file=sys.stderr, flush=True)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    if whole_process:
        _MODE = None


def configure_from_args(args):
    """Apply ``args.predict_fused_kernel`` (False, "exact" or "fast").

    Commands without the flags -- the pipelines, ``train``, ``prep``,
    ``contribs_bw`` -- leave it off. ``contribs_bw`` deliberately has none:
    DeepLIFT needs autograd and hooks on the ``nn.ReLU`` modules, which a
    fused kernel cannot provide.
    """
    return configure(getattr(args, "predict_fused_kernel", None))
