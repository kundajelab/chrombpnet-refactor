"""Which optimizations are active, in three independently switchable phases.

Every optimization in this package can be turned off, so that a run can be
compared against the unoptimized code path in the same working tree rather
than by checking out an older revision. The phases group them by origin:

``1`` -- levers ported from the ChromBPNet optimization kit at
github.com/anthropics/uplifting-biomolecular-modeling (Apache-2.0), excluding
its Triton forward: memory-mapped featurization and the one-hot lookup table
(``prefetch``), the finish-stage pre-import (``preimport``), the vectorized
profile JSD (``metrics_stage``), running the bigWig write and the metrics in
a worker process (``bigwig_writer`` / ``metrics_stage``), and skipping
interpreter teardown once every output is closed and synced (``exit_fast``).
Byte-identical output.

``2`` -- the kit's Triton forward (``forward_route``, the ``chrombpnet_k1``
kernels). Fp32-exact rather than the TF32 that cuDNN uses by default, so the
outputs move by ~2e-5 in the profile head; see ``chrombpnet.triton_forward``.

``3`` -- optimizations that are **not** from the kit: sharing the region,
sequence and observed-count work across ``pred_bw``'s models, building the
bigWig writer's arguments with numpy, collecting the signal only when a
stats file is requested, casting prediction batches as they are needed, and
writing bigWigs as fixed-step blocks. All byte-identical except the
fixed-step writer, which stores the same values in a different on-disk
section type.

The setting is process-global rather than threaded through call signatures.
Phase 1 reaches into ``one_hot`` and ``sequences``, which sit several frames
below the CLI and are shared with training; passing a flag down to them would
mean changing a dozen signatures that have nothing else to do with it.
``configure`` is called once, from the CLI, before any work starts.
"""

from __future__ import annotations

import os

#: Read when a subcommand has no ``--optimize`` of its own. The flag is only
#: on ``pred_bw``, ``contribs_bw`` and ``footprints``; the pipeline commands
#: do not carry it, because ``train.py`` dumps ``vars(args)`` into
#: ``<prefix>.args.json`` and a new flag would change that recorded artifact.
#: This lets a whole ``chrombpnet pipeline`` run be measured under a phase
#: setting without altering any command's argument surface. An explicit
#: ``--optimize`` always wins over it.
ENVIRONMENT_VARIABLE = "CHROMBPNET_OPTIMIZE"

#: Phase identifiers as they appear on the command line.
PHASES = ("1", "2", "3")

#: What each phase covers, for `--help` and for error messages.
PHASE_SUMMARY = {
    "1": "Anthropic kit levers other than the Triton forward (byte-identical)",
    "2": "the Anthropic kit's Triton forward (fp32-exact; output moves ~2e-5)",
    "3": "optimizations not from the kit (fixed-step bigWigs change file bytes)",
}


class Optimizations:
    """An immutable record of which phases are on."""

    __slots__ = ("_active",)

    def __init__(self, active=()):
        self._active = frozenset(str(phase) for phase in active)
        unknown = self._active - set(PHASES)
        if unknown:
            raise ValueError(
                f"unknown optimization phase(s): {sorted(unknown)}; "
                f"expected any of {list(PHASES)}")

    def __contains__(self, phase):
        return str(phase) in self._active

    def __repr__(self):
        return f"Optimizations({sorted(self._active) or 'none'})"

    @property
    def active(self):
        return tuple(sorted(self._active))

    # -- Phase 1: Anthropic kit, excluding the Triton forward ---------------
    @property
    def prefetch(self):
        """Memory-mapped window reads and the one-hot lookup table."""
        return "1" in self._active

    @property
    def preimport(self):
        """Load the metrics stage's imports during the forward."""
        return "1" in self._active

    @property
    def vectorized_metrics(self):
        """Compute the profile JSD over all regions at once."""
        return "1" in self._active

    @property
    def overlapped_finish(self):
        """Run each model's write and metrics in a worker process."""
        return "1" in self._active

    @property
    def exit_fast(self):
        """Skip interpreter and CUDA teardown once outputs are durable."""
        return "1" in self._active

    # -- Phase 2: the kit's Triton forward ----------------------------------
    @property
    def triton_forward(self):
        return "2" in self._active

    # -- Phase 3: not from the kit ------------------------------------------
    @property
    def shared_inputs(self):
        """Featurize and read the observed bigWig once, not once per model."""
        return "3" in self._active

    @property
    def numpy_writer_args(self):
        """Feed pyBigWig numpy arrays rather than Python lists."""
        return "3" in self._active

    @property
    def lazy_stats(self):
        """Accumulate the written signal only when a stats file is wanted."""
        return "3" in self._active

    @property
    def batched_cast(self):
        """Cast each prediction batch as it is needed."""
        return "3" in self._active

    @property
    def fixed_step_bigwig(self):
        """Write fixed-step blocks instead of one interval per base."""
        return "3" in self._active


#: The process-wide setting. Replaced by `configure`.
_ACTIVE = Optimizations(())


def parse(value):
    """Turn a ``--optimize`` value into an :class:`Optimizations`.

    Accepts ``none``, ``all``, or a comma-separated list of phase numbers, so
    a phase can be exercised alone or in union with others.
    """
    if value is None:
        return Optimizations(())
    text = str(value).strip().lower()
    if text in ("", "none", "off"):
        return Optimizations(())
    if text == "all":
        return Optimizations(PHASES)
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        return Optimizations(())
    if "all" in tokens:
        return Optimizations(PHASES)
    if "none" in tokens:
        raise ValueError("'none' cannot be combined with other phases")
    return Optimizations(tokens)


def configure(value):
    """Set the process-wide optimizations and return them."""
    global _ACTIVE
    _ACTIVE = parse(value)
    return _ACTIVE


def active():
    """The optimizations in force for this process."""
    return _ACTIVE


def configure_from_args(args):
    """Apply ``args.optimize``, else ``$CHROMBPNET_OPTIMIZE``, else nothing.

    The command line wins: a subcommand that declares ``--optimize`` has a
    default, so it never falls through to the environment. Commands that do
    not declare it -- the pipelines, ``train``, ``qc``, ``prep`` -- can still
    be placed under a phase setting this way, which is what makes a
    whole-pipeline runtime measurement possible.

    A setting that arrives from the environment is announced, because
    invisible state that changes numerics is worse than none.
    """
    value = getattr(args, "optimize", None)
    if value is not None:
        return configure(value)
    value = os.environ.get(ENVIRONMENT_VARIABLE)
    active = configure(value)
    if active.active:
        print(f"[chrombpnet] optimizations {','.join(active.active)} "
              f"(from ${ENVIRONMENT_VARIABLE})", flush=True)
    return active
