"""Reproducibility levels: how much of a training run is held fixed.

This package is a port of the TensorFlow ChromBPNet, so its training
*settings* are TensorFlow's: Glorot/Xavier-uniform initialization with zero
biases, Adam epsilon ``1e-7``, sample-weighted epoch losses the way Keras
aggregates them, batches visited in the shuffled order Keras uses, a seeded
count-threshold nonpeak draw, and restoration of the best-validation
checkpoint. Every engine always trains that way. Jitter, reverse
complementation, per-epoch negative subsampling and within-epoch shuffling
come from the global NumPy stream, which each engine seeds from ``--seed``
when training starts: repeatable on one engine, but each engine consumes the
stream in its own order, so the engines' schedules differ. Two flags hold more
fixed:

``--reproducibility``
    A pinned example schedule. All generator randomness is drawn from a
    dedicated, framework-independent NumPy stream seeded from ``--seed``, so
    every engine realizes the same jitter, reverse complements and negative
    subsamples; epoch-start shuffling and the Keras batch-order shuffle are
    disabled; validation generators are immutable.

``--reproducibility-max``
    ``--reproducibility`` plus everything else that can be held constant:
    every engine receives the same concrete initial weight arrays rather than
    merely the same initializer family, the augmentation schedule for epoch
    ``k`` is re-derived from ``(seed, mode, k)`` so it cannot drift,
    deterministic kernels are requested, TF32 is disabled, and Adam's
    foreach/fused dispatch is turned off.

Internally these are three levels -- ``tensorflow`` (neither flag),
``medium`` and ``max`` -- which is what checkpoints record and what the
generators are handed. They replace ``--benchmark``, whose ``pytorch`` level
(PyTorch's own initialization and Adam settings) is gone.
"""

from __future__ import annotations


import numpy as np


BENCHMARK_LEVELS = ("tensorflow", "medium", "max")

# The level with neither flag -- from the command line, and equally from a
# namespace that never had the attributes (qc, library callers).
BENCHMARK_IMPLICIT_LEVEL = "tensorflow"

# Level selected by a bare ``True``.
BENCHMARK_DEFAULT_LEVEL = "max"

# Older names, still read so that recorded levels keep loading: ``lite`` was
# ``tensorflow`` before it became the default.
_DEPRECATED_LEVEL_ALIASES = {"lite": "tensorflow"}

_LEVEL_RANK = {name: rank for rank, name in enumerate(BENCHMARK_LEVELS, start=1)}

# Minimum level at which each individual control becomes active. The
# ``tensorflow`` ones are therefore always on.
BENCHMARK_CONTROLS = {
    # always: the TensorFlow implementation's training settings
    "keras_initializers": "tensorflow",
    "keras_adam": "tensorflow",
    "keras_loss_aggregation": "tensorflow",
    "keras_batch_order_shuffle": "tensorflow",
    "seeded_threshold_draw": "tensorflow",
    # Restoring the best-validation checkpoint is what save_best_only and
    # restore_best_weights already intend; without it a TensorFlow run that
    # exhausts max_epochs saves its last epoch while PyTorch saves its best.
    "restore_best_checkpoint": "tensorflow",
    # --reproducibility: the example schedule is pinned and shared
    "generator_rng": "medium",
    "fixed_example_order": "medium",
    "frozen_validation": "medium",
    # --reproducibility-max: identical starting tensors and numerics
    "shared_initial_weights": "max",
    "replayable_augmentation": "max",
    "deterministic_kernels": "max",
    "scalar_adam_dispatch": "max",
}

# Keeps the train/valid/test augmentation streams independent of one another.
_MODE_OFFSETS = {"train": 0, "valid": 1, "test": 2}


def normalize_benchmark_level(value):
    """Map a level name (or ``None``/``True``/``False``) onto a level.

    ``None`` and ``False`` mean no extra reproducibility, i.e.
    :data:`BENCHMARK_IMPLICIT_LEVEL`; ``True`` means ``max``.
    """
    if value is None or value is False:
        return BENCHMARK_IMPLICIT_LEVEL
    if value is True:
        return BENCHMARK_DEFAULT_LEVEL
    text = str(value).strip().lower()
    if text in ("", "false"):
        return BENCHMARK_IMPLICIT_LEVEL
    if text == "true":
        return BENCHMARK_DEFAULT_LEVEL
    text = _DEPRECATED_LEVEL_ALIASES.get(text, text)
    if text not in _LEVEL_RANK:
        raise ValueError(
            "Unknown reproducibility level {!r}; expected one of {}".format(
                value, ", ".join(BENCHMARK_LEVELS)
            )
        )
    return text


def level_from_flags(reproducibility=None, reproducibility_max=None):
    """The level ``--reproducibility`` / ``--reproducibility-max`` select."""
    if reproducibility_max:
        return "max"
    if reproducibility:
        return "medium"
    return BENCHMARK_IMPLICIT_LEVEL


def benchmark_level(source):
    """The reproducibility level of an argument namespace or a raw value.

    A namespace is read through its ``reproducibility`` and
    ``reproducibility_max`` flags; one with neither -- ``qc``, library
    callers -- gets the default level a training run gets. Library and test
    callers may instead carry a level name as ``benchmark``.
    """
    if source is None or isinstance(source, (str, bool)):
        return normalize_benchmark_level(source)
    if getattr(source, "reproducibility_max", None) or getattr(source, "reproducibility", None):
        return level_from_flags(getattr(source, "reproducibility", None),
                                getattr(source, "reproducibility_max", None))
    return normalize_benchmark_level(getattr(source, "benchmark", None))


def benchmark_at_least(source, level):
    return _LEVEL_RANK[benchmark_level(source)] >= _LEVEL_RANK[level]


def benchmark_enables(source, control):
    """True when ``control`` is active at the level requested by ``source``."""
    return benchmark_at_least(source, BENCHMARK_CONTROLS[control])


REPRODUCIBILITY_HELP = (
    "Pin the example schedule: jitter, reverse complements and negative "
    "subsamples come from a stream seeded by --seed and shared by every "
    "engine, epoch-start shuffling is off and the validation set is frozen. "
    "Without it, training uses the TensorFlow implementation's settings with "
    "normal shuffling and augmentation.")

REPRODUCIBILITY_MAX_HELP = (
    "--reproducibility plus identical initial weight arrays in every engine, "
    "a replayable augmentation schedule, deterministic kernels, TF32 off and "
    "scalar Adam dispatch. Slower.")


def add_reproducibility_arguments(parser):
    """Register ``--reproducibility`` and ``--reproducibility-max``.

    Both default to ``None``, meaning *absent*, and are dropped from the
    namespace when absent (``parsers.ABSENT_WHEN_OMITTED``), so
    ``<prefix>.args.json`` records only what was typed.
    """
    parser.add_argument("--reproducibility", action="store_true", default=None,
                        help=REPRODUCIBILITY_HELP)
    parser.add_argument("--reproducibility-max", action="store_true", default=None,
                        help=REPRODUCIBILITY_MAX_HELP)


def use_keras_batch_order_shuffle(source):
    """Keras shuffles Sequence batch indices; --reproducibility pins the order."""
    return benchmark_enables(source, "keras_batch_order_shuffle") and not benchmark_enables(
        source, "fixed_example_order"
    )


def generator_seed(seed, mode, epoch=0):
    """Stable augmentation seed shared by both frameworks for one epoch."""
    offset = _MODE_OFFSETS.get(mode, len(_MODE_OFFSETS))
    return (int(seed) + 1000003 * offset + 7919 * int(epoch)) % (2 ** 32)


def make_generator_rng(source, seed, mode, epoch=0):
    """Dedicated NumPy stream for a generator, or ``np.random`` when disabled."""
    if seed is None or not benchmark_enables(source, "generator_rng"):
        return np.random
    return np.random.RandomState(generator_seed(seed, mode, epoch))


def _glorot_uniform(rng, shape, fan_in, fan_out):
    limit = np.sqrt(6.0 / float(fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def canonical_bpnet_weights(filters, n_dilation_layers, seed):
    """Return Keras-layout Glorot weights from a shared NumPy RNG stream."""
    filters = int(filters)
    n_dilation_layers = int(n_dilation_layers)
    rng = np.random.RandomState(int(seed))
    values = {
        "first_kernel": _glorot_uniform(
            rng, (21, 4, filters), 21 * 4, 21 * filters
        ),
        "first_bias": np.zeros(filters, dtype=np.float32),
    }
    for layer in range(1, n_dilation_layers + 1):
        values[f"dilated_{layer}_kernel"] = _glorot_uniform(
            rng, (3, filters, filters), 3 * filters, 3 * filters
        )
        values[f"dilated_{layer}_bias"] = np.zeros(filters, dtype=np.float32)
    values["profile_kernel"] = _glorot_uniform(
        rng, (75, filters, 1), 75 * filters, 75
    )
    values["profile_bias"] = np.zeros(1, dtype=np.float32)
    values["count_kernel"] = _glorot_uniform(
        rng, (filters, 1), filters, 1
    )
    values["count_bias"] = np.zeros(1, dtype=np.float32)
    return values


