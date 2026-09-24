"""Framework-neutral fixtures for controlled TensorFlow/PyTorch benchmarks.

This package is a port of the TensorFlow ChromBPNet, so matching that
implementation's *settings* is the ordinary case, not a special mode. It is
therefore what you get when ``--benchmark`` is omitted. The flag exists to
hold progressively more constant than that, or to opt out of it entirely.

``pytorch``
    No parity controls at all: PyTorch's own native initializations, Adam
    defaults and loss aggregation. Not a level, an opt-out.

``tensorflow`` (**the default** -- what you get with no ``--benchmark``)
    Make the PyTorch training *settings* equivalent to the TensorFlow/Keras
    ones, while leaving normal training stochasticity in place. Conv1D/Dense
    weights use Glorot/Xavier-uniform initialization with zero biases, Adam
    uses epsilon ``1e-7``, epoch losses are sample-weighted the way Keras
    aggregates them, batches are visited in the shuffled order Keras uses by
    default, and the count-threshold nonpeak draw is seeded so both frameworks
    prepare byte-identical training inputs. Jitter, reverse complementation,
    per-epoch negative subsampling, and within-epoch shuffling still vary.

``medium``
    ``tensorflow`` plus a frozen example schedule. All generator randomness is
    drawn from a dedicated, framework-independent NumPy stream seeded from
    ``--seed``, so both frameworks realize the same jitter, reverse
    complements, and negative subsamples; epoch-start shuffling and the Keras
    batch-order shuffle are disabled; validation generators are immutable; and
    both frameworks restore the best-validation checkpoint before saving.

``max``
    ``medium`` plus everything else that can be held constant: both frameworks
    receive the same concrete initial weight arrays rather than merely the same
    initializer family, the augmentation schedule for epoch ``k`` is re-derived
    from ``(seed, mode, k)`` so it cannot drift, deterministic kernels are
    requested, TF32 is disabled, and Adam's foreach/fused dispatch is turned
    off.

A bare ``--benchmark`` with no value means ``max``.

This level was once called ``lite`` and was reachable only as the argparse
default, so a caller that built an argument namespace by hand -- ``qc`` does,
and so does every library user -- silently got no controls instead. The
default and the library fallback are now the same thing.
"""

from __future__ import annotations


import numpy as np


BENCHMARK_LEVELS = ("tensorflow", "medium", "max")

# Level used for a bare ``--benchmark`` with no value.
BENCHMARK_DEFAULT_LEVEL = "max"

# Level used when ``--benchmark`` is absent - from the command line, and
# equally from a namespace that never had the attribute. ``tensorflow`` only
# changes settings that bring this package in line with the TensorFlow
# implementation it is a port of, so it is the right default for ordinary runs.
BENCHMARK_IMPLICIT_LEVEL = "tensorflow"

# Opting out of parity altogether. Accepted on the command line but not a
# level, so it is kept out of BENCHMARK_LEVELS and the rank table.
BENCHMARK_NATIVE = "pytorch"

BENCHMARK_CHOICES = (BENCHMARK_NATIVE,) + BENCHMARK_LEVELS

# ``tensorflow`` was called ``lite`` until the default was inverted. The name
# is gone from the command line - argparse rejects it and lists the valid ones
# - but checkpoints record the normalized level in their provenance, so the
# normalizer still reads it. Dropping it here would make old .pt files
# unloadable for a renaming.
_DEPRECATED_LEVEL_ALIASES = {"lite": "tensorflow"}

_LEVEL_RANK = {name: rank for rank, name in enumerate(BENCHMARK_LEVELS, start=1)}

# Minimum level at which each individual control becomes active. A
# TensorFlow-side fork carrying the same table must rename `lite` to
# `tensorflow` with it, or a level name stops meaning the same thing on both
# sides of the comparison.
BENCHMARK_CONTROLS = {
    # tensorflow: settings parity between the frameworks
    "keras_initializers": "tensorflow",
    "keras_adam": "tensorflow",
    "keras_loss_aggregation": "tensorflow",
    "keras_batch_order_shuffle": "tensorflow",
    "seeded_threshold_draw": "tensorflow",
    # Restoring the best-validation checkpoint is what save_best_only and
    # restore_best_weights already intend; without it a TensorFlow run that
    # exhausts max_epochs saves its last epoch while PyTorch saves its best.
    "restore_best_checkpoint": "tensorflow",
    # medium: the example schedule is pinned and shared
    "generator_rng": "medium",
    "fixed_example_order": "medium",
    "frozen_validation": "medium",
    # max: identical starting tensors and numerics
    "shared_initial_weights": "max",
    "replayable_augmentation": "max",
    "deterministic_kernels": "max",
    "scalar_adam_dispatch": "max",
}

# Keeps the train/valid/test augmentation streams independent of one another.
_MODE_OFFSETS = {"train": 0, "valid": 1, "test": 2}


def normalize_benchmark_level(value):
    """Map a raw ``--benchmark`` value onto a level, or ``None`` for no controls.

    ``None`` means *absent*, not *off*, and resolves to
    :data:`BENCHMARK_IMPLICIT_LEVEL`. Only ``pytorch`` (or an explicit
    ``False``) selects no controls at all.
    """
    if value is None:
        return BENCHMARK_IMPLICIT_LEVEL
    if value is False:
        return None
    if value is True:
        return BENCHMARK_DEFAULT_LEVEL
    text = str(value).strip().lower()
    if text in ("", "true"):
        return BENCHMARK_DEFAULT_LEVEL
    if text in ("false", BENCHMARK_NATIVE):
        return None
    text = _DEPRECATED_LEVEL_ALIASES.get(text, text)
    if text not in _LEVEL_RANK:
        raise ValueError(
            "Unknown benchmark level {!r}; expected one of {}".format(
                value, ", ".join(BENCHMARK_CHOICES)
            )
        )
    return text


def benchmark_level(source):
    """Read a benchmark level from an argument namespace or a raw value.

    A namespace without a ``benchmark`` attribute is treated exactly like an
    omitted flag, so ``qc`` and library callers get the same parity settings a
    training run gets.
    """
    if source is None or isinstance(source, (str, bool)):
        return normalize_benchmark_level(source)
    return normalize_benchmark_level(getattr(source, "benchmark", None))


def benchmark_at_least(source, level):
    current = benchmark_level(source)
    if current is None:
        return False
    return _LEVEL_RANK[current] >= _LEVEL_RANK[level]


def benchmark_enables(source, control):
    """True when ``control`` is active at the level requested by ``source``."""
    return benchmark_at_least(source, BENCHMARK_CONTROLS[control])


BENCHMARK_HELP = (
    "How much to hold constant for a controlled TensorFlow/PyTorch comparison. "
    "Omit the flag for the default, 'tensorflow': match the TensorFlow "
    "model/optimizer settings (Glorot/Xavier Conv1D initialization, Adam "
    "epsilon 1e-7, Keras loss aggregation and batch order) while normal "
    "shuffling and augmentation stay random. "
    "medium: tensorflow plus a pinned, framework-shared example schedule. "
    "max: hold as much constant as possible, including identical initial weight "
    "arrays and deterministic kernels. "
    "pytorch: PyTorch's own native defaults, with no TensorFlow parity. "
    "A bare --benchmark means max."
)


def add_benchmark_argument(parser):
    """Register ``--benchmark [pytorch|tensorflow|medium|max]`` on a parser.

    The default is ``None``, meaning *absent*, which
    :func:`normalize_benchmark_level` resolves to
    :data:`BENCHMARK_IMPLICIT_LEVEL`. Recording ``None`` rather than the level
    name keeps ``<prefix>.args.json`` a faithful record of what was typed.
    A bare ``--benchmark`` selects :data:`BENCHMARK_DEFAULT_LEVEL`.
    """
    return parser.add_argument(
        "--benchmark",
        nargs="?",
        const=BENCHMARK_DEFAULT_LEVEL,
        default=None,
        choices=BENCHMARK_CHOICES,
        help=BENCHMARK_HELP,
    )


def use_keras_batch_order_shuffle(source):
    """Keras shuffles Sequence batch indices; medium and above pin the order."""
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


