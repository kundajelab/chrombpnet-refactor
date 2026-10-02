"""Which deep-learning engine builds, trains, loads and explains the models.

ChromBPNet's preprocessing, data generators, metrics, bigwig writing,
footprinting, motif discovery and reports are framework-neutral and shared.
Everything that needs a deep-learning framework sits behind
:class:`chrombpnet.engines.base.Engine`, with one implementation per framework.

The choice is process-global for the same reason ``--predict-fk`` is (see
:mod:`chrombpnet.fused_kernels`): the code that loads and saves models runs
several frames below the CLI, in stages shared by every pipeline, and threading
an engine argument through all of them would change signatures that have
nothing else to do with it. More importantly it would change the argument
namespaces those stages deep-copy and dump to ``<prefix>.args.json``.
:func:`configure_from_args` is called once, from ``CHROMBPNET.main``, before any
work starts.

Loading is the exception. A checkpoint's suffix names the engine that wrote it,
so :func:`for_model` and :func:`load_model_wrapper` load with the configured
engine when it can read the file, and otherwise with the engine whose suffix
it has. The JAX engine, for one, also reads the original ChromBPNet's ``.h5``
models.
"""

from __future__ import annotations

import importlib
import os

from chrombpnet.engines.base import Engine, EngineNotImplemented, EngineUnavailable

#: Every engine name, in the order ``--engine`` lists them.
ENGINE_NAMES = ("pytorch", "tensorflow", "jax")

#: What an omitted ``--engine`` means when no model path says otherwise.
DEFAULT_ENGINE = "pytorch"

_MODULES = {
    "pytorch": "chrombpnet.engines.pytorch",
    "tensorflow": "chrombpnet.engines.tensorflow",
    "jax": "chrombpnet.engines.jax",
}

#: Checkpoint suffix -> engine that writes it.
_SUFFIX_TO_ENGINE = {
    ".pt": "pytorch",
    ".h5": "tensorflow",
    ".hdf5": "tensorflow",
    ".keras": "jax",
}

#: Argument-namespace attributes that hold an input model path. Their suffixes
#: decide the engine when ``--engine`` is omitted.
MODEL_PATH_ATTRIBUTES = (
    "bias_model_path",   # pipeline/train: the pretrained bias model
    "model",             # predict
    "model_h5",          # contribs_bw, footprints (legacy option name)
    "bias_model",        # pred_bw, bias qc
    "chrombpnet_model",  # pred_bw, qc
    "chrombpnet_model_nb",
)

#: Training options only the JAX engine has (``parsers._add_jax_training_options``).
#: Like ``--engine`` they default to ``None`` and are dropped when absent.
JAX_ONLY_OPTIONS = ("optimizer", "muon_lr", "ema", "ema_momentum",
                    "lr_schedule", "precision")

_INSTANCES = {}
_CURRENT = DEFAULT_ENGINE

__all__ = [
    "DEFAULT_ENGINE",
    "ENGINE_NAMES",
    "Engine",
    "EngineNotImplemented",
    "EngineUnavailable",
    "configure",
    "configure_from_args",
    "current",
    "for_model",
    "get",
    "infer_from_path",
    "load_model_wrapper",
    "resolve_name",
]


def get(name):
    """The engine object for ``name``. Importing it imports no framework."""
    if name not in _MODULES:
        raise ValueError(
            f"Unknown engine {name!r}; expected one of {', '.join(ENGINE_NAMES)}")
    if name not in _INSTANCES:
        _INSTANCES[name] = importlib.import_module(_MODULES[name]).ENGINE
    return _INSTANCES[name]


def configure(name):
    """Make ``name`` the process-wide engine and return it."""
    global _CURRENT
    engine = get(name)
    _CURRENT = name
    return engine


def current():
    """The process-wide engine (``pytorch`` unless configured otherwise)."""
    return get(_CURRENT)


def infer_from_path(path):
    """The engine whose checkpoints end like ``path``, or ``None``."""
    if not path or not isinstance(path, str):
        return None
    return _SUFFIX_TO_ENGINE.get(os.path.splitext(path)[1].lower())


def for_model(path):
    """The engine that loads ``path``: the configured one if it can read it,
    else the one whose suffix it has, else the configured one."""
    engine = current()
    if engine.can_read(path):
        return engine
    name = infer_from_path(path)
    return get(name) if name else engine


def resolve_name(args):
    """Decide the engine for one command from its parsed arguments.

    An explicit ``--engine`` wins, but it must be able to read every input
    model. Otherwise the input models' suffixes decide; when they name more
    than one engine, the one engine that reads them all (JAX reads ``.h5`` as
    well as ``.keras``) is chosen. With neither, :data:`DEFAULT_ENGINE`.
    """
    explicit = getattr(args, "engine", None)
    inferred = {}
    for attribute in MODEL_PATH_ATTRIBUTES:
        name = infer_from_path(getattr(args, attribute, None))
        if name:
            inferred[attribute] = name
    if explicit:
        engine = get(explicit)
        conflicts = {a: n for a, n in inferred.items()
                     if n != explicit and not engine.can_read(getattr(args, a))}
        if conflicts:
            detail = ", ".join(
                f"--{a.replace('_', '-')} {getattr(args, a)!r} is a {n} checkpoint"
                for a, n in conflicts.items())
            raise ValueError(f"--engine {explicit} was given, but {detail}")
        return explicit
    names = sorted(set(inferred.values()))
    if len(names) > 1:
        readers = [name for name in ENGINE_NAMES
                   if all(get(name).can_read(getattr(args, a)) for a in inferred)]
        if readers:
            return readers[0]
        raise ValueError(
            "Input models come from different engines: "
            + ", ".join(f"{a}={getattr(args, a)!r} ({n})" for a, n in inferred.items()))
    return names[0] if names else DEFAULT_ENGINE


def configure_from_args(args, require=True):
    """Resolve and configure the engine for a parsed command line.

    With ``require`` the engine's framework is imported here, so an
    unavailable engine fails before any output directory is created. Commands
    that never touch a model (``prep``, ``modisco_motifs``) pass
    ``require=False`` and keep working in any engine's environment.

    An omitted ``--engine`` (``None``) is removed from ``args`` afterwards, so
    the namespaces later dumped to ``<prefix>.args.json`` are exactly what they
    were before the flag existed; an explicit one stays and is recorded. The
    same goes for the JAX-only training options.
    """
    engine = configure(resolve_name(args))
    _check_combinations(engine, args)
    if require:
        engine.require()
    for attribute in ("engine",) + JAX_ONLY_OPTIONS:
        if hasattr(args, attribute) and getattr(args, attribute) is None:
            delattr(args, attribute)
    return engine


def _check_combinations(engine, args):
    """Reject flags that only mean something for another engine."""
    if engine.name != "jax":
        given = [a for a in JAX_ONLY_OPTIONS if getattr(args, a, None) is not None]
        if given:
            flags = ", ".join("--" + a.replace("_", "-") for a in given)
            raise ValueError(
                f"{flags} only apply to the JAX engine, not the {engine.name} engine")
    if engine.name == "pytorch":
        return
    if getattr(args, "predict_fk", False):
        raise ValueError(
            f"--predict-fk runs PyTorch's fused Triton forward and cannot be "
            f"combined with the {engine.name} engine")
    if getattr(args, "benchmark", None) == "pytorch":
        raise ValueError(
            f"--benchmark pytorch opts out of PyTorch's TensorFlow-parity "
            f"settings and has no meaning for the {engine.name} engine")


def load_model_wrapper(path, device=None, *, announce=True, summary=True):
    """Load a checkpoint for prediction with whichever engine wrote it."""
    return for_model(path).load_predictor(
        path, device, announce=announce, summary=summary)
