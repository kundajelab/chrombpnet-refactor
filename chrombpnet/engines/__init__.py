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
so :func:`for_model` and :func:`load_model_wrapper` pick the engine from the
path and fall back to the configured one only for an unrecognised suffix.
"""

from __future__ import annotations

import importlib
import os

from chrombpnet.engines.base import Engine, EngineNotImplemented, EngineUnavailable

#: Every engine name, in the order ``--engine`` lists them.
ENGINE_NAMES = ("pytorch",)

#: What an omitted ``--engine`` means when no model path says otherwise.
DEFAULT_ENGINE = "pytorch"

_MODULES = {
    "pytorch": "chrombpnet.engines.pytorch",
}

#: Checkpoint suffix -> engine that writes it.
_SUFFIX_TO_ENGINE = {
    ".pt": "pytorch",
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
    """The engine that loads ``path``: by suffix, else the configured one."""
    name = infer_from_path(path)
    return get(name) if name else current()


def resolve_name(args):
    """Decide the engine for one command from its parsed arguments.

    An explicit ``--engine`` wins, but every input model must then agree with
    it. Otherwise the input models' suffixes decide, and they must agree with
    each other. With neither, :data:`DEFAULT_ENGINE`.
    """
    explicit = getattr(args, "engine", None)
    inferred = {}
    for attribute in MODEL_PATH_ATTRIBUTES:
        name = infer_from_path(getattr(args, attribute, None))
        if name:
            inferred[attribute] = name
    if explicit:
        conflicts = {a: n for a, n in inferred.items() if n != explicit}
        if conflicts:
            detail = ", ".join(
                f"--{a.replace('_', '-')} {getattr(args, a)!r} is a {n} checkpoint"
                for a, n in conflicts.items())
            raise ValueError(f"--engine {explicit} was given, but {detail}")
        return explicit
    names = sorted(set(inferred.values()))
    if len(names) > 1:
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
    were before the flag existed; an explicit one stays and is recorded.
    """
    engine = configure(resolve_name(args))
    if require:
        engine.require()
    if hasattr(args, "engine") and args.engine is None:
        delattr(args, "engine")
    return engine


def load_model_wrapper(path, device=None, *, announce=True, summary=True):
    """Load a checkpoint for prediction with whichever engine wrote it."""
    return for_model(path).load_predictor(
        path, device, announce=announce, summary=summary)
