"""Which deep-learning engine builds, trains, loads and explains the models.

ChromBPNet's preprocessing, data generators, metrics, bigwig writing,
footprinting, motif discovery and reports are framework-neutral and shared.
Everything that needs a deep-learning framework sits behind
:class:`chrombpnet.engines.base.Engine`, with one implementation per framework.

The choice is process-global for the same reason the fused-kernel flags are (see
:mod:`chrombpnet.fused_kernels`): the code that loads and saves models runs
several frames below the CLI, in stages shared by every pipeline, and threading
an engine argument through all of them would change signatures that have
nothing else to do with it. More importantly it would change the per-step
namespaces those stages dump to ``<prefix>.args.json``.
:func:`configure_from_args` is called once, from ``CHROMBPNET.main``, before any
work starts and before the options are frozen.

Loading is the exception. A checkpoint's suffix names the engine that wrote it,
so :func:`for_model` and :func:`load_model_wrapper` load with the configured
engine when it can read the file, and otherwise with the engine whose suffix
it has. The JAX engine, for one, also reads the original ChromBPNet's ``.h5``
models. ``.keras`` checkpoints are Keras 3's, which both Keras 3 engines read
(``jax``, and ``tensorflow``, the JAX engine's code on the TensorFlow backend);
the one whose framework is installed loads them. ``.h5`` files are written by
``tensorflow-old``, chrombpnet 1.x's Keras 2 code.
"""

from __future__ import annotations

import importlib
import importlib.util
import os

from chrombpnet.engines.base import Engine, EngineNotImplemented, EngineUnavailable

#: Every engine name, in the order ``--engine`` lists them.
ENGINE_NAMES = ("pytorch", "tensorflow", "jax", "tensorflow-old")

#: The engines that run the Keras 3 code (``chrombpnet.engines.jax``) and share its checkpoints and options.
KERAS3_ENGINES = ("jax", "tensorflow")

#: What an omitted ``--engine`` means when no model path says otherwise.
DEFAULT_ENGINE = "pytorch"

_MODULES = {
    "pytorch": "chrombpnet.engines.pytorch",
    "tensorflow": "chrombpnet.engines.tensorflow",
    "jax": "chrombpnet.engines.jax",
    "tensorflow-old": "chrombpnet.engines.tensorflow_old",
}

#: The module each engine's framework is imported as (to tell, without importing it, whether it is installed).
_FRAMEWORKS = {
    "pytorch": "torch",
    "tensorflow": "tensorflow",
    "jax": "jax",
    "tensorflow-old": "tensorflow",
}

#: Checkpoint suffix -> engine that writes it.
_SUFFIX_TO_ENGINE = {
    ".pt": "pytorch",
    ".h5": "tensorflow-old",
    ".hdf5": "tensorflow-old",
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

#: Training options only the Keras 3 engines have (``parsers._add_jax_training_options``).
#: Like ``--engine`` they default to ``None`` and are dropped when absent.
JAX_ONLY_OPTIONS = ("optimizer", "muon_lr", "ema", "ema_momentum",
                    "lr_schedule", "precision")

#: Training options only the PyTorch engine has (``--bias-profile-only``).
#: Also default ``None`` and dropped when absent.
PYTORCH_ONLY_OPTIONS = ("bias_profile_only",)

_INSTANCES = {}
_CURRENT = DEFAULT_ENGINE

__all__ = [
    "DEFAULT_ENGINE",
    "ENGINE_NAMES",
    "KERAS3_ENGINES",
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


def installed(name):
    """True if engine ``name``'s framework can be imported here (checked without importing it)."""
    try:
        return importlib.util.find_spec(_FRAMEWORKS[name]) is not None
    except (ImportError, ValueError):
        return False


def keras3_engine():
    """The Keras 3 engine for this environment: ``jax`` where JAX is installed, else
    ``tensorflow`` where TensorFlow is, else ``jax`` (whose error names its environment)."""
    if not installed("jax") and installed("tensorflow"):
        return "tensorflow"
    return "jax"


def infer_from_path(path):
    """The engine whose checkpoints end like ``path``, or ``None``. A Keras 3
    ``.keras`` file is the installed Keras 3 engine's (``keras3_engine``)."""
    if not path or not isinstance(path, str):
        return None
    name = _SUFFIX_TO_ENGINE.get(os.path.splitext(path)[1].lower())
    return keras3_engine() if name == "jax" else name


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
        # installed engines first, then the Keras 3 engine a .keras file alone
        # would get, then the --engine order
        preferred = keras3_engine()
        readers = sorted((name for name in ENGINE_NAMES
                          if all(get(name).can_read(getattr(args, a)) for a in inferred)),
                         key=lambda name: (not installed(name), name != preferred))
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
    same goes for the Keras 3 engines' training options.
    """
    engine = configure(resolve_name(args))
    _check_combinations(engine, args)
    engine.prepare(args)
    if require:
        engine.require()
    for attribute in ("engine",) + JAX_ONLY_OPTIONS + PYTORCH_ONLY_OPTIONS:
        if hasattr(args, attribute) and getattr(args, attribute) is None:
            delattr(args, attribute)
    return engine


def _check_combinations(engine, args):
    """Reject flags that only mean something for another engine."""
    if engine.name not in KERAS3_ENGINES:
        given = [a for a in JAX_ONLY_OPTIONS if getattr(args, a, None) is not None]
        if given:
            flags = ", ".join("--" + a.replace("_", "-") for a in given)
            raise ValueError(
                f"{flags} only apply to the Keras 3 engines (jax, tensorflow), "
                f"not the {engine.name} engine")
    if engine.name == "pytorch":
        return
    given = [a for a in PYTORCH_ONLY_OPTIONS if getattr(args, a, None) is not None]
    if given:
        flags = ", ".join("--" + a.replace("_", "-") for a in given)
        raise ValueError(
            f"{flags} only apply to the PyTorch engine, not the {engine.name} engine")
    fused = getattr(args, "predict_fused_kernel", False)
    if fused:
        flag = "--predict-fused-kernel-" + (fused if isinstance(fused, str) else "exact")
        raise ValueError(
            f"{flag} runs PyTorch's fused Triton forward and cannot be "
            f"combined with the {engine.name} engine")


def load_model_wrapper(path, device=None, *, announce=True, summary=True):
    """Load a checkpoint for prediction with whichever engine wrote it."""
    return for_model(path).load_predictor(
        path, device, announce=announce, summary=summary)
