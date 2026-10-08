"""Read-only command-line options, and the inputs each pipeline step adds.

The parsed command line is frozen into an :class:`Options` once it is
complete (``CHROMBPNET.main``), so nothing downstream can overwrite what the
user typed. Values the pipeline decides -- a bigwig it generated, an output
prefix, the filtered peaks a step trains on, a geometry read off a
checkpoint -- are passed to each step explicitly. Where a step hands one
namespace to shared code that reads attributes (the generators, the engines,
the step modules' ``main``), it builds a new one with :meth:`Options.step`,
which leaves the original untouched and refuses to shadow a value the user
gave unless the step declares it replaces it.

This replaces the original package's pattern of deep-copying the argument
namespace and assigning step values into the copy -- or into the namespace
itself, as preprocessing did -- which let one step's values leak into every
later step and into the ``<prefix>.args.json`` records, and could silently
override user input.
"""

from __future__ import annotations

import argparse
import copy


class Options:
    """An immutable attribute namespace: argparse's, frozen.

    Attribute access, ``getattr(..., default)``, ``hasattr`` and ``vars()``
    work as on an ``argparse.Namespace``; assignment and deletion raise.
    """

    def __init__(self, values=None):
        self.__dict__.update(dict(values or {}))

    @classmethod
    def from_namespace(cls, namespace):
        """Freeze ``namespace`` (copying its attributes; it is not changed)."""
        if isinstance(namespace, Options):
            return namespace
        return cls(vars(namespace))

    def __setattr__(self, name, value):
        raise AttributeError(
            f"options are read-only: pass {name!r} to the step that needs it "
            f"(Options.step) instead of assigning it")

    def __delattr__(self, name):
        raise AttributeError(f"options are read-only: cannot delete {name!r}")

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Options({items})"

    def __eq__(self, other):
        if isinstance(other, (Options, argparse.Namespace)):
            return vars(self) == vars(other)
        return NotImplemented

    def step(self, *, replacing=None, **inputs):
        """These options plus one step's own inputs, as new Options.

        ``inputs`` are values the step adds. One that names an option the user
        gave (anything but ``None``) with a different value is refused: a step
        must not quietly override user input. ``replacing`` holds the values a
        step deliberately substitutes for an option -- the filtered peaks a
        model trains on instead of the ``-p`` file, a sub-prefix of ``-o`` --
        so each such substitution is named where it happens.
        """
        for name, value in inputs.items():
            given = self.__dict__.get(name)
            if given is not None and given != value:
                raise ValueError(
                    f"step input {name}={value!r} would override the option "
                    f"{name}={given!r}; pass it in replacing= if the step "
                    f"means to substitute it")
        values = dict(self.__dict__)
        values.update(inputs)
        values.update(replacing or {})
        return Options(values)

    def replace(self, **values):
        """A copy with ``values`` substituted, unchecked: for an engine's own
        short-lived variants (a schedule it compiles later), not for steps."""
        return Options({**self.__dict__, **values})


def as_options(args):
    """``args`` as Options: library callers may still pass a Namespace."""
    return Options.from_namespace(args)


def with_values(args, **values):
    """A copy of ``args`` (Options or Namespace) with ``values`` substituted."""
    if isinstance(args, Options):
        return args.replace(**values)
    variant = copy.copy(args)
    for name, value in values.items():
        setattr(variant, name, value)
    return variant
