"""Pin the entire command-line surface against a committed snapshot.

The CLI is the package's contract: flag names, short options, defaults,
required-ness, choices and nargs are what users' scripts depend on. `parsers.py`
is a single 300-line function containing byte-identical argument blocks repeated
up to six times, so it is a prime candidate for consolidation -- and a
consolidation that silently drops a short option or flips a default would be
invisible to every other test.

This walks the whole subparser tree and compares all ~240 actions to
`cli_surface.json`. Unlike the artifact digests in tests/regression, this
snapshot is environment-independent, so it is committed.

To accept a deliberate CLI change, regenerate it:

    python -m tests.test_cli_surface > tests/cli_surface.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("cli_surface.json")


def build_parser() -> argparse.ArgumentParser:
    """Return the fully built parser without consuming argv.

    `read_parser()` builds the tree and calls `parse_args()` in one function, so
    there is no way to get the parser object out of it. Intercepting
    `parse_args` is the least invasive way to capture it; splitting a
    `build_parser()` out of `read_parser()` would remove the need for this.
    """
    import chrombpnet.parsers as parsers

    captured: dict[str, argparse.ArgumentParser] = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        captured.setdefault("parser", self)
        return argparse.Namespace()

    argparse.ArgumentParser.parse_args = capture
    try:
        parsers.read_parser()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


def describe(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()) -> dict:
    """Flatten every action in the tree to ``"<subcommand path> <dest>"``."""
    described: dict[str, dict] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                described.update(describe(subparser, path + (name,)))
            continue
        key = " ".join(path + (action.dest,))
        options = set(action.option_strings)
        if key in described:
            # An alias kept as its own, help-suppressed action shares the
            # flag's dest; record its spelling with the flag's.
            options |= set(described[key]["options"])
        described[key] = {
            "options": sorted(options),
            "default": repr(action.default),
            "required": bool(action.required),
            "choices": sorted(map(str, action.choices)) if action.choices else None,
            "nargs": repr(action.nargs),
            "type": getattr(action.type, "__name__", repr(action.type)),
        }
    return described


def test_cli_surface_matches_the_committed_snapshot():
    current = describe(build_parser())
    expected = json.loads(SNAPSHOT.read_text())

    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    assert not missing, f"arguments disappeared from the CLI: {missing}"
    assert not added, (
        f"new arguments appeared; regenerate the snapshot if intended: {added}")

    for key in sorted(expected):
        assert current[key] == expected[key], (
            f"{key} changed:\n  expected {expected[key]}\n  current  {current[key]}")


def test_every_advertised_subcommand_can_be_dispatched():
    """The help text must not advertise a command CHROMBPNET.py cannot run.

    It used to list `snp_score`, whose subparser was commented out and whose
    dispatch branch imported a module that does not exist.
    """
    import chrombpnet.CHROMBPNET as cli

    parser = build_parser()
    subparsers = [action for action in parser._actions
                  if isinstance(action, argparse._SubParsersAction)]
    assert len(subparsers) == 1
    commands = set(subparsers[0].choices)

    source = Path(cli.__file__).read_text()
    for command in sorted(commands):
        assert f'"{command}"' in source, (
            f"{command} is exposed by the parser but never dispatched")

    advertised = subparsers[0].help or ""
    for quoted in advertised.replace('"', "'").split("'")[1::2]:
        if quoted and quoted.isidentifier():
            assert quoted in commands, (
                f"the subparser help advertises {quoted!r}, which is not a "
                "registered subcommand")


if __name__ == "__main__":
    print(json.dumps(describe(build_parser()), indent=2, sort_keys=True))
