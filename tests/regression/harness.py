"""Run the CLI scenarios and compare their artifacts against a baseline.

Usage::

    # Record what the current working tree produces.
    python -m tests.regression.harness capture --out /path/to/baseline

    # Later, after refactoring, check nothing moved.
    python -m tests.regression.harness compare --baseline /path/to/baseline

    # Or do both in one step against a git ref (uses a temporary worktree):
    python -m tests.regression.harness against --ref main

    # The TensorFlow engines, from the chrombpnet-tensorflow environment:
    python -m tests.regression.harness capture --engine tensorflow --out DIR       # Keras 3
    python -m tests.regression.harness capture --engine tensorflow-old --out DIR   # 1.x's Keras 2

    # The JAX engine, from the chrombpnet-jax environment:
    python -m tests.regression.harness capture --engine jax --out DIR

Baselines are captured locally rather than committed. The artifacts depend on
the installed PyTorch build, so a committed digest would be a per-environment
constant that goes stale on every dependency bump -- and the question this
harness answers is "did *my change* move the output", which only ever compares
two runs in one environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import artifacts, fixture, scenarios
from .scenarios import Scenario

#: Held fixed because several artifacts embed absolute paths: the WeasyPrint
#: PDFs reference their images by absolute path, and <prefix>.args.json is a
#: dump of the argparse namespace. Varying the workspace (as pytest's tmp_path
#: would) makes those files differ for reasons that have nothing to do with the
#: code under test.
DEFAULT_WORKSPACE = Path("/tmp/chrombpnet-regression") / f"uid{os.getuid()}"

#: Every run -- baseline or current -- executes here, so the absolute paths
#: baked into artifacts are identical and drop out of the comparison. A captured
#: baseline keeps a *copy* of this tree so mismatches can still be explained.
#:
#: Override with CHROMBPNET_REGRESSION_RUN_DIR to run two environments side by
#: side; it must hold the *same* value for a capture and the compare that reads
#: that capture, or every path-bearing artifact will differ spuriously.
RUN_DIR = Path(os.environ.get("CHROMBPNET_REGRESSION_RUN_DIR",
                              str(DEFAULT_WORKSPACE / "run")))

#: Scratch for `against --ref`, derived from RUN_DIR so that overriding the run
#: directory isolates a whole invocation. Two runs that shared these would
#: overwrite each other's baseline and report meaningless path diffs.
SCRATCH = RUN_DIR.parent / f"{RUN_DIR.name}-scratch"

#: Pinned so CPU reduction order, hashing and plot rendering cannot drift.
PINNED_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
    "CUDA_VISIBLE_DEVICES": "",
}

#: Engines the sweep can run, and the checkpoint suffix each one writes.
ENGINE_SUFFIXES = {"pytorch": ".pt", "tensorflow": ".keras", "jax": ".keras", "tensorflow-old": ".h5"}

#: TensorFlow sizes its own thread pools rather than following OMP_NUM_THREADS;
#: with one thread each its CPU kernels reduce in a fixed order. Added only for
#: --engine tensorflow-old, so the PyTorch environment is exactly what it was.
ENGINE_PINNED_ENV = {
    "tensorflow-old": {
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        # WeasyPrint 61.2 (the newest for Python 3.8) embeds fontTools font
        # subsets whose `head.modified` is the time of writing, so every PDF
        # differs run to run by a few bytes. fontTools takes that timestamp
        # from SOURCE_DATE_EPOCH when it is set.
        "SOURCE_DATE_EPOCH": "0",
    },
    # JAX on the CPU platform only (no CUDA context even where a GPU is
    # visible), with XLA's CPU thread pool at one thread, so reductions run in
    # a fixed order.
    "jax": {
        "KERAS_BACKEND": "jax",
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "SOURCE_DATE_EPOCH": "0",
    },
    # The JAX engine's Keras 3 code on TensorFlow: TensorFlow's thread pins.
    "tensorflow": {
        "KERAS_BACKEND": "tensorflow",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "SOURCE_DATE_EPOCH": "0",
    },
}


def modiscolite_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("modiscolite") is not None


def _placeholders(workspace: Path, paths: dict, scenario: Scenario) -> dict[str, str]:
    from chrombpnet.data import DefaultDataFile, get_default_data_path

    values = {key: str(value) for key, value in paths.items()}
    values["root"] = str(workspace)
    values["out"] = str(workspace / scenario.output)
    values["motifs_to_pwm"] = str(
        get_default_data_path(DefaultDataFile.motif_to_pwm_atac))
    return values


def _accepts_engine(argv: list[str]) -> bool:
    """--engine exists on every command except prep and modisco_motifs."""
    return argv[0] not in ("prep", "modisco_motifs")


def _for_engine(argv: list[str], engine: str) -> list[str]:
    """Adapt a scenario's argv to an engine.

    PyTorch runs the scenarios exactly as written: not even ``--engine
    pytorch`` is added, because an explicit flag is recorded in
    ``<prefix>.args.json`` and would move four artifacts. Any other engine
    gets its checkpoint suffix in place of ``.pt`` and an explicit
    ``--engine``, which ``bias train`` needs since it has no input model to
    infer the engine from.
    """
    if engine == "pytorch":
        return argv
    suffix = ENGINE_SUFFIXES[engine]
    argv = [part[:-len(".pt")] + suffix if part.endswith(".pt") else part
            for part in argv]
    if _accepts_engine(argv):
        argv = argv + ["--engine", engine]
    return argv


def run_scenarios(workspace: Path, names: list[str] | None = None,
                  reproducibility: str | None = None,
                  verbose: bool = True,
                  engine: str = "pytorch") -> dict[str, dict[str, str]]:
    """Execute the scenarios and return ``{scenario: {artifact: digest}}``."""
    workspace = Path(workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    paths = fixture.build(workspace / "fixture")

    env = dict(os.environ)
    env.update(PINNED_ENV)
    env.update(ENGINE_PINNED_ENV.get(engine, {}))
    _verify_package_root(env, workspace)

    # The architecture module's absolute path reaches <prefix>.args.json and
    # the parameter TSVs, and differs between two checkouts of the same code.
    # Both the resolved and unresolved roots are substituted: pipelines.py
    # records `module.__file__`, which is not resolved, so when the package is
    # exposed through the staging symlink used by `against --ref` the two
    # differ.
    import chrombpnet

    package_file = Path(chrombpnet.__file__)
    roots = {str(package_file.parent.parent), str(package_file.resolve().parent.parent)}
    substitutions = tuple((root, "<package-root>") for root in sorted(roots, reverse=True))

    if names is None:
        selected = list(scenarios.SCENARIOS)
    else:
        expanded = scenarios.with_dependencies(names)
        if set(expanded) != set(names):
            added = sorted(set(expanded) - set(names))
            print(f"[deps] also running {', '.join(added)}")
        selected = [s for s in scenarios.SCENARIOS if s.name in expanded]
    manifests: dict[str, dict[str, str]] = {}
    for scenario in selected:
        if scenario.needs_modiscolite and not modiscolite_available():
            if verbose:
                print(f"[skip] {scenario.name}: modiscolite is not installed")
            continue
        output = workspace / scenario.output
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)

        values = _placeholders(workspace, paths, scenario)
        argv = _for_engine([part.format(**values) for part in scenario.argv], engine)
        if reproducibility is not None and _accepts_reproducibility(argv):
            argv += [REPRODUCIBILITY_FLAGS[reproducibility]]

        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-m", "chrombpnet.CHROMBPNET", *argv],
            env=env, capture_output=True, text=True, cwd=workspace)
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"scenario {scenario.name} failed ({completed.returncode})\n"
                f"argv: {' '.join(argv)}\n"
                f"--- stdout tail ---\n{completed.stdout[-3000:]}\n"
                f"--- stderr tail ---\n{completed.stderr[-3000:]}")
        if scenario.nondeterministic:
            if verbose:
                print(f"[ran]  {scenario.name}: completed in {elapsed:.1f}s; "
                      f"artifacts not compared ({scenario.nondeterministic})")
            continue
        manifests[scenario.name] = artifacts.manifest(
            output, scenario.ignore, substitutions)
        if verbose:
            print(f"[ok]   {scenario.name}: "
                  f"{len(manifests[scenario.name])} artifacts in {elapsed:.1f}s")
    return manifests


def _verify_package_root(env: dict[str, str], cwd: Path) -> None:
    """Fail loudly if the CLI subprocesses will not import the intended tree.

    ``against --ref`` captures its baseline by putting a git worktree on
    PYTHONPATH, but an editable install can register an import finder that
    outranks PYTHONPATH. If that happened silently the "baseline" would be the
    working tree and the comparison would always pass, so check it instead of
    assuming.
    """
    expected = env.get("CHROMBPNET_EXPECT_PACKAGE_ROOT")
    if not expected:
        return
    completed = subprocess.run(
        [sys.executable, "-c", "import chrombpnet; print(chrombpnet.__file__)"],
        env=env, cwd=cwd, capture_output=True, text=True, check=True)
    resolved = Path(completed.stdout.strip()).resolve()
    root = Path(expected).resolve()
    if root not in resolved.parents:
        raise RuntimeError(
            f"expected the CLI to import chrombpnet from {root}, but it "
            f"resolved to {resolved}. An editable-install import finder is "
            "probably shadowing PYTHONPATH; run the baseline capture from a "
            "checkout of that revision instead of using 'against --ref'.")


#: ``--reproducibility`` values for the harness, and the CLI flag each adds.
REPRODUCIBILITY_FLAGS = {"on": "--reproducibility", "max": "--reproducibility-max"}


def _accepts_reproducibility(argv: list[str]) -> bool:
    """The reproducibility flags exist on the four training parsers (and
    predict, which no scenario runs)."""
    head = [part for part in argv[:2] if not part.startswith("-")]
    return head[:2] in (["pipeline"], ["train"], ["bias", "pipeline"],
                        ["bias", "train"])


#: Records which engine a baseline was captured with. A sidecar file rather
#: than a manifest key, so manifests keep the shape older harnesses read.
ENGINE_RECORD = "engine.txt"


def capture(out: Path, reproducibility: str | None, names: list[str] | None,
            engine: str = "pytorch") -> Path:
    """Run the scenarios in RUN_DIR and archive the result under ``out``."""
    out = Path(out)
    manifests = run_scenarios(RUN_DIR, names=names, reproducibility=reproducibility,
                              engine=engine)
    manifest_path = out / "manifest.json"
    artifacts.write_manifest(manifest_path, manifests)
    (out / ENGINE_RECORD).write_text(engine + "\n")

    # Archive a copy so a later mismatch can be explained artifact-by-artifact.
    archive = out / "workspace"
    if archive.exists():
        shutil.rmtree(archive)
    shutil.copytree(RUN_DIR, archive, symlinks=True)

    total = sum(len(entries) for entries in manifests.values())
    print(f"\ncaptured {total} artifacts across {len(manifests)} scenarios")
    print(f"manifest: {manifest_path}")
    print(f"archived for diffing: {archive}")
    return manifest_path


def compare(baseline_dir: Path, reproducibility: str | None,
            names: list[str] | None, engine: str = "pytorch") -> int:
    """Re-run the scenarios and diff against a captured baseline."""
    baseline_dir = Path(baseline_dir)
    baseline = artifacts.read_manifest(baseline_dir / "manifest.json")
    baseline_workspace = baseline_dir / "workspace"
    record = baseline_dir / ENGINE_RECORD
    # Baselines from before --engine existed are PyTorch ones.
    baseline_engine = record.read_text().strip() if record.exists() else "pytorch"
    if baseline_engine != engine:
        raise SystemExit(f"baseline was captured with --engine {baseline_engine}, "
                         f"not {engine}")

    current_workspace = RUN_DIR
    current = run_scenarios(current_workspace, names=names, reproducibility=reproducibility,
                            engine=engine)

    problems: list[str] = []
    for name in sorted(set(baseline) | set(current)):
        if name not in current:
            print(f"[skip] {name}: not run now (present in baseline)")
            continue
        if name not in baseline:
            problems.append(f"{name}: ran now but absent from the baseline")
            continue
        before, after = baseline[name], current[name]
        scenario = scenarios.BY_NAME[name]
        for artifact in sorted(set(before) | set(after)):
            if before.get(artifact) == after.get(artifact):
                continue
            detail = artifacts.explain(
                baseline_workspace / scenario.output / artifact,
                current_workspace / scenario.output / artifact)
            problems.append(f"{name}/{artifact}\n{detail}")

    if problems:
        print(f"\n{len(problems)} artifact(s) changed:\n")
        for problem in problems[:25]:
            print(f"  * {problem}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        return 1
    total = sum(len(entries) for entries in current.values())
    print(f"\nall {total} artifacts identical across {len(current)} scenarios")
    return 0


def against_ref(ref: str, reproducibility: str | None,
                names: list[str] | None, engine: str = "pytorch") -> int:
    """Capture a baseline from ``ref`` in a temporary worktree, then compare.

    Only the *package under test* comes from ``ref``; the harness itself always
    runs from the working tree. Running the ref's own harness would be circular
    and impossible for any revision older than the harness -- which is every
    revision worth comparing against -- and, more subtly, it would let the two
    sides digest artifacts with different code.

    Achieving that split needs care, because PYTHONPATH entries expose every
    top-level name in a directory. Pointing it at the worktree would resolve
    `chrombpnet` *and* `tests.regression` there. So the worktree's package is
    exposed through a staging directory holding nothing but a symlink to it:
    `chrombpnet` resolves from the stage, and `tests` falls through to the repo.
    """
    repo = Path(__file__).resolve().parents[2]
    worktree = SCRATCH / f"ref-{ref.replace('/', '_')}"
    stage = SCRATCH / f"stage-{ref.replace('/', '_')}"
    baseline = SCRATCH / "baseline"

    if worktree.exists():
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force",
                        str(worktree)], check=False, capture_output=True)
        shutil.rmtree(worktree, ignore_errors=True)
    shutil.rmtree(stage, ignore_errors=True)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--quiet",
                    "--detach", str(worktree), ref], check=True)
    stage.mkdir(parents=True)
    (stage / "chrombpnet").symlink_to(worktree / "chrombpnet",
                                      target_is_directory=True)

    neutral_cwd = SCRATCH / "cwd"
    neutral_cwd.mkdir(parents=True, exist_ok=True)
    try:
        print(f"--- capturing baseline from {ref} ---")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(stage), str(repo)] +
            ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        # The stage only holds a symlink, so a resolved __file__ lands in the
        # worktree itself; that is what the check must expect.
        env["CHROMBPNET_EXPECT_PACKAGE_ROOT"] = str(worktree)
        argv = [sys.executable, "-m", "tests.regression.harness", "capture",
                "--out", str(baseline)]
        if reproducibility:
            # The ref's own harness runs the baseline: a ref from before
            # --reproducibility replaced --benchmark does not know this flag.
            argv += ["--reproducibility", reproducibility]
        if engine != "pytorch":
            argv += ["--engine", engine]
        for name in names or []:
            argv += ["--scenario", name]
        subprocess.run(argv, cwd=neutral_cwd, env=env, check=True)
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force",
                        str(worktree)], check=False, capture_output=True)

    print(f"\n--- comparing working tree against {ref} ---")
    return compare(baseline, reproducibility, names, engine)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "compare", "against"):
        child = sub.add_parser(name)
        child.add_argument("--reproducibility", default=None, choices=sorted(REPRODUCIBILITY_FLAGS),
                           help="pass --reproducibility (on) or --reproducibility-max (max) "
                                "to the training commands (omit for neither)")
        child.add_argument("--scenario", action="append", dest="scenarios",
                           help="run only this scenario (repeatable)")
        child.add_argument("--engine", default="pytorch", choices=sorted(ENGINE_SUFFIXES),
                           help="deep-learning engine to run the scenarios with; "
                                "run from that engine's environment")
        if name == "capture":
            child.add_argument("--out", required=True, type=Path)
        elif name == "compare":
            child.add_argument("--baseline", required=True, type=Path)
        else:
            child.add_argument("--ref", default="main")
    args = parser.parse_args(argv)

    if args.command == "capture":
        capture(args.out, args.reproducibility, args.scenarios, args.engine)
        return 0
    if args.command == "compare":
        return compare(args.baseline, args.reproducibility, args.scenarios, args.engine)
    return against_ref(args.ref, args.reproducibility, args.scenarios, args.engine)


if __name__ == "__main__":
    raise SystemExit(main())
