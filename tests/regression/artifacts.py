"""Hashing, and human-readable explanations when a hash changes.

Detection is a plain SHA-256 of every produced file. That is sufficient because
every artifact the CLI writes -- including ``.pt`` checkpoints, matplotlib PNGs,
WeasyPrint PDFs, bigwigs and the HTML reports -- is byte-identical across runs
provided three things are held fixed: the output *path* (several artifacts embed
it), the device, and the thread count. See ``tests/regression/README.md``.

A hash tells you *that* something changed but not *what*, so when one differs
:func:`explain` reopens both copies with a format-aware reader and reports the
first meaningful difference -- which HDF5 dataset, which tensor, the max
absolute error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root: Path, ignore: tuple[str, ...] = ()) -> dict[str, str]:
    """Map every file under ``root`` to its digest, keyed by relative path."""
    root = Path(root)
    skip = set(ignore)
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in skip or any(part == "__pycache__" for part in path.parts):
            continue
        entries[relative] = sha256(path)
    return entries


def write_manifest(path: Path, entries: dict[str, dict[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------

def _explain_text(a: Path, b: Path) -> str:
    import difflib

    left = a.read_text(errors="replace").splitlines()
    right = b.read_text(errors="replace").splitlines()
    diff = list(difflib.unified_diff(left, right, "baseline", "current", n=1))
    return "\n".join(diff[:40]) or "no textual difference (trailing bytes?)"


def _explain_json(a: Path, b: Path) -> str:
    left, right = json.loads(a.read_text()), json.loads(b.read_text())
    if left == right:
        return "parsed JSON is equal; only formatting differs"
    lines = []

    def walk(x, y, prefix=""):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(x.get(key), y.get(key), f"{prefix}.{key}" if prefix else key)
        elif x != y:
            lines.append(f"  {prefix}: baseline={x!r} current={y!r}")

    walk(left, right)
    return "\n".join(lines[:40])


def _explain_hdf5(a: Path, b: Path) -> str:
    import h5py
    import numpy as np

    lines = []
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        names_a, names_b = [], []
        fa.visit(names_a.append)
        fb.visit(names_b.append)
        only_a = sorted(set(names_a) - set(names_b))
        only_b = sorted(set(names_b) - set(names_a))
        if only_a:
            lines.append(f"  only in baseline: {only_a[:10]}")
        if only_b:
            lines.append(f"  only in current:  {only_b[:10]}")
        for name in sorted(set(names_a) & set(names_b)):
            if not isinstance(fa[name], h5py.Dataset):
                continue
            left, right = fa[name][()], fb[name][()]
            if getattr(left, "shape", None) != getattr(right, "shape", None):
                lines.append(f"  {name}: shape {left.shape} -> {right.shape}")
                continue
            try:
                if np.array_equal(left, right):
                    continue
                error = np.nanmax(np.abs(np.asarray(left, dtype="float64")
                                         - np.asarray(right, dtype="float64")))
                lines.append(f"  {name}: differs, max|delta| = {error:.6g}")
            except (TypeError, ValueError):
                lines.append(f"  {name}: differs (non-numeric)")
    return "\n".join(lines[:40]) or "  datasets equal; container metadata differs"


def _explain_checkpoint(a: Path, b: Path) -> str:
    import torch

    left = torch.load(str(a), map_location="cpu", weights_only=False)
    right = torch.load(str(b), map_location="cpu", weights_only=False)
    lines = []
    for key in ("format_version", "model_type", "config", "bias_config", "provenance"):
        if left.get(key) != right.get(key):
            lines.append(f"  {key}: baseline={left.get(key)!r} current={right.get(key)!r}")
    sa, sb = left.get("state_dict", {}), right.get("state_dict", {})
    only = sorted(set(sa) ^ set(sb))
    if only:
        lines.append(f"  state_dict keys differ: {only[:10]}")
    for key in sorted(set(sa) & set(sb)):
        if sa[key].shape != sb[key].shape:
            lines.append(f"  {key}: shape {tuple(sa[key].shape)} -> {tuple(sb[key].shape)}")
        elif not torch.equal(sa[key], sb[key]):
            error = (sa[key].double() - sb[key].double()).abs().max().item()
            lines.append(f"  {key}: differs, max|delta| = {error:.6g}")
    return "\n".join(lines[:40]) or "  payloads equal; serialization differs"


def _explain_bigwig(a: Path, b: Path) -> str:
    import numpy as np
    import pyBigWig

    lines = []
    ha, hb = pyBigWig.open(str(a)), pyBigWig.open(str(b))
    try:
        if ha.chroms() != hb.chroms():
            return f"  headers differ: {ha.chroms()} -> {hb.chroms()}"
        for chrom, length in sorted(ha.chroms().items()):
            left = np.nan_to_num(np.asarray(ha.values(chrom, 0, length)))
            right = np.nan_to_num(np.asarray(hb.values(chrom, 0, length)))
            if not np.array_equal(left, right):
                error = np.nanmax(np.abs(left - right))
                count = int((left != right).sum())
                lines.append(f"  {chrom}: {count} positions differ, max|delta| = {error:.6g}")
    finally:
        ha.close()
        hb.close()
    return "\n".join(lines[:40]) or "  values equal; header metadata differs"


def _explain_image(a: Path, b: Path) -> str:
    import numpy as np
    from PIL import Image

    left = np.asarray(Image.open(a).convert("RGBA"), dtype="int16")
    right = np.asarray(Image.open(b).convert("RGBA"), dtype="int16")
    if left.shape != right.shape:
        return f"  image shape {left.shape} -> {right.shape}"
    if np.array_equal(left, right):
        return "  pixels equal; only PNG metadata differs"
    changed = int((left != right).any(axis=-1).sum())
    return (f"  {changed} pixels differ, max channel delta = "
            f"{int(np.abs(left - right).max())}")


def _explain_pdf(a: Path, b: Path) -> str:
    # Avoid a parser dependency: PDFs here embed the absolute output path, so a
    # size/page-count summary plus the byte delta is the useful signal.
    pages_a = a.read_bytes().count(b"/Type /Page")
    pages_b = b.read_bytes().count(b"/Type /Page")
    return (f"  size {a.stat().st_size} -> {b.stat().st_size} bytes, "
            f"page markers {pages_a} -> {pages_b}")


_EXPLAINERS = {
    ".h5": _explain_hdf5,
    ".hdf5": _explain_hdf5,
    ".pt": _explain_checkpoint,
    ".bw": _explain_bigwig,
    ".png": _explain_image,
    ".pdf": _explain_pdf,
    ".json": _explain_json,
}
_TEXT_SUFFIXES = {".bed", ".tsv", ".txt", ".log", ".batch", ".html", ".csv", ".fa",
                  ".sizes", ".narrowPeak"}


def explain(baseline: Path, current: Path) -> str:
    """Describe how two copies of the same artifact differ."""
    baseline, current = Path(baseline), Path(current)
    if not baseline.exists():
        return "  missing from baseline"
    if not current.exists():
        return "  missing from current run"
    explainer = _EXPLAINERS.get(baseline.suffix.lower())
    if explainer is None and (baseline.suffix.lower() in _TEXT_SUFFIXES
                              or baseline.name.endswith(".log.batch")):
        explainer = _explain_text
    if explainer is None:
        return (f"  binary difference: {baseline.stat().st_size} -> "
                f"{current.stat().st_size} bytes")
    try:
        return explainer(baseline, current)
    except Exception as error:  # pragma: no cover - diagnostics must not mask
        return f"  could not explain ({type(error).__name__}: {error})"
