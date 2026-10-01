"""Hashing, and human-readable explanations when a hash changes.

Detection is a SHA-256 of every produced file, because every artifact the CLI
writes -- ``.pt`` checkpoints, matplotlib PNGs, WeasyPrint PDFs, bigwigs, the
HTML reports -- is byte-identical across runs provided three things are held
fixed: the output *path* (several artifacts embed it), the device, and the
thread count. See ``tests/regression/README.md``.

HDF5 is the one exception. Its container layout is not a function of the
logical content: two write sequences that produce identical objects, dtypes,
chunking and values can still allocate metadata differently and differ by a few
hundred bytes. Hashing the raw file therefore reports changes that do not exist.
``.h5`` files get :func:`hdf5_digest` instead, which hashes the object tree and
the dataset values rather than the container.

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


def _code_behaviour(code) -> tuple:
    """A code object's instructions, constants and names -- not its origin."""
    import types

    consts = tuple(_code_behaviour(c) if isinstance(c, types.CodeType) else repr(c)
                   for c in code.co_consts)
    return (code.co_code.hex(), consts, code.co_names, code.co_varnames,
            code.co_argcount, code.co_freevars, code.co_cellvars)


def _normalize_keras_lambdas(text: str) -> str:
    """Keras model JSON with each ``Lambda``'s bytecode reduced to behaviour.

    Keras serializes a ``Lambda`` layer's function as marshalled bytecode,
    which embeds the absolute path and line number of the file that defined
    it. For the TensorFlow engine's bias-factorized model that file is in the
    package, so two checkouts of identical code would otherwise digest
    differently -- the HDF5 analogue of the package-root substitution that
    ``text_digest`` applies to ``args.json``.
    """
    import base64
    import marshal

    def walk(node):
        if isinstance(node, dict):
            if node.get("class_name") == "Lambda":
                config = dict(node.get("config", {}))
                function = config.get("function")
                if isinstance(function, list) and function and isinstance(function[0], str):
                    try:
                        code = marshal.loads(base64.decodebytes(function[0].encode("ascii")))
                        config["function"] = [repr(_code_behaviour(code))] + function[1:]
                    except (ValueError, EOFError, TypeError):
                        pass  # written by another Python version; leave as is
                node = dict(node, config=config)
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return json.dumps(walk(json.loads(text)), sort_keys=True)


def _attr_text(key: str, value) -> str:
    """How one HDF5 attribute enters the digest.

    Unchanged -- ``repr`` of the value -- for everything except a Keras
    ``model_config`` that contains a ``Lambda`` layer, so the digests of every
    PyTorch-engine artifact are exactly what they were.
    """
    if key == "model_config":
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        if isinstance(text, str) and '"class_name": "Lambda"' in text:
            return f"attr:{key}={_normalize_keras_lambdas(text)!r}"
    return f"attr:{key}={value!r}"


def hdf5_digest(path: Path) -> str:
    """Digest an HDF5 file's logical content, ignoring container layout.

    Covers the full object tree, each dataset's dtype, shape, chunking and
    compression settings, and the raw value bytes -- everything a consumer can
    observe -- but not where HDF5 chose to put its metadata.
    """
    import h5py
    import numpy as np

    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        names: list[str] = []
        handle.visit(names.append)
        for name in sorted(names):
            node = handle[name]
            digest.update(name.encode())
            for key in sorted(node.attrs):
                digest.update(_attr_text(key, node.attrs[key]).encode())
            if not isinstance(node, h5py.Dataset):
                digest.update(b"<group>")
                continue
            digest.update(
                f"|{node.dtype}|{node.shape}|{node.chunks}|"
                f"{node.compression}|{node.compression_opts}|".encode())
            values = node[()]
            if isinstance(values, np.ndarray) and values.dtype == object:
                # Variable-length strings: hash the decoded text.
                for item in values.ravel().tolist():
                    digest.update(repr(item).encode())
            else:
                digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


_CONTENT_DIGESTS = {".h5": hdf5_digest, ".hdf5": hdf5_digest}

#: Text artifacts that embed absolute paths and so need substitution.
_SUBSTITUTED_SUFFIXES = {".json", ".tsv", ".html"}


def text_digest(path: Path, substitutions: tuple[tuple[str, str], ...] = ()) -> str:
    """Digest a text artifact after rewriting environment-specific paths.

    ``<prefix>.args.json`` is a dump of the argparse namespace and records the
    absolute path of the architecture module, which legitimately differs
    between two checkouts of the same code. Substituting the package root keeps
    that from masquerading as an output change.
    """
    text = Path(path).read_text(errors="replace")
    for needle, placeholder in substitutions:
        text = text.replace(needle, placeholder)
    return hashlib.sha256(text.encode()).hexdigest()


def manifest(root: Path, ignore: tuple[str, ...] = (),
             substitutions: tuple[tuple[str, str], ...] = ()) -> dict[str, str]:
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
        suffix = path.suffix.lower()
        if suffix in _CONTENT_DIGESTS:
            entries[relative] = _CONTENT_DIGESTS[suffix](path)
        elif substitutions and suffix in _SUBSTITUTED_SUFFIXES:
            entries[relative] = text_digest(path, substitutions)
        else:
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
