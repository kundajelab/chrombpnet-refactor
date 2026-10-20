"""Maximum compression for the HDF5 files chrombpnet writes.

Every dataset is gzip level 9 (h5py's ``compression="gzip"`` alone is level
4). Numeric datasets also go behind the byte-shuffle filter, which groups the
bytes of each value so floats compress far better; strings get gzip only.
Scalars and empty datasets cannot be chunked, so they stay as they are. Any
HDF5 reader decompresses these transparently, PyTables included.

``recompress_h5`` rewrites a file someone else wrote (TF-MoDISco's results)
the same way, and replaces it only after every dataset reads back identical.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

GZIP_LEVEL = 9


def h5_filters(data, dtype=None):
    """Keyword arguments for ``create_dataset``: gzip 9, plus shuffle for numbers."""
    array = np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
    if array.ndim == 0 or array.size == 0:
        return {}
    filters = {"compression": "gzip", "compression_opts": GZIP_LEVEL}
    if array.dtype.kind in "biuf":
        filters["shuffle"] = True
    return filters


def _copy_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _datasets(path):
    values = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, node: values.__setitem__(name, node[()])
                          if isinstance(node, h5py.Dataset) else None)
    return values


def _identical(first, second):
    if first.keys() != second.keys():
        return False
    return all(np.array_equal(first[key], second[key]) for key in first)


def recompress_h5(path):
    """Rewrite ``path`` with ``h5_filters`` on every dataset; returns (old, new) bytes."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with h5py.File(path, "r") as source, h5py.File(temporary, "w") as target:
        _copy_attrs(source, target)

        def copy(name, node):
            if isinstance(node, h5py.Group):
                _copy_attrs(node, target.require_group(name))
                return
            data = node[()]
            created = target.create_dataset(name, data=data, dtype=node.dtype,
                                            **h5_filters(data, node.dtype))
            _copy_attrs(node, created)

        source.visititems(copy)
    if not _identical(_datasets(path), _datasets(temporary)):
        temporary.unlink()
        raise RuntimeError(f"Recompressed {path} does not read back identical; left unchanged")
    before = path.stat().st_size
    os.replace(temporary, path)
    return before, path.stat().st_size
