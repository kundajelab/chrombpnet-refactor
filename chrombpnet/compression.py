"""Maximum compression for the HDF5 files chrombpnet writes.

Every dataset is gzip level 9 (h5py's ``compression="gzip"`` alone is level
4). Numeric datasets also go behind the byte-shuffle filter, which groups the
bytes of each value so floats compress far better; strings get gzip only.
Scalars and empty datasets cannot be chunked, so they stay as they are. Any
HDF5 reader decompresses these transparently, PyTables included.

``create_dataset`` writes a dataset that way. Large numeric ones are filtered
chunk by chunk on a thread pool (the optimization kit's ``h5_writer``, at this
package's level and filters): gzip level 9 on one thread took ~35 s per
20,000-region predictions file, most of ``pred_bw``'s finish stage.

``recompress_h5`` rewrites a file someone else wrote (TF-MoDISco's results)
the same way, and replaces it only after every dataset reads back identical.
"""

from __future__ import annotations

import itertools
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

GZIP_LEVEL = 9

#: Numeric datasets at least this large are compressed on threads; below it,
#: thread start-up costs more than it saves.
THREADED_MIN_BYTES = 8 << 20

#: Upper bound on compression threads (the CPUs this process may run on caps
#: it further).
MAX_THREADS = 8


def h5_filters(data, dtype=None):
    """Keyword arguments for ``create_dataset``: gzip 9, plus shuffle for numbers."""
    array = np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
    if array.ndim == 0 or array.size == 0:
        return {}
    filters = {"compression": "gzip", "compression_opts": GZIP_LEVEL}
    if array.dtype.kind in "biuf":
        filters["shuffle"] = True
    return filters


def _threads():
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(MAX_THREADS, available))


def create_dataset(group, name, data, dtype=None):
    """``group.create_dataset(name, data=data, dtype=dtype, **h5_filters(...))``.

    Numeric data of at least ``THREADED_MIN_BYTES`` gets the same dataset --
    h5py's own chunk shape, fill value, shuffle and gzip filters -- but each
    chunk goes through those filters (the byte shuffle, then deflate) on a
    thread pool, as HDF5's filter pipeline would, and is written as is with
    ``write_direct_chunk``. zlib releases the GIL, so the chunks compress in
    parallel. Readers see the same dataset; only where the chunks sit in the
    file can differ.
    """
    array = np.asarray(data) if dtype is None else np.asarray(data, dtype=dtype)
    filters = h5_filters(array)
    if not filters or array.dtype.kind not in "biuf" or array.nbytes < THREADED_MIN_BYTES:
        return group.create_dataset(name, data=data, dtype=dtype, **filters)
    dataset = group.create_dataset(name, shape=array.shape, dtype=array.dtype, **filters)
    chunks = dataset.chunks
    shuffle = bool(filters.get("shuffle")) and array.dtype.itemsize > 1

    def encode(offset):
        block = array[tuple(slice(o, o + c) for o, c in zip(offset, chunks))]
        if block.shape != chunks:
            # An edge chunk: HDF5 stores the whole chunk, the fill value (0)
            # beyond the dataset's edge.
            full = np.zeros(chunks, dtype=array.dtype)
            full[tuple(slice(0, n) for n in block.shape)] = block
            block = full
        raw = np.ascontiguousarray(block)
        if shuffle:
            # HDF5's shuffle filter: byte 0 of every value, then byte 1, ...
            raw = np.ascontiguousarray(raw.view(np.uint8).reshape(-1, array.dtype.itemsize).T)
        return zlib.compress(raw, GZIP_LEVEL)

    offsets = list(itertools.product(*(range(0, n, c) for n, c in zip(array.shape, chunks))))
    with ThreadPoolExecutor(_threads()) as pool:
        for offset, blob in zip(offsets, pool.map(encode, offsets)):
            dataset.id.write_direct_chunk(offset, blob, 0)
    return dataset


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
            created = create_dataset(target, name, data, node.dtype)
            _copy_attrs(node, created)

        source.visititems(copy)
    if not _identical(_datasets(path), _datasets(temporary)):
        temporary.unlink()
        raise RuntimeError(f"Recompressed {path} does not read back identical; left unchanged")
    before = path.stat().st_size
    os.replace(temporary, path)
    return before, path.stat().st_size
