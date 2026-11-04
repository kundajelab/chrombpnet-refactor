"""glibc memory-allocator settings for ``pred_bw``.

The ``malloc_env`` lever of the ChromBPNet optimization kit at
github.com/anthropics/uplifting-biomolecular-modeling (Apache-2.0;
``chrombpnet/opt/kit_ho/tf/chrombpnet_fastkit/malloc_env.py``), with the
kit's three values:

    M_MMAP_MAX        0        large blocks come from the heap instead of
                               fresh mmap'd pages, so a new array reuses
                               memory a freed one left
    M_TRIM_THRESHOLD  -1       freed memory is kept for reuse, never handed
                               back to the system
    M_TOP_PAD         512 MiB  the heap grows 512 MiB at a time

They are applied to this process with ``mallopt``, as the kit does, and
exported as ``MALLOC_MMAP_MAX_``, ``MALLOC_TRIM_THRESHOLD_`` and
``MALLOC_TOP_PAD_``, which glibc reads when a process starts, so a child
started by exec gets them too; ``pred_bw``'s forked side processes inherit
the settings directly. They change how host memory is allocated, never what
is computed, and do not touch GPU memory (CUDA's and the frameworks' own
allocators).

What differs from the kit: a setting already in the environment is left as
the user set it, and where ``mallopt`` is unavailable (a C library other than
glibc) nothing is applied and ``pred_bw`` runs as it otherwise would; the kit
refuses its mode in both cases.

Measured for blog 2026-015 (``pred_bw`` with three models on 20,000 peaks):
on an L40S 76.3 s -> 71.2 s, and the largest process's peak RAM 2.20 ->
2.39 GB; on an H200 66 s -> 59 s. A CPU-only run (2,000 peaks) ran about
twice as fast but held about twice the RAM, since there every activation
lives in host memory.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys

#: Environment name -> (glibc ``mallopt`` parameter, value). The parameter
#: numbers are glibc's (malloc.h): M_TRIM_THRESHOLD -1, M_TOP_PAD -2,
#: M_MMAP_MAX -4.
SETTINGS = {
    "MALLOC_MMAP_MAX_": (-4, 0),
    "MALLOC_TRIM_THRESHOLD_": (-1, -1),
    "MALLOC_TOP_PAD_": (-2, 512 << 20),
}


def _mallopt():
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        mallopt = libc.mallopt
    except (OSError, AttributeError):
        return None
    mallopt.argtypes = (ctypes.c_int, ctypes.c_int)
    mallopt.restype = ctypes.c_int
    return mallopt


def apply():
    """Apply the settings to this process and export them for its children.

    Returns ``{environment name: value}`` for each setting applied: none
    where ``mallopt`` is unavailable, and none for a setting the environment
    already holds (glibc applied the user's own value at start).
    """
    mallopt = _mallopt()
    if mallopt is None:
        return {}
    applied = {}
    for name, (parameter, value) in SETTINGS.items():
        if name in os.environ:
            continue
        if mallopt(parameter, value) == 1:
            os.environ[name] = str(value)
            applied[name] = value
    return applied
