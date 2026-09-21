"""Skipping interpreter and CUDA teardown once the outputs are durable.

The `exit_fast` lever of the ChromBPNet optimization kit at
github.com/anthropics/uplifting-biomolecular-modeling (Apache-2.0): "after
every output is closed, re-opened non-empty and synced and the side processes
are joined, `os._exit(0)` skips interpreter and CUDA teardown; failure paths
exit normally."

Teardown is about 1.2s of a 43s `pred_bw` run -- freeing a few GB of CUDA
allocations, running finalizers, unloading the driver. None of it affects
what was written.

`os._exit` is genuinely dangerous: it skips `atexit` handlers and, more to
the point, it does not flush Python's own buffers. A file left in a buffer
would simply not exist. So this module never guesses -- it is given the list
of outputs the run promised, and it checks each one is present, non-empty and
`fsync`ed before it will skip anything. If a check fails it returns and the
interpreter exits normally, paying the teardown and running every finalizer.
"""

from __future__ import annotations

import os
import sys


def _is_durable(path):
    """True once `path` exists, is non-empty, and its bytes are on disk."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
        handle = os.open(path, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
        return True
    except OSError:
        return False


def exit_now(outputs, status=0):
    """Verify `outputs`, then `os._exit(status)`. Returns if it cannot.

    Returning is the safe outcome: the caller falls off the end of `main`
    and the interpreter shuts down normally.
    """
    if status != 0:
        return False
    outputs = [path for path in outputs if path]
    if not outputs:
        return False

    # Flush anything this process still holds before inspecting the files;
    # os._exit would discard it.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        return False

    if not all(_is_durable(path) for path in outputs):
        return False

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
