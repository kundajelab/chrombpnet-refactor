"""Loading the finish stage's imports while the GPU is still busy.

Adapted from the ``preimport`` lever of the ChromBPNet optimization kit at
github.com/anthropics/uplifting-biomolecular-modeling (Apache-2.0).

``pred_bw`` pays about six seconds importing SciPy and Matplotlib, and it pays
them at the worst possible moment: ``compare_with_observed`` imports
``chrombpnet.training.metrics`` lazily, so the cost lands *after* the forward
and the bigWig write, on the critical path, with the GPU idle.

Starting those imports on a daemon thread up front moves them under the
forward. The overlap is real despite the GIL because the main thread spends
the forward blocked in CUDA calls that release it.

This changes no output. Imports have no observable effect beyond their own
side effects, and the one side effect that matters -- ``metrics`` selecting
Matplotlib's Agg backend at import time -- happens either way, just earlier.
"""

from __future__ import annotations

import threading

#: What the finish stage of a `pred_bw` run with `-bw` ends up importing.
#: `chrombpnet.training.metrics` pulls in scipy.stats and matplotlib.pyplot,
#: which together are the bulk of it.
FINISH_STAGE_MODULES = ("chrombpnet.training.metrics",)


def start(module_names=FINISH_STAGE_MODULES):
    """Import ``module_names`` on a daemon thread; return the thread.

    Failures are swallowed deliberately. This is a prefetch: if a module
    cannot be imported here it will fail again, with its own traceback, at
    the point that actually needs it. Raising from a background thread would
    only obscure that.
    """
    def _load():
        import importlib

        for name in module_names:
            try:
                importlib.import_module(name)
            except Exception:
                pass

    thread = threading.Thread(target=_load, name="chrombpnet-preimport",
                              daemon=True)
    thread.start()
    return thread
