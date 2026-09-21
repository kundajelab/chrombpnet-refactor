"""Running each model's bigWig write and metrics while the next one predicts.

Adapted from the `bigwig_writer` / `metrics_stage` levers of the ChromBPNet
optimization kit at github.com/anthropics/uplifting-biomolecular-modeling
(Apache-2.0), which run the writer and the metrics outside the forward's
process so the two overlap.

`pred_bw` runs up to three models strictly in sequence: predict, write the
bigWig, compute the metrics, then move to the next model. Profiled on 5,000
regions with three models, the write is 14.4s and the metrics 8.7s against
9.3s of forward -- so every second of GPU work has roughly two and a half
seconds of single-threaded CPU work sitting behind it, and neither overlaps.

A worker process fixes that: the forward stays in the parent, which owns the
GPU, and the finish work for model N runs alongside the forward for N+1.

Threads cannot do this. `pyBigWig`'s extension never releases the GIL (its
`.so` exports no `PyEval_SaveThread`), so a writer thread would block the
main thread between CUDA calls.

**The pool must be created before the first CUDA context exists.** These
workers are forked, and forking a process that has initialized CUDA gives a
child whose CUDA state is unusable -- and, worse, one that can wedge the
parent's driver. `start()` is therefore called at the top of
`predict_to_bigwig.main`, before any model is loaded. The workers never touch
the GPU themselves.

Outputs are unchanged: the same functions write the same files from the same
arguments, and each model writes to its own prefix, so there is no ordering
interaction between them.
"""

from __future__ import annotations

import multiprocessing


def _worker_init():
    """Load the finish stage's imports in the worker, right after the fork.

    The pool is forked before CUDA exists, which is also before the parent's
    `preimport` thread has run, so the worker inherits an interpreter with
    neither SciPy nor Matplotlib loaded. Without this the worker paid that
    ~6s import on its first job -- on the critical path, since the last job
    has no forward left to hide behind, which made the "overlapped" version
    slower than the serial one.

    Running it here means the worker imports while the parent is still
    loading the first model and running its forward.
    """
    try:
        import chrombpnet.training.metrics  # noqa: F401
    except Exception:
        # Same reasoning as chrombpnet.preimport: a prefetch must never be
        # the thing that fails a run. The real import in _run_finish will
        # raise with its own traceback.
        pass


def _run_finish(payload):
    """Write one model's bigWig and, if asked, its metrics. Runs in a worker.

    Imported lazily inside the worker so the parent never pays for SciPy or
    Matplotlib on its own critical path.
    """
    import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
    from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import (
        compare_with_observed,
    )

    bigwig_helper.write_bigwig(
        payload["data"], payload["regions"], payload["chrom_sizes"],
        payload["bigwig_out"], outstats_file=payload["outstats_file"],
        debug_chr=payload["debug_chr"], use_tqdm=payload["use_tqdm"])

    if payload["observed_bigwig"]:
        compare_with_observed(
            payload["observed_bigwig"], payload["used_regions_df"],
            payload["regions"], payload["outputlen"], payload["pred_logits"],
            payload["pred_logcts"], payload["output_prefix"],
            obs_data=payload["obs_data"])
    return payload["output_prefix"]


class FinishStage:
    """Dispatches finish work to a worker, or runs it inline.

    Inline is the fallback and the single-model case: with one model there is
    no following forward to hide the work behind, so a worker would only add
    serialization cost.
    """

    def __init__(self, enabled):
        self._pending = []
        self._pool = None
        if enabled:
            # "fork" deliberately, and only before CUDA is initialized: it is
            # the cheapest start method and the workers inherit the already
            # imported numpy/pandas. "spawn" would re-import everything in
            # each worker, which is most of what we are trying to avoid.
            context = multiprocessing.get_context("fork")
            self._pool = context.Pool(processes=1, initializer=_worker_init)

    def submit(self, payload):
        if self._pool is None:
            _run_finish(payload)
        else:
            self._pending.append(self._pool.apply_async(_run_finish, (payload,)))

    def close(self):
        """Wait for every submitted job and re-raise the first failure.

        A worker exception must reach the caller: silently dropping one would
        leave a run that looks successful with a missing or truncated bigWig.
        """
        try:
            for result in self._pending:
                result.get()
        finally:
            self._pending = []
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # The run is already failing; do not mask it by waiting on work
            # that may never complete.
            if self._pool is not None:
                self._pool.terminate()
                self._pool.join()
                self._pool = None
            return False
        self.close()
        return False


def start(model_count):
    """A FinishStage that uses a worker only when there is work to hide it behind."""
    return FinishStage(enabled=model_count > 1)
