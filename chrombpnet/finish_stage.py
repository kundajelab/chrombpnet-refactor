"""Writing each model's bigWig and metrics while that model predicts.

Adapted from the `bigwig_writer`, `metrics_stage` and pipelined-predict
levers of the ChromBPNet optimization kit at
github.com/anthropics/uplifting-biomolecular-modeling (Apache-2.0), which run
the writer and the metrics outside the forward's process and feed them each
forward chunk as it lands.

`pred_bw` predicts over the regions with up to three models, and each model's
outputs -- the bigWig, and with `-bw` the observed-signal comparison (the
predictions HDF5, the metrics JSON and two plots) -- are single-threaded CPU
work the size of the forward or larger. This module runs them in two forked
worker processes, one per output family, while the parent keeps the GPU:

  bigwig   the counts-scaled prediction bigWig and its stats file
           (bigwig_helper.BigWigWriter), written down the regions' sorted
           order as far as the arrived rows allow;
  metrics  the observed signal read and the shuffled baseline drawn as soon
           as the model's pass opens, the softmax per arriving chunk, then
           the HDF5, the metrics and the plots once the last chunk is in
           (predict_to_bigwig.ObservedComparison).

The parent predicts in chunks of whole batches and hands each one to both
workers (`FinishStage.open(...).add(...)`), so the work overlaps the same
model's forward. The previous version handed a model's finished predictions
to one worker in one piece, so model N's work overlapped model N+1's forward
and a single-model run did all of it after its forward, inline.

Threads cannot do this. `pyBigWig`'s extension never releases the GIL (its
`.so` exports no `PyEval_SaveThread`), so a writer thread would block the
main thread between CUDA calls.

**The workers must be created before the first CUDA context exists.** They
are forked, and forking a process that has initialized CUDA gives a child
whose CUDA state is unusable -- and, worse, one that can wedge the parent's
driver. `start()` is therefore called at the top of `predict_to_bigwig.main`,
before any model is loaded. The workers never touch the GPU themselves.

Outputs are unchanged: the same functions write the same files from the same
values, the chunks are the batches a single predict call makes, and each model
writes to its own prefix. A worker's exception reaches the caller from
`close()`, with the worker's traceback.
"""

from __future__ import annotations

import multiprocessing
import queue
import traceback

#: The worker processes, one per output family.
ROLES = ("bigwig", "metrics")

#: How often `close()` checks that a worker it is still waiting on is alive.
_POLL_SECONDS = 1.0


def _worker_init(role):
    """Load a worker's imports right after the fork.

    The workers are forked before CUDA exists and before any model is
    loaded, so the metrics worker inherits an interpreter without SciPy or
    Matplotlib. Importing them here (~6 s) lands while the parent is still
    loading the first model, instead of on the first job.
    """
    if role != "metrics":
        return
    try:
        import chrombpnet.training.metrics  # noqa: F401
    except Exception:
        # A prefetch must never be the thing that fails a run. The real
        # import in the comparison will raise with its own traceback.
        pass


class _BigWigTask:
    """One model's bigWig: the softmax times the counts, per arriving chunk."""

    def __init__(self, job):
        self._job = job
        self._writer = None

    def add(self, start, pred_logits, pred_logcts):
        import numpy as np

        from chrombpnet.evaluation.make_bigwigs import bigwig_helper
        from chrombpnet.predictions import softmax

        if self._writer is None:
            # Opened on the first rows, not with the pass, so a model that
            # fails its first forward (a profile-only bias model) leaves no
            # empty bigWig behind.
            job = self._job
            self._writer = bigwig_helper.BigWigWriter(
                job["regions"], job["chrom_sizes"], job["bigwig_out"],
                debug_chr=job["debug_chr"], use_tqdm=job["use_tqdm"],
                outstats_file=job["outstats_file"])
        # expm1, because the training target is log(1 + counts) (defect 4).
        self._writer.add(start, softmax(pred_logits) * np.expm1(pred_logcts))

    def close(self):
        if self._writer is None:
            raise ValueError(f"no predictions arrived for {self._job['bigwig_out']}")
        self._writer.close()


def _metrics_task(job, cache):
    """One model's observed comparison, or None without an observed bigWig.

    `cache` lives as long as the worker: the observed counts and the
    shuffled baseline depend only on the arguments, the model geometry and
    the seed, so with the usual three same-geometry models they are read and
    drawn once, not three times.
    """
    if not job["observed_bigwig"]:
        return None
    import chrombpnet.training.metrics as metrics
    from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import (
        ObservedComparison,
        read_observed_counts,
    )

    key = job["geometry"]
    if key not in cache:
        counts = read_observed_counts(job["observed_bigwig"], job["used_regions_df"],
                                      job["outputlen"])
        # Read-only: every model's comparison shares this array.
        counts.flags.writeable = False
        cache[key] = (counts, {})
    counts, baselines = cache[key]
    if job["seed"] not in baselines:
        baselines[job["seed"]] = metrics.ProfileBaseline(counts, seed=job["seed"])
    return ObservedComparison(
        job["observed_bigwig"], job["used_regions_df"], job["regions"], job["outputlen"],
        job["output_prefix"], obs_data=counts, seed=job["seed"],
        baseline=baselines[job["seed"]])


def _open_task(role, job, cache):
    if role == "bigwig":
        return _BigWigTask(job)
    return _metrics_task(job, cache)


class _Failed:
    def __init__(self, text):
        self.text = text


def _serve(role, inbox, results):
    """A worker's loop: open, rows and close messages until stop."""
    _worker_init(role)
    cache = {}
    tasks = {}
    while True:
        message = inbox.get()
        kind = message[0]
        if kind == "stop":
            break
        job_id = message[1]
        if kind == "open":
            try:
                tasks[job_id] = _open_task(role, message[2], cache)
            except BaseException:
                tasks[job_id] = _Failed(traceback.format_exc())
        elif kind == "rows":
            task = tasks[job_id]
            if task is not None and not isinstance(task, _Failed):
                try:
                    task.add(*message[2:])
                except BaseException:
                    tasks[job_id] = _Failed(traceback.format_exc())
        elif kind == "close":
            task = tasks.pop(job_id)
            if isinstance(task, _Failed):
                results.put((job_id, role, task.text))
                continue
            try:
                if task is not None:
                    task.close()
            except BaseException:
                results.put((job_id, role, traceback.format_exc()))
            else:
                results.put((job_id, role, None))


class _InlineStream:
    """A model's finish work run in this process, as each call arrives."""

    def __init__(self, job, cache):
        self._tasks = [task for task in (_open_task(role, job, cache) for role in ROLES)
                       if task is not None]

    def add(self, start, pred_logits, pred_logcts):
        for task in self._tasks:
            task.add(start, pred_logits, pred_logcts)

    def close(self):
        for task in self._tasks:
            task.close()


class _WorkerStream:
    """A model's finish work, sent to the workers."""

    def __init__(self, stage, job_id):
        self._stage = stage
        self._job_id = job_id

    def add(self, start, pred_logits, pred_logcts):
        """Predictions for regions ``start`` onward, in the regions' order.

        The arrays are pickled for the workers on the queues' feeder threads,
        after this returns, so the caller must not modify them.
        """
        self._stage._send(("rows", self._job_id, start, pred_logits, pred_logcts))

    def close(self):
        """No more rows for this model. Returns at once; the work finishes in
        the workers, and `FinishStage.close` waits for it."""
        self._stage._send(("close", self._job_id))
        self._stage._closed[self._job_id] = self._stage._names[self._job_id]


class FinishStage:
    """Runs each model's finish work in the workers, or inline.

    Inline (``enabled=False``) runs the same tasks in this process as the
    rows arrive; the tests use it to check that the workers change nothing.
    """

    def __init__(self, enabled=True):
        self._workers = {}
        self._results = None
        self._cache = {}
        self._names = {}
        self._closed = {}
        if enabled:
            # "fork" deliberately, and only before CUDA is initialized: it is
            # the cheapest start method and the workers inherit the already
            # imported numpy/pandas. "spawn" would re-import everything in
            # each worker, which is most of what we are trying to avoid.
            context = multiprocessing.get_context("fork")
            self._results = context.Queue()
            for role in ROLES:
                inbox = context.Queue()
                process = context.Process(
                    target=_serve, args=(role, inbox, self._results),
                    name=f"chrombpnet-finish-{role}", daemon=True)
                process.start()
                self._workers[role] = (process, inbox)

    def open(self, job):
        """Start one model's finish work; returns a stream to feed its rows."""
        job_id = len(self._names)
        self._names[job_id] = job.get("output_prefix") or job.get("bigwig_out")
        if not self._workers:
            return _InlineStream(job, self._cache)
        self._send(("open", job_id, job))
        return _WorkerStream(self, job_id)

    def _send(self, message):
        for _, inbox in self._workers.values():
            inbox.put(message)

    def close(self):
        """Wait for every closed model's work and re-raise the first failure.

        A worker exception must reach the caller: silently dropping one would
        leave a run that looks successful with a missing or truncated bigWig.
        """
        if not self._workers:
            return
        try:
            self._send(("stop",))
            waiting = {(job_id, role) for job_id in self._closed for role in ROLES}
            failures = []
            dead_polls = 0
            while waiting:
                try:
                    job_id, role, error = self._results.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    dead = [role for role, (process, _) in self._workers.items()
                            if not process.is_alive() and any(r == role for _, r in waiting)]
                    # One more poll first: a worker that exits right after its
                    # last result can be seen dead before that result is read.
                    dead_polls = dead_polls + 1 if dead else 0
                    if dead_polls > 1:
                        process = self._workers[dead[0]][0]
                        raise RuntimeError(
                            f"the {dead[0]} finish worker exited (code {process.exitcode}) "
                            f"before finishing its work")
                    continue
                dead_polls = 0
                waiting.discard((job_id, role))
                if error is not None:
                    failures.append(
                        f"{role} work for {self._names.get(job_id)} failed in its worker:\n{error}")
            if failures:
                raise RuntimeError("\n".join(failures))
            for process, _ in self._workers.values():
                process.join()
        finally:
            self._stop_workers()

    def _stop_workers(self):
        for process, inbox in self._workers.values():
            if process.is_alive():
                process.terminate()
            process.join()
            # Do not let interpreter exit block on rows still queued for a
            # worker that is gone.
            inbox.cancel_join_thread()
            inbox.close()
        self._workers = {}
        if self._results is not None:
            self._results.close()
            self._results = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # The run is already failing; do not mask it by waiting on work
            # that may never complete.
            self._stop_workers()
            return False
        self.close()
        return False


def start():
    """A FinishStage with its worker processes, forked now."""
    return FinishStage(enabled=True)
