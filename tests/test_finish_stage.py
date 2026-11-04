"""The finish stage must produce identical files whether inline or in its
workers, and whether the predictions arrive whole or in chunks."""

import hashlib
import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from chrombpnet import finish_stage
from chrombpnet.evaluation.make_bigwigs import bigwig_helper
from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import compare_with_observed

WIDTH = 100


def _regions(order="sorted"):
    # Neighbours overlap by 30 bases, so the midpoint rule runs, and every
    # third region is on chr2, so a sorted write interleaves the two.
    regions = [["chr2" if index % 3 == 0 else "chr1", start, start + WIDTH, start + WIDTH // 2]
               for index, start in enumerate(range(1000, 21000, 70))]
    if order == "sorted":
        regions.sort(key=lambda r: (r[0] != "chr1", r[1]))
    else:
        np.random.default_rng(3).shuffle(regions)
    return regions


def _predictions(count, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(count, WIDTH)).astype(np.float32)
    logcts = rng.uniform(1, 6, size=(count, 1)).astype(np.float32)
    return logits, logcts


def _observed(regions, seed=1):
    counts = np.random.default_rng(seed).poisson(2.0, size=(len(regions), WIDTH)).astype(float)
    counts[3] = 0.0          # a region whose observed profile cannot normalize
    return counts


def _job(tmp_path, name, regions, observed=None):
    frame = pd.DataFrame({"chr": [r[0] for r in regions],
                          "start": [r[1] for r in regions],
                          "summit": [WIDTH // 2] * len(regions)})
    return {
        "name": name,
        "regions": regions,
        "chrom_sizes": [("chr1", 1_000_000), ("chr2", 1_000_000)],
        "bigwig_out": str(tmp_path / f"{name}.bw"),
        "outstats_file": str(tmp_path / f"{name}_stats.txt"),
        "debug_chr": None,
        "use_tqdm": 0,
        "observed_bigwig": "observed.bw" if observed is not None else None,
        "used_regions_df": frame,
        "geometry": (2 * WIDTH, WIDTH),
        "outputlen": WIDTH,
        "output_prefix": str(tmp_path / name),
        "seed": 5,
    }


@pytest.fixture
def observed_reads(monkeypatch):
    """Serve the observed counts without a bigWig (the workers are forked
    after this patch, so they see it too)."""
    from chrombpnet.evaluation.make_bigwigs import predict_to_bigwig

    table = {}

    def read(bigwig, regions_df, outputlen):
        return table["counts"].copy()

    monkeypatch.setattr(predict_to_bigwig, "read_observed_counts", read)
    return table


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _h5_content(path):
    with h5py.File(path, "r") as f:
        return {name: f[name][()].tobytes() if f[name].dtype.kind != "O" else list(f[name][()])
                for name in ("coords/coords_chrom", "coords/coords_center",
                             "predictions/profs", "predictions/logcounts")}


def _outputs(tmp_path, name, with_metrics):
    found = {
        "bw": _digest(tmp_path / f"{name}.bw"),
        "stats": (tmp_path / f"{name}_stats.txt").read_text(),
    }
    if with_metrics:
        found["h5"] = _h5_content(tmp_path / f"{name}_predictions.h5")
        found["json"] = (tmp_path / f"{name}_metrics.json").read_text()
        for png in (".counts_pearsonr.png", ".profile_jsd.png"):
            found[png] = _digest(tmp_path / f"{name}{png}")
    return found


def _stream(stage, job, logits, logcts, chunk):
    stream = stage.open(job)
    for start in range(0, len(logits), chunk):
        stream.add(start, logits[start:start + chunk], logcts[start:start + chunk])
    stream.close()


def _one_shot(tmp_path, name, regions, logits, logcts, observed):
    """The pre-streaming finish: the whole arrays, one write, one comparison."""
    from chrombpnet.predictions import softmax

    job = _job(tmp_path, name, regions, observed)
    bigwig_helper.write_bigwig(softmax(logits) * np.expm1(logcts), regions,
                               job["chrom_sizes"], job["bigwig_out"],
                               outstats_file=job["outstats_file"])
    if observed is not None:
        compare_with_observed(None, None, regions, WIDTH, logits, logcts,
                              job["output_prefix"], obs_data=observed, seed=job["seed"])


@pytest.mark.parametrize("order", ["sorted", "shuffled"])
@pytest.mark.parametrize("with_metrics", [False, True])
def test_workers_and_chunks_change_no_output(tmp_path, observed_reads, order, with_metrics):
    regions = _regions(order)
    logits, logcts = _predictions(len(regions))
    observed = _observed(regions) if with_metrics else None
    observed_reads["counts"] = observed

    _one_shot(tmp_path, "whole", regions, logits, logcts, observed)
    with finish_stage.FinishStage(enabled=False) as inline:
        _stream(inline, _job(tmp_path, "inline", regions, observed), logits, logcts, chunk=7)
    with finish_stage.FinishStage(enabled=True) as workers:
        _stream(workers, _job(tmp_path, "workers", regions, observed), logits, logcts, chunk=16)

    expected = _outputs(tmp_path, "whole", with_metrics)
    assert _outputs(tmp_path, "inline", with_metrics) == expected
    assert _outputs(tmp_path, "workers", with_metrics) == expected


def test_several_models_share_the_observed_read(tmp_path, observed_reads):
    """The metrics worker reads the observed counts once per geometry."""
    regions = _regions()
    observed_reads["counts"] = _observed(regions)
    calls = []
    real = observed_reads["counts"]
    from chrombpnet.evaluation.make_bigwigs import predict_to_bigwig

    def counting(bigwig, regions_df, outputlen):
        calls.append(1)
        return real.copy()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(predict_to_bigwig, "read_observed_counts", counting)
        with finish_stage.FinishStage(enabled=False) as stage:
            for name, seed in (("a", 0), ("b", 1), ("c", 2)):
                logits, logcts = _predictions(len(regions), seed)
                _stream(stage, _job(tmp_path, name, regions, real), logits, logcts, chunk=32)
    assert len(calls) == 1
    for name in "abc":
        assert json.loads((tmp_path / f"{name}_metrics.json").read_text())


def test_a_worker_failure_reaches_the_caller(tmp_path):
    """Silently dropping one would leave a run that looks successful with a
    missing bigwig."""
    regions = _regions()
    logits, logcts = _predictions(len(regions))
    job = _job(tmp_path, "bad", regions)
    job["chrom_sizes"] = []          # addHeader with no chromosomes
    with pytest.raises(RuntimeError, match="bigwig work for .*bad failed"):
        with finish_stage.FinishStage(enabled=True) as stage:
            _stream(stage, job, logits, logcts, chunk=16)


def test_missing_rows_are_an_error(tmp_path):
    regions = _regions()
    logits, logcts = _predictions(len(regions))
    with pytest.raises(RuntimeError, match="never arrived"):
        with finish_stage.FinishStage(enabled=True) as stage:
            stream = stage.open(_job(tmp_path, "short", regions))
            stream.add(0, logits[:10], logcts[:10])
            stream.close()


def test_an_error_in_the_parent_does_not_hang_on_the_workers(tmp_path):
    """The context manager must not wait on queued work while unwinding."""
    regions = _regions()
    logits, logcts = _predictions(len(regions))
    with pytest.raises(ValueError, match="parent failed"):
        with finish_stage.FinishStage(enabled=True) as stage:
            stream = stage.open(_job(tmp_path, "queued", regions))
            stream.add(0, logits, logcts)
            raise ValueError("parent failed")


def test_a_dead_worker_is_reported_not_waited_on(tmp_path):
    regions = _regions()
    logits, logcts = _predictions(len(regions))
    stage = finish_stage.FinishStage(enabled=True)
    try:
        stream = stage.open(_job(tmp_path, "orphan", regions))
        stream.add(0, logits, logcts)
        process = stage._workers["bigwig"][0]
        process.kill()
        process.join()
        stream.close()
        with pytest.raises(RuntimeError, match="bigwig finish worker exited"):
            stage.close()
    finally:
        stage._stop_workers()


def test_close_is_idempotent(tmp_path):
    regions = _regions()
    logits, logcts = _predictions(len(regions))
    stage = finish_stage.FinishStage(enabled=True)
    _stream(stage, _job(tmp_path, "once", regions), logits, logcts, chunk=64)
    stage.close()
    stage.close()
    assert os.path.getsize(tmp_path / "once.bw") > 0


def test_pred_bw_always_has_workers():
    """With one model the work still overlaps that model's own forward."""
    with finish_stage.start() as stage:
        assert set(stage._workers) == set(finish_stage.ROLES)
