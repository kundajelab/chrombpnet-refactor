"""The finish stage must produce identical files whether inline or in a worker."""

import hashlib

import numpy as np
import pytest

from chrombpnet import finish_stage


def _payload(tmp_path, name):
    rng = np.random.default_rng(0)
    regions = [["chr1", s, s + 100, s + 50] for s in range(1000, 41000, 400)]
    return {
        "data": rng.random((len(regions), 100)).astype(np.float32),
        "regions": regions,
        "chrom_sizes": [("chr1", 1_000_000)],
        "bigwig_out": str(tmp_path / f"{name}.bw"),
        "outstats_file": str(tmp_path / f"{name}_stats.txt"),
        "debug_chr": None,
        "use_tqdm": 0,
        "observed_bigwig": None,   # keep the metrics stage out of this test
        "used_regions_df": None,
        "outputlen": 100,
        "pred_logits": None,
        "pred_logcts": None,
        "output_prefix": str(tmp_path / name),
        "obs_data": None,
    }


def _digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_worker_and_inline_produce_identical_files(tmp_path):
    with finish_stage.FinishStage(enabled=False) as inline:
        inline.submit(_payload(tmp_path, "inline"))
    with finish_stage.FinishStage(enabled=True) as worker:
        worker.submit(_payload(tmp_path, "worker"))

    assert _digest(tmp_path / "inline.bw") == _digest(tmp_path / "worker.bw")
    assert ((tmp_path / "inline_stats.txt").read_text()
            == (tmp_path / "worker_stats.txt").read_text())


def test_a_single_model_runs_inline():
    """With one model there is no following forward to hide the work behind,
    so a worker would only add cost."""
    with finish_stage.start(1) as stage:
        assert stage._pool is None
    with finish_stage.start(3) as stage:
        assert stage._pool is not None


def test_a_worker_failure_reaches_the_caller(tmp_path):
    """Silently dropping one would leave a run that looks successful with a
    missing bigwig."""
    payload = _payload(tmp_path, "bad")
    payload["chrom_sizes"] = []          # addHeader with no chromosomes
    with pytest.raises(Exception):
        with finish_stage.FinishStage(enabled=True) as stage:
            stage.submit(payload)


def test_an_error_in_the_parent_does_not_hang_on_the_worker(tmp_path):
    """The context manager must not wait on queued work while unwinding."""
    with pytest.raises(ValueError, match="parent failed"):
        with finish_stage.FinishStage(enabled=True) as stage:
            stage.submit(_payload(tmp_path, "queued"))
            raise ValueError("parent failed")


def test_close_is_idempotent(tmp_path):
    stage = finish_stage.FinishStage(enabled=True)
    stage.submit(_payload(tmp_path, "once"))
    stage.close()
    stage.close()
