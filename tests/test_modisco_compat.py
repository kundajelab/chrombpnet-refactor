import importlib.metadata
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from chrombpnet.evaluation.modisco import api


def _write_scores(path, n=2, length=12):
    sequence = np.zeros((n, 4, length), dtype=np.int8)
    sequence[:, 0, :] = 1
    shap = np.arange(n * 4 * length, dtype=np.float32).reshape(n, 4, length)
    with h5py.File(path, "w") as handle:
        for name, data in (("raw", sequence), ("shap", shap),
                           ("projected_shap", sequence * shap)):
            handle.create_group(name).create_dataset("seq", data=data)
    return sequence, shap


def _write_results(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["window_size"] = 8
        pattern = handle.create_group("pos_patterns/pattern_0")
        pattern.create_dataset("sequence", data=np.full((3, 4), 0.25))
        pattern.create_dataset("contrib_scores", data=np.arange(12).reshape(3, 4))
        pattern.create_dataset("hypothetical_contribs", data=np.arange(12).reshape(3, 4))
        seqlets = pattern.create_group("seqlets")
        seqlets.create_dataset("n_seqlets", data=[1])
        seqlets.create_dataset("start", data=[2])
        seqlets.create_dataset("end", data=[5])
        seqlets.create_dataset("example_idx", data=[0])
        seqlets.create_dataset("is_revcomp", data=[False])
        seqlets.create_dataset("sequence", data=np.eye(4, dtype=np.int8)[None, :3])


def test_centered_slice_rejects_ambiguous_windows():
    assert api.centered_slice(12, 8) == slice(2, 10)
    with pytest.raises(ValueError, match="even"):
        api.centered_slice(12, 7)
    with pytest.raises(ValueError, match="exceeds"):
        api.centered_slice(12, 14)


def test_load_chrombpnet_scores_transposes_and_crops(tmp_path):
    score_path = tmp_path / "scores.h5"
    sequence, shap = _write_scores(score_path)
    loaded_sequence, loaded_shap = api.load_contribution_scores(score_path, 8)
    np.testing.assert_array_equal(loaded_sequence, sequence[:, :, 2:10].transpose(0, 2, 1))
    np.testing.assert_array_equal(loaded_shap, shap[:, :, 2:10].transpose(0, 2, 1))
    assert loaded_shap.dtype == np.float32


def test_iter_patterns_and_seqlet_fasta_use_26_schema(tmp_path):
    results_path = tmp_path / "modisco.h5"
    fasta_path = tmp_path / "seqlets.fa"
    _write_results(results_path)
    patterns = list(api.iter_patterns(results_path))
    assert len(patterns) == 1
    assert patterns[0].tag == "pos_patterns.pattern_0"
    assert patterns[0].n_seqlets == 1
    api.write_seqlets_fasta(results_path, fasta_path)
    assert fasta_path.read_text().splitlines() == [
        ">pos_patterns.pattern_0:example0:2-5(+)", "ACG"]


def test_run_motifs_calls_current_python_api(monkeypatch, tmp_path):
    score_path = tmp_path / "scores.h5"
    output_path = tmp_path / "results.h5"
    _write_scores(score_path)
    calls = {}

    class Tfmodisco:
        @staticmethod
        def TFMoDISco(**kwargs):
            calls["run"] = kwargs
            return ["positive"], ["negative"]

    class Io:
        @staticmethod
        def save_hdf5(path, positive, negative, window):
            calls["save"] = (path, positive, negative, window)

    monkeypatch.setattr(api, "_modisco_modules", lambda: (object(), Tfmodisco, Io, object()))
    api.run_motifs(score_path, output_path, max_seqlets=123, window=8)
    assert calls["run"]["one_hot"].shape == (2, 8, 4)
    assert calls["run"]["max_seqlets_per_metacluster"] == 123
    assert calls["run"]["n_leiden_runs"] == 2
    assert calls["save"] == (str(output_path), ["positive"], ["negative"], 8)


def test_requested_modisco_checkout_is_installed():
    modisco, tfmodisco, io, report = api._modisco_modules()
    assert tuple(map(int, importlib.metadata.version("modisco").split(".")[:2])) >= (2, 6)
    assert Path(modisco.__file__).resolve().is_relative_to(
        Path("/path/to/workdir/software/tfmodisco").resolve())
    assert callable(tfmodisco.TFMoDISco)
    assert callable(io.save_hdf5)
    assert callable(report.report_motifs)


def test_adapter_cli_exposes_workflow_commands():
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.evaluation.modisco.api", "--help"],
        check=True, capture_output=True, text=True)
    assert "motifs" in completed.stdout
    assert "report" in completed.stdout
