import inspect
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


def test_installed_modisco_exposes_the_api_this_adapter_calls():
    """Pin the TF-MoDISco surface the adapter uses, not a version or a path.

    The pinned checkout still reports version 2.5.2 while providing the 2.6
    API, and it may live anywhere on sys.path, so asserting either would fail
    for a working install. What actually has to hold is that the three
    entry points exist and accept the arguments api.py passes them.
    """
    pytest.importorskip("modiscolite")
    _, tfmodisco, io, report = api._modisco_modules()

    assert callable(tfmodisco.TFMoDISco)
    assert callable(io.save_hdf5)
    assert callable(report.report_motifs)

    # run_arrays passes these by keyword; a rename upstream must fail loudly.
    discovery = inspect.signature(tfmodisco.TFMoDISco).parameters
    for name in ("one_hot", "hypothetical_contribs", "max_seqlets_per_metacluster",
                 "sliding_window_size", "flank_size", "trim_to_window_size",
                 "initial_flank_to_add", "final_flank_to_add",
                 "target_seqlet_fdr", "n_leiden_runs", "verbose"):
        assert name in discovery, name

    # io.save_hdf5(path, pos, neg, window) is the current 4-argument form.
    assert len(inspect.signature(io.save_hdf5).parameters) == 4

    # generate_report relies on is_writing_tomtom_matrix being accepted.
    assert "is_writing_tomtom_matrix" in inspect.signature(report.report_motifs).parameters


def test_adapter_cli_exposes_workflow_commands():
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.evaluation.modisco.api", "--help"],
        check=True, capture_output=True, text=True)
    assert "motifs" in completed.stdout
    assert "report" in completed.stdout


class _ReportWithTtl:
    calls = []

    @staticmethod
    def report_motifs(modisco_h5py, output_dir, img_path_suffix, meme_motif_db,
                      is_writing_tomtom_matrix, top_n_matches=3, trim_threshold=0.3, ttl=False):
        _ReportWithTtl.calls.append({"meme": meme_motif_db, "ttl": ttl})


class _ReportWithoutTtl:
    @staticmethod
    def report_motifs(modisco_h5py, output_dir, img_path_suffix, meme_motif_db,
                      is_writing_tomtom_matrix, top_n_matches=3, trim_threshold=0.3,
                      trim_min_length=3):
        raise AssertionError("must not be called")


def _empty_results(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["window_size"] = 8


def test_tomtom_lite_is_passed_only_when_asked_and_there_are_patterns(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_modisco_modules", lambda: (object(), object(), object(), _ReportWithTtl))
    _ReportWithTtl.calls.clear()
    results, empty = tmp_path / "results.h5", tmp_path / "empty.h5"
    _write_results(results)
    _empty_results(empty)
    api.generate_report(results, tmp_path / "a", "motifs.meme")
    api.generate_report(results, tmp_path / "b", "motifs.meme", tomtom_lite=True)
    # memelite fails on an empty pattern list, and there is nothing to match
    api.generate_report(empty, tmp_path / "c", "motifs.meme", tomtom_lite=True)
    assert [call["ttl"] for call in _ReportWithTtl.calls] == [False, True, False]


def test_tomtom_lite_is_refused_without_ttl_support(monkeypatch, tmp_path):
    """modisco-lite 2.3.2 (the TensorFlow environment) has no ttl."""
    monkeypatch.setattr(api, "_modisco_modules", lambda: (object(), object(), object(), _ReportWithoutTtl))
    assert not api.supports_tomtom_lite()
    results = tmp_path / "results.h5"
    _write_results(results)
    with pytest.raises(RuntimeError, match="TF-MoDISco 2.5"):
        api.generate_report(results, tmp_path / "out", "motifs.meme", tomtom_lite=True)


def test_the_cli_refuses_tomtom_lite_without_ttl_support(monkeypatch):
    from argparse import Namespace
    from chrombpnet import CHROMBPNET

    monkeypatch.setattr(api, "_modisco_modules", lambda: (object(), object(), object(), _ReportWithoutTtl))
    with pytest.raises(SystemExit, match="--tomtom-lite needs TF-MoDISco 2.5"):
        CHROMBPNET.check_tomtom(Namespace(cmd="qc", tomtom_lite=True))
    # and nothing is checked for commands without a report
    CHROMBPNET.check_tomtom(Namespace(cmd="train", tomtom_lite=True))


def _table(stat):
    header = "".join("<th>{}</th>".format(c) for c in
                     ["pattern", "num_seqlets", "modisco_cwm_fwd", "modisco_cwm_rev",
                      "match0", stat + "0", "match0_logo"])
    return '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      ' \
           + header + "\n    </tr>\n  </thead>\n</table>"


def test_report_wording_follows_the_table():
    from chrombpnet.helpers.generate_reports import modisco_table

    assert modisco_table.match_stat(_table("qval")) == "qval"
    assert modisco_table.match_stat(_table("pval")) == "pval"
    assert modisco_table.match_stat("<table></table>") is None
    # q-values (and no database): the wording the reports have always used
    assert modisco_table.stat_description("qval") == "qvals (qval0,qval1,qval2)"
    assert modisco_table.stat_description(None) == "qvals (qval0,qval1,qval2)"
    assert modisco_table.stat_name("qval") == "qvals"
    assert modisco_table.stat_name("pval") == "p-values"
    assert "not corrected for multiple testing" in modisco_table.stat_description("pval")


def _motifs_table(patterns, matches=3):
    """motifs.html as modiscolite writes it (pandas, no index), with the pattern
    names shortened the way make_html_bias does (``>neg_patterns.pattern`` ->
    ``>neg_``). The logo paths still contain ``neg_patterns``."""
    import pandas as pd

    def logo(path):
        return '<img src="{}" width="240" >'.format(path)

    columns = {"pattern": [], "num_seqlets": [], "modisco_cwm_fwd": [], "modisco_cwm_rev": []}
    for i in range(matches):
        columns.update({"match{}".format(i): [], "qval{}".format(i): [], "match{}_logo".format(i): []})
    for name in patterns:
        tag = name.replace("_", "_patterns.pattern_", 1)
        columns["pattern"].append(tag)
        columns["num_seqlets"].append(100 + int(name.split("_")[1]))
        for strand in ("fwd", "rev"):
            columns["modisco_cwm_" + strand].append("./trimmed_logos/{}.cwm.{}.png".format(tag, strand))
        for i in range(matches):
            columns["match{}".format(i)].append("TN5_{}".format(i))
            columns["qval{}".format(i)].append(0.01 * (i + 1))
            columns["match{}_logo".format(i)].append("./TN5_{}.png".format(i))
    html = pd.DataFrame(columns).to_html(
        escape=False, index=False,
        formatters={c: logo for c in columns if c.startswith("modisco_cwm") or c.endswith("_logo")})
    return html.replace(">pos_patterns.pattern", ">pos_").replace(">neg_patterns.pattern", ">neg_")


@pytest.mark.parametrize("matches", [3, 1, 0], ids=["13-columns", "7-columns", "no-database"])
@pytest.mark.parametrize("patterns", [
    ["pos_0", "pos_1"],
    ["neg_0", "neg_1"],
    ["pos_0", "neg_0", "pos_1", "neg_1", "neg_2", "pos_2"],
], ids=["positive-only", "negative-only", "mixed"])
def test_negative_pattern_rows_are_removed_whole(patterns, matches):
    from chrombpnet.helpers.generate_reports import modisco_table

    positive = [name for name in patterns if name.startswith("pos_")]
    kept = modisco_table.drop_negative_patterns(_motifs_table(patterns, matches))
    assert kept == _motifs_table(positive, matches)
    assert "  </tbody>" in kept and ">neg_" not in kept


def test_negative_pattern_removal_leaves_other_text_alone():
    from chrombpnet.helpers.generate_reports import modisco_table

    for text in ("<table></table>", "", "no table\nat all"):
        assert modisco_table.drop_negative_patterns(text) == text


def test_run_modisco_forwards_only_the_given_settings(monkeypatch, tmp_path):
    from argparse import Namespace
    from types import SimpleNamespace

    import chrombpnet.evaluation.modisco.convert_html_to_pdf as convert_html_to_pdf
    from chrombpnet import stages

    calls = []
    monkeypatch.setattr(api, "run_motifs", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(api, "generate_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(convert_html_to_pdf, "main", lambda *args, **kwargs: None)
    layout = SimpleNamespace(interpret_subsample=lambda name: str(tmp_path / name),
                             modisco_dir=lambda kind: str(tmp_path) + "/",
                             evaluation=lambda name: str(tmp_path / name))
    stages.run_modisco(Namespace(), layout, "bias", "counts", "bias_counts.pdf")
    stages.run_modisco(Namespace(modisco_max_seqlets=2000, modisco_window=400),
                       layout, "bias", "counts", "bias_counts.pdf")
    assert calls == [{}, {"max_seqlets": 2000, "window": 400}]


def test_interpretation_subsample_size(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd
    from chrombpnet import stages

    peaks = pd.DataFrame({"chrom": "chr1", "start": range(0, 5000, 100),
                          "end": range(100, 5100, 100)})
    peaks_path = tmp_path / "peaks.bed"
    peaks.to_csv(peaks_path, sep="\t", header=False, index=False)
    layout = SimpleNamespace(aux=lambda name: str(tmp_path / name),
                             create_interpret_subsample_dir=lambda: None)

    def subsample(**kwargs):
        path = stages.subsample_peaks_for_interpretation(layout, peaks_path, **kwargs)
        assert Path(path).name == "30K_subsample_peaks.bed"
        return pd.read_csv(path, sep="\t", header=None, names=list(peaks.columns))

    # fewer peaks than the default size: all of them, in order
    pd.testing.assert_frame_equal(subsample(), peaks)
    pd.testing.assert_frame_equal(subsample(size=None), peaks)
    expected = peaks.sample(20, random_state=stages.INTERPRET_SUBSAMPLE_SEED)
    pd.testing.assert_frame_equal(subsample(size=20), expected.reset_index(drop=True))
