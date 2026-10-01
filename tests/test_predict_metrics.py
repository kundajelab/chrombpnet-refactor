"""Profile correlation in evaluation JSON, with unchanged HTML/PDF tables."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from chrombpnet.training import predict


@pytest.fixture
def prediction_inputs(tmp_path, monkeypatch):
    counts = np.tile([1., 2., 3., 4.], (6, 1))
    probabilities = np.array([
        [1., 2., 3., 4.], [2., 4., 6., 8.], [4., 3., 2., 1.],
        [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]])
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    coords = np.array([["chr1", str(100 + i * 10), "f", flag]
                       for i, flag in enumerate(["1"] * 4 + ["0"] * 2)])
    monkeypatch.setattr(predict, "load_model_wrapper", lambda *a, **k: None)
    monkeypatch.setattr(predict.initializers, "initialize_generators",
                        lambda *a, **k: None)
    monkeypatch.setattr(predict, "predict_on_batch_wrapper", lambda *a: (
        counts, probabilities, np.arange(6.), np.arange(6.), coords))
    monkeypatch.setattr(predict.metrics, "counts_metrics", lambda *a: (0.8, 0.9, 0.1))
    monkeypatch.setattr(predict.metrics, "plot_histogram", lambda *a: None)
    return SimpleNamespace(model_h5="model.pt", peaks="peaks.bed",
                           nonpeaks="nonpeaks.bed",
                           output_prefix=str(tmp_path / "chrombpnet"))


def test_json_profile_pearsonr_uses_subset_medians_and_null(prediction_inputs):
    args = prediction_inputs
    predict.main(args)
    with open(args.output_prefix + "_metrics.json") as handle:
        data = json.load(handle)
    assert data["profile_metrics"]["peaks"]["median_pearsonr"] == pytest.approx(1.)
    assert data["profile_metrics"]["nonpeaks"]["median_pearsonr"] is None
    assert data["profile_metrics"]["peaks_and_nonpeaks"]["median_pearsonr"] == pytest.approx(1.)
    assert data["counts_metrics"]["peaks"]["pearsonr"] == 0.9


@pytest.mark.parametrize("pipeline,outputs", [
    ("chrombpnet_train_pipeline", ["bias", "run_chrombpnet"]),
    ("chrombpnet_qc", ["run_chrombpnet"]),
    ("train_bias_pipeline", ["run_bias"]),
    ("bias_model_qc", ["run_bias"]),
])
def test_pipeline_json_includes_profile_pearsonr(
        prediction_inputs, tmp_path, monkeypatch, pipeline, outputs):
    import importlib.util

    from chrombpnet import engines, pipelines, stages
    from chrombpnet.layout import RunLayout
    from chrombpnet.training import train

    if importlib.util.find_spec("torch") is None:
        # The training pipelines ask the engine for its architecture file;
        # use an engine this environment has. Nothing here builds a model.
        monkeypatch.setattr(engines, "_CURRENT", "tensorflow")

    layout = RunLayout(str(tmp_path), "run")
    layout.create_training_dirs()
    (tmp_path / "logs" / "run_chrombpnet_model_params.tsv").write_text("")
    monkeypatch.setattr(RunLayout, "move", lambda *a: None)
    for name in ("preprocess_and_qc_bigwig", "find_hyperparameters",
                 "check_bias_model_quality", "write_bias_motif_table",
                 "run_marginal_footprints", "subsample_peaks_for_interpretation",
                 "run_interpret", "run_modisco", "render_report"):
        monkeypatch.setattr(stages, name, lambda *a, **k: None)
    monkeypatch.setattr(train, "main", lambda *a: None)
    monkeypatch.setattr(pipelines, "load_model_wrapper", lambda *a, **k:
                        SimpleNamespace(input_shape=(None, 4, 4),
                                        output_shape=[(None, 4), (None, 1)]))
    args = SimpleNamespace(output_dir=str(tmp_path), bias_model_path="bias.pt",
                           architecture_from_file=None, cmd="pipeline",
                           cmd_bias="pipeline", file_prefix="run",
                           bias_model="bias.pt", chrombpnet_model="chrombpnet.pt",
                           chrombpnet_model_nb="chrombpnet_nobias.pt",
                           peaks="peaks.bed", nonpeaks="nonpeaks.bed")
    getattr(pipelines, pipeline)(args)
    for output in outputs:
        data = json.loads((tmp_path / "evaluation" / f"{output}_metrics.json").read_text())
        assert data["profile_metrics"]["peaks"]["median_pearsonr"] == pytest.approx(1.)
        for subset in data["profile_metrics"].values():
            assert "median_pearsonr" in subset


@pytest.mark.parametrize("report", ["chrombpnet_qc", "chrombpnet_train", "bias_qc"])
def test_html_report_keeps_existing_metrics(tmp_path, monkeypatch, report):
    from chrombpnet.helpers.generate_reports import make_html, make_html_bias, report_shell

    evaluation = tmp_path / "evaluation"
    (evaluation / "modisco_profile").mkdir(parents=True)
    (evaluation / "modisco_profile" / "motifs.html").write_text("<table></table>")
    (evaluation / "modisco_counts").mkdir()
    (evaluation / "modisco_counts" / "motifs.html").write_text("<table></table>")
    (evaluation / "chrombpnet_nobias_max_bias_response.txt").write_text("corrected_0.001")
    data = {"counts_metrics": {"peaks": {"pearsonr": 0.9}},
            "profile_metrics": {"peaks": {"median_jsd": 0.2,
                                          "median_norm_jsd": 0.8}}}
    if report == "chrombpnet_qc":
        metrics_path = evaluation / "chrombpnet_metrics.json"
        render = lambda: make_html.qc_report("", str(tmp_path), "ATAC")
    elif report == "chrombpnet_train":
        metrics_path = evaluation / "bias_metrics.json"
        monkeypatch.setattr(report_shell, "loss_curve", lambda *a: "./epoch_loss.png")
        render = lambda: make_html.train_report("", str(tmp_path))
    else:
        metrics_path = evaluation / "bias_metrics.json"
        render = lambda: make_html_bias.qc_report("", str(tmp_path))
        data["profile_metrics"]["nonpeaks"] = {"median_jsd": 0.3, "median_norm_jsd": 0.7}
    metrics_path.write_text(json.dumps(data))
    original = render()
    for subset in data["profile_metrics"].values():
        subset["median_pearsonr"] = 0.75
    metrics_path.write_text(json.dumps(data))
    assert render() == original


@pytest.mark.parametrize("constant", [False, True])
def test_pred_bw_json_profile_pearsonr(tmp_path, monkeypatch, constant):
    from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import compare_with_observed

    counts = np.array([[1., 2., 3., 4.], [2., 4., 6., 8.], [4., 3., 2., 1.]])
    probabilities = np.tile([0.1, 0.2, 0.3, 0.4], (3, 1))
    if constant:
        probabilities[:] = 0.25
    monkeypatch.setattr(predict.metrics, "counts_metrics", lambda *a: (0.8, 0.9, 0.1))
    monkeypatch.setattr(predict.metrics, "plot_histogram", lambda *a: None)
    prefix = str(tmp_path / "prediction_chrombpnet")
    regions = [["chr1", i * 10, i * 10 + 4, i * 10 + 2] for i in range(3)]
    compare_with_observed(None, None, regions, 4, np.log(probabilities),
                          np.log1p(counts.sum(axis=1)), prefix, obs_data=counts)
    data = json.loads((tmp_path / "prediction_chrombpnet_metrics.json").read_text())
    result = data["profile_metrics"]["regions"]["median_pearsonr"]
    if constant:
        assert result is None
    else:
        assert result == pytest.approx(1.)
    assert data["counts_metrics"]["regions"]["pearsonr"] == 0.9
