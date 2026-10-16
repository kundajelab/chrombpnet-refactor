"""--seed reaches every random draw of the command that takes it."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chrombpnet.seed import DEFAULT_SEED


def _peaks(tmp_path):
    peaks = pd.DataFrame({"chrom": "chr1", "start": range(0, 5000, 100),
                          "end": range(100, 5100, 100)})
    path = tmp_path / "peaks.bed"
    peaks.to_csv(path, sep="\t", header=False, index=False)
    return peaks, path


def test_interpret_subsample_follows_seed(tmp_path):
    from chrombpnet import stages

    peaks, path = _peaks(tmp_path)
    layout = SimpleNamespace(aux=lambda name: str(tmp_path / name),
                             create_interpret_subsample_dir=lambda: None)

    def subsample(**kwargs):
        out = stages.subsample_peaks_for_interpretation(layout, path, size=20, **kwargs)
        return pd.read_csv(out, sep="\t", header=None, names=list(peaks.columns))

    def expected(seed):
        return peaks.sample(20, random_state=seed).reset_index(drop=True)

    pd.testing.assert_frame_equal(subsample(seed=7), expected(7))
    pd.testing.assert_frame_equal(subsample(), expected(DEFAULT_SEED))
    assert not subsample(seed=7).equals(subsample(seed=8))


@pytest.mark.parametrize("pipeline", ["chrombpnet_train_pipeline", "chrombpnet_qc",
                                      "train_bias_pipeline", "bias_model_qc"])
def test_pipelines_subsample_with_seed(tmp_path, monkeypatch, pipeline):
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
    for name in ("preprocess_and_qc_bigwig", "find_hyperparameters", "run_predictions",
                 "check_bias_model_quality", "write_bias_motif_table",
                 "run_marginal_footprints", "run_interpret", "run_modisco", "render_report"):
        monkeypatch.setattr(stages, name, lambda *a, **k: None)
    seeds = []
    monkeypatch.setattr(stages, "subsample_peaks_for_interpretation",
                        lambda *a, **k: seeds.append(k.get("seed")))
    monkeypatch.setattr(train, "main", lambda *a: None)
    monkeypatch.setattr(pipelines, "load_model_wrapper", lambda *a, **k:
                        SimpleNamespace(input_shape=(None, 4, 4),
                                        output_shape=[(None, 4), (None, 1)]))
    args = SimpleNamespace(output_dir=str(tmp_path), bias_model_path="bias.pt",
                           architecture_from_file=None, cmd="pipeline",
                           cmd_bias="pipeline", file_prefix="run",
                           bias_model="bias.pt", chrombpnet_model="chrombpnet.pt",
                           chrombpnet_model_nb="chrombpnet_nobias.pt",
                           peaks="peaks.bed", nonpeaks="nonpeaks.bed", seed=7)
    getattr(pipelines, pipeline)(args)
    assert seeds == [7]


def _capture_profile_metrics(monkeypatch, module):
    seen = []
    real = module.metrics.profile_metrics

    def profile_metrics(*args, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(module.metrics, "profile_metrics", profile_metrics)
    monkeypatch.setattr(module.metrics, "counts_metrics", lambda *a: (0.8, 0.9, 0.1))
    monkeypatch.setattr(module.metrics, "plot_histogram", lambda *a: None)
    return seen


@pytest.mark.parametrize("seed,expected", [(9, 9), (None, DEFAULT_SEED)])
def test_predict_shuffles_profiles_with_seed(tmp_path, monkeypatch, seed, expected):
    from chrombpnet.training import predict

    counts = np.tile([1., 2., 3., 4.], (4, 1))
    probabilities = np.full((4, 4), 0.25)
    coords = np.array([["chr1", str(100 + i * 10), "f", "1"] for i in range(4)])
    monkeypatch.setattr(predict, "load_model_wrapper", lambda *a, **k: None)
    monkeypatch.setattr(predict.initializers, "initialize_generators", lambda *a, **k: None)
    monkeypatch.setattr(predict, "predict_on_batch_wrapper", lambda *a: (
        counts, probabilities, np.arange(4.), np.arange(4.), coords))
    seen = _capture_profile_metrics(monkeypatch, predict)
    args = SimpleNamespace(model_h5="model.pt", peaks="peaks.bed", nonpeaks="None",
                           output_prefix=str(tmp_path / "chrombpnet"))
    if seed is not None:
        args.seed = seed
    predict.main(args)
    assert seen == [expected]


def test_pred_bw_shuffles_profiles_with_seed(tmp_path, monkeypatch):
    from chrombpnet import finish_stage
    from chrombpnet.evaluation.make_bigwigs import bigwig_helper

    import chrombpnet.training.metrics as metrics  # imported lazily by compare_with_observed

    seen = _capture_profile_metrics(monkeypatch, SimpleNamespace(metrics=metrics))
    monkeypatch.setattr(bigwig_helper, "write_bigwig", lambda *a, **k: None)
    counts = np.array([[1., 2., 3., 4.], [2., 4., 6., 8.]])
    finish_stage._run_finish({
        "data": None, "regions": [["chr1", 0, 4, 2], ["chr1", 10, 14, 12]],
        "chrom_sizes": None, "bigwig_out": None, "outstats_file": None,
        "debug_chr": None, "use_tqdm": False, "observed_bigwig": "observed.bw",
        "used_regions_df": None, "outputlen": 4,
        "pred_logits": np.log(np.full((2, 4), 0.25)), "pred_logcts": np.log1p(counts.sum(1)),
        "output_prefix": str(tmp_path / "prediction"), "obs_data": counts, "seed": 5})
    assert seen == [5]
