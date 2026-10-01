from types import SimpleNamespace

import h5py
import numpy as np
import pyBigWig
import pytest

# PyTorch-engine tests: skipped in the TensorFlow and JAX environments.
pytest.importorskip("torch")

from chrombpnet.evaluation.interpret.interpret import interpret
from chrombpnet.evaluation.make_bigwigs.bigwig_helper import write_bigwig
from chrombpnet.model import BiasModel, ModelConfig, PredictionAdapter, load_model
from chrombpnet.training.train import fit_and_evaluate
from chrombpnet.training.predict import write_predictions_h5py


class SyntheticGenerator:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        bases = rng.randint(0, 4, size=(2, 128))
        self.sequences = np.eye(4, dtype=np.float32)[bases]
        self.profiles = rng.poisson(1.0, size=(2, 20)).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        logcounts = np.log1p(self.profiles.sum(-1, keepdims=True))
        return self.sequences, [self.profiles, logcounts]

    def on_epoch_end(self):
        pass


def test_synthetic_training_prediction_attribution_and_genomics_artifacts(tmp_path):
    config = ModelConfig(128, 20, filters=4, n_dilation_layers=1)
    model = BiasModel(config)
    prefix = str(tmp_path / "synthetic")
    args = SimpleNamespace(
        output_prefix=prefix,
        device="cpu",
        learning_rate=0.001,
        epochs=1,
        early_stop=1,
        seed=11,
    )
    architecture = SimpleNamespace(save_model_without_bias=lambda model, prefix: None)
    generator = SyntheticGenerator(4)
    fit_and_evaluate(
        model,
        generator,
        generator,
        args,
        architecture,
        {"counts_loss_weight": "2.0"},
    )
    restored = load_model(prefix + ".pt")
    profile, logcounts = PredictionAdapter(restored).predict(generator.sequences)
    assert profile.shape == (2, 20)
    assert logcounts.shape == (2, 1)

    coords = np.array([["chr1", "30", "f", "1"], ["chr1", "60", "f", "1"]])
    probabilities = np.exp(profile - profile.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    write_predictions_h5py(prefix, probabilities, logcounts[:, 0], coords)
    assert (tmp_path / "synthetic_predictions.h5").is_file()

    bw_path = tmp_path / "synthetic.bw"
    write_bigwig(
        probabilities,
        [["chr1", 20, 40, 30], ["chr1", 50, 70, 60]],
        [("chr1", 100)],
        str(bw_path),
    )
    with pyBigWig.open(str(bw_path)) as bigwig:
        assert len(bigwig.values("chr1", 20, 40)) == 20

    try:
        import tangermeme  # noqa: F401
    except Exception as error:
        pytest.skip("Tangermeme is unavailable in this test environment: {}".format(error))
    contribution_prefix = str(tmp_path / "synthetic_contrib")
    interpret(
        restored,
        generator.sequences[:1],
        contribution_prefix,
        ["counts"],
        device="cpu",
        batch_size=1,
        num_shuffles=1,
        seed=3,
    )
    with h5py.File(contribution_prefix + ".counts_scores.h5") as handle:
        assert set(handle) == {"raw", "shap", "projected_shap"}
        assert handle["shap/seq"].shape == (1, 4, 128)
