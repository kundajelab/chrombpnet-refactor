import numpy as np
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace

from chrombpnet.model import (
    BPNet,
    BiasModel,
    ChromBPNet,
    ModelConfig,
    PredictionAdapter,
    load_model,
    resolve_device,
    save_model,
)
from chrombpnet.training.utils.losses import chrombpnet_loss, multinomial_nll
from chrombpnet.training.train import get_model


SMALL = ModelConfig(input_length=128, output_length=20, filters=8, n_dilation_layers=2)


def test_bpnet_shapes_and_native_ncl_layout():
    model = BPNet(SMALL)
    profile, logcounts = model(torch.zeros(3, 4, 128))
    assert profile.shape == (3, 20)
    assert logcounts.shape == (3, 1)
    with pytest.raises(ValueError, match="batch, 4, length"):
        model(torch.zeros(3, 128, 4))


def test_receptive_field_and_crop_are_exact():
    model = BPNet(SMALL)
    feature = model._features(torch.zeros(1, 4, 128))
    # 21-mer valid conv removes 20; dilation 2 and 4 remove 4 and 8.
    assert feature.shape[-1] == 96
    assert model.profile_conv(feature).shape[-1] == 22


def test_multinomial_and_weighted_count_loss_formula():
    true = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    logits = torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.7, 0.1]])
    logp = torch.log_softmax(logits, dim=-1)
    n = true.sum(-1)
    expected = -(
        torch.lgamma(n + 1)
        - torch.lgamma(true + 1).sum(-1)
        + (true * logp).sum(-1)
    ).mean()
    assert torch.allclose(multinomial_nll(true, logits), expected)

    pred_counts = torch.tensor([[0.0], [1.0]])
    true_counts = torch.tensor([[1.0], [3.0]])
    total, profile, counts = chrombpnet_loss(
        logits, pred_counts, true, true_counts, counts_loss_weight=7.5
    )
    assert torch.allclose(total, profile + 7.5 * counts)


def test_combined_model_freezes_bias_and_uses_logsumexp():
    bias_cfg = ModelConfig(128, 20, filters=4, n_dilation_layers=1)
    bias = BiasModel(bias_cfg)
    model = ChromBPNet(SMALL, bias)
    assert all(not parameter.requires_grad for parameter in model.bias_model.parameters())
    x = torch.randn(2, 4, 128)
    with torch.no_grad():
        bias_profile, bias_counts = model.bias_model(x)
        tf_profile, tf_counts = model.tf_model(x)
        profile, counts = model(x)
    assert torch.allclose(profile, bias_profile + tf_profile)
    assert torch.allclose(
        counts, torch.logsumexp(torch.cat([tf_counts, bias_counts], -1), -1, keepdim=True)
    )


def test_structured_checkpoint_round_trip_with_distinct_branch_configs(tmp_path):
    bias = BiasModel(ModelConfig(128, 20, filters=4, n_dilation_layers=1))
    model = ChromBPNet(SMALL, bias)
    path = tmp_path / "combined.pt"
    save_model(model, path, provenance={"test": True})
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 1
    assert payload["model_type"] == "chrombpnet"
    assert payload["provenance"]["source_repository_commit"].startswith("09938fd")
    assert payload["bias_config"]["filters"] == 4
    restored = load_model(path)
    assert restored.bias_model.config.filters == 4
    x = torch.randn(2, 4, 128)
    with torch.no_grad():
        for actual, expected in zip(restored(x), model(x)):
            assert torch.equal(actual, expected)


def test_prediction_adapter_accepts_legacy_nlc():
    model = BPNet(SMALL)
    nlc = np.zeros((2, 128, 4), dtype=np.float32)
    nlc[:, :, 0] = 1
    profile, counts = PredictionAdapter(model).predict(nlc, batch_size=1)
    assert profile.shape == (2, 20)
    assert counts.shape == (2, 1)


def test_prediction_adapter_accepts_negative_stride_reverse_complement_view():
    model = BPNet(SMALL)
    nlc = np.zeros((2, 128, 4), dtype=np.float32)
    nlc[:, :, 0] = 1
    reverse_complement = nlc[:, ::-1, ::-1]
    assert any(stride < 0 for stride in reverse_complement.strides)

    adapter = PredictionAdapter(model)
    actual = adapter.predict(reverse_complement, batch_size=1)
    expected = adapter.predict(reverse_complement.copy(), batch_size=1)

    for actual_output, expected_output in zip(actual, expected):
        assert np.array_equal(actual_output, expected_output)


def test_device_resolution_cpu_and_unavailable_cuda():
    assert resolve_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="not available"):
            resolve_device("cuda")


def test_chrombpnet_seed_controls_trainable_initialization(tmp_path):
    bias_path = tmp_path / "bias.pt"
    save_model(BiasModel(SMALL), bias_path)
    architecture = Path(__file__).parents[1] / "chrombpnet/training/models/chrombpnet_with_bias_model.py"
    args = SimpleNamespace(seed=91, architecture_from_file=str(architecture))
    parameters = {
        "input_length": SMALL.input_length,
        "output_length": SMALL.output_length,
        "filters": SMALL.filters,
        "n_dilation_layers": SMALL.n_dilation_layers,
        "counts_loss_weight": SMALL.counts_loss_weight,
        "bias_model_path": str(bias_path),
    }
    first, _ = get_model(args, parameters)
    second, _ = get_model(args, parameters)
    for left, right in zip(first.tf_model.parameters(), second.tf_model.parameters()):
        assert torch.equal(left, right)
