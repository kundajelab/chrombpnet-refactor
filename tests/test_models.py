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
    initialize_for_benchmark,
    initialize_keras_compatible,
    initialize_keras_default,
    load_model,
    resolve_device,
    save_model,
)
from chrombpnet.training.utils.losses import chrombpnet_loss, multinomial_nll
from chrombpnet.training.train import _build_optimizer, get_model
from chrombpnet.training.utils.benchmark_utils import (
    BENCHMARK_CONTROLS,
    BENCHMARK_LEVELS,
    benchmark_enables,
    benchmark_level,
    canonical_bpnet_weights,
    generator_seed,
    make_generator_rng,
    normalize_benchmark_level,
    use_keras_batch_order_shuffle,
)


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


def test_benchmark_initializer_uses_canonical_glorot_arrays():
    model = BPNet(SMALL)
    initialize_keras_compatible(model, seed=19)
    expected = canonical_bpnet_weights(
        SMALL.filters, SMALL.n_dilation_layers, seed=19
    )
    np.testing.assert_array_equal(
        model.first_conv.weight.detach().numpy(),
        expected["first_kernel"].transpose(2, 1, 0),
    )
    np.testing.assert_array_equal(
        model.profile_conv.weight.detach().numpy(),
        expected["profile_kernel"].transpose(2, 1, 0),
    )
    np.testing.assert_array_equal(
        model.count_dense.weight.detach().numpy(), expected["count_kernel"].T
    )



def test_benchmark_levels_are_ordered_and_default_to_max():
    assert BENCHMARK_LEVELS == ("lite", "medium", "max")
    assert normalize_benchmark_level(True) == "max"
    assert normalize_benchmark_level(None) is None
    assert normalize_benchmark_level(False) is None
    assert benchmark_level(SimpleNamespace(seed=1)) is None
    with pytest.raises(ValueError, match="Unknown benchmark level"):
        normalize_benchmark_level("huge")
    # Every control is monotone: once active it stays active at stricter levels.
    for control in BENCHMARK_CONTROLS:
        active = [benchmark_enables(level, control) for level in BENCHMARK_LEVELS]
        assert active == sorted(active)


def test_benchmark_level_selects_the_expected_controls():
    assert benchmark_enables("lite", "keras_adam")
    assert benchmark_enables("lite", "seeded_threshold_draw")
    assert benchmark_enables("lite", "restore_best_checkpoint")
    assert not benchmark_enables("lite", "fixed_example_order")
    assert not benchmark_enables("lite", "generator_rng")
    assert benchmark_enables("medium", "fixed_example_order")
    assert benchmark_enables("medium", "generator_rng")
    assert not benchmark_enables("medium", "shared_initial_weights")
    assert not benchmark_enables("medium", "deterministic_kernels")
    assert benchmark_enables("max", "shared_initial_weights")
    assert benchmark_enables("max", "replayable_augmentation")
    # Keras shuffles batch indices by default; medium and above pin the order.
    assert use_keras_batch_order_shuffle("lite")
    assert not use_keras_batch_order_shuffle("medium")
    assert not use_keras_batch_order_shuffle(None)


def test_lite_initializer_is_glorot_but_not_the_shared_arrays():
    model = BPNet(SMALL)
    initialize_keras_default(model, seed=19)
    shared = canonical_bpnet_weights(SMALL.filters, SMALL.n_dilation_layers, seed=19)
    # Same initializer family and bounds as Keras...
    for weight, (fan_in, fan_out) in (
        (model.first_conv.weight, (21 * 4, 21 * SMALL.filters)),
        (model.count_dense.weight, (SMALL.filters, 1)),
    ):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert float(weight.detach().abs().max()) <= limit
    assert torch.all(model.first_conv.bias == 0)
    assert torch.all(model.count_dense.bias == 0)
    # ...but a different concrete draw than the cross-framework arrays.
    assert not np.allclose(
        model.first_conv.weight.detach().numpy(),
        shared["first_kernel"].transpose(2, 1, 0),
    )
    # The draw is reproducible from the seed alone.
    other = BPNet(SMALL)
    initialize_keras_default(other, seed=19)
    assert torch.equal(model.first_conv.weight, other.first_conv.weight)


def test_initialize_for_benchmark_dispatches_on_level():
    assert initialize_for_benchmark(BPNet(SMALL), SimpleNamespace(seed=3)) is None
    for level, expected in (
        ("lite", "keras_default_glorot"),
        ("medium", "keras_default_glorot"),
        ("max", "shared_canonical_arrays"),
    ):
        args = SimpleNamespace(seed=3, benchmark=level)
        assert initialize_for_benchmark(BPNet(SMALL), args) == expected
    model = BPNet(SMALL)
    initialize_for_benchmark(model, SimpleNamespace(seed=3, benchmark="max"))
    expected = canonical_bpnet_weights(SMALL.filters, SMALL.n_dilation_layers, 3)
    np.testing.assert_array_equal(
        model.first_conv.weight.detach().numpy(),
        expected["first_kernel"].transpose(2, 1, 0),
    )


def test_benchmark_adam_matches_keras_epsilon():
    model = BPNet(SMALL)
    args = SimpleNamespace(learning_rate=0.001, benchmark=True)
    optimizer = _build_optimizer(model, args)
    assert optimizer.defaults["eps"] == 1e-7
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is False


def test_lite_adam_matches_keras_epsilon_without_pinning_dispatch():
    model = BPNet(SMALL)
    lite = _build_optimizer(model, SimpleNamespace(learning_rate=0.001, benchmark="lite"))
    assert lite.defaults["eps"] == 1e-7
    assert lite.defaults["betas"] == (0.9, 0.999)
    assert lite.defaults["amsgrad"] is False
    assert lite.defaults["foreach"] is None
    assert lite.defaults["fused"] is None
    plain = _build_optimizer(model, SimpleNamespace(learning_rate=0.001))
    assert plain.defaults["eps"] == 1e-8


def test_generator_rng_is_dedicated_from_medium_and_replayable_at_max():
    assert make_generator_rng("lite", 1234, "train") is np.random
    assert make_generator_rng(None, 1234, "train") is np.random
    stream = make_generator_rng("medium", 1234, "train")
    assert isinstance(stream, np.random.RandomState)
    # train/valid streams are independent, and epoch k is a pure function.
    assert generator_seed(1234, "train", 0) != generator_seed(1234, "valid", 0)
    assert generator_seed(1234, "train", 1) != generator_seed(1234, "train", 0)
    left = make_generator_rng("max", 1234, "train", epoch=2)
    right = make_generator_rng("max", 1234, "train", epoch=2)
    np.testing.assert_array_equal(left.permutation(10), right.permutation(10))
