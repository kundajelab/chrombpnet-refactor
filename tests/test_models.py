import numpy as np
import pytest
# PyTorch-engine tests: skipped in the TensorFlow and JAX environments.
torch = pytest.importorskip("torch")
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
    level_from_flags,
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



def test_reproducibility_levels_are_ordered_and_never_opt_out():
    assert BENCHMARK_LEVELS == ("tensorflow", "medium", "max")
    assert normalize_benchmark_level(True) == "max"
    # Absent means the TensorFlow settings, for a raw None, for False and for
    # a namespace that never had the flags alike. Nothing opts out of them.
    assert normalize_benchmark_level(None) == "tensorflow"
    assert normalize_benchmark_level(False) == "tensorflow"
    assert benchmark_level(SimpleNamespace(seed=1)) == "tensorflow"
    # The PyTorch-native level is gone.
    with pytest.raises(ValueError, match="Unknown reproducibility level"):
        normalize_benchmark_level("pytorch")
    # "lite" is still readable, so a checkpoint whose provenance records it
    # keeps loading.
    assert normalize_benchmark_level("lite") == "tensorflow"
    with pytest.raises(ValueError, match="Unknown reproducibility level"):
        normalize_benchmark_level("huge")
    # Every control is monotone: once active it stays active at stricter levels.
    for control in BENCHMARK_CONTROLS:
        active = [benchmark_enables(level, control) for level in BENCHMARK_LEVELS]
        assert active == sorted(active)


def test_benchmark_level_selects_the_expected_controls():
    assert benchmark_enables("tensorflow", "keras_adam")
    assert benchmark_enables("tensorflow", "seeded_threshold_draw")
    assert benchmark_enables("tensorflow", "restore_best_checkpoint")
    assert not benchmark_enables("tensorflow", "fixed_example_order")
    assert not benchmark_enables("tensorflow", "generator_rng")
    assert benchmark_enables("medium", "fixed_example_order")
    assert benchmark_enables("medium", "generator_rng")
    assert not benchmark_enables("medium", "shared_initial_weights")
    assert not benchmark_enables("medium", "deterministic_kernels")
    assert benchmark_enables("max", "shared_initial_weights")
    assert benchmark_enables("max", "replayable_augmentation")
    # Keras shuffles batch indices by default; medium and above pin the order.
    assert use_keras_batch_order_shuffle("tensorflow")
    assert not use_keras_batch_order_shuffle("medium")
    # None is absent, so it behaves as the default level.
    assert use_keras_batch_order_shuffle(None)


def test_the_flags_select_the_levels():
    assert level_from_flags() == "tensorflow"
    assert level_from_flags(reproducibility=True) == "medium"
    assert level_from_flags(reproducibility_max=True) == "max"
    assert level_from_flags(True, True) == "max"
    # A namespace is read through its flags; drop_absent may have removed them.
    assert benchmark_level(SimpleNamespace(reproducibility=True)) == "medium"
    assert benchmark_level(SimpleNamespace(reproducibility_max=True)) == "max"
    assert benchmark_level(SimpleNamespace(reproducibility=None, reproducibility_max=None)) == "tensorflow"
    assert benchmark_enables(SimpleNamespace(reproducibility=True), "generator_rng")
    assert not benchmark_enables(SimpleNamespace(reproducibility=True), "deterministic_kernels")
    assert benchmark_enables(SimpleNamespace(reproducibility_max=True), "deterministic_kernels")


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
    # Never PyTorch's own initialization: a namespace with neither flag gets
    # the Keras initializer family.
    assert initialize_for_benchmark(
        BPNet(SMALL), SimpleNamespace(seed=3)) == "keras_default_glorot"
    for level, expected in (
        ("tensorflow", "keras_default_glorot"),
        ("medium", "keras_default_glorot"),
        ("max", "shared_canonical_arrays"),
    ):
        args = SimpleNamespace(seed=3, benchmark=level)
        assert initialize_for_benchmark(BPNet(SMALL), args) == expected
    assert initialize_for_benchmark(
        BPNet(SMALL), SimpleNamespace(seed=3, reproducibility_max=True)) == "shared_canonical_arrays"
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


def test_tensorflow_adam_matches_keras_epsilon_without_pinning_dispatch():
    model = BPNet(SMALL)
    args = SimpleNamespace(learning_rate=0.001, benchmark="tensorflow")
    parity = _build_optimizer(model, args)
    assert parity.defaults["eps"] == 1e-7
    assert parity.defaults["betas"] == (0.9, 0.999)
    assert parity.defaults["amsgrad"] is False
    assert parity.defaults["foreach"] is None
    assert parity.defaults["fused"] is None
    # Omitting the flags is the same thing, which is the point of the default.
    implicit = _build_optimizer(model, SimpleNamespace(learning_rate=0.001))
    assert implicit.defaults["eps"] == 1e-7
    # --reproducibility-max additionally pins Adam's dispatch.
    pinned = _build_optimizer(
        model, SimpleNamespace(learning_rate=0.001, reproducibility_max=True))
    assert pinned.defaults["eps"] == 1e-7 and pinned.defaults["foreach"] is False


def test_generator_rng_is_dedicated_from_medium_and_replayable_at_max():
    assert make_generator_rng("tensorflow", 1234, "train") is np.random
    assert make_generator_rng(None, 1234, "train") is np.random
    stream = make_generator_rng("medium", 1234, "train")
    assert isinstance(stream, np.random.RandomState)
    # train/valid streams are independent, and epoch k is a pure function.
    assert generator_seed(1234, "train", 0) != generator_seed(1234, "valid", 0)
    assert generator_seed(1234, "train", 1) != generator_seed(1234, "train", 0)
    left = make_generator_rng("max", 1234, "train", epoch=2)
    right = make_generator_rng("max", 1234, "train", epoch=2)
    np.testing.assert_array_equal(left.permutation(10), right.permutation(10))


def _predict_before_batched_casting(adapter, values, batch_size=64):
    """The pre-rewrite `PredictionAdapter.predict`, kept as the oracle.

    It cast the whole input to one float32 CPU tensor up front, sliced a
    permuted non-contiguous view, ran under `no_grad`, and concatenated
    per-batch lists at the end.
    """
    if isinstance(values, list):
        values = values[0]
    if isinstance(values, np.ndarray) and any(stride < 0 for stride in values.strides):
        values = np.ascontiguousarray(values)
    array = torch.as_tensor(values, dtype=torch.float32)
    if array.shape[-1] == 4:
        array = array.permute(0, 2, 1)
    profiles, logcounts = [], []
    adapter.model.eval()
    with torch.no_grad():
        for start in range(0, array.shape[0], int(batch_size)):
            profile, counts = adapter.model(
                array[start:start + int(batch_size)].to(adapter.device))
            profiles.append(profile.cpu())
            logcounts.append(counts.cpu())
    return torch.cat(profiles).numpy(), torch.cat(logcounts).numpy()


@pytest.fixture
def _adapter():
    torch.manual_seed(0)
    model = BPNet(ModelConfig(400, 100, filters=8, n_dilation_layers=2)).eval()
    return PredictionAdapter(model, device="cpu")


@pytest.mark.parametrize("batch_size", [8, 64])
@pytest.mark.parametrize("case", ["nlc_int8", "nlc_float32", "ncl_int8",
                                  "reverse_complement_view", "single", "ragged"])
def test_predict_is_bitwise_unchanged_by_the_batched_cast(_adapter, case, batch_size):
    rng = np.random.default_rng(0)
    values = {
        "nlc_int8": lambda: rng.integers(0, 2, (37, 400, 4)).astype(np.int8),
        "nlc_float32": lambda: rng.random((37, 400, 4)).astype(np.float32),
        "ncl_int8": lambda: rng.integers(0, 2, (37, 4, 400)).astype(np.int8),
        # Negative strides: torch cannot wrap these, so the adapter has always
        # had to materialize them. marginal_footprinting feeds exactly this.
        "reverse_complement_view":
            lambda: rng.integers(0, 2, (37, 400, 4)).astype(np.int8)[:, ::-1, ::-1],
        "single": lambda: rng.integers(0, 2, (1, 400, 4)).astype(np.int8),
        "ragged": lambda: rng.integers(0, 2, (130, 400, 4)).astype(np.int8),
    }[case]()

    produced = _adapter.predict([values], batch_size=batch_size)
    expected = _predict_before_batched_casting(_adapter, [values], batch_size=batch_size)
    assert np.array_equal(produced[0], expected[0])
    assert np.array_equal(produced[1], expected[1])


def test_predict_still_rejects_an_empty_input(_adapter):
    """`torch.cat([])` used to raise here; preallocation must not swallow it."""
    with pytest.raises(RuntimeError, match="empty input"):
        _adapter.predict([np.zeros((0, 400, 4), dtype=np.int8)])


def test_predict_returns_writable_arrays(_adapter):
    """inference_mode outputs are poison if they escape; these are copies."""
    profiles, _ = _adapter.predict(
        [np.zeros((4, 400, 4), dtype=np.int8)])
    profiles[0, 0] = 1.0


def test_predict_rejects_input_without_four_channels(_adapter):
    with pytest.raises(ValueError, match="four sequence channels"):
        _adapter.predict([np.zeros((3, 5, 400), dtype=np.int8)])
