"""``pipeline`` / ``train --posthoc``: a plain BPNet whose profile is
bias-corrected after training (:mod:`chrombpnet.engines.pytorch.posthoc`)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

# PyTorch-engine tests: skipped in the TensorFlow and JAX environments.
torch = pytest.importorskip("torch")

import chrombpnet.CHROMBPNET as cli
import chrombpnet.training.models.chrombpnet_posthoc_model as posthoc_architecture
from chrombpnet import engines, parsers, pipelines, stages
from chrombpnet.engines.pytorch import distill
from chrombpnet.engines.pytorch.model import (
    BiasModel,
    BPNet,
    ChromBPNet,
    ModelConfig,
    PredictionAdapter,
    UncorrectedBPNet,
    _load_payload,
    load_model,
    save_model,
)
from chrombpnet.engines.pytorch.posthoc import (
    SHRINK_EPS,
    SHRINK_WINDOW,
    PostHocChromBPNet,
    _running_mean,
    correct_logits,
)
from chrombpnet.engines.pytorch.train import fit_and_evaluate
from chrombpnet.options import Options

INPUT, OUTPUT = 200, 50
TRAIN = "train -g g.fa -c c.sizes -ibw x.bw -o out -d ATAC -p p.bed -n n.bed -fl f.json"


def _models(seed=0, bias_counts_head=True):
    torch.manual_seed(seed)
    model = BPNet(ModelConfig(INPUT, OUTPUT, filters=6, n_dilation_layers=1))
    bias = BiasModel(ModelConfig(INPUT, OUTPUT, filters=4, n_dilation_layers=1,
                                 counts_head=bias_counts_head))
    return model, bias


def _peaked_models(seed=0):
    """Models whose profiles span orders of magnitude, so that the bias model
    predicts almost nothing at some bases and shrink differs from plain."""
    model, bias = _models(seed)
    with torch.no_grad():
        model.profile_conv.weight.mul_(30.0)
        bias.profile_conv.weight.mul_(30.0)
    return model, bias


def _sequences(n, seed=0):
    bases = np.random.RandomState(seed).randint(0, 4, size=(n, INPUT))
    return torch.from_numpy(np.eye(4, dtype=np.float32)[bases]).permute(0, 2, 1).contiguous()


def _logits(n=3, length=OUTPUT, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (2 * torch.randn(n, length, generator=generator),
            2 * torch.randn(n, length, generator=generator))


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.exp(values - values.max(-1, keepdims=True))
    return values / values.sum(-1, keepdims=True)


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_plain_subtracts_the_bias_logits():
    logits, bias_logits = _logits()
    assert torch.equal(correct_logits(logits, bias_logits, "plain"), logits - bias_logits)


def test_shrink_is_its_formula():
    logits, bias_logits = _logits()
    q, p = _softmax(logits.numpy()), _softmax(bias_logits.numpy())
    c, half = SHRINK_EPS / OUTPUT, SHRINK_WINDOW // 2

    def running_mean(values):
        # Reflection without the edge value: position -1 is position 1.
        padded = np.concatenate([values[:, half:0:-1], values, values[:, -2:-2 - half:-1]], axis=1)
        return np.stack([padded[:, i:i + SHRINK_WINDOW].mean(1) for i in range(OUTPUT)], axis=1)

    r = (running_mean(q) + c) / (running_mean(p) + c)
    corrected = correct_logits(logits, bias_logits).double().numpy()
    np.testing.assert_allclose(corrected, np.log((q + c * r) / (p + c)), rtol=0, atol=1e-5)
    # The same thing, as the per-base ratio weighted against the neighbourhood's.
    w = p / (p + c)
    np.testing.assert_allclose(np.exp(corrected), w * q / p + (1 - w) * r, rtol=1e-4)


def test_a_logit_offset_does_not_move_the_corrected_profile():
    logits, bias_logits = _logits()
    for rule in ("shrink", "plain"):
        reference = torch.softmax(correct_logits(logits, bias_logits, rule), dim=-1)
        shifted = torch.softmax(correct_logits(logits + 3.0, bias_logits - 7.0, rule), dim=-1)
        assert torch.allclose(reference, shifted, atol=1e-6)


def test_shrink_fills_a_bias_hole_that_plain_division_turns_into_a_spike():
    length = 1000
    q = torch.ones(length)
    q[495:505] = 0.3      # a footprint
    q[300] = 1e-3         # a base the enzyme hardly cuts ...
    p = torch.ones(length)
    p[300] = 1e-5         # ... which the bias model predicts 100 times lower still
    corrected = {
        rule: length * torch.softmax(
            correct_logits(q.log()[None], p.log()[None], rule)[0].double(), dim=-1)
        for rule in ("plain", "shrink")}
    assert corrected["plain"][300] > 50          # times a uniform profile's value
    assert 0.9 < corrected["shrink"][300] < 1.2
    for rule in ("plain", "shrink"):             # the footprint is kept either way
        assert corrected[rule][500] / corrected[rule][100] == pytest.approx(0.3, abs=0.02)


@pytest.mark.parametrize("kwargs", [
    dict(rule="divide"), dict(eps=0.0), dict(eps=float("inf")), dict(eps=float("nan")),
    dict(window=8), dict(window=9.0), dict(window=2 * OUTPUT + 1)])
def test_the_rule_refuses_settings_it_cannot_apply(kwargs):
    logits, bias_logits = _logits()
    with pytest.raises(ValueError):
        correct_logits(logits, bias_logits, **kwargs)


@pytest.mark.parametrize("window", [1, 3, 9, 2 * OUTPUT - 1])
def test_the_running_mean_reflects_at_the_ends(window):
    values = torch.softmax(_logits()[0].double(), dim=-1).requires_grad_()
    half = window // 2
    padded = torch.nn.functional.pad(values.unsqueeze(1), (half, half), mode="reflect")
    expected = torch.nn.functional.avg_pool1d(padded, window, stride=1).squeeze(1)
    got = _running_mean(values, window)
    assert torch.equal(got, expected)
    # ... and has the same gradient, without reflection padding's backward.
    weights = torch.randn(got.shape, dtype=got.dtype, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(torch.autograd.grad((got * weights).sum(), values)[0],
                          torch.autograd.grad((expected * weights).sum(), values)[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_shrink_is_computed_in_float64_and_returned_in_the_logits_dtype(dtype):
    logits, bias_logits = (values.to(dtype) for values in _logits())
    corrected = correct_logits(logits, bias_logits)
    assert corrected.dtype == dtype
    exact = correct_logits(logits.double(), bias_logits.double())
    assert torch.equal(corrected, exact.to(dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_shrink_has_a_deterministic_backward_on_cuda():
    logits, bias_logits = (values.cuda().requires_grad_() for values in _logits())
    torch.use_deterministic_algorithms(True)
    try:
        correct_logits(logits, bias_logits).sum().backward()
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.isfinite(logits.grad).all() and torch.isfinite(bias_logits.grad).all()


def test_the_rule_refuses_profiles_of_different_shapes():
    logits, bias_logits = _logits()
    with pytest.raises(ValueError, match="batch, length"):
        correct_logits(logits, bias_logits[:, :-1])
    with pytest.raises(ValueError, match="batch, length"):
        correct_logits(logits[0], bias_logits[0], "plain")


# --------------------------------------------------------------------------
# The corrected model and its checkpoint
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["shrink", "plain"])
def test_the_profile_is_corrected_and_the_counts_are_the_uncorrected_models(rule):
    model, bias = _models()
    sequence = _sequences(3)
    profile, logcounts = PostHocChromBPNet(model, bias, rule=rule)(sequence)
    uncorrected_profile, uncorrected_logcounts = model(sequence)
    assert torch.equal(logcounts, uncorrected_logcounts)
    assert torch.allclose(
        profile, correct_logits(uncorrected_profile, bias(sequence)[0], rule))
    assert not torch.allclose(profile, uncorrected_profile)


def test_both_models_are_frozen_but_gradients_reach_the_input_through_both():
    model, bias = _models()
    corrected = PostHocChromBPNet(model, bias, rule="plain")
    assert not any(parameter.requires_grad for parameter in corrected.parameters())
    assert not corrected.training

    def input_gradient(module):
        sequence = _sequences(2).requires_grad_()
        module(sequence)[0][:, OUTPUT // 2].sum().backward()
        return sequence.grad

    # Attribution needs the bias model's dependence on the sequence too.
    assert torch.allclose(input_gradient(corrected),
                          input_gradient(model) - input_gradient(bias), atol=1e-6)


def test_a_profile_only_bias_model_corrects_as_well(tmp_path):
    model, bias = _models(bias_counts_head=False)
    corrected = PostHocChromBPNet(model, bias)
    sequence = _sequences(2)
    profile, logcounts = corrected(sequence)
    assert torch.allclose(profile, correct_logits(model(sequence)[0], bias(sequence)[0]))
    assert torch.equal(logcounts, model(sequence)[1])
    save_model(corrected, str(tmp_path / "corrected.pt"))
    restored = load_model(str(tmp_path / "corrected.pt"))
    assert restored.bias_model.count_dense is None
    assert torch.equal(restored(sequence)[0], profile)


def test_models_that_cannot_be_combined_are_refused():
    model, bias = _models()
    with pytest.raises(ValueError, match="lengths differ"):
        PostHocChromBPNet(model, BiasModel(ModelConfig(INPUT + 2, OUTPUT, filters=4,
                                                       n_dilation_layers=1)))
    with pytest.raises(ValueError, match="counts head"):
        PostHocChromBPNet(_models(bias_counts_head=False)[1], bias)
    with pytest.raises(ValueError, match="rule"):
        PostHocChromBPNet(model, bias, rule="divide")
    with pytest.raises(ValueError, match="window"):
        PostHocChromBPNet(model, bias, window=2 * OUTPUT + 1)


def test_only_a_plain_bpnet_and_a_bias_model_are_saved_together(tmp_path):
    model, bias = _models()
    full = ChromBPNet(model.config, _models(seed=2)[1], model)
    with pytest.raises(TypeError, match="plain BPNet"):
        save_model(PostHocChromBPNet(full, bias), str(tmp_path / "corrected.pt"))


def test_the_rules_settings_are_kept_in_the_checkpoint(tmp_path):
    model, bias = _peaked_models()
    corrected = PostHocChromBPNet(model, bias, eps=0.2, window=5)
    path = str(tmp_path / "corrected.pt")
    save_model(corrected, path)
    assert _load_payload(path, "cpu")["posthoc"] == {"rule": "shrink", "eps": 0.2, "window": 5}
    sequence = _sequences(3)
    restored = load_model(path)
    assert torch.equal(restored(sequence)[0], corrected(sequence)[0])
    assert not torch.allclose(restored(sequence)[0], PostHocChromBPNet(model, bias)(sequence)[0])


@pytest.mark.parametrize("rule", ["shrink", "plain"])
def test_checkpoints_round_trip_and_predict_like_any_no_bias_model(tmp_path, rule):
    model, bias = _models()
    corrected = PostHocChromBPNet(model, bias, rule=rule)
    path = str(tmp_path / "chrombpnet_nobias.pt")
    save_model(corrected, path)

    payload = _load_payload(path, "cpu")
    assert payload["model_type"] == "chrombpnet_posthoc"
    assert payload["posthoc"] == {"rule": rule, "eps": SHRINK_EPS, "window": SHRINK_WINDOW}
    assert payload["bias_config"]["filters"] == 4 and payload["config"]["filters"] == 6

    restored = load_model(path)
    assert isinstance(restored, PostHocChromBPNet) and restored.rule == rule
    sequence = _sequences(3)
    for expected, got in zip(corrected(sequence), restored(sequence)):
        assert torch.equal(expected, got)

    # What the shared commands see: two outputs, read off the checkpoint.
    assert stages.has_counts_head(path)
    assert stages.contribution_heads(path) == ["counts", "profile"]
    adapter = PredictionAdapter(restored)
    assert adapter.input_shape == (None, INPUT, 4)
    profile, logcounts = adapter.predict(sequence.permute(0, 2, 1).numpy())
    np.testing.assert_allclose(profile, corrected(sequence)[0].numpy(), atol=1e-6)
    assert logcounts.shape == (3, 1)


# --------------------------------------------------------------------------
# Training: the architecture hook
# --------------------------------------------------------------------------


class _Generator:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        self.sequences = np.eye(4, dtype=np.float32)[rng.randint(0, 4, size=(4, INPUT))]
        self.profiles = rng.poisson(1.0, size=(4, OUTPUT)).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sequences, [self.profiles, np.log1p(self.profiles.sum(-1, keepdims=True))]

    def on_epoch_end(self):
        pass


@pytest.mark.parametrize("rule", ["shrink", "plain"])
def test_training_writes_the_uncorrected_model_and_its_corrected_no_bias_model(tmp_path, rule):
    _, bias = _models(seed=5)
    bias_path = str(tmp_path / "bias_model_scaled.pt")
    save_model(bias, bias_path)
    prefix = str(tmp_path / "chrombpnet")
    args = SimpleNamespace(output_prefix=prefix, device="cpu", learning_rate=0.001, epochs=2,
                           early_stop=2, seed=11, posthoc=rule)
    parameters = {"inputlen": INPUT, "outputlen": OUTPUT, "filters": 6, "n_dil_layers": 1,
                  "counts_loss_weight": "2.0", "bias_model_path": bias_path}

    model = posthoc_architecture.build_model(args, parameters)
    assert type(model) is UncorrectedBPNet
    generator = _Generator(4)
    fit_and_evaluate(model, generator, generator, args, posthoc_architecture, parameters)

    # <prefix>.pt is the trained model itself: a plain BPNet with no bias in
    # it, under a type of its own, so nothing takes it for a bias-free model.
    uncorrected = load_model(prefix + ".pt")
    assert type(uncorrected) is UncorrectedBPNet
    assert _load_payload(prefix + ".pt", "cpu")["model_type"] == "chrombpnet_uncorrected"
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), uncorrected.state_dict()[key])
    # Building the corrected model did not freeze the one that was trained.
    assert all(parameter.requires_grad for parameter in model.parameters())

    corrected = load_model(prefix + "_nobias.pt")
    assert isinstance(corrected, PostHocChromBPNet) and corrected.rule == rule
    assert _load_payload(prefix + "_nobias.pt", "cpu")["provenance"]["posthoc_bias_model"] == bias_path
    sequence = _sequences(3)
    profile, logcounts = corrected(sequence)
    assert torch.equal(logcounts, uncorrected(sequence)[1])
    assert torch.allclose(
        profile, correct_logits(uncorrected(sequence)[0], bias(sequence)[0], rule))


def test_the_hook_needs_the_bias_model():
    args = SimpleNamespace(seed=1, posthoc="shrink")
    with pytest.raises(ValueError, match="bias_model_path"):
        posthoc_architecture.build_model(args, {"inputlen": INPUT, "outputlen": OUTPUT})


def test_the_hook_named_by_an_architecture_file_alone_corrects_with_shrink():
    # -a given this file: no --posthoc, so the namespace has no rule.
    parameters = {"inputlen": INPUT, "outputlen": OUTPUT, "filters": 6, "n_dil_layers": 1,
                  "bias_model_path": "bias.pt"}
    model = posthoc_architecture.build_model(SimpleNamespace(seed=1), parameters)
    assert model.posthoc["rule"] == "shrink"


def test_a_profile_too_short_for_the_rule_is_refused_before_training():
    # 4 bp of profile cannot hold shrink's 9 bp running mean.
    parameters = {"inputlen": 154, "outputlen": 4, "filters": 6, "n_dil_layers": 1,
                  "bias_model_path": "bias.pt"}
    with pytest.raises(ValueError, match="window"):
        posthoc_architecture.build_model(SimpleNamespace(seed=1, posthoc="shrink"), parameters)
    posthoc_architecture.build_model(SimpleNamespace(seed=1, posthoc="plain"), parameters)


def test_shrink_is_refused_on_mps_before_training():
    parameters = {"inputlen": INPUT, "outputlen": OUTPUT, "filters": 6, "n_dil_layers": 1,
                  "bias_model_path": "bias.pt"}
    with pytest.raises(ValueError, match="MPS"):
        posthoc_architecture.build_model(
            SimpleNamespace(seed=1, posthoc="shrink", device="mps"), parameters)
    posthoc_architecture.build_model(SimpleNamespace(seed=1, posthoc="plain", device="mps"), parameters)


def test_distillation_takes_the_corrected_model_and_refuses_the_uncorrected_one(tmp_path):
    model, bias = _models()
    uncorrected = UncorrectedBPNet(model.config)
    uncorrected.load_state_dict(model.state_dict())
    with torch.no_grad():
        uncorrected.count_dense.bias.fill_(3.0)   # some predicted reads to distribute
    save_model(uncorrected, str(tmp_path / "chrombpnet.pt"))
    save_model(PostHocChromBPNet(uncorrected, bias), str(tmp_path / "chrombpnet_nobias.pt"))

    with pytest.raises(ValueError, match="uncorrected model of a --posthoc run"):
        distill.load_teachers([str(tmp_path / "chrombpnet.pt")], "cpu")
    teachers = distill.load_teachers([str(tmp_path / "chrombpnet_nobias.pt")], "cpu")
    assert isinstance(teachers[0], PostHocChromBPNet)
    sequence = _sequences(2)
    profile, logcounts = distill.TeacherTargets(teachers)(sequence)
    corrected_profile, corrected_logcounts = teachers[0](sequence)
    assert torch.equal(logcounts, corrected_logcounts)
    assert (torch.expm1(logcounts) > 1).all()
    assert torch.allclose(profile, torch.softmax(corrected_profile, dim=-1) * torch.expm1(logcounts),
                          rtol=1e-5)


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["train", "pipeline"])
def test_posthoc_is_shrink_unless_a_rule_is_named(command):
    line = TRAIN.replace("train", command, 1).split() + ["-b", "bias.pt"]
    assert parsers.read_parser(line + ["--posthoc"]).posthoc == "shrink"
    assert parsers.read_parser(line + ["--posthoc", "plain"]).posthoc == "plain"
    with pytest.raises(SystemExit):
        parsers.read_parser(line + ["--posthoc", "divide"])


def test_a_run_without_posthoc_records_nothing_about_it():
    args = parsers.read_parser(TRAIN.split() + ["-b", "bias.pt"])
    assert args.posthoc is None
    engines.configure_from_args(args, require=False)
    assert not hasattr(args, "posthoc")
    engines.configure(engines.DEFAULT_ENGINE)


def test_posthoc_is_refused_by_the_other_engines():
    args = parsers.read_parser(TRAIN.split() + ["-b", "bias.keras", "--posthoc", "--engine", "jax"])
    with pytest.raises(ValueError, match="--posthoc only apply to the PyTorch engine"):
        engines.configure_from_args(args, require=False)
    engines.configure(engines.DEFAULT_ENGINE)


def test_posthoc_is_refused_with_an_architecture_file():
    line = TRAIN.split() + ["-b", "bias.pt", "--posthoc"]
    cli.check_posthoc(parsers.read_parser(line))
    with pytest.raises(SystemExit, match="--posthoc"):
        cli.check_posthoc(parsers.read_parser(line + ["-a", "architecture.py"]))


def test_the_pytorch_engine_trains_the_posthoc_architecture():
    engine = engines.get("pytorch")
    assert engine.default_architecture("chrombpnet_posthoc") == posthoc_architecture.__file__
    assert engine.default_architecture("chrombpnet") != posthoc_architecture.__file__


@pytest.mark.parametrize("posthoc", [None, "shrink"])
def test_the_training_pipeline_picks_the_architecture_by_the_flag(tmp_path, monkeypatch, posthoc):
    import chrombpnet.training.train as train

    for directory in ("auxiliary", "logs", "models", "evaluation"):
        (tmp_path / directory).mkdir()
    for name in ("bias_model_scaled.pt", "chrombpnet_model_params.tsv", "chrombpnet_data_params.tsv"):
        (tmp_path / "auxiliary" / name).write_text("")
    for name in ("chrombpnet.log", "chrombpnet.log.batch", "chrombpnet.args.json"):
        (tmp_path / "models" / name).write_text("")
    trained = {}
    monkeypatch.setattr(stages, "preprocess_and_qc_bigwig", lambda options, layout: "x.bw")
    for stage in ("find_hyperparameters", "run_predictions", "check_bias_model_quality", "render_report"):
        monkeypatch.setattr(stages, stage, lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "main", lambda options: trained.update(vars(options)))
    engines.configure("pytorch")

    values = dict(cmd="train", output_dir=str(tmp_path), file_prefix=None,
                  architecture_from_file=None, bias_model_path="bias.pt", peaks="p.bed",
                  nonpeaks="n.bed")
    if posthoc:
        values["posthoc"] = posthoc
    pipelines.chrombpnet_train_pipeline(Options(values))
    expected = "chrombpnet_posthoc" if posthoc else "chrombpnet"
    assert trained["architecture_from_file"] == engines.get("pytorch").default_architecture(expected)


# --------------------------------------------------------------------------
# Contribution scores
# --------------------------------------------------------------------------


def _summation_to_delta(corrected, ops, n_shuffles=4):
    """Largest |change in the profile target - sum of its DeepLIFT scores| over
    (sequence, reference) pairs, and the largest change, for scale."""
    from tangermeme.deep_lift_shap import deep_lift_shap

    from chrombpnet.engines.pytorch.attribution import ProfileWrapper

    wrapper = ProfileWrapper(corrected).eval()
    sequence = _sequences(3)
    multipliers, references = deep_lift_shap(
        wrapper, sequence, n_shuffles=n_shuffles, random_state=3, raw_outputs=True,
        return_references=True, device="cpu", additional_nonlinear_ops=ops,
        warning_threshold=float("inf"))
    inputs = sequence[:, None].expand_as(references).reshape(-1, 4, INPUT)
    references = references.reshape(-1, 4, INPUT).to(inputs.dtype)
    with torch.no_grad():
        # Inputs then references, as DeepLIFT pairs them for the target's weights.
        target, reference_target = wrapper(torch.cat([inputs, references])).chunk(2)
    change = (target - reference_target)[:, 0]
    scores = ((inputs - references) * multipliers.reshape(-1, 4, INPUT)).sum(dim=(1, 2))
    return (change - scores).abs().max().item(), change.abs().max().item()


@pytest.mark.parametrize("rule", ["shrink", "plain"])
def test_profile_scores_sum_to_the_change_in_their_target(rule):
    pytest.importorskip("tangermeme")
    corrected = PostHocChromBPNet(*_peaked_models(), rule=rule)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residual, change = _summation_to_delta(corrected, corrected.deep_lift_ops())
        unruled, _ = _summation_to_delta(corrected, None)
    assert change > 1.0                       # the pairs really differ
    assert residual < 1e-3 * change
    if rule == "shrink":
        # The reason shrink brings a rule: without one it is taken as linear.
        assert unruled > 100 * residual
    else:
        assert corrected.deep_lift_ops() is None


def test_the_rule_is_refused_where_deeplift_could_not_attach_it():
    pytest.importorskip("tangermeme")
    corrected = PostHocChromBPNet(*_models())
    assert list(corrected.deep_lift_ops()) == [type(corrected.correction)]
    # Tangermeme skips a module that already has a backward hook.
    corrected.correction.register_full_backward_hook(lambda module, grad_input, grad_output: None)
    with pytest.raises(RuntimeError, match="backward hook"):
        corrected.deep_lift_ops()


@pytest.mark.parametrize("rule", ["shrink", "plain"])
def test_contribution_scores_run_through_the_corrected_model(rule):
    pytest.importorskip("tangermeme")
    from chrombpnet.engines.pytorch import attribution

    model, bias = _peaked_models()
    corrected = PostHocChromBPNet(model, bias, rule=rule)
    sequences = _sequences(2).permute(0, 2, 1).numpy()
    settings = dict(device="cpu", batch_size=8, num_shuffles=4, seed=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = dict(attribution.attribute(corrected, sequences, ["counts", "profile"], **settings))
    assert not [w for w in caught if "onvergence" in str(w.message)]
    assert scores["profile"].shape == sequences.shape and np.isfinite(scores["profile"]).all()
    assert np.abs(scores["profile"]).max() > 0
    # The counts are the uncorrected model's, and so are their scores.
    expected = dict(attribution.attribute(model, sequences, ["counts"], **settings))
    np.testing.assert_allclose(scores["counts"], expected["counts"], atol=1e-6)
