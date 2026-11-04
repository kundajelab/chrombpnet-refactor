"""The opt-in PyTorch training changes: a profile-only bias model
(``--bias-profile-only``) and ``chrombpnet distill``. Epoch selection stays
chrombpnet 1.x's lowest validation loss."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

# PyTorch-engine tests: skipped in the TensorFlow and JAX environments.
torch = pytest.importorskip("torch")

from chrombpnet import engines, parsers
from chrombpnet.engines.pytorch import distill
from chrombpnet.engines.pytorch.losses import chrombpnet_loss, multinomial_nll
from chrombpnet.engines.pytorch.model import (
    BiasModel,
    BPNet,
    ChromBPNet,
    ModelConfig,
    PredictionAdapter,
    _load_payload,
    load_model,
    save_model,
)
from chrombpnet.engines.pytorch.train import _run_epoch, fit_and_evaluate
from chrombpnet.options import Options

INPUT, OUTPUT = 200, 50


def _config(**changes):
    values = dict(input_length=INPUT, output_length=OUTPUT, filters=4,
                  n_dilation_layers=1, counts_loss_weight=2.0)
    values.update(changes)
    return ModelConfig(**values)


def _sequences(n, seed=0, length=INPUT):
    bases = np.random.RandomState(seed).randint(0, 4, size=(n, length))
    return np.eye(4, dtype=np.float32)[bases]


def _ncl(values):
    return torch.from_numpy(values).permute(0, 2, 1).contiguous()


# --------------------------------------------------------------------------
# 1. Profile-only bias model
# --------------------------------------------------------------------------


def test_profile_only_bias_has_no_counts_head():
    torch.manual_seed(0)
    bias = BiasModel(_config(counts_head=False))
    assert bias.count_dense is None
    assert not any(name.startswith("count_dense") for name in bias.state_dict())
    assert bias.output_shape == ((None, OUTPUT), None)
    profile, logcounts = bias(_ncl(_sequences(3)))
    assert profile.shape == (3, OUTPUT)
    assert logcounts is None


def test_profile_only_bias_joins_the_profile_head_only():
    torch.manual_seed(0)
    bias = BiasModel(_config(counts_head=False))
    model = ChromBPNet(_config(), bias)
    sequence = _ncl(_sequences(3))
    with torch.no_grad():
        profile, logcounts = model(sequence)
        tf_profile, tf_logcounts = model.tf_model(sequence)
        bias_profile, _ = bias(sequence)
    torch.testing.assert_close(profile, tf_profile + bias_profile, rtol=0, atol=0)
    torch.testing.assert_close(logcounts, tf_logcounts, rtol=0, atol=0)


def test_a_counts_bias_still_joins_both_heads():
    torch.manual_seed(0)
    bias = BiasModel(_config())
    model = ChromBPNet(_config(), bias)
    sequence = _ncl(_sequences(3))
    with torch.no_grad():
        _, logcounts = model(sequence)
        _, tf_logcounts = model.tf_model(sequence)
        _, bias_logcounts = bias(sequence)
    expected = torch.logsumexp(torch.cat((tf_logcounts, bias_logcounts), -1), -1, keepdim=True)
    torch.testing.assert_close(logcounts, expected)


def test_profile_only_checkpoints_round_trip_and_old_ones_have_a_counts_head(tmp_path):
    torch.manual_seed(0)
    bias = BiasModel(_config(counts_head=False))
    model = ChromBPNet(_config(), bias)
    save_model(bias, tmp_path / "bias.pt")
    save_model(model, tmp_path / "chrombpnet.pt")
    assert load_model(tmp_path / "bias.pt").config.counts_head is False
    restored = load_model(tmp_path / "chrombpnet.pt")
    assert restored.bias_model.count_dense is None
    sequence = _ncl(_sequences(2))
    with torch.no_grad():
        for got, want in zip(restored(sequence), model.eval()(sequence)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    # A checkpoint written before the field existed.
    counts_bias = BiasModel(_config())
    payload = _load_payload(str(tmp_path / "bias.pt"), "cpu")
    payload["config"].pop("counts_head")
    payload["state_dict"] = counts_bias.state_dict()
    torch.save(payload, str(tmp_path / "old.pt"))
    old = load_model(tmp_path / "old.pt")
    assert old.config.counts_head is True and old.count_dense is not None


def test_profile_only_loss_is_the_profile_nll():
    logits = torch.randn(3, OUTPUT)
    counts = torch.poisson(torch.ones(3, OUTPUT))
    total, profile, count = chrombpnet_loss(logits, None, counts, torch.zeros(3, 1), 15.0)
    torch.testing.assert_close(total, multinomial_nll(counts, logits))
    torch.testing.assert_close(profile, total)
    assert float(count) == 0.0


def test_prediction_adapter_returns_no_counts_for_a_profile_only_bias():
    adapter = PredictionAdapter(BiasModel(_config(counts_head=False)))
    assert adapter.output_shape == [(None, OUTPUT), None]
    profile, logcounts = adapter.predict(_sequences(5), batch_size=2)
    assert profile.shape == (5, OUTPUT)
    assert logcounts is None


def test_the_bias_builder_honours_bias_profile_only():
    import chrombpnet.training.models.bpnet_model as architecture

    params = {"inputlen": INPUT, "outputlen": OUTPUT, "filters": 4,
              "n_dil_layers": 1, "counts_loss_weight": 1.0}
    args = SimpleNamespace(seed=1, bias_profile_only=True)
    assert architecture.build_model(args, params).count_dense is None
    args = SimpleNamespace(seed=1)
    assert architecture.build_model(args, params).count_dense is not None


def test_rescaling_saves_a_profile_only_bias_unchanged(tmp_path):
    torch.manual_seed(0)
    bias = BiasModel(_config(counts_head=False))
    save_model(bias, tmp_path / "bias.pt")
    engines.get("pytorch").rescale_bias_logcounts(
        str(tmp_path / "bias.pt"), str(tmp_path / "scaled.pt"),
        _sequences(4), np.ones(4))
    scaled = load_model(tmp_path / "scaled.pt")
    for name, value in bias.state_dict().items():
        assert torch.equal(scaled.state_dict()[name], value)


def test_the_bias_quality_gate_is_skipped_without_counts(tmp_path, capsys):
    from chrombpnet import stages

    path = tmp_path / "bias_metrics.json"
    path.write_text(json.dumps({"counts_metrics": {}, "profile_metrics": {}}))
    layout = SimpleNamespace(unprefixed=lambda *parts: str(path))
    stages.check_bias_model_quality(layout)
    assert "does not apply" in capsys.readouterr().out


def test_contribution_heads_follow_the_checkpoint(tmp_path):
    from chrombpnet import stages

    save_model(BiasModel(_config(counts_head=False)), tmp_path / "profile_only.pt")
    save_model(BiasModel(_config()), tmp_path / "counts.pt")
    assert stages.contribution_heads(str(tmp_path / "profile_only.pt")) == ["profile"]
    assert stages.contribution_heads(str(tmp_path / "counts.pt")) == ["counts", "profile"]
    assert stages.contribution_heads(str(tmp_path / "model.h5")) == ["counts", "profile"]


def test_the_fused_kernels_refuse_a_profile_only_bias():
    triton_forward = pytest.importorskip("chrombpnet.engines.pytorch.triton_forward")
    model = ChromBPNet(_config(), BiasModel(_config(counts_head=False)))
    with pytest.raises(triton_forward.TritonRouteRefused, match="counts head"):
        triton_forward._require_model(model)


# --------------------------------------------------------------------------
# 2. The training loop
# --------------------------------------------------------------------------


class _Generator:
    """One batch of fixed sequences and Poisson profiles, as the training and
    validation generators return them."""

    def __init__(self, seed, n=8):
        rng = np.random.RandomState(seed)
        self.sequences = _sequences(n, seed)
        self.profiles = rng.poisson(1.0, size=(n, OUTPUT)).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        logcounts = np.log1p(self.profiles.sum(-1, keepdims=True))
        return self.sequences, [self.profiles, logcounts]

    def on_epoch_end(self):
        pass


def _fit_args(tmp_path, **values):
    defaults = dict(output_prefix=str(tmp_path / "model"), device="cpu",
                    learning_rate=0.01, epochs=3, early_stop=3, seed=11)
    defaults.update(values)
    return SimpleNamespace(**defaults)


_NO_EXPORT = SimpleNamespace(save_model_without_bias=lambda model, prefix: None)


def test_the_kept_epoch_and_early_stopping_follow_the_validation_loss(tmp_path):
    torch.manual_seed(0)
    args = _fit_args(tmp_path, epochs=6, early_stop=1)
    fit_and_evaluate(BPNet(_config()), _Generator(1), _Generator(2), args, _NO_EXPORT,
                     {"counts_loss_weight": 2.0})
    with open(args.output_prefix + ".log") as handle:
        losses = [float(row["val_loss"]) for row in csv.DictReader(handle)]
    # Patience 1: training stops at the first epoch that does not improve.
    worse = [i for i in range(1, len(losses)) if losses[i] >= min(losses[:i])]
    assert len(losses) == (worse[0] + 1 if worse else args.epochs)
    # The weights kept are the lowest-validation-loss epoch's.
    kept, _ = _run_epoch(load_model(args.output_prefix + ".pt"), _Generator(2),
                         torch.device("cpu"), 2.0)
    assert kept[0] == pytest.approx(min(losses), rel=1e-5)


def test_default_selection_leaves_the_log_and_checkpoint_as_they_were(tmp_path):
    torch.manual_seed(0)
    args = _fit_args(tmp_path, epochs=2)
    fit_and_evaluate(BPNet(_config()), _Generator(1), _Generator(2), args, _NO_EXPORT,
                     {"counts_loss_weight": 2.0})
    with open(args.output_prefix + ".log") as handle:
        header = next(csv.reader(handle))
    assert header[-1] == "val_logcount_predictions_loss"
    provenance = _load_payload(args.output_prefix + ".pt", "cpu")["provenance"]
    assert set(provenance) == {"source_repository_commit", "seed",
                               "epochs_completed", "benchmark"}


def test_pytorch_only_options_are_refused_by_the_other_engines():
    for name in ("tensorflow", "jax", "tensorflow-old"):
        engine = SimpleNamespace(name=name)
        with pytest.raises(ValueError, match="only apply to the PyTorch engine"):
            engines._check_combinations(engine, SimpleNamespace(bias_profile_only=True))
    engines._check_combinations(SimpleNamespace(name="pytorch"),
                                SimpleNamespace(bias_profile_only=True))


def test_absent_pytorch_only_options_are_not_recorded():
    args = SimpleNamespace(bias_profile_only=None)
    engines.configure_from_args(args, require=False)
    assert vars(args) == {}
    args = SimpleNamespace(bias_profile_only=True)
    engines.configure_from_args(args, require=False)
    assert vars(args) == {"bias_profile_only": True}


# --------------------------------------------------------------------------
# 3. chrombpnet distill
# --------------------------------------------------------------------------


def _teachers(count=3, seed=0, **changes):
    torch.manual_seed(seed)
    return [BPNet(_config(**changes)).eval() for _ in range(count)]


def test_teacher_targets_are_mean_probabilities_times_mean_counts():
    teachers = _teachers()
    sequence = _ncl(_sequences(4))
    profile_target, logcount_target = distill.TeacherTargets(teachers)(sequence)
    with torch.no_grad():
        outputs = [teacher(sequence) for teacher in teachers]
    probabilities = torch.stack([torch.softmax(p, -1) for p, _ in outputs]).mean(0)
    logcounts = torch.stack([c for _, c in outputs]).mean(0)
    # A negative mean log count (an untrained teacher's) is a zero total.
    totals = torch.expm1(logcounts).clamp_min(0.0)
    torch.testing.assert_close(logcount_target, logcounts)
    torch.testing.assert_close(profile_target, probabilities * totals)
    torch.testing.assert_close(profile_target.sum(-1, keepdim=True), totals)
    assert not profile_target.requires_grad

    with torch.no_grad():
        for teacher in teachers:
            teacher.count_dense.bias.add_(5.0)
    profile_target, logcount_target = distill.TeacherTargets(teachers)(sequence)
    assert (logcount_target > 0).all()
    torch.testing.assert_close(profile_target.sum(-1, keepdim=True),
                               torch.expm1(logcount_target))


def test_one_teacher_is_its_own_target():
    teacher = _teachers(1)[0]
    with torch.no_grad():
        teacher.count_dense.bias.add_(5.0)   # a non-zero total to divide by
    sequence = _ncl(_sequences(2))
    profile_target, logcount_target = distill.TeacherTargets([teacher])(sequence)
    with torch.no_grad():
        profile, logcounts = teacher(sequence)
    torch.testing.assert_close(logcount_target, logcounts)
    torch.testing.assert_close(profile_target / profile_target.sum(-1, keepdim=True),
                               torch.softmax(profile, -1))


def test_load_teachers_takes_nobias_or_full_models_and_refuses_bias(tmp_path):
    teacher = _teachers(1)[0]
    save_model(teacher, tmp_path / "nobias.pt")
    save_model(ChromBPNet(_config(), BiasModel(_config()), teacher), tmp_path / "full.pt")
    save_model(BiasModel(_config()), tmp_path / "bias.pt")
    loaded = distill.load_teachers([tmp_path / "nobias.pt", tmp_path / "full.pt"], "cpu")
    for model in loaded:
        assert type(model) is BPNet and not model.training
        assert not any(p.requires_grad for p in model.parameters())
        for name, value in teacher.state_dict().items():
            assert torch.equal(model.state_dict()[name], value)
    with pytest.raises(ValueError, match="bias model"):
        distill.load_teachers([tmp_path / "bias.pt"], "cpu")
    with pytest.raises(ValueError, match=".pt"):
        distill.load_teachers([tmp_path / "model.h5"], "cpu")


def test_all_chromosome_folds_trains_on_every_chromosome(tmp_path):
    source = tmp_path / "fold.json"
    source.write_text(json.dumps({"train": ["chr1", "chr2"], "valid": ["chr3"],
                                  "test": ["chr4"]}))
    path = distill.all_chromosome_folds(str(source), str(tmp_path / "all.json"))
    assert json.loads(open(path).read()) == {
        "train": ["chr1", "chr2", "chr3", "chr4"], "valid": [], "test": []}


CHROM_SIZES = {"chr1": 4000, "chr2": 4000, "chr3": 4000}


def _genomics(directory):
    """A three-chromosome genome, a bigwig and peak/nonpeak BEDs."""
    import pyBigWig

    rng = np.random.RandomState(0)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "genome.fa").write_text("".join(
        f">{chrom}\n" + "".join(rng.choice(list("ACGT"), size)) + "\n"
        for chrom, size in CHROM_SIZES.items()))
    (directory / "folds.json").write_text(json.dumps(
        {"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}))
    handle = pyBigWig.open(str(directory / "data.bw"), "w")
    handle.addHeader(list(CHROM_SIZES.items()))
    for chrom, size in CHROM_SIZES.items():
        handle.addEntries(chrom, 0, values=rng.poisson(1.0, size).astype(np.float64),
                          span=1, step=1)
    handle.close()
    for name, offset in (("peaks", 0), ("nonpeaks", 150)):
        with open(directory / f"{name}.bed", "w") as bed:
            for chrom in CHROM_SIZES:
                for index in range(8):
                    start = 400 + 400 * index + offset - INPUT // 2
                    bed.write("\t".join(str(field) for field in (
                        chrom, start, start + INPUT,
                        ".", ".", ".", ".", ".", ".", INPUT // 2)) + "\n")
    return directory


def _distill_options(directory, prefix, teachers, *extra):
    argv = ["distill", "-t", *map(str, teachers), "-g", str(directory / "genome.fa"),
            "-bw", str(directory / "data.bw"), "-p", str(directory / "peaks.bed"),
            "-n", str(directory / "nonpeaks.bed"), "-fl", str(directory / "folds.json"),
            "-op", str(prefix), "-d", "ATAC", "-bs", "4", "-j", "10", "-sr", "1.0",
            "--device", "cpu", *extra]
    args = parsers.read_parser(argv)
    engines.configure_from_args(args)
    parsers.drop_absent(args)
    return Options.from_namespace(args)


def test_a_tiny_distillation_run(tmp_path):
    directory = _genomics(tmp_path / "data")
    teacher_paths = []
    for index, teacher in enumerate(_teachers(2, counts_loss_weight=3.0)):
        teacher_paths.append(tmp_path / f"teacher_{index}.pt")
        save_model(teacher, teacher_paths[-1])

    prefix = tmp_path / "lofo" / "student"
    distill.main(_distill_options(directory, prefix, teacher_paths, "-e", "2", "-es", "1"))
    payload = _load_payload(str(prefix) + ".pt", "cpu")
    assert payload["model_type"] == "chrombpnet_nobias"
    assert payload["config"]["filters"] == 4
    assert payload["config"]["counts_loss_weight"] == 3.0
    assert payload["provenance"]["distilled_from"] == [str(p) for p in teacher_paths]
    assert payload["provenance"]["all_chromosomes"] is False
    with open(str(prefix) + ".log") as handle:
        rows = list(csv.DictReader(handle))
    assert 1 <= len(rows) <= 2
    assert all(np.isfinite(float(row["val_loss"])) for row in rows)
    assert json.load(open(str(prefix) + ".args.json"))["teachers"] == [
        str(p) for p in teacher_paths]
    # The student is an ordinary bias-free model.
    student = load_model(str(prefix) + ".pt")
    assert type(student) is BPNet

    final = tmp_path / "final" / "student"
    distill.main(_distill_options(directory, final, teacher_paths, "-e", "2",
                                  "--all-chromosomes", "-fil", "6"))
    payload = _load_payload(str(final) + ".pt", "cpu")
    assert payload["provenance"]["all_chromosomes"] is True
    assert payload["provenance"]["epochs_completed"] == 2
    assert payload["provenance"]["best_epoch"] == 1
    assert payload["config"]["filters"] == 6
    assert json.load(open(str(final) + ".folds.json"))["train"] == ["chr1", "chr2", "chr3"]
    with open(str(final) + ".log") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and all(row["val_loss"] == "nan" for row in rows)


def test_the_student_learns_its_teachers(tmp_path):
    """Distilling one teacher pulls the student's predictions toward it."""
    teacher = _teachers(1, seed=5)[0]
    generator = _Generator(7, n=16)
    sequence = _ncl(generator.sequences)

    def distance(model):
        with torch.no_grad():
            p, c = model(sequence)
            q, d = teacher(sequence)
        return float(((torch.softmax(p, -1) - torch.softmax(q, -1)) ** 2).sum()
                     + ((c - d) ** 2).sum())

    torch.manual_seed(1)
    student = BPNet(_config())
    before = distance(student)
    fit_and_evaluate(student, generator, generator,
                     _fit_args(tmp_path, epochs=30, early_stop=30), _NO_EXPORT,
                     {"counts_loss_weight": 2.0},
                     target_fn=distill.TeacherTargets([teacher]))
    assert distance(load_model(str(tmp_path / "model.pt"))) < 0.5 * before
