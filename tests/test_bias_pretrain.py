"""``chrombpnet bias pretrain`` and ``bias pipeline --pretrained-bias``: a
multitask profile-only bias trunk, and a bias model whose only trained part
is a new profile head over that trunk, frozen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# PyTorch-engine tests: skipped in the TensorFlow and JAX environments.
torch = pytest.importorskip("torch")

from chrombpnet import engines, parsers
from chrombpnet.engines.pytorch import bias_pretrain
from chrombpnet.engines.pytorch.losses import multinomial_nll
from chrombpnet.engines.pytorch.model import (
    BiasModel,
    ModelConfig,
    MultitaskBias,
    PredictionAdapter,
    _load_payload,
    load_model,
    save_model,
)
from chrombpnet.engines.pytorch.train import fit_and_evaluate, get_model

INPUT, OUTPUT = 200, 50
FOLD = {"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}


def _config(**changes):
    values = dict(input_length=INPUT, output_length=OUTPUT, filters=4,
                  n_dilation_layers=1, counts_loss_weight=2.0, counts_head=False)
    values.update(changes)
    return ModelConfig(**values)


def _sequences(n, seed=0):
    bases = np.random.RandomState(seed).randint(0, 4, size=(n, INPUT))
    return torch.from_numpy(np.eye(4, dtype=np.float32)[bases]).permute(0, 2, 1).contiguous()


def _pretrained(tmp_path, fold=FOLD, assay="ATAC", **changes):
    torch.manual_seed(0)
    model = MultitaskBias(_config(**changes), ["dsA", "dsB"], assay, fold)
    path = tmp_path / "multitask_bias.pt"
    save_model(model, path)
    return model, path


def _adapt_args(tmp_path, path, assay="ATAC", fold=FOLD):
    (tmp_path / "fold.json").write_text(json.dumps(fold))
    return SimpleNamespace(pretrained_bias=str(path), data_type=assay,
                           chr_fold_path=str(tmp_path / "fold.json"), seed=3)


def _trunk_equal(pretrained, model):
    return all(torch.equal(value, model.state_dict()[key])
               for key, value in pretrained.trunk.state_dict().items())


# --------------------------------------------------------------------------
# The multitask model and its checkpoint
# --------------------------------------------------------------------------


def test_each_dataset_has_its_own_head_over_one_trunk():
    model = MultitaskBias(_config(), ["dsA", "dsB"], "ATAC")
    sequence = _sequences(3)
    assert model(sequence, "dsA").shape == (3, OUTPUT)
    assert not hasattr(model.trunk, "profile_conv")
    # A dataset's loss trains the trunk and its own head, not the other's.
    multinomial_nll(torch.ones(3, OUTPUT), model(sequence, "dsA")).backward()
    assert model.trunk.first_conv.weight.grad is not None
    assert model.heads["dsA"].weight.grad is not None
    assert model.heads["dsB"].weight.grad is None


@pytest.mark.parametrize("tasks, assay", [(["dsA"], "ATAC"), (["dsA", "dsA"], "ATAC"),
                                          (["ds.A", "dsB"], "ATAC"), (["dsA", "dsB"], "RNA")])
def test_a_multitask_bias_needs_two_named_datasets_of_an_assay(tasks, assay):
    with pytest.raises(ValueError):
        MultitaskBias(_config(), tasks, assay)


def test_multitask_checkpoints_round_trip(tmp_path):
    model, path = _pretrained(tmp_path)
    assert _load_payload(path, "cpu")["model_type"] == "multitask_bias"
    loaded = load_model(path)
    assert isinstance(loaded, MultitaskBias)
    assert (loaded.tasks, loaded.assay, loaded.fold) == (["dsA", "dsB"], "ATAC", FOLD)
    assert not loaded.config.counts_head
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_prediction_refuses_a_multitask_bias(tmp_path):
    _, path = _pretrained(tmp_path)
    with pytest.raises(ValueError, match="bias pretrain"):
        PredictionAdapter(load_model(path))


# --------------------------------------------------------------------------
# Adaptation: --pretrained-bias
# --------------------------------------------------------------------------


def test_adapt_copies_and_freezes_the_trunk_under_a_zero_head(tmp_path):
    pretrained, path = _pretrained(tmp_path)
    torch.manual_seed(9)
    model = bias_pretrain.adapt(BiasModel(_config()), path, _adapt_args(tmp_path, path))
    assert _trunk_equal(pretrained, model)
    assert not model.profile_conv.weight.any() and not model.profile_conv.bias.any()
    trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    assert trainable == ["profile_conv.bias", "profile_conv.weight"]


@pytest.mark.parametrize("change, message", [
    ({"assay": "DNASE"}, "pretrained on ATAC"),
    ({"fold": {"train": ["chr2"], "valid": ["chr1"], "test": ["chr3"]}}, "another fold"),
])
def test_adapt_refuses_another_assay_or_fold(tmp_path, change, message):
    _, path = _pretrained(tmp_path)
    args = _adapt_args(tmp_path, path, **change)
    with pytest.raises(ValueError, match=message):
        bias_pretrain.adapt(BiasModel(_config()), path, args)


def test_adapt_refuses_another_architecture_or_a_counts_head(tmp_path):
    _, path = _pretrained(tmp_path)
    args = _adapt_args(tmp_path, path)
    with pytest.raises(ValueError, match="pass -fil 4"):
        bias_pretrain.adapt(BiasModel(_config(filters=6)), path, args)
    with pytest.raises(ValueError, match="profile-only"):
        bias_pretrain.adapt(BiasModel(_config(counts_head=True)), path, args)
    save_model(BiasModel(_config()), tmp_path / "bias.pt")
    with pytest.raises(ValueError, match="not a `chrombpnet bias pretrain`"):
        bias_pretrain.adapt(BiasModel(_config()), tmp_path / "bias.pt",
                            _adapt_args(tmp_path, tmp_path / "bias.pt"))


def test_the_bias_architecture_builds_an_adapted_profile_only_model(tmp_path):
    import chrombpnet.training.models.bpnet_model as bias_architecture

    pretrained, path = _pretrained(tmp_path)
    args = _adapt_args(tmp_path, path)
    args.architecture_from_file = bias_architecture.__file__
    parameters = {"inputlen": INPUT, "outputlen": OUTPUT, "filters": 4, "n_dil_layers": 1,
                  "counts_loss_weight": 1.0}
    model, _ = get_model(args, parameters)
    assert isinstance(model, BiasModel) and not model.config.counts_head
    assert _trunk_equal(pretrained, model)


class _Generator:
    """One fixed batch of sequences and Poisson profiles."""

    def __init__(self, seed, n=8):
        self.sequences = _sequences(n, seed).permute(0, 2, 1).numpy()
        self.profiles = np.random.RandomState(seed).poisson(1.0, (n, OUTPUT)).astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sequences, [self.profiles, np.log1p(self.profiles.sum(-1, keepdims=True))]

    def on_epoch_end(self):
        pass


def test_training_an_adapted_bias_moves_only_its_head(tmp_path):
    pretrained, path = _pretrained(tmp_path)
    model = bias_pretrain.adapt(BiasModel(_config()), path, _adapt_args(tmp_path, path))
    args = SimpleNamespace(output_prefix=str(tmp_path / "bias"), device="cpu",
                           learning_rate=0.01, epochs=3, early_stop=3, seed=11)
    fit_and_evaluate(model, _Generator(1), _Generator(2), args,
                     SimpleNamespace(save_model_without_bias=lambda m, p: None),
                     {"counts_loss_weight": 1.0})
    trained = load_model(str(tmp_path / "bias.pt"))
    assert _trunk_equal(pretrained, trained)
    assert trained.profile_conv.weight.any()


# --------------------------------------------------------------------------
# The manifest and the command line
# --------------------------------------------------------------------------


def _manifest(tmp_path, rows, header="dataset\tbigwig\tpeaks\tnonpeaks"):
    for name in ("data.bw", "peaks.bed", "nonpeaks.bed"):
        (tmp_path / name).write_text("")
    path = tmp_path / "manifest.tsv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_the_manifest_resolves_relative_paths(tmp_path):
    path = _manifest(tmp_path, ["dsA\tdata.bw\tpeaks.bed\tnonpeaks.bed",
                                f"dsB\t{tmp_path / 'data.bw'}\tpeaks.bed\tnonpeaks.bed"])
    rows = bias_pretrain.read_manifest(path)
    assert [row["dataset"] for row in rows] == ["dsA", "dsB"]
    assert all(row["bigwig"] == str(tmp_path / "data.bw") for row in rows)


@pytest.mark.parametrize("rows, header, message", [
    (["dsA\tdata.bw\tpeaks.bed\tnonpeaks.bed"], None, "two or more"),
    (["dsA\tdata.bw\tpeaks.bed\tnonpeaks.bed"] * 2, None, "two or more"),
    (["ds A\tdata.bw\tpeaks.bed\tnonpeaks.bed", "dsB\tdata.bw\tpeaks.bed\tnonpeaks.bed"], None,
     "letters, digits"),
    (["dsA\tmissing.bw\tpeaks.bed\tnonpeaks.bed", "dsB\tdata.bw\tpeaks.bed\tnonpeaks.bed"], None,
     "does not exist"),
    (["dsA\tdata.bw\tpeaks.bed", "dsB\tdata.bw\tpeaks.bed"], "dataset\tbigwig\tpeaks", "header"),
])
def test_the_manifest_refuses(tmp_path, rows, header, message):
    path = _manifest(tmp_path, rows, *([header] if header else []))
    with pytest.raises(ValueError, match=message):
        bias_pretrain.read_manifest(path)


def test_pretrained_bias_is_pytorch_only_and_unrecorded_when_absent():
    for name in ("tensorflow", "jax", "tensorflow-old"):
        with pytest.raises(ValueError, match="only apply to the PyTorch engine"):
            engines._check_combinations(SimpleNamespace(name=name),
                                        SimpleNamespace(pretrained_bias="x.pt"))
    args = SimpleNamespace(pretrained_bias=None)
    engines.configure_from_args(args, require=False)
    assert vars(args) == {}


def test_bias_pretrain_takes_the_bias_architecture_defaults():
    args = parsers.read_parser(["bias", "pretrain", "-m", "m.tsv", "-g", "g.fa", "-d", "ATAC",
                                "-fl", "f.json", "-b", "0.5", "-o", "out"])
    assert (args.filters, args.n_dilation_layers, args.max_jitter) == (128, 4, 0)
    assert (args.inputlen, args.outputlen, args.batch_size) == (2114, 1000, 64)
    train = parsers.read_parser(["bias", "train", "-ibw", "d.bw", "-d", "ATAC", "-g", "g.fa",
                                 "-c", "c.sizes", "-p", "p", "-n", "n", "-fl", "f.json",
                                 "-b", "0.5", "-o", "out", "--pretrained-bias", "x.pt"])
    assert train.pretrained_bias == "x.pt"


# --------------------------------------------------------------------------
# End to end: bias pretrain, then bias train --pretrained-bias
# --------------------------------------------------------------------------


def test_pretrain_then_adapt_end_to_end(tmp_path):
    from tests.regression import fixture
    from tests.regression.scenarios import GEOMETRY

    paths = fixture.build(tmp_path / "fixture")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("dataset\tbigwig\tpeaks\tnonpeaks\n" + "".join(
        f"{name}\t{paths['bigwig']}\t{paths['peaks']}\t{paths['nonpeaks']}\n"
        for name in ("dsA", "dsB")))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0", MPLBACKEND="Agg")

    def chrombpnet(*argv):
        subprocess.run([sys.executable, "-m", "chrombpnet.CHROMBPNET", *argv],
                       env=env, check=True, cwd=tmp_path, stdout=subprocess.DEVNULL)

    common = ["-g", str(paths["genome"]), "-d", "ATAC", "-fl", str(paths["folds"]),
              "-b", "0.5", "--device", "cpu", *GEOMETRY]
    chrombpnet("bias", "pretrain", "-m", str(manifest), "-o", str(tmp_path / "pre"), *common)
    checkpoint = tmp_path / "pre" / "models" / "multitask_bias.pt"
    pretrained = load_model(checkpoint)
    assert pretrained.tasks == ["dsA", "dsB"]
    assert pretrained.fold == json.loads(paths["folds"].read_text())
    log = (tmp_path / "pre" / "logs" / "multitask_bias.log").read_text().splitlines()
    assert log[0].split(",") == ["epoch", "loss", "val_loss", "loss_dsA", "loss_dsB",
                                 "val_loss_dsA", "val_loss_dsB"]

    chrombpnet("bias", "train", "-ibw", str(paths["bigwig"]), "-c", str(paths["chrom_sizes"]),
               "-p", str(paths["peaks"]), "-n", str(paths["nonpeaks"]), "-j", "0",
               "-o", str(tmp_path / "adapt"), "--pretrained-bias", str(checkpoint), *common)
    bias = load_model(tmp_path / "adapt" / "models" / "bias.pt")
    assert isinstance(bias, BiasModel) and not bias.config.counts_head
    assert _trunk_equal(pretrained, bias)
    assert bias.profile_conv.weight.any()
    record = json.loads((tmp_path / "adapt" / "logs" / "bias.args.json").read_text())
    assert record["pretrained_bias"] == str(checkpoint)
