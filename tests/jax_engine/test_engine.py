"""The JAX engine behind the engine interface (chrombpnet-engines' own tests).

Agreement with the original implementation is ``tests/parity/jax_engine.py``'s
job; these pin the engine's behaviour on tiny models, on the CPU.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import chrombpnet.engines.jax  # noqa: F401  (sets KERAS_BACKEND=jax before keras is imported)
import keras

from chrombpnet import engines
from chrombpnet.engines.jax.attribution import SequentialReferences, sequences_per_step

from .test_dinuc_shuffle import deeplift_dinuc_shuffle

INPUTLEN, OUTPUTLEN, FILTERS, N_DIL = 164, 20, 4, 2
K = 3


@pytest.fixture(scope="module")
def engine():
    engine = engines.get("jax")
    engine.require()
    return engine


@pytest.fixture(autouse=True)
def _restore_engine():
    previous = engines._CURRENT
    yield
    engines._CURRENT = previous


def onehot(rng, n, length=INPUTLEN):
    return np.eye(4, dtype=np.int8)[rng.randint(0, 4, size=(n, length))]


def parameters(**extra):
    values = {"filters": str(FILTERS), "n_dil_layers": str(N_DIL), "counts_loss_weight": "1.0",
              "inputlen": str(INPUTLEN), "outputlen": str(OUTPUTLEN)}
    values.update(extra)
    return values


def training_args(engine, kind, prefix, **extra):
    values = dict(seed=7, learning_rate=0.01, epochs=4, early_stop=10, device="cpu",
                  output_prefix=str(prefix), architecture_from_file=engine.default_architecture(kind))
    values.update(extra)
    return SimpleNamespace(**values)


class Generator:
    """The shared generator's shape: (x, [profile_counts, log1p_total])."""

    def __init__(self, seed, n_batches=3, batch_size=4):
        rng = np.random.RandomState(seed)
        self.x = onehot(rng, n_batches * batch_size).astype(np.float32)
        self.cts = rng.poisson(1.0, (n_batches * batch_size, OUTPUTLEN)).astype(np.float32)
        self.batch_size = batch_size
        self.epochs_ended = 0

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, index):
        s = slice(index * self.batch_size, (index + 1) * self.batch_size)
        return self.x[s], [self.cts[s], np.log1p(self.cts[s].sum(-1, keepdims=True))]

    def on_epoch_end(self):
        self.epochs_ended += 1


# -- DeepSHAP references -------------------------------------------------------

def test_references_are_the_tensorflow_engines_stream():
    """One RandomState(seed), K single deeplift shuffles per sequence, in order."""
    seqs = onehot(np.random.RandomState(0), 5)
    rng = np.random.RandomState(1234)
    want = np.stack([np.stack([deeplift_dinuc_shuffle(s, rng=rng) for _ in range(K)]) for s in seqs])
    got = SequentialReferences(K, 1234)(seqs, 0, len(seqs))
    assert got.dtype == np.int8
    np.testing.assert_array_equal(got, want.astype(np.int8))


def test_references_survive_prefetch_and_out_of_memory_retries():
    """iter_explain prefetches the next chunk, then may retry the current one
    in smaller pieces; every sequence keeps the references it was first given."""
    seqs = onehot(np.random.RandomState(0), 12)
    want = SequentialReferences(K, 5)(seqs, 0, 12)
    refs = SequentialReferences(K, 5)
    calls = [(0, 4), (4, 8),            # chunk 0, prefetch chunk 1
             (0, 2), (2, 4),            # OOM: chunk 0 again, halved
             (4, 6), (6, 8), (8, 10), (10, 12)]
    for start, stop in calls:
        np.testing.assert_array_equal(refs(seqs[start:stop], start, stop), want[start:stop])


def test_interpret_batch_size_counts_pairs():
    assert sequences_per_step(16, 20) == 1
    assert sequences_per_step(80, 20) == 4
    assert sequences_per_step(64, 3) == 21


# -- training ------------------------------------------------------------------

def test_bias_training_writes_keras_checkpoints_and_restores_the_best_epoch(engine, tmp_path):
    prefix = tmp_path / "bias"
    args = training_args(engine, "bias", prefix)
    model, module = engine.get_model(args, parameters())
    train, valid = Generator(0), Generator(1)
    engine.fit(model, train, valid, args, parameters(), module)
    for suffix in (".keras", ".log", ".log.batch"):
        assert os.path.exists(str(prefix) + suffix), suffix
    # one on_epoch_end per completed pass, as under TF-Keras (see sequence_adapter)
    assert train.epochs_ended == args.epochs
    assert valid.epochs_ended == args.epochs
    log =np.genfromtxt(str(prefix) + ".log", delimiter=",", names=True)
    assert len(log) == args.epochs
    # The saved model is the best validation epoch, whatever the last one was.
    from chrombpnet.engines.jax import model_io
    from chrombpnet.engines.jax.sequence_adapter import KerasPyDataset

    saved = model_io.load_model(str(prefix) + ".keras", compile=True)
    val_loss = saved.evaluate(KerasPyDataset(valid), verbose=0, return_dict=True)["loss"]
    np.testing.assert_allclose(val_loss, np.min(log["val_loss"]), rtol=1e-4)
    # fit gives the cosine schedule its step count without touching args
    assert not hasattr(args, "total_steps")


def test_training_extras_together(engine, tmp_path):
    """--optimizer muon --lr-schedule cosine --ema --precision bf16, through
    get_model and fit: the cosine schedule gets its step count from fit, the
    saved model is float32, and the process-wide precision is restored."""
    import jax

    prefix = tmp_path / "bias"
    args = training_args(engine, "bias", prefix, epochs=2, optimizer="muon", lr_schedule="cosine",
                         ema=True, precision="bf16")
    model, module = engine.get_model(args, parameters())
    assert keras.config.dtype_policy().name == "mixed_bfloat16"
    engine.fit(model, Generator(0), Generator(1), args, parameters(), module)
    assert keras.config.dtype_policy().name == "float32"
    assert jax.config.jax_default_matmul_precision is None
    assert not hasattr(args, "total_steps")
    from chrombpnet.engines.jax import model_io
    saved = model_io.load_model(str(prefix) + ".keras")
    assert {layer.dtype_policy.name for layer in saved.layers} == {"float32"}


def test_seed_sets_the_bias_factorized_tower(engine, tmp_path):
    bias, module = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    bias_path = str(tmp_path / "bias.keras")
    bias.save(bias_path)

    def tower(seed):
        args = training_args(engine, "chrombpnet", tmp_path / "c", seed=seed)
        model, _ = engine.get_model(args, parameters(bias_model_path=bias_path))
        return model.get_layer("model_wo_bias").get_weights()

    first, again, other = tower(7), tower(7), tower(8)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_chrombpnet_training_writes_the_nobias_model(engine, tmp_path):
    bias, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    bias_path = str(tmp_path / "bias.keras")
    bias.save(bias_path)
    prefix = tmp_path / "chrombpnet"
    args = training_args(engine, "chrombpnet", prefix, epochs=2)
    params = parameters(bias_model_path=bias_path)
    model, module = engine.get_model(args, params)
    engine.fit(model, Generator(0), Generator(1), args, params, module)
    full = engine.load_predictor(str(prefix) + ".keras", announce=False, summary=False)
    nobias = engine.load_predictor(str(prefix) + "_nobias.keras", announce=False, summary=False)
    bias = engine.load_predictor(bias_path, announce=False, summary=False)
    x = Generator(2)[0][0]
    p, c = full.predict(x, verbose=0)
    p_nb, c_nb = nobias.predict(x, verbose=0)
    p_b, c_b = bias.predict(x, verbose=0)
    np.testing.assert_allclose(p, p_nb + p_b, atol=1e-5)
    np.testing.assert_allclose(c, np.logaddexp(c_nb, c_b), atol=1e-5)


# -- prediction and rescale -----------------------------------------------------

def test_predictor_pads_a_short_final_batch(engine, tmp_path):
    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    path = str(tmp_path / "bias.keras")
    model.save(path)
    predictor = engine.load_predictor(path, announce=False, summary=False)
    assert predictor.input_shape == (None, INPUTLEN, 4)
    x = Generator(0, n_batches=2)[0][0]
    full = predictor.predict_on_batch(x)
    short = predictor.predict_on_batch(x[:3])
    assert short[0].shape == (3, OUTPUTLEN) and short[1].shape == (3, 1)
    np.testing.assert_allclose(short[0], full[0][:3], atol=1e-6)
    np.testing.assert_allclose(short[1], full[1][:3], atol=1e-6)


def test_rescale_shifts_the_count_head_and_saves_keras(engine, tmp_path):
    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    path = str(tmp_path / "bias.keras")
    model.save(path)
    gen = Generator(0)
    x, cts = gen.x, gen.cts.sum(-1)
    out = str(tmp_path / "bias_model_scaled.keras")
    engine.rescale_bias_logcounts(path, out, x, cts)
    before = engine.load_predictor(path, announce=False, summary=False).predict(x, verbose=0)[1]
    after = engine.load_predictor(out, announce=False, summary=False).predict(x, verbose=0)[1]
    np.testing.assert_allclose(np.mean(after), np.mean(np.log1p(cts)), atol=1e-4)
    np.testing.assert_allclose(after - before, np.full_like(after, np.mean(after - before)), atol=1e-5)


# -- export --------------------------------------------------------------------

def test_export_writes_a_tf_keras_2_file_from_the_cli(engine, tmp_path):
    import subprocess
    import sys

    import h5py

    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    source, target = str(tmp_path / "bias.keras"), str(tmp_path / "bias.legacy.h5")
    model.save(source)
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.CHROMBPNET", "export", "--legacy-h5", "-m", source, "-o", target],
        capture_output=True, text=True, env=dict(os.environ, JAX_PLATFORMS="cpu"))
    assert completed.returncode == 0, completed.stderr
    with h5py.File(target, "r") as handle:
        assert str(handle.attrs["keras_version"]).startswith("2.")
        assert handle.attrs["backend"] == "tensorflow"
    # and the JAX engine reads its own export back
    exported = engine.load_predictor(target, announce=False, summary=False)
    x = Generator(0)[0][0]
    np.testing.assert_allclose(exported.predict(x, verbose=0)[0], model.predict(x, verbose=0)[0], atol=1e-6)


# -- attribution ----------------------------------------------------------------

def test_attribute_yields_counts_then_profile(engine, tmp_path):
    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    path = str(tmp_path / "bias.keras")
    model.save(path)
    loaded = engine.load_attribution_model(path, "cpu")
    assert engine.input_length(loaded) == INPUTLEN
    seqs = onehot(np.random.RandomState(3), 3)
    heads = list(engine.attribute(loaded, seqs, ["profile", "counts"], device="cpu",
                                  batch_size=2 * K, num_shuffles=K, seed=11))
    assert [head for head, _ in heads] == ["counts", "profile"]
    for _, scores in heads:
        assert scores.shape == seqs.shape and scores.dtype == np.float32
    # The same seed gives the same scores, whatever the chunking.
    again = dict(engine.attribute(loaded, seqs, ["counts"], device="cpu",
                                  batch_size=K, num_shuffles=K, seed=11))
    np.testing.assert_allclose(again["counts"], heads[0][1], rtol=1e-5, atol=1e-7)


def test_attribute_weights_the_profile_as_the_original_unless_stopgrad_is_asked_for(engine, tmp_path):
    """By default the profile logits are weighted by the shap fork's midpoint
    ("chrombpnet", as chrombpnet 1.x computes them); with
    --contribs-profile-stopgrad, by softmax(mn_x) ("softmax_x")."""
    from chrombpnet.engines.jax.attribution.explainer import DeepLiftShap

    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    seqs = onehot(np.random.RandomState(3), 3)
    def attribute(stop_gradient):
        return dict(engine.attribute(model, seqs, ["profile"], device="cpu",
                                     batch_size=K, num_shuffles=K, seed=11,
                                     profile_stop_gradient=stop_gradient))["profile"]

    def explain(weighting):
        explainer = DeepLiftShap(model, heads=("profile",), profile_weighting=weighting,
                                 num_shuffles=K)
        return explainer.explain(seqs, references=SequentialReferences(K, 11))["profile"]

    midpoint, honoured = explain("chrombpnet"), explain("softmax_x")
    np.testing.assert_allclose(attribute(False), midpoint, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(attribute(True), honoured, rtol=1e-5, atol=1e-7)
    assert np.linalg.norm(honoured - midpoint) > 1e-3 * np.linalg.norm(midpoint)
