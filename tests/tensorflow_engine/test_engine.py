"""The tensorflow engine (Keras 3 on TensorFlow) behind the engine interface, on the CPU.

The JAX engine's own behavioural tests (``tests/jax_engine/test_engine.py``) are imported and run again here with
this engine as their ``engine`` fixture: the same Keras 3 code, on TensorFlow. Three of them read JAX state or
the JAX explainer and have TensorFlow versions below. Agreement with the original implementation and with the
other engines is ``tests/parity``'s (``jax_engine.py --engine tensorflow``, ``deepshap_engines.py``).
"""

import os
import subprocess
import sys

import keras
import numpy as np
import pytest

from chrombpnet import engines
from chrombpnet.engines.jax.attribution import SequentialReferences
from tests.jax_engine.test_engine import (  # noqa: F401  (collected here, with this module's engine fixture)
    INPUTLEN, K, Generator, onehot, parameters, training_args,
    test_attribute_yields_counts_then_profile,
    test_bias_training_writes_keras_checkpoints_and_restores_the_best_epoch,
    test_chrombpnet_training_writes_the_nobias_model,
    test_interpret_batch_size_counts_pairs,
    test_predictor_pads_a_short_final_batch,
    test_references_are_the_tensorflow_engines_stream,
    test_references_survive_prefetch_and_out_of_memory_retries,
    test_rescale_shifts_the_count_head_and_saves_keras,
    test_seed_sets_the_bias_factorized_tower,
)


@pytest.fixture(scope="module")
def engine():
    engine = engines.get("tensorflow")
    engine.require()
    return engine


@pytest.fixture(autouse=True)
def _restore_engine():
    previous = engines._CURRENT
    yield
    engines._CURRENT = previous


def test_keras_runs_on_tensorflow(engine):
    """Keras 3 on TensorFlow; the models are Keras 3's (keras.src), not tf-keras's."""
    assert keras.backend.backend() == "tensorflow" and keras.__version__.startswith("3.")
    assert engine.name == "tensorflow" and engine.checkpoint_suffix == ".keras"
    model, _ = engine.get_model(training_args(engine, "bias", "unused"), parameters())
    assert isinstance(model, keras.Model)
    assert type(model).__module__.startswith("keras.src")


def test_keras3_and_tf_keras_share_one_process():
    """The tensorflow engine sets TF_USE_LEGACY_KERAS=1 when imported; that redirects tf.keras only, so a
    process (the engine registry imports every engine) still builds Keras 3 models after it."""
    code = ("import chrombpnet.engines.tensorflow_old, os; assert os.environ['TF_USE_LEGACY_KERAS'] == '1'; "
            "from chrombpnet import engines; e = engines.get('tensorflow'); e.require(); "
            "import keras, tensorflow as tf; assert keras.backend.backend() == 'tensorflow'; "
            "assert tf.keras.Model.__module__.startswith('tf_keras'), tf.keras.Model.__module__; "
            "print(keras.Sequential([keras.Input((3,)), keras.layers.Dense(1)])(tf.ones((2, 3))).shape)")
    env = {k: v for k, v in os.environ.items() if k not in ("KERAS_BACKEND", "TF_USE_LEGACY_KERAS")}
    completed = subprocess.run([sys.executable, "-c", code], env=dict(env, CUDA_VISIBLE_DEVICES=""),
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "(2, 1)" in completed.stdout


def test_training_extras_together(engine, tmp_path):
    """The JAX test of --optimizer muon --lr-schedule cosine --ema --precision bf16 on TensorFlow, where the
    precision that fit restores is TensorFlow's TF32 switch."""
    import tensorflow as tf

    before = tf.config.experimental.tensor_float_32_execution_enabled()
    prefix = tmp_path / "bias"
    args = training_args(engine, "bias", prefix, epochs=2, optimizer="muon", lr_schedule="cosine",
                         ema=True, precision="bf16")
    model, module = engine.get_model(args, parameters())
    assert keras.config.dtype_policy().name == "mixed_bfloat16"
    engine.fit(model, Generator(0), Generator(1), args, parameters(), module)
    assert keras.config.dtype_policy().name == "float32"
    assert tf.config.experimental.tensor_float_32_execution_enabled() == before
    assert not hasattr(args, "total_steps")
    from chrombpnet.engines.jax import model_io
    saved = model_io.load_model(str(prefix) + ".keras")
    assert {layer.dtype_policy.name for layer in saved.layers} == {"float32"}


def test_highest_precision_turns_tf32_off_for_training_only(engine, tmp_path):
    import tensorflow as tf

    from chrombpnet.engines.jax import runtime

    assert runtime.configure_precision("highest") == "highest"
    assert tf.config.experimental.tensor_float_32_execution_enabled() is False
    assert runtime.configure_precision(None) == "default"
    assert tf.config.experimental.tensor_float_32_execution_enabled() is True
    args = training_args(engine, "bias", tmp_path / "bias", epochs=1, precision="highest")
    model, module = engine.get_model(args, parameters())
    assert tf.config.experimental.tensor_float_32_execution_enabled() is False
    engine.fit(model, Generator(0), Generator(1), args, parameters(), module)
    assert tf.config.experimental.tensor_float_32_execution_enabled() is True
    info = runtime.runtime_info()
    assert info["keras_backend"] == "tensorflow" and info["tensor_float_32"] is True


def test_device_values_are_checked(engine):
    from chrombpnet.engines.jax import runtime

    assert runtime.configure_device(None) is None
    with pytest.raises(ValueError, match="Unknown device"):
        runtime.configure_device("tpu")


def test_a_jax_written_checkpoint_and_this_engines_are_the_same_format(engine, tmp_path):
    """The engines share .keras files: what one saves, the other loads (the JAX side is checked by the
    parity harness, which runs both); here a save and load through this engine round-trips exactly."""
    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    path = str(tmp_path / "bias.keras")
    model.save(path)
    assert engines.for_model(path).name in engines.KERAS3_ENGINES
    loaded = engine.load_predictor(path, announce=False, summary=False)
    x = Generator(0)[0][0]
    for a, b in zip(model.predict(x, verbose=0), loaded.predict(x, verbose=0)):
        np.testing.assert_array_equal(a, b)


def test_export_writes_a_tf_keras_2_file_from_the_cli(engine, tmp_path):
    """``export --engine tensorflow``: the 1.x .h5 layout, written from the TensorFlow environment."""
    import h5py

    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    source, target = str(tmp_path / "bias.keras"), str(tmp_path / "bias.legacy.h5")
    model.save(source)
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.CHROMBPNET", "export", "--legacy-h5", "--engine", "tensorflow",
         "-m", source, "-o", target],
        capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert completed.returncode == 0, completed.stderr
    with h5py.File(target, "r") as handle:
        assert str(handle.attrs["keras_version"]).startswith("2.")
        assert handle.attrs["backend"] == "tensorflow"
    exported = engine.load_predictor(target, announce=False, summary=False)
    x = Generator(0)[0][0]
    np.testing.assert_allclose(exported.predict(x, verbose=0)[0], model.predict(x, verbose=0)[0], atol=1e-6)


def test_attribute_weights_the_profile_as_the_original_unless_stopgrad_is_asked_for(engine, tmp_path):
    """The JAX test, with this engine's (TensorFlow) explainer."""
    from chrombpnet.engines.tensorflow.explainer import DeepLiftShap

    model, _ = engine.get_model(training_args(engine, "bias", tmp_path / "b"), parameters())
    seqs = onehot(np.random.RandomState(3), 3)

    def attribute(stop_gradient):
        return dict(engine.attribute(model, seqs, ["profile"], device="cpu",
                                     batch_size=K, num_shuffles=K, seed=11,
                                     profile_stop_gradient=stop_gradient))["profile"]

    def explain(weighting):
        explainer = DeepLiftShap(model, heads=("profile",), profile_weighting=weighting, num_shuffles=K)
        return explainer.explain(seqs, references=SequentialReferences(K, 11))["profile"]

    midpoint, honoured = explain("chrombpnet"), explain("softmax_x")
    np.testing.assert_allclose(attribute(False), midpoint, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(attribute(True), honoured, rtol=1e-5, atol=1e-7)
    assert np.linalg.norm(honoured - midpoint) > 1e-3 * np.linalg.norm(midpoint)
