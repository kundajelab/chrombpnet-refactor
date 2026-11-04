"""Loading Keras checkpoints, and placing TensorFlow on a device.

A Keras model already *is* the predictor the shared code expects:
``input_shape == (None, inputlen, 4)``, ``output_shape`` a list of two shapes,
and ``predict``/``predict_on_batch`` returning ``[profile_logits, logcounts]``.
So, as in the original implementation, prediction gets the model itself.

The custom objects and ``compile`` flags follow the original's loaders:
``compile=False`` for prediction and attribution
(``training/predict.py``, ``evaluation/interpret/input_utils.py``), and the
default ``compile=True`` where a model is loaded to be modified and saved again
(``helpers/hyperparameters/param_utils.py``,
``chrombpnet_with_bias_model.load_pretrained_bias``), because only a compiled
model saves its optimizer and training configuration.
"""

from __future__ import annotations

import warnings


def custom_objects():
    """The original loaders' ``tf`` and ``multinomial_nll``, and the count
    head's Lambda, which loads 1.x's Python 3.8 bytecode head on any Python
    (``layers``)."""
    import tensorflow as tf
    from chrombpnet.engines.tensorflow_old import layers, losses
    return {"tf": tf, "multinomial_nll": losses.multinomial_nll,
            "Lambda": layers.CountHeadLambda,
            layers.chrombpnet_logsumexp.__name__: layers.chrombpnet_logsumexp}


def refuse_keras3(path):
    """Fail clearly on a Keras 3 ``.h5`` (ChromBPNet 2.0's format).

    tf-keras (Keras 2) cannot deserialize Keras 3 model configs and would fail
    deep inside its loader; the JAX engine reads these files.
    """
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            version = handle.attrs.get("keras_version", "")
    except OSError:
        return
    if isinstance(version, bytes):
        version = version.decode("utf-8")
    if str(version).startswith("3."):
        raise ValueError(
            f"{path} was written by Keras {version} (e.g. ChromBPNet 2.0) and "
            f"cannot be loaded by the tensorflow-old engine's Keras 2 (tf-keras); use --engine tensorflow or jax")


def load_model(path, compile=True):
    from chrombpnet.engines.tensorflow_old import require_tf_keras

    require_tf_keras()
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.utils import get_custom_objects

    refuse_keras3(path)
    get_custom_objects().update(custom_objects())
    return keras_load_model(path, compile=compile)


def configure_device(value=None):
    """Restrict TensorFlow to the ``--device`` the user asked for.

    ``None`` leaves TensorFlow's own placement alone (a GPU when one is
    visible), which is what the original implementation did. ``cpu`` hides
    every GPU; ``cuda``/``cuda:N`` keeps only GPU N. TensorFlow allows this only
    before it has initialized its devices; afterwards the existing placement
    stands and a warning says so.
    """
    if value is None:
        return None
    import tensorflow as tf

    text = str(value).strip().lower()
    if text == "cpu":
        visible = []
    elif text == "cuda" or text.startswith("cuda:"):
        index = int(text.split(":", 1)[1]) if ":" in text else 0
        gpus = tf.config.list_physical_devices("GPU")
        if index >= len(gpus):
            raise RuntimeError("CUDA was requested but is not available")
        visible = [gpus[index]]
    else:
        raise ValueError(f"Unknown device {value!r}; expected cpu, cuda or cuda:N")
    try:
        tf.config.set_visible_devices(visible, "GPU")
    except RuntimeError as error:
        warnings.warn(
            f"--device {value} could not be applied because TensorFlow has "
            f"already initialized its devices ({error})", RuntimeWarning)
    return text


def load_model_wrapper(path, device=None, *, announce=True, summary=True):
    """Load a Keras checkpoint for prediction, with the CLI's console output."""
    configure_device(device)
    model = load_model(path, compile=False)
    if announce:
        print("got the model")
    if summary:
        model.summary()
    return model
