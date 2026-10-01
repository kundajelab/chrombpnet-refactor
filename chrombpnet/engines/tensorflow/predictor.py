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
    import tensorflow as tf
    from chrombpnet.engines.tensorflow import losses
    return {"tf": tf, "multinomial_nll": losses.multinomial_nll}


def load_model(path, compile=True):
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.utils import get_custom_objects

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
