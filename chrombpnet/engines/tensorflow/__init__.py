"""The TensorFlow engine: the original Keras ChromBPNet, behind the engine interface.

The model definitions, loss, callbacks, training call, bias rescale and
DeepSHAP code are the original implementation's (kundajelab/chrombpnet
@09938fd); each module says what, if anything, it changes. It runs them on
TensorFlow 2.21 with the Keras 2 API (tf-keras, ``TF_USE_LEGACY_KERAS=1``),
where the original ran TF 2.8: given the same batches and initial weights it
reproduces that implementation's outputs within float tolerance, which
``tests/parity/`` checks against the original in its own environment. Where
it departs from the original it fixes a defect: ``--seed`` reaches the
bias-factorized tower, the best validation epoch is kept when training runs
out of epochs, and DeepSHAP's references are seeded. The profile DeepSHAP
target's ``stop_gradient`` is ignored by default, as in the original, and
honoured with ``--contribs-profile-stopgrad``. Original ``.h5`` models load
(``layers``), and those it saves load in the original. It needs the
``chrombpnet-tensorflow`` environment (Python 3.12, TF 2.21).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from chrombpnet.engines import bundled_cuda
from chrombpnet.engines.base import Engine

_ARCHITECTURES = {
    "bias": "bpnet_model.py",
    "chrombpnet": "chrombpnet_with_bias_model.py",
}

# GPU memory is allocated as needed instead of reserved whole when TensorFlow
# starts. DeepSHAP runs in a spawned child process (attribution.py) while
# this one still holds the GPU it trained or predicted on; under TensorFlow's
# default the child finds the device already taken. Set when the engine is
# imported, before TensorFlow can initialize a GPU, so that the child
# inherits it too. An explicit TF_FORCE_GPU_ALLOW_GROWTH still wins. It
# changes how memory is reserved, not what is computed.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# The engine's models, losses and checkpoints are Keras 2 code. TensorFlow
# 2.16 and newer give ``tf.keras`` as Keras 3 unless this is set before
# TensorFlow is imported, and then as tf-keras, the Keras 2 package. Set here
# for the same reasons as above, and checked in ``require``.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _prefer_bundled_cuda():
    """Load TensorFlow's pip-installed CUDA libraries where a system CUDA could
    shadow them (``bundled_cuda``), unless no GPU is visible.

    TensorFlow loads cuDNN by name. A system cuDNN 9 on LD_LIBRARY_PATH would
    replace the 9.3 the environment pins, and with it the kernels for Pascal
    and Volta GPUs, or be older than the 9.3 TensorFlow 2.21 needs.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) in ("", "-1"):
        return
    bundled_cuda.prefer_bundled("tensorflow")


_prefer_bundled_cuda()


def require_tf_keras():
    """Fail clearly unless ``tf.keras`` is tf-keras (Keras 2), both as an
    attribute and as what ``import tensorflow.keras...`` statements load.

    The two can disagree: TensorFlow resolves the attribute when it is first
    used, but the import statements from a search path it fixes when it is
    imported. TensorFlow imported before TF_USE_LEGACY_KERAS was set gives
    Keras 3 modules to the engine's ``from tensorflow.keras... import``
    lines even where ``tf.keras.Model`` is tf-keras's. Each is judged by where
    it lives (``tf.keras.__name__`` reads ``tensorflow.keras`` either way).
    """
    import importlib
    from pathlib import Path

    import tensorflow as tf

    model = getattr(getattr(tf.keras, "Model", None), "__module__", "") or "nowhere"
    imported = getattr(importlib.import_module("tensorflow.keras"), "__file__", None) or "nowhere"
    if not model.startswith("tf_keras") or "tf_keras" not in Path(imported).parts:
        raise RuntimeError(
            "The TensorFlow engine needs tf.keras to be tf-keras (Keras 2), but tf.keras.Model "
            f"comes from {model!r} and `import tensorflow.keras` loads {imported!r}: TensorFlow "
            "was imported before the engine set TF_USE_LEGACY_KERAS=1, or tf-keras is not "
            "installed. Import chrombpnet.engines.tensorflow before TensorFlow, or export "
            "TF_USE_LEGACY_KERAS=1.")


@dataclass(frozen=True)
class AttributionModel:
    """What attribution needs in this process: the model runs in a child."""
    path: str
    input_length: int
    device: object = None


def _input_length_from_h5(path):
    """The model's sequence length, read without building a TensorFlow graph."""
    import h5py

    from chrombpnet.engines.tensorflow.predictor import refuse_keras3

    refuse_keras3(path)
    with h5py.File(path, "r") as handle:
        config = handle.attrs["model_config"]
    if isinstance(config, bytes):
        config = config.decode("utf-8")
    layers = json.loads(config)["config"]["layers"]
    for layer in layers:
        if layer["class_name"] == "InputLayer":
            return int(layer["config"]["batch_input_shape"][1])
    raise ValueError(f"No input layer in {path}")


class TensorFlowEngine(Engine):
    name = "tensorflow"
    checkpoint_suffix = ".h5"
    readable_suffixes = (".h5", ".hdf5")
    environment = "chrombpnet-tensorflow"

    def require(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError as error:
            raise self._unavailable(error) from error
        require_tf_keras()

    # -- training ----------------------------------------------------------
    def default_architecture(self, kind):
        if kind not in _ARCHITECTURES:
            raise ValueError(f"Unknown model kind {kind!r}")
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "architectures", _ARCHITECTURES[kind])

    def get_model(self, args, parameters):
        from chrombpnet.engines.tensorflow import train
        return train.get_model(args, parameters)

    def fit(self, model, train_gen, valid_gen, args, parameters, architecture_module):
        from chrombpnet.engines.tensorflow import train
        return train.fit_and_evaluate(
            model, train_gen, valid_gen, args, architecture_module)

    # -- inference ---------------------------------------------------------
    def resolve_device(self, value=None):
        # Applied where TensorFlow starts (configure_device); the value is
        # carried through unchanged until then.
        return value

    def load_predictor(self, path, device=None, *, announce=True, summary=True):
        from chrombpnet.engines.tensorflow.predictor import load_model_wrapper
        return load_model_wrapper(path, device, announce=announce, summary=summary)

    def rescale_bias_logcounts(self, bias_path, output_path, seqs, cts):
        from chrombpnet.engines.tensorflow.hyperparameters import rescale_bias_logcounts
        rescale_bias_logcounts(bias_path, output_path, seqs, cts)

    # -- attribution -------------------------------------------------------
    def load_attribution_model(self, path, device):
        return AttributionModel(str(path), _input_length_from_h5(path), device)

    def input_length(self, model):
        return model.input_length

    def attribute(self, model, seqs, profile_or_counts, *, device, batch_size,
                  num_shuffles, seed, profile_stop_gradient=False):
        from chrombpnet.engines.tensorflow import attribution
        return attribution.attribute(
            model.path, seqs, profile_or_counts, device=device,
            num_shuffles=num_shuffles, seed=seed,
            profile_stop_gradient=profile_stop_gradient)


ENGINE = TensorFlowEngine()
