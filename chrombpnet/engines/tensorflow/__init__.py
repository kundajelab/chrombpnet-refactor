"""The TensorFlow engine: the original Keras ChromBPNet, behind the engine interface.

The model definitions, loss, callbacks, training call, bias rescale and
DeepSHAP code are the original implementation's (kundajelab/chrombpnet
@09938fd); each module says what, if anything, it changes. Given the same
batches and initial weights it reproduces that implementation's outputs
byte for byte, which ``tests/parity/`` checks. Where it departs from the
original it fixes a defect: ``--seed`` reaches the bias-factorized tower, the
best validation epoch is kept when training runs out of epochs, DeepSHAP's
references are seeded, and the profile DeepSHAP target's ``stop_gradient`` is
honoured, as in the PyTorch engine. It needs the ``chrombpnet-tensorflow``
environment (Python 3.8, TF 2.8.0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from chrombpnet.engines.base import Engine

_ARCHITECTURES = {
    "bias": "bpnet_model.py",
    "chrombpnet": "chrombpnet_with_bias_model.py",
}


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
                  num_shuffles, seed):
        from chrombpnet.engines.tensorflow import attribution
        return attribution.attribute(
            model.path, seqs, profile_or_counts, device=device,
            num_shuffles=num_shuffles, seed=seed)


ENGINE = TensorFlowEngine()
