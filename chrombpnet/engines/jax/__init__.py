"""The JAX engine: ChromBPNet on Keras 3 with the JAX backend.

The model definitions, loss, callbacks, training call, model loader, bias
rescale and the native JAX DeepSHAP are kundajelab/chrombpnet#284's
("ChromBPNet 2.0", NNFC-GMD/chrombpnet@14ca773); each module says what, if
anything, it changes. The architectures are the original ChromBPNet's, with
the same layer names, and the defaults are the original training recipe
(Adam, constant learning rate). It reproduces the original implementation to
within float32 tolerances rather than byte for byte; ``tests/parity/`` checks
how closely. It needs the ``chrombpnet-jax`` environment (Python 3.12,
Keras 3.15, JAX 0.11).

Checkpoints are Keras 3 ``.keras`` archives. The engine also reads the
original ChromBPNet's ``.h5`` models (a pretrained bias model, say) and
ChromBPNet 2.0's Keras 3 ``.h5`` files.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from chrombpnet.engines import bundled_cuda
from chrombpnet.engines.base import Engine, EngineUnavailable

_ARCHITECTURES = {
    "bias": "bpnet_model.py",
    "chrombpnet": "chrombpnet_with_bias_model.py",
}

# Keras picks its backend when it is first imported, and every module of this
# package imports keras: choose JAX before any of them can. Importing this
# package imports no framework, and an explicit KERAS_BACKEND still wins. Only
# where JAX is installed: in a TensorFlow environment the tensorflow
# engine imports these modules to run them on TensorFlow.
# GPU memory is allocated on demand instead of reserving 75% of the device at
# start-up (set XLA_PYTHON_CLIENT_MEM_FRACTION to cap it).
if "keras" not in sys.modules and importlib.util.find_spec("jax") is not None:
    os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _prefer_bundled_cuda():
    """Load jax's pip-installed CUDA libraries where a system CUDA could shadow
    them (``bundled_cuda``), unless the run is CPU-only.

    Without it, a system CUDA or cuDNN on LD_LIBRARY_PATH makes JAX report
    "Unable to load cuSPARSE" and fall back to the CPU, or fail every
    convolution with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
    """
    if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        return
    bundled_cuda.prefer_bundled("jax")


_prefer_bundled_cuda()


#: XLA's switch for deterministic GPU kernels. XLA reads XLA_FLAGS once, when
#: JAX creates its GPU client -- at the first device use, which the bias scaling
#: and hyperparameter steps reach before training -- so it is set from the
#: command line, in :meth:`JaxEngine.prepare`.
DETERMINISTIC_XLA_FLAG = "--xla_gpu_deterministic_ops=true"


class JaxEngine(Engine):
    name = "jax"
    checkpoint_suffix = ".keras"
    #: The original ChromBPNet's models are .h5, and so are ChromBPNet 2.0's.
    readable_suffixes = (".keras", ".h5", ".hdf5")
    environment = "chrombpnet-jax"

    def prepare(self, args):
        from chrombpnet.training.utils.benchmark_utils import benchmark_enables

        if benchmark_enables(args, "deterministic_kernels"):
            flags = os.environ.get("XLA_FLAGS", "")
            if DETERMINISTIC_XLA_FLAG not in flags:
                os.environ["XLA_FLAGS"] = (flags + " " + DETERMINISTIC_XLA_FLAG).strip()

    def require(self):
        try:
            import jax  # noqa: F401
            import keras
        except ImportError as error:
            raise self._unavailable(error) from error
        if keras.backend.backend() != "jax":
            raise EngineUnavailable(
                f"--engine jax needs the Keras JAX backend, but Keras is using "
                f"{keras.backend.backend()!r}; unset KERAS_BACKEND or set it to jax")

    # -- training ----------------------------------------------------------
    def default_architecture(self, kind):
        if kind not in _ARCHITECTURES:
            raise ValueError(f"Unknown model kind {kind!r}")
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "architectures", _ARCHITECTURES[kind])

    def get_model(self, args, parameters):
        from chrombpnet.engines.jax import train
        return train.get_model(args, parameters)

    def fit(self, model, train_gen, valid_gen, args, parameters, architecture_module):
        from chrombpnet.engines.jax import train
        return train.fit_and_evaluate(
            model, train_gen, valid_gen, args, architecture_module)

    # -- inference ---------------------------------------------------------
    def resolve_device(self, value=None):
        # Applied where JAX starts (runtime.configure_device); the value is
        # carried through unchanged until then.
        return value

    def load_predictor(self, path, device=None, *, announce=True, summary=True):
        from chrombpnet.engines.jax.predictor import load_model_wrapper
        return load_model_wrapper(path, device, announce=announce, summary=summary)

    def rescale_bias_logcounts(self, bias_path, output_path, seqs, cts):
        from chrombpnet.engines.jax.hyperparameters import rescale_bias_logcounts
        rescale_bias_logcounts(bias_path, output_path, seqs, cts)

    # -- attribution -------------------------------------------------------
    def load_attribution_model(self, path, device):
        from chrombpnet.engines.jax import model_io
        from chrombpnet.engines.jax.runtime import configure_device

        configure_device(device)
        model = model_io.load_model(path, compile=False)
        print("got the model")
        model.summary()
        return model

    def input_length(self, model):
        return int(model.inputs[0].shape[1])

    def attribute(self, model, seqs, profile_or_counts, *, device, batch_size,
                  num_shuffles, seed, profile_stop_gradient=False):
        from chrombpnet.engines.jax import attribution
        return attribution.attribute(
            model, seqs, profile_or_counts, batch_size=batch_size,
            num_shuffles=num_shuffles, seed=seed,
            profile_stop_gradient=profile_stop_gradient)


ENGINE = JaxEngine()
