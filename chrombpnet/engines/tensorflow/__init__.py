"""The tensorflow engine: the JAX engine's Keras 3 ChromBPNet, on the TensorFlow backend.

Keras 3 runs one model definition on several backends. This engine runs the JAX
engine's (``chrombpnet.engines.jax``: kundajelab/chrombpnet#284's Keras 3 code)
with ``KERAS_BACKEND=tensorflow``: the same architectures, loss, optimizers,
callbacks, training call, ``.keras`` checkpoints and model loader, which also
reads the original ChromBPNet's ``.h5`` models. A model trained by either
engine loads in the other. The JAX-specific parts have TensorFlow
counterparts: device and precision settings (``jax.runtime``'s TensorFlow
branches) and DeepSHAP, whose DeepLIFT rules are ported from the JAX
engine's to TensorFlow (``deeplift_tf``, ``explainer``).

It is not the ``tensorflow-old`` engine (``chrombpnet.engines.tensorflow_old``).
That one is chrombpnet 1.x's Keras 2 code on tf-keras: its ``.h5``
checkpoints are read by chrombpnet 1.x, variant-scorer and bpnet-lite, and its
DeepSHAP is the original's TF1-graph code. This one writes Keras 3 ``.keras``
files (``chrombpnet export`` converts them to the 1.x layout), and its DeepSHAP
matches tensorflow-old's (``tests/parity/deepshap_engines.py``). It needs the
``chrombpnet-tensorflow`` environment, which installs Keras 3 beside tf-keras,
or, on glibc 2.17 hosts (CentOS 7), ``chrombpnet-tensorflow-glibc``.
"""

from __future__ import annotations

import os
import sys

from chrombpnet.engines import bundled_cuda
from chrombpnet.engines.base import EngineUnavailable
from chrombpnet.engines.jax import JaxEngine

#: The Keras backend this engine runs.
BACKEND = "tensorflow"


def select_backend():
    """Start Keras on TensorFlow: set KERAS_BACKEND, unless Keras is imported already (``require``
    then reports the backend it has). Called for the configured engine only, never at import: the
    engine registry imports every engine, and the JAX engine needs KERAS_BACKEND=jax."""
    if "keras" not in sys.modules:
        os.environ["KERAS_BACKEND"] = BACKEND
    # GPU memory as needed, as the tensorflow engine: DeepSHAP and the next
    # pipeline step can then share the device with a model already loaded.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
        bundled_cuda.prefer_bundled("tensorflow")


class TensorFlowEngine(JaxEngine):
    name = "tensorflow"
    checkpoint_suffix = ".keras"
    #: Its own .keras files, the JAX engine's, and the original ChromBPNet's .h5 models.
    readable_suffixes = (".keras", ".h5", ".hdf5")
    environment = "chrombpnet-tensorflow"

    def prepare(self, args):
        # --reproducibility-max: TensorFlow's op determinism, switched on when
        # the model is built (jax.train._enable_deterministic_kernels)
        select_backend()

    def require(self):
        select_backend()
        try:
            import keras
            import tensorflow  # noqa: F401
        except ImportError as error:
            raise self._unavailable(error) from error
        if not keras.__version__.startswith("3."):
            raise EngineUnavailable(
                f"--engine {self.name} needs Keras 3, but keras {keras.__version__} is installed")
        if keras.backend.backend() != BACKEND:
            raise EngineUnavailable(
                f"--engine {self.name} needs the Keras TensorFlow backend, but Keras is using "
                f"{keras.backend.backend()!r}; unset KERAS_BACKEND or set it to tensorflow")

    # Training, prediction, the bias rescale and model loading are the JAX
    # engine's (inherited); only DeepSHAP is backend-specific.
    def attribute(self, model, seqs, profile_or_counts, *, device, batch_size,
                  num_shuffles, seed, profile_stop_gradient=False):
        from chrombpnet.engines.tensorflow import attribution
        return attribution.attribute(
            model, seqs, profile_or_counts, batch_size=batch_size,
            num_shuffles=num_shuffles, seed=seed,
            profile_stop_gradient=profile_stop_gradient)


ENGINE = TensorFlowEngine()
