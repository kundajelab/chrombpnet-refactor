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

import os
import sys

from chrombpnet.engines.base import Engine, EngineUnavailable

_ARCHITECTURES = {
    "bias": "bpnet_model.py",
    "chrombpnet": "chrombpnet_with_bias_model.py",
}

# Keras picks its backend when it is first imported, and every module of this
# package imports keras: choose JAX before any of them can. Importing this
# package imports no framework, and an explicit KERAS_BACKEND still wins.
# GPU memory is allocated on demand instead of reserving 75% of the device at
# start-up (set XLA_PYTHON_CLIENT_MEM_FRACTION to cap it).
if "keras" not in sys.modules:
    os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


#: Load order for bundled CUDA libraries that need loading: each before those
#: that link against it (cuSPARSE and cuSOLVER need nvJitLink, the cuDNN
#: libraries resolve libcudnn_graph's symbols). Others follow.
_CUDA_LOAD_ORDER = ("libcudart", "libnvJitLink", "libnvrtc", "libcublasLt", "libcublas",
                    "libcusparse", "libcusolver", "libcufft", "libcupti", "libnccl",
                    "libnvshmem_host", "libcudnn_graph", "libcudnn")


def _cuda_load_rank(name):
    stem = name.split(".so")[0]
    if stem in _CUDA_LOAD_ORDER:
        return _CUDA_LOAD_ORDER.index(stem)
    if stem.startswith("libcudnn"):
        return _CUDA_LOAD_ORDER.index("libcudnn")
    return len(_CUDA_LOAD_ORDER)


def _prefer_bundled_cuda():
    """Load jax's pip-installed CUDA libraries where a system CUDA could shadow them.

    jax[cuda12] and jax[cuda13] bring CUDA and cuDNN as pip wheels (the
    ``nvidia`` packages). They find one another through RUNPATH, which the
    loader searches *after* LD_LIBRARY_PATH, so a system CUDA or cuDNN there --
    common on clusters -- supplies libraries of another release under the
    same names:

    * The jax plugin loads cuSPARSE by path but not the nvJitLink it links
      against, which then comes from the system. Too old for the bundled
      cuSPARSE, it fails to load, and JAX reports "Unable to load cuSPARSE"
      and falls back to the CPU.
    * cuDNN 9's libcudnn_graph loads its engine libraries by name, gets the
      system's, and every convolution fails with
      CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.

    A library already loaded is found by its SONAME before any search, so
    loading each shadowed bundled library by path here, dependencies first,
    makes those lookups resolve to it. Only done where JAX is installed, the
    run is not CPU-only, and a directory on LD_LIBRARY_PATH holds a library
    with the name of a bundled one.
    """
    import ctypes
    import glob
    import importlib.util

    if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        return
    if importlib.util.find_spec("jax") is None:
        return
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ModuleNotFoundError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return
    bundled = {}
    for root in spec.submodule_search_locations:
        for path in glob.glob(os.path.join(root, "*", "lib", "lib*.so*")):
            bundled.setdefault(os.path.basename(path), path)
    own_dirs = {os.path.realpath(os.path.dirname(path)) for path in bundled.values()}
    shadowed = set()
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not directory or not os.path.isdir(directory) or os.path.realpath(directory) in own_dirs:
            continue
        try:
            shadowed.update(name for name in os.listdir(directory) if name in bundled)
        except OSError:
            continue
    # Dependencies first; retry what failed until nothing more loads.
    pending = [bundled[name] for name in sorted(shadowed, key=lambda n: (_cuda_load_rank(n), n))]
    while pending:
        failed = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
        if len(failed) == len(pending):
            break
        pending = failed


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
