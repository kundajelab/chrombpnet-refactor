# JAX engine. From kundajelab/chrombpnet#284 (NNFC-GMD/chrombpnet@14ca773),
# chrombpnet/training/runtime.py, except that assert_gpu_if_requested (--device auto|gpu|cpu)
# is replaced by configure_device for this package's --device values (cpu, cuda, cuda:N).
"""Process-wide runtime settings: matmul precision, mixed precision and the device."""
import contextlib
import warnings

import keras

PRECISIONS = ("default", "highest", "bf16")
MIXED_BF16 = "mixed_bfloat16"


def configure_precision(precision="default"):
    """Apply a --precision choice.

    default  leave JAX's default (TF32 for float32 matmuls/convs on Ampere and newer GPUs, like TF 2.x)
    highest  full float32 matmuls/convs (jax_default_matmul_precision='highest'); used for parity runs
    bf16     keras mixed_bfloat16 policy (bfloat16 compute, float32 variables) for models built afterwards;
             the architecture files keep the output heads in float32 (see head_dtype)
    """
    precision = precision or "default"
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of {}, got {!r}".format(PRECISIONS, precision))
    if precision in ("default", "highest"):
        import jax
        # 'default' resets a 'highest' left by an earlier run in the same process
        jax.config.update("jax_default_matmul_precision", "highest" if precision == "highest" else None)
    else:
        keras.config.set_dtype_policy(MIXED_BF16)
    print("precision: {} (dtype policy: {})".format(precision, keras.config.dtype_policy().name))
    return precision


def bf16_active():
    return keras.config.dtype_policy().name == MIXED_BF16


def head_dtype():
    """dtype for the output-head layers: float32 under mixed_bfloat16, else None (the global policy)."""
    return "float32" if bf16_active() else None


def set_float32_policy(model):
    """Switch every layer of `model` (nested models included) to the float32 policy, e.g. before saving a model
    trained with mixed_bfloat16 so that predict / interpret run it in float32. Variables are float32 already.
    Returns [(layer, previous policy)] for the layers it changed."""
    changed = []
    for layer in model._flatten_layers(include_self=True, recursive=True):
        if layer.dtype_policy.name != "float32":
            changed.append((layer, layer.dtype_policy))
            layer.dtype_policy = "float32"
    return changed


@contextlib.contextmanager
def float32_policy(model):
    """Give `model` float32 policies inside the block and restore each layer's own policy afterwards, e.g. to
    write a float32 checkpoint in the middle of mixed_bfloat16 training."""
    changed = set_float32_policy(model)
    try:
        yield model
    finally:
        for layer, policy in changed:
            layer.dtype_policy = policy


GPU_BACKENDS = ("gpu", "cuda", "rocm")


def backends_initialized():
    """True once JAX has created its backend clients (on a CUDA install, a context on every visible GPU)."""
    try:
        from jax._src import xla_bridge
        return xla_bridge.backends_are_initialized()
    except (ImportError, AttributeError):
        return True


def log_backend():
    """Print (and return) the JAX backend and devices. Initialises the JAX backends if needed."""
    import jax
    backend = jax.default_backend()
    print("jax backend: {}, devices: {}".format(backend, jax.devices()))
    return backend


def configure_device(device=None):
    """Apply a --device value (this package's spelling) and return it.

    None       leaves the choice to JAX (a GPU when the CUDA build sees one)
    cpu        restricts JAX to the CPU platform (jax_platforms='cpu') before any backend is created, so no CUDA
               context is opened on the GPUs; if the backends exist already, makes the CPU the default device
    cuda[:N]   makes GPU N (default 0) the default JAX device, failing if JAX has no such GPU
    """
    import jax
    if device is None:
        return None
    text = str(device).strip().lower()
    if text == "cpu":
        if not backends_initialized():
            jax.config.update("jax_platforms", "cpu")
        elif jax.default_backend() != "cpu":
            warnings.warn("--device cpu: the JAX {!r} backend was initialised already; running on the CPU as the "
                          "default device instead.".format(jax.default_backend()))
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        return text
    if text == "cuda" or text.startswith("cuda:"):
        index = int(text.split(":", 1)[1]) if ":" in text else 0
        try:
            gpus = jax.devices("gpu")
        except RuntimeError:
            gpus = []
        if index >= len(gpus):
            raise RuntimeError(
                "CUDA was requested but is not available: JAX sees {} GPU(s) (devices: {}). Check that the CUDA "
                "build of jax is installed (envs/chrombpnet-jax.yml) and that a GPU is visible (nvidia-smi)."
                .format(len(gpus), jax.devices()))
        jax.config.update("jax_default_device", gpus[index])
        return text
    raise ValueError("Unknown device {!r}; expected cpu, cuda or cuda:N".format(device))


def runtime_info():
    """Versions, devices and precision settings, for the args.json of a run (JSON-serializable)."""
    import jax
    return {"keras": keras.__version__, "jax": jax.__version__, "keras_backend": keras.backend.backend(),
            "jax_backend": jax.default_backend(), "jax_devices": [str(d) for d in jax.devices()],
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "dtype_policy": keras.config.dtype_policy().name}
