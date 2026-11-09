"""Exits non-zero unless a framework in the active env reaches the GPU.

    python -I gpu_check.py tensorflow|torch

TensorFlow 2.8 without its CUDA libraries runs on the CPU without an error,
which would time the original package on the CPU. PyTorch is also checked for
running its own cuDNN: cuDNN 9 looks for its sub-libraries on LD_LIBRARY_PATH
before its own directory, so another cuDNN 9 there can get mixed in (on a
cluster node, 2026-10-21, the kit env's torch reported cuDNN 9.4, not its own
9.1, and the kit's exact mode died loading cuDNN).
Prints the CUDA libraries the process loaded, for the log.
"""

import sys


def loaded(fragment):
    with open("/proc/self/maps") as maps:
        return sorted({line.split()[-1] for line in maps
                       if fragment in line and "/" in line})


def check_tensorflow():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow", tf.__version__, "GPUs:", gpus)
    for path in loaded("libcud"):
        print("  " + path)
    if not gpus:
        sys.exit("TensorFlow sees no GPU")


def wheel_cudnn():
    """The env's own cuDNN wheel and its CUDNN_VERSION number, or (None, None)."""
    from importlib import metadata
    for name in ("nvidia-cudnn-cu13", "nvidia-cudnn-cu12", "nvidia-cudnn-cu11"):
        try:
            major, minor, patch = (int(part) for part in metadata.version(name).split(".")[:3])
        except metadata.PackageNotFoundError:
            continue
        return name, major * (10000 if major >= 9 else 1000) + minor * 100 + patch
    return None, None


def check_torch():
    import torch
    if not torch.cuda.is_available():
        sys.exit("PyTorch sees no GPU")
    running = torch.backends.cudnn.version()
    name, own = wheel_cudnn()
    print("PyTorch", torch.__version__, "on", torch.cuda.get_device_name(0),
          "cuDNN", running, "(env: {} {})".format(name, own) if name else "")
    libraries = loaded("libcudnn")
    for path in libraries:
        print("  " + path)
    if own and running != own:
        sys.exit("PyTorch runs cuDNN {}, not its own {} ({}): another cuDNN on "
                 "LD_LIBRARY_PATH was mixed in. Loaded: {}".format(
                     running, own, name, ", ".join(libraries)))


if __name__ == "__main__":
    {"tensorflow": check_tensorflow, "torch": check_torch}[sys.argv[1]]()
