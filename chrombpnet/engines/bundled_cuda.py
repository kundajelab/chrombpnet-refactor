"""Load a framework's pip-installed CUDA libraries ahead of a system CUDA.

jax[cuda12] and tensorflow[and-cuda] bring CUDA and cuDNN as pip wheels (the
``nvidia`` packages). They find one another through RUNPATH, which the loader
searches *after* LD_LIBRARY_PATH, so a system CUDA or cuDNN there -- common
on clusters -- supplies libraries of another release under the same names:

* The jax plugin loads cuSPARSE by path but not the nvJitLink it links
  against, which then comes from the system. Too old for the bundled
  cuSPARSE, it fails to load, and JAX falls back to the CPU.
* cuDNN 9's libcudnn_graph loads its engine libraries by name, and
  TensorFlow loads libcudnn itself by name: both get the system's, and
  convolutions fail -- with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED, or on a
  GPU the system's cuDNN has no kernels for.

A library already loaded is found by its SONAME before any search, so
loading each shadowed bundled library by path, dependencies first, makes
those lookups resolve to it. Imports no framework.
"""

import os

#: Load order for bundled CUDA libraries that need loading: each before those
#: that link against it (cuSPARSE and cuSOLVER need nvJitLink, the cuDNN
#: libraries resolve libcudnn_graph's symbols). Others follow.
LOAD_ORDER = ("libcudart", "libnvJitLink", "libnvrtc", "libcublasLt", "libcublas",
              "libcusparse", "libcusolver", "libcufft", "libcupti", "libnccl",
              "libnvshmem_host", "libcudnn_graph", "libcudnn")


def _load_rank(name):
    stem = name.split(".so")[0]
    if stem in LOAD_ORDER:
        return LOAD_ORDER.index(stem)
    if stem.startswith("libcudnn"):
        return LOAD_ORDER.index("libcudnn")
    return len(LOAD_ORDER)


def prefer_bundled(framework):
    """Load the shadowed pip CUDA libraries, where ``framework`` is installed
    and a directory on LD_LIBRARY_PATH holds a library with the name of a
    bundled one."""
    import ctypes
    import glob
    import importlib.util

    if importlib.util.find_spec(framework) is None:
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
    pending = [bundled[name] for name in sorted(shadowed, key=lambda n: (_load_rank(n), n))]
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


