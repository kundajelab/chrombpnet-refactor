"""Engine selection: the registry, ``--engine`` resolution and its guards.

None of these tests needs a deep-learning framework, so they run in every
engine's environment.
"""

import os
import re
import subprocess
import sys
from argparse import Namespace

import pytest

from chrombpnet import engines


@pytest.fixture(autouse=True)
def _restore_engine():
    previous = engines._CURRENT
    yield
    engines._CURRENT = previous


def _environment(monkeypatch, *frameworks):
    """Pretend only these frameworks are installed (an environment's own: torch;
    jax and keras; tensorflow and keras)."""
    monkeypatch.setattr(engines, "installed", lambda name: engines._FRAMEWORKS[name] in frameworks)


@pytest.fixture
def every_framework(monkeypatch):
    _environment(monkeypatch, "torch", "jax", "tensorflow")


#: tensorflow-old shares the TensorFlow environment, which has tf-keras beside Keras 3.
ENVIRONMENTS = {"tensorflow-old": "chrombpnet-tensorflow"}


def test_every_engine_is_registered_and_importable_without_its_framework():
    for name in engines.ENGINE_NAMES:
        engine = engines.get(name)
        assert engine.name == name
        assert engine.environment == ENVIRONMENTS.get(name, f"chrombpnet-{name}")


def test_keras3_files_go_to_the_installed_keras3_engine(monkeypatch):
    """Both Keras 3 engines read .keras; JAX where it is installed, else TensorFlow's."""
    def installed(present):
        return lambda name: engines._FRAMEWORKS[name] in present
    monkeypatch.setattr(engines, "installed", installed({"jax", "tensorflow"}))
    assert engines.keras3_engine() == "jax"
    monkeypatch.setattr(engines, "installed", installed({"tensorflow"}))
    assert engines.keras3_engine() == "tensorflow"
    assert engines.infer_from_path("m.keras") == "tensorflow"
    assert engines.infer_from_path("m.h5") == "tensorflow-old"
    mixed = Namespace(bias_model="original_bias.h5", chrombpnet_model="chrombpnet.keras")
    assert engines.resolve_name(mixed) == "tensorflow"
    monkeypatch.setattr(engines, "installed", installed(set()))
    assert engines.keras3_engine() == "jax"   # whose error names its environment


def test_the_default_is_pytorch(every_framework):
    assert engines.DEFAULT_ENGINE == "pytorch"
    assert engines.resolve(Namespace()) == ("pytorch", None)
    assert engines.current().name == "pytorch"


@pytest.mark.parametrize("path,expected", [
    ("models/bias.pt", "pytorch"),
    ("models/bias.h5", "tensorflow-old"),
    ("models/BIAS.H5", "tensorflow-old"),
    ("models/bias.hdf5", "tensorflow-old"),
    ("models/bias.keras", engines.keras3_engine()),
    ("models/bias.ckpt", None),
    ("None", None),
    (None, None),
])
def test_checkpoint_suffix_names_the_engine(path, expected):
    assert engines.infer_from_path(path) == expected


def test_an_omitted_engine_is_pytorch_whatever_the_suffix(every_framework):
    """The original's .h5 models run on PyTorch too (engines/pytorch/keras_h5)."""
    args = Namespace(engine=None, chrombpnet_model="a.h5", chrombpnet_model_nb="b.h5")
    assert engines.resolve_name(args) == "pytorch"
    assert engines.resolve_name(Namespace(engine=None, bias_model_path="b.pt")) == "pytorch"
    with pytest.raises(ValueError, match="pytorch engine cannot read --model-h5 'm.keras'"):
        engines.resolve_name(Namespace(engine=None, model_h5="m.keras"))


def test_an_explicit_engine_must_agree_with_the_input_models():
    with pytest.raises(ValueError, match="--model-h5 'm.pt' is a pytorch checkpoint"):
        engines.resolve_name(Namespace(engine="tensorflow-old", model_h5="m.pt"))
    with pytest.raises(ValueError, match=f"--model-h5 'm.keras' is a {engines.keras3_engine()} checkpoint"):
        engines.resolve_name(Namespace(engine="tensorflow-old", model_h5="m.keras"))
    # An unrecognised suffix is trusted to the explicit engine.
    assert engines.resolve_name(Namespace(engine="tensorflow-old", model_h5="m.ckpt")) == "tensorflow-old"
    # The Keras 3 tensorflow engine reads .h5 (1.x models) as well as .keras.
    assert engines.resolve_name(Namespace(engine="tensorflow", model_h5="m.h5")) == "tensorflow"


def test_an_explicit_engine_may_read_another_engines_checkpoints():
    """The Keras 3 engines load the original ChromBPNet's .h5 models, and each other's .keras files."""
    for name in engines.KERAS3_ENGINES:
        assert engines.resolve_name(Namespace(engine=name, bias_model_path="original_bias.h5")) == name
        assert engines.resolve_name(Namespace(engine=name, model_h5="m.keras")) == name


def test_mixed_input_models_need_an_engine_that_reads_them_all(every_framework):
    args = Namespace(bias_model="original_bias.h5", chrombpnet_model="chrombpnet.keras")
    with pytest.raises(ValueError, match="cannot read"):
        engines.resolve_name(args)
    args.engine = "jax"
    assert engines.resolve_name(args) == "jax"


def test_input_models_no_engine_reads_together_are_refused(every_framework):
    with pytest.raises(ValueError, match="cannot read --chrombpnet-model 'c.keras'"):
        engines.resolve_name(Namespace(bias_model="b.pt", chrombpnet_model="c.keras"))


def test_a_pt_and_an_original_h5_go_to_pytorch(every_framework):
    """The PyTorch engine reads chrombpnet 1.x's .h5 models too (keras_h5)."""
    assert engines.resolve_name(Namespace(bias_model="b.h5", chrombpnet_model="c.pt")) == "pytorch"
    assert engines.resolve_name(Namespace(engine="pytorch", chrombpnet_model="c.h5")) == "pytorch"


def test_an_omitted_engine_leaves_no_trace_in_the_namespace():
    """Stages deep-copy ``args`` and dump it to ``<prefix>.args.json``; an
    attribute the user never typed must not appear there."""
    args = Namespace(engine=None, model_h5="m.pt")
    engines.configure_from_args(args, require=False)
    assert not hasattr(args, "engine")

    explicit = Namespace(engine="pytorch", model_h5="m.pt")
    engines.configure_from_args(explicit, require=False)
    assert explicit.engine == "pytorch"


def test_pytorch_only_flags_are_refused_for_other_engines():
    with pytest.raises(ValueError, match="--predict-fused-kernel"):
        engines.configure_from_args(
            Namespace(engine="tensorflow-old", predict_fused_kernel=True), require=False)
    # A fallback engine gets the same check (here TensorFlow's environment).
    import unittest.mock
    with unittest.mock.patch.object(engines, "installed",
                                    lambda name: engines._FRAMEWORKS[name] == "tensorflow"):
        with pytest.raises(ValueError, match="--predict-fused-kernel"):
            engines.configure_from_args(
                Namespace(model_h5="m.h5", predict_fused_kernel=True), require=False)


def test_reproducibility_max_reaches_xla_before_jax_starts(monkeypatch):
    """XLA reads XLA_FLAGS once, when JAX creates its GPU client, which the
    pipelines reach before training; so the flag is set from the command
    line, before the framework is imported (require=False here: no JAX
    needed to check it)."""
    from chrombpnet.engines.jax import DETERMINISTIC_XLA_FLAG

    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/x")
    engines.configure_from_args(Namespace(engine="jax", reproducibility=True), require=False)
    assert DETERMINISTIC_XLA_FLAG not in os.environ["XLA_FLAGS"]
    engines.configure_from_args(Namespace(engine="jax", reproducibility_max=True), require=False)
    assert os.environ["XLA_FLAGS"] == "--xla_dump_to=/tmp/x " + DETERMINISTIC_XLA_FLAG
    # Idempotent, and no other engine touches it.
    engines.configure_from_args(Namespace(engine="jax", reproducibility_max=True), require=False)
    assert os.environ["XLA_FLAGS"].count(DETERMINISTIC_XLA_FLAG) == 1
    monkeypatch.delenv("XLA_FLAGS")
    engines.configure_from_args(Namespace(engine="tensorflow-old", reproducibility_max=True), require=False)
    assert "XLA_FLAGS" not in os.environ
    monkeypatch.setenv("KERAS_BACKEND", "jax")   # restored afterwards: prepare() sets it
    engines.configure_from_args(Namespace(engine="tensorflow", reproducibility_max=True), require=False)
    assert "XLA_FLAGS" not in os.environ
    engines.configure("pytorch")


def test_jax_only_training_options_are_refused_for_other_engines():
    for engine in ("pytorch", "tensorflow-old"):
        with pytest.raises(ValueError, match=r"--optimizer, --ema only apply to the Keras 3 engines \(jax, tensorflow\)"):
            engines.configure_from_args(
                Namespace(engine=engine, optimizer="muon", ema=True), require=False)


def test_keras3_training_options_apply_to_the_tensorflow_engine_too(monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "jax")   # restored afterwards: prepare() sets it
    given = Namespace(engine="tensorflow", optimizer="muon", ema=True)
    engines.configure_from_args(given, require=False)
    assert given.optimizer == "muon" and given.ema is True


def test_absent_jax_options_leave_no_trace_in_the_namespace():
    absent = {name: None for name in engines.JAX_ONLY_OPTIONS}
    args = Namespace(engine=None, bias_model_path="bias.pt", **absent)
    engines.configure_from_args(args, require=False)
    assert not any(hasattr(args, name) for name in engines.JAX_ONLY_OPTIONS)

    given = Namespace(engine="jax", **dict(absent, optimizer="muon", precision="highest"))
    engines.configure_from_args(given, require=False)
    assert given.optimizer == "muon" and given.precision == "highest"
    assert not hasattr(given, "ema")


def test_model_loading_follows_the_checkpoint_not_the_configured_engine():
    engines.configure("tensorflow-old")
    assert engines.for_model("bias.pt").name == "pytorch"
    assert engines.for_model("bias.keras").name == engines.keras3_engine()
    assert engines.for_model("bias.unknown").name == "tensorflow-old"


def test_the_configured_engine_loads_what_it_can_read():
    engines.configure("jax")
    assert engines.for_model("original_bias.h5").name == "jax"
    assert engines.for_model("bias.pt").name == "pytorch"
    engines.configure("tensorflow")
    assert engines.for_model("original_bias.h5").name == "tensorflow"
    assert engines.for_model("bias.keras").name == "tensorflow"
    engines.configure("pytorch")
    assert engines.for_model("original_bias.h5").name == "pytorch"
    assert engines.for_model("bias.keras").name == engines.keras3_engine()


def _jax_installed():
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("jax", "keras"))


@pytest.mark.skipif(_jax_installed(), reason="the JAX engine is available here")
def test_an_unavailable_engine_fails_before_creating_any_output(tmp_path):
    output_dir = tmp_path / "out"
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.CHROMBPNET", "bias", "train",
         "-g", "genome.fa", "-c", "chrom.sizes", "-ibam", "reads.bam",
         "-o", str(output_dir), "-d", "ATAC", "-p", "peaks.bed",
         "-n", "nonpeaks.bed", "-fl", "folds.json", "-b", "0.5",
         "--engine", "jax"],
        capture_output=True, text=True, cwd=tmp_path)
    assert completed.returncode != 0
    assert "chrombpnet-jax conda environment" in completed.stderr
    assert not output_dir.exists()


def _tensorflow_installed():
    import importlib.util
    return importlib.util.find_spec("tensorflow") is not None


@pytest.mark.skipif(_tensorflow_installed(), reason="TensorFlow is installed here")
def test_the_tensorflow_engines_without_tensorflow_name_their_environment(tmp_path):
    output_dir = tmp_path / "out"
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.CHROMBPNET", "bias", "train",
         "-g", "genome.fa", "-c", "chrom.sizes", "-ibam", "reads.bam",
         "-o", str(output_dir), "-d", "ATAC", "-p", "peaks.bed",
         "-n", "nonpeaks.bed", "-fl", "folds.json", "-b", "0.5",
         "--engine", "tensorflow"],
        capture_output=True, text=True, cwd=tmp_path)
    assert completed.returncode != 0
    assert "chrombpnet-tensorflow conda environment" in completed.stderr
    assert not output_dir.exists()


def test_importing_the_package_imports_no_framework():
    code = ("import sys, chrombpnet, chrombpnet.CHROMBPNET, chrombpnet.engines; "
            "loaded = [m for m in ('torch', 'tensorflow', 'jax') if m in sys.modules]; "
            "assert not loaded, loaded")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_engine_help_describes_every_engine_and_its_suffix(capsys):
    """It once still called the JAX engine a placeholder after it shipped."""
    import chrombpnet.parsers as parsers

    with pytest.raises(SystemExit):
        parsers.read_parser(["pred_bw", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    help_text = text[text.index("--engine"):]
    assert "placeholder" not in help_text and "not implemented" not in help_text
    for name in engines.ENGINE_NAMES:
        assert re.search(rf"\b{name}\b", help_text), name
        suffix = engines.get(name).checkpoint_suffix
        assert f"{suffix} " in help_text, (name, suffix)


@pytest.mark.parametrize("given, expected", [(None, "true"), ("false", "false")])
def test_the_tensorflow_old_engine_allocates_gpu_memory_as_needed(given, expected):
    """Its DeepSHAP child process shares the GPU with the process that
    trained; an explicit setting still wins."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "TF_FORCE_GPU_ALLOW_GROWTH"}
    if given is not None:
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = given
    code = ("import os, chrombpnet.engines.tensorflow_old; "
            "print(os.environ['TF_FORCE_GPU_ALLOW_GROWTH'])")
    completed = subprocess.run([sys.executable, "-c", code], env=env,
                               capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == expected


@pytest.mark.parametrize("module,framework,cpu_only", [("jax", "jax", ("JAX_PLATFORMS", "cpu")),
                                                       ("tensorflow_old", "tensorflow", ("CUDA_VISIBLE_DEVICES", ""))])
def test_engines_load_shadowed_cuda_libraries_dependencies_first(tmp_path, monkeypatch, module, framework,
                                                                 cpu_only):
    """A system CUDA on LD_LIBRARY_PATH would shadow the CUDA wheels jax and
    TensorFlow bring; the engine loads the shadowed ones by path,
    dependencies first, and nothing else."""
    import ctypes
    import importlib
    import importlib.util
    import os
    import types

    engine = importlib.import_module("chrombpnet.engines." + module)
    variable, cpu_value = cpu_only

    nvidia = tmp_path / "nvidia"
    for package, libraries in {
            "cusparse": ["libcusparse.so.12"], "nvjitlink": ["libnvJitLink.so.12"],
            "cublas": ["libcublas.so.12", "libcublasLt.so.12"],
            "cudnn": ["libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_ops.so.9"]}.items():
        (nvidia / package / "lib").mkdir(parents=True)
        for library in libraries:
            (nvidia / package / "lib" / library).write_bytes(b"")
    system = tmp_path / "system"
    system.mkdir()
    for library in ["libcusparse.so.12", "libnvJitLink.so.12", "libcudnn_ops.so.9",
                    "libcudnn_graph.so.9", "libcudart.so.12"]:
        (system / library).write_bytes(b"")

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == framework:
            return object()
        if name == "nvidia":
            return types.SimpleNamespace(submodule_search_locations=[str(nvidia)])
        return real_find_spec(name, *args, **kwargs)

    loaded = []
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: loaded.append(os.path.basename(path)))
    monkeypatch.delenv(variable, raising=False)

    monkeypatch.setenv("LD_LIBRARY_PATH", f"{system}:{nvidia / 'cublas' / 'lib'}")
    engine._prefer_bundled_cuda()
    # nvJitLink before the cuSPARSE that links against it, libcudnn_graph
    # before the other cuDNN libraries; cuBLAS is not shadowed, and the
    # system's libcudart is not bundled.
    assert loaded == ["libnvJitLink.so.12", "libcusparse.so.12",
                      "libcudnn_graph.so.9", "libcudnn_ops.so.9"]

    for ld_library_path, value in ((str(tmp_path / "elsewhere"), None), (str(system), cpu_value)):
        loaded.clear()
        monkeypatch.setenv("LD_LIBRARY_PATH", ld_library_path)
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
        engine._prefer_bundled_cuda()
        assert loaded == [], (ld_library_path, value)


@pytest.mark.parametrize("frameworks, expected", [
    (("jax",), "jax"),                 # chrombpnet-jax
    (("tensorflow",), "tensorflow"),   # chrombpnet-tensorflow: Keras 3 before the original's Keras 2 code
])
def test_an_omitted_engine_runs_on_the_environments_own_engine(monkeypatch, capsys, frameworks, expected):
    _environment(monkeypatch, *frameworks)
    args = Namespace(engine=None, chrombpnet_model="model.chrombpnet.h5")
    name, notice = engines.resolve(args)
    assert name == expected and "pytorch engine is not installed" in notice and expected in notice
    engine = engines.configure_from_args(args, require=False)
    assert engine.name == expected and not hasattr(args, "engine")
    assert f"running on {expected}" in capsys.readouterr().err


def test_an_engine_asked_for_by_name_is_never_swapped(monkeypatch):
    """--engine pytorch in the JAX environment fails in require, naming its environment."""
    _environment(monkeypatch, "jax")
    assert engines.resolve(Namespace(engine="pytorch", chrombpnet_model="m.h5")) == ("pytorch", None)


def test_no_fallback_engine_reads_a_pt(monkeypatch):
    """A .pt in the JAX environment stays PyTorch's, whose require names its environment."""
    _environment(monkeypatch, "jax")
    assert engines.resolve(Namespace(engine=None, model_h5="m.pt")) == ("pytorch", None)


def test_the_fallback_order():
    assert engines.fallback_order()[0] == "pytorch" and engines.fallback_order()[-1] == "tensorflow-old"
