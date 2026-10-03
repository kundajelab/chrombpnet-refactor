"""Engine selection: the registry, ``--engine`` resolution and its guards.

None of these tests needs a deep-learning framework, so they run in every
engine's environment.
"""

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


def test_every_engine_is_registered_and_importable_without_its_framework():
    for name in engines.ENGINE_NAMES:
        engine = engines.get(name)
        assert engine.name == name
        assert engine.environment == f"chrombpnet-{name}"


def test_the_default_is_pytorch():
    assert engines.DEFAULT_ENGINE == "pytorch"
    assert engines.resolve_name(Namespace()) == "pytorch"
    assert engines.current().name == "pytorch"


@pytest.mark.parametrize("path,expected", [
    ("models/bias.pt", "pytorch"),
    ("models/bias.h5", "tensorflow"),
    ("models/BIAS.H5", "tensorflow"),
    ("models/bias.hdf5", "tensorflow"),
    ("models/bias.keras", "jax"),
    ("models/bias.ckpt", None),
    ("None", None),
    (None, None),
])
def test_checkpoint_suffix_names_the_engine(path, expected):
    assert engines.infer_from_path(path) == expected


def test_an_omitted_engine_follows_the_input_models():
    args = Namespace(engine=None, chrombpnet_model="a.h5", chrombpnet_model_nb="b.h5")
    assert engines.resolve_name(args) == "tensorflow"
    assert engines.resolve_name(Namespace(engine=None, bias_model_path="b.pt")) == "pytorch"
    assert engines.resolve_name(Namespace(engine=None, model_h5="m.keras")) == "jax"


def test_an_explicit_engine_must_agree_with_the_input_models():
    with pytest.raises(ValueError, match="--model-h5 'm.pt' is a pytorch checkpoint"):
        engines.resolve_name(Namespace(engine="tensorflow", model_h5="m.pt"))
    with pytest.raises(ValueError, match="--model-h5 'm.keras' is a jax checkpoint"):
        engines.resolve_name(Namespace(engine="tensorflow", model_h5="m.keras"))
    # An unrecognised suffix is trusted to the explicit engine.
    assert engines.resolve_name(Namespace(engine="tensorflow", model_h5="m.ckpt")) == "tensorflow"


def test_an_explicit_engine_may_read_another_engines_checkpoints():
    """The JAX engine loads the original ChromBPNet's .h5 models."""
    args = Namespace(engine="jax", bias_model_path="original_bias.h5")
    assert engines.resolve_name(args) == "jax"


def test_mixed_input_models_go_to_the_engine_that_reads_them_all():
    args = Namespace(bias_model="original_bias.h5", chrombpnet_model="chrombpnet.keras")
    assert engines.resolve_name(args) == "jax"


def test_input_models_from_two_engines_are_refused():
    with pytest.raises(ValueError, match="different engines"):
        engines.resolve_name(Namespace(bias_model="b.pt", chrombpnet_model="c.h5"))


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
    with pytest.raises(ValueError, match="--predict-fk"):
        engines.configure_from_args(
            Namespace(engine="tensorflow", predict_fk=True), require=False)
    with pytest.raises(ValueError, match="--benchmark pytorch"):
        engines.configure_from_args(
            Namespace(engine="tensorflow", benchmark="pytorch"), require=False)
    # Inferred engines get the same check.
    with pytest.raises(ValueError, match="--predict-fk"):
        engines.configure_from_args(
            Namespace(model_h5="m.h5", predict_fk=True), require=False)


def test_jax_only_training_options_are_refused_for_other_engines():
    for engine in ("pytorch", "tensorflow"):
        with pytest.raises(ValueError, match="--optimizer, --ema only apply to the JAX engine"):
            engines.configure_from_args(
                Namespace(engine=engine, optimizer="muon", ema=True), require=False)


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
    engines.configure("tensorflow")
    assert engines.for_model("bias.pt").name == "pytorch"
    assert engines.for_model("bias.keras").name == "jax"
    assert engines.for_model("bias.unknown").name == "tensorflow"


def test_the_configured_engine_loads_what_it_can_read():
    engines.configure("jax")
    assert engines.for_model("original_bias.h5").name == "jax"
    assert engines.for_model("bias.pt").name == "pytorch"
    engines.configure("pytorch")
    assert engines.for_model("original_bias.h5").name == "tensorflow"


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
