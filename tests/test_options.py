"""The read-only options and the per-step namespaces built from them."""

import argparse

import pytest

from chrombpnet.options import Options, as_options, with_values


def test_options_are_read_only():
    options = Options({"peaks": "p.bed", "bigwig": None})
    with pytest.raises(AttributeError, match="read-only"):
        options.peaks = "other.bed"
    with pytest.raises(AttributeError, match="read-only"):
        del options.peaks
    assert options.peaks == "p.bed"


def test_options_read_like_a_namespace():
    namespace = argparse.Namespace(peaks="p.bed", seed=1234)
    options = as_options(namespace)
    assert options == namespace
    assert vars(options) == {"peaks": "p.bed", "seed": 1234}
    assert getattr(options, "absent", "default") == "default"
    assert not hasattr(options, "absent")
    assert as_options(options) is options
    # Freezing copies: the namespace stays assignable, and the options do
    # not follow it.
    namespace.peaks = "other.bed"
    assert options.peaks == "p.bed"


def test_options_survive_pickling_and_copying():
    """Assignment is refused, but pickle and copy rebuild ``__dict__``
    directly, so options can still cross a process boundary or be copied."""
    import copy
    import pickle

    options = Options({"peaks": "p.bed", "trackables": ["loss"]})
    for clone in (pickle.loads(pickle.dumps(options)), copy.deepcopy(options),
                  copy.copy(options)):
        assert isinstance(clone, Options) and clone == options
        with pytest.raises(AttributeError):
            clone.peaks = "other.bed"


def test_a_step_adds_its_inputs_to_a_new_namespace():
    options = Options({"peaks": "p.bed", "bigwig": None})
    step = options.step(output_prefix="out/x", bigwig="data.bw")
    assert (step.output_prefix, step.bigwig, step.peaks) == ("out/x", "data.bw", "p.bed")
    assert vars(options) == {"peaks": "p.bed", "bigwig": None}
    # The next step starts from the options again, without the first's inputs.
    assert not hasattr(options.step(regions="r.bed"), "output_prefix")


def test_a_step_cannot_quietly_override_user_input():
    options = Options({"peaks": "p.bed", "inputlen": 2114})
    with pytest.raises(ValueError, match="would override the option peaks"):
        options.step(peaks="filtered.bed")
    # The same value is not an override, and a declared substitution is
    # allowed.
    assert options.step(inputlen=2114).inputlen == 2114
    assert options.step(replacing={"peaks": "filtered.bed"}).peaks == "filtered.bed"


def test_with_values_leaves_its_input_alone():
    namespace = argparse.Namespace(lr_schedule="cosine")
    variant = with_values(namespace, lr_schedule="constant")
    assert (namespace.lr_schedule, variant.lr_schedule) == ("cosine", "constant")
    options = Options({"lr_schedule": "cosine"})
    variant = with_values(options, lr_schedule="constant", total_steps=10)
    assert isinstance(variant, Options)
    assert (options.lr_schedule, variant.lr_schedule, variant.total_steps) == (
        "cosine", "constant", 10)


def test_preprocessing_inputs_reach_only_their_steps(monkeypatch, tmp_path):
    """The shift and PWM settings preprocessing picks go to its own two steps,
    and the generated bigwig comes back to the caller; none of it is written
    into the options, so none of it reaches training's args.json."""
    import json
    from types import SimpleNamespace

    import chrombpnet.helpers.preprocessing.analysis.build_pwm_from_bigwig as build_pwm_from_bigwig
    import chrombpnet.helpers.preprocessing.reads_to_bigwig as reads_to_bigwig
    from chrombpnet import stages

    seen = {}
    monkeypatch.setattr(reads_to_bigwig, "main", lambda ns: seen.setdefault("reads", ns))
    monkeypatch.setattr(build_pwm_from_bigwig, "main", lambda ns: seen.setdefault("pwm", ns))
    folds = tmp_path / "fold_0.json"
    folds.write_text(json.dumps({"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}))
    layout = SimpleNamespace(aux=lambda name: f"aux/{name}",
                             evaluation=lambda name: f"evaluation/{name}")
    options = Options({"chr_fold_path": str(folds), "data_type": "ATAC"})

    bigwig = stages.preprocess_and_qc_bigwig(options, layout)

    assert bigwig == "aux/data_unstranded.bw"
    assert vars(options) == {"chr_fold_path": str(folds), "data_type": "ATAC"}
    reads, pwm = seen["reads"], seen["pwm"]
    assert (reads.output_prefix, reads.plus_shift, reads.minus_shift) == ("aux/data", None, None)
    assert not hasattr(reads, "chr")
    assert (pwm.bigwig, pwm.output_prefix, pwm.chr, pwm.pwm_width) == (
        bigwig, "evaluation/bw_shift_qc", "chr2", stages.PWM_WIDTH)
    assert not hasattr(pwm, "plus_shift")
