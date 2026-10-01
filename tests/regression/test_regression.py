"""Pytest front end for the golden-artifact harness.

The harness is normally driven from the command line (see
``tests/regression/README.md``); these tests cover the parts that are cheap and
that silently invalidate everything else when they break.

The full artifact sweep is *not* a default test: it takes minutes, shells out to
bedtools and bedGraphToBigWig, and needs a baseline captured from whichever
revision you are comparing against. It runs only when a baseline path is given
via ``CHROMBPNET_REGRESSION_BASELINE``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from . import artifacts, fixture, harness, scenarios


def test_planted_consensus_yields_a_reproducible_standard_shift(tmp_path):
    """The fixture must satisfy auto_shift_detect, or no training scenario runs.

    ``compute_shift`` demands that every reference motif agree on one shift and
    that the shift be standard, and it samples reads through an unseeded
    ``shuf``. Both properties are fixture responsibilities, so pin them here
    rather than discovering the breakage six scenarios deep.
    """
    from chrombpnet.data import DefaultDataFile, get_default_data_path
    from chrombpnet.helpers.preprocessing import auto_shift_detect

    paths = fixture.build(tmp_path / "fixture")
    reference = get_default_data_path(DefaultDataFile.atac_ref_motifs)
    estimates = {
        auto_shift_detect.compute_shift(
            None, None, str(paths["reads"]), 10_000,
            str(paths["genome"]), "ATAC", reference)
        for _ in range(3)
    }
    assert estimates == {(0, 0)}, (
        f"expected a stable +0/-0 estimate, got {estimates}. A single value "
        "that is not (0, 0) means the planting offsets drifted; several values "
        "mean the read count crossed MAX_REPRODUCIBLE_READS."
    )


def test_read_count_stays_below_the_shuf_ceiling(tmp_path):
    import gzip

    path = fixture.write_reads(tmp_path)
    with gzip.open(path, "rt") as handle:
        count = sum(1 for _ in handle)
    assert count < fixture.MAX_REPRODUCIBLE_READS, count


def test_fixture_geometry_satisfies_the_packages_hardcoded_limits():
    """Guard the constants that other modules hardcode against the fixture."""
    # marginal_footprinting.py:145 slices [outputlen//2-100 : outputlen//2+100]
    # and plots it against range(200).
    assert fixture.OUTPUTLEN >= 200
    # pipelines._run_modisco_motifs defaults window=500 and is not configurable.
    assert fixture.INPUTLEN >= 500
    # BPNet receptive field: first conv 20 + dilated 2*2**1 + profile conv 74.
    profile_length = fixture.INPUTLEN - (20 + 2 * 2 ** 1 + 74)
    assert profile_length >= fixture.OUTPUTLEN
    assert (profile_length - fixture.OUTPUTLEN) % 2 == 0
    # Jittered windows must stay inside the chromosome.
    assert fixture.REGION_MARGIN > fixture.INPUTLEN // 2 + fixture.MAX_JITTER


def test_every_scenario_placeholder_can_be_resolved(tmp_path):
    """A typo in a scenario's argv should fail here, not minutes into a sweep."""
    paths = {key: tmp_path / key for key in
             ("genome", "chrom_sizes", "reads", "peaks", "nonpeaks", "bigwig",
              "folds")}
    for scenario in scenarios.SCENARIOS:
        values = harness._placeholders(tmp_path, paths, scenario)
        for part in scenario.argv:
            try:
                part.format(**values)
            except KeyError as error:
                pytest.fail(f"{scenario.name}: unknown placeholder {error} in {part!r}")


def test_scenario_outputs_are_unique():
    outputs = [scenario.output for scenario in scenarios.SCENARIOS]
    assert len(outputs) == len(set(outputs))


def test_explain_reports_the_tensor_that_moved(tmp_path):
    """The diff explainer is the harness's whole value once a hash mismatches."""
    torch = pytest.importorskip("torch")  # PyTorch checkpoints

    from chrombpnet.model import BiasModel, ModelConfig, save_model

    config = ModelConfig(input_length=200, output_length=100,
                         filters=2, n_dilation_layers=1)
    model = BiasModel(config)
    before, after = tmp_path / "a.pt", tmp_path / "b.pt"
    save_model(model, before)
    with torch.no_grad():
        model.count_dense.bias.add_(0.25)
    save_model(model, after)

    report = artifacts.explain(before, after)
    assert "count_dense.bias" in report
    assert "0.25" in report


def test_hdf5_digest_ignores_container_layout_but_not_content(tmp_path):
    """HDF5 files are digested by content, so this must not become a blind spot.

    HDF5's container layout is not a function of the logical content: writing
    the same datasets in a different creation order yields the same objects and
    values in a byte-different file. A raw byte hash therefore reports changes
    that do not exist -- which is exactly what it did when the shared
    write_predictions_h5py landed, for a file whose every dataset was
    bit-identical. The content digest has to see through that without also
    seeing through real changes.
    """
    import h5py
    import numpy as np

    values = np.arange(40, dtype=float).reshape(4, 10)

    def write(path, *, names, payload):
        with h5py.File(path, "w") as handle:
            group = handle.create_group("predictions")
            for name in names:
                group.create_dataset(name, data=payload, dtype=float,
                                     compression="gzip")

    forward = tmp_path / "forward.h5"
    reversed_order = tmp_path / "reversed.h5"
    changed = tmp_path / "changed.h5"
    write(forward, names=("profs", "logcounts"), payload=values)
    write(reversed_order, names=("logcounts", "profs"), payload=values)
    write(changed, names=("profs", "logcounts"), payload=values + 1.0)

    assert artifacts.sha256(forward) != artifacts.sha256(reversed_order), (
        "expected creation order to change the container bytes; if this fails "
        "the test no longer exercises the case it was written for")
    assert artifacts.hdf5_digest(forward) == artifacts.hdf5_digest(reversed_order)
    assert artifacts.hdf5_digest(forward) != artifacts.hdf5_digest(changed)

    # The manifest must route .h5 through the content digest, not sha256.
    assert artifacts.manifest(tmp_path)["forward.h5"] == artifacts.hdf5_digest(forward)


def test_explain_handles_a_missing_artifact(tmp_path):
    present = tmp_path / "present.bed"
    present.write_text("chr1\t0\t10\n")
    assert "missing from current run" in artifacts.explain(present, tmp_path / "gone.bed")
    assert "missing from baseline" in artifacts.explain(tmp_path / "gone.bed", present)


def test_pytorch_scenarios_run_exactly_as_written(tmp_path):
    """--engine pytorch must not change one byte of what the CLI is given.

    An explicit ``--engine`` is recorded in ``<prefix>.args.json``, so adding
    even the default would move artifacts of every PyTorch baseline.
    """
    for scenario in scenarios.SCENARIOS:
        assert harness._for_engine(list(scenario.argv), "pytorch") == scenario.argv


def test_tensorflow_scenarios_swap_checkpoints_and_name_the_engine():
    for scenario in scenarios.SCENARIOS:
        argv = harness._for_engine(list(scenario.argv), "tensorflow")
        assert not any(part.endswith(".pt") for part in argv), scenario.name
        takes_engine = scenario.argv[0] not in ("prep", "modisco_motifs")
        assert (argv[-2:] == ["--engine", "tensorflow"]) == takes_engine, scenario.name
    train = harness._for_engine(list(scenarios.BY_NAME["04_train"].argv), "tensorflow")
    assert "{root}/03_bias_train/models/bias.h5" in train


def test_hdf5_attributes_digest_as_before_unless_they_hold_a_keras_lambda():
    """PyTorch-engine HDF5 digests must not move when the TensorFlow
    engine's Lambda normalization is added -- only Keras model configs that
    contain a Lambda are rewritten."""
    import json
    import numpy as np

    for key, value in (("version", 3), ("names", np.array([b"a", b"b"])),
                       ("model_config", json.dumps({"class_name": "Functional"}))):
        assert artifacts._attr_text(key, value) == f"attr:{key}={value!r}"


def test_keras_lambda_bytecode_is_compared_without_its_origin():
    import base64
    import json
    import marshal

    def config_for(source, filename):
        code = compile(source, filename, "eval").co_consts[0]
        payload = base64.encodebytes(marshal.dumps(code)).decode("ascii")
        return json.dumps({"class_name": "Functional", "config": {"layers": [
            {"class_name": "Lambda", "config": {"function": [payload, None, None]}}]}})

    here = config_for("lambda x: x + 1", "/checkout/a/model.py")
    there = config_for("\n\nlambda x: x + 1", "/checkout/b/model.py")
    other = config_for("lambda x: x + 2", "/checkout/a/model.py")
    assert here != there
    assert artifacts._attr_text("model_config", here) == artifacts._attr_text("model_config", there)
    assert artifacts._attr_text("model_config", here) != artifacts._attr_text("model_config", other)


@pytest.mark.regression
@pytest.mark.skipif(
    not os.environ.get("CHROMBPNET_REGRESSION_BASELINE"),
    reason="set CHROMBPNET_REGRESSION_BASELINE=<dir> to run the full sweep "
           "(capture one first: python -m tests.regression.harness capture --out <dir>)")
def test_artifacts_match_the_captured_baseline():
    baseline = Path(os.environ["CHROMBPNET_REGRESSION_BASELINE"])
    benchmark = os.environ.get("CHROMBPNET_REGRESSION_BENCHMARK") or None
    assert harness.compare(baseline, benchmark, None) == 0, (
        "CLI artifacts changed; see the per-artifact report above")
