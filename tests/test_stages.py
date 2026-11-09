"""Tests for the shared pipeline stages."""

from __future__ import annotations

import csv

import pytest

from chrombpnet.data import DefaultDataFile, get_default_data_path
from chrombpnet.stages import BIAS_MOTIFS

PACKAGED = {
    "ATAC": DefaultDataFile.motif_to_pwm_atac,
    "DNASE": DefaultDataFile.motif_to_pwm_dnase,
}


def _read_packaged(key):
    with open(get_default_data_path(key), newline="") as handle:
        return [row for row in csv.reader(handle, delimiter="\t") if row]


def test_bias_motifs_match_the_packaged_tables():
    """The inserted motifs must match the packaged data files.

    These had drifted: the pipeline's DNASE table listed dnase_2 with
    dnase_1's sequence, so every DNase run footprinted the same motif twice and
    never footprinted the second one. Comparing parsed rows rather than file
    bytes, because the packaged files use CRLF while the generated auxiliary
    TSV is LF.
    """
    for assay, key in PACKAGED.items():
        assert BIAS_MOTIFS[assay] == _read_packaged(key), assay


def test_bias_motif_names_and_sequences_are_distinct():
    """A repeated sequence means one footprint is computed twice."""
    for assay, motifs in BIAS_MOTIFS.items():
        names = [name for name, _ in motifs]
        sequences = [sequence for _, sequence in motifs]
        assert len(names) == len(set(names)), f"{assay}: duplicate motif names"
        assert len(sequences) == len(set(sequences)), (
            f"{assay}: duplicate motif sequences {sequences}")
        for sequence in sequences:
            assert set(sequence) <= set("ACGT"), f"{assay}: {sequence!r}"


def test_write_bias_motif_table_rejects_an_unknown_assay(tmp_path):
    """The assay dispatch used to leave a variable unbound and crash later."""
    import pytest

    from chrombpnet.layout import RunLayout
    from chrombpnet.stages import write_bias_motif_table
    from types import SimpleNamespace

    layout = RunLayout(str(tmp_path))
    (tmp_path / "auxiliary").mkdir()
    with pytest.raises(ValueError, match="unknown data type"):
        write_bias_motif_table(SimpleNamespace(data_type="RNA"), layout)

    path = write_bias_motif_table(SimpleNamespace(data_type="ATAC"), layout)
    rows = [line.split("\t") for line in
            open(path).read().splitlines() if line]
    assert [row[0] for row in rows] == [name for name, _ in BIAS_MOTIFS["ATAC"]]


# -- The bias-model quality gate and --ignore-bias-flag --------------------

def _bias_layout(tmp_path, pearsonr):
    """A layout whose bias_metrics.json reports `pearsonr`."""
    import json

    (tmp_path / "evaluation").mkdir(exist_ok=True)
    (tmp_path / "evaluation" / "bias_metrics.json").write_text(
        json.dumps({"counts_metrics": {"peaks": {"pearsonr": pearsonr}}}))

    class Layout:
        def unprefixed(self, *parts):
            return str(tmp_path.joinpath(*parts))

    return Layout()


@pytest.mark.parametrize("pearsonr", [0.25, 0.0, -0.49])
def test_an_acceptable_bias_model_passes_either_way(tmp_path, pearsonr):
    """The flag must not change the verdict for a model that was fine."""
    from chrombpnet import stages

    for ignore in (False, True):
        stages.check_bias_model_quality(_bias_layout(tmp_path, pearsonr),
                                        ignore=ignore)


@pytest.mark.parametrize("pearsonr", [-0.5, -0.6, -0.95])
def test_an_anticorrelated_bias_model_is_refused_when_asked(tmp_path, pearsonr):
    """--no-ignore-bias-flag makes the check a hard stop -- the number is in
    the message so the log says how bad it was."""
    from chrombpnet import stages

    with pytest.raises(ValueError, match="strongly negative peak count"):
        stages.check_bias_model_quality(_bias_layout(tmp_path, pearsonr),
                                        ignore=False)


def test_by_default_a_failing_check_only_warns(tmp_path):
    """Going on has to be audible: a silent bypass of this check would let
    an untrustworthy bias correction reach the results unnoticed."""
    import warnings

    from chrombpnet import stages

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stages.check_bias_model_quality(_bias_layout(tmp_path, -0.6))
    assert any(issubclass(w.category, RuntimeWarning)
               and "strongly negative peak count" in str(w.message)
               for w in caught), "going on must still warn"


def test_the_flag_reaches_the_check_from_the_command_line():
    """`pipelines.chrombpnet_train_pipeline` reads it off its options, so a
    namespace without the attribute must behave as the default (warn)."""
    import inspect
    from types import SimpleNamespace

    from chrombpnet import pipelines
    from chrombpnet.parsers import read_parser

    source = inspect.getsource(pipelines.chrombpnet_train_pipeline)
    assert 'getattr(options, "ignore_bias_flag", True)' in source
    assert getattr(SimpleNamespace(), "ignore_bias_flag", True) is True

    base = ["pipeline", "-ibam", "b", "-d", "DNASE", "-g", "g", "-c", "c",
            "-p", "p", "-n", "n", "-fl", "f", "-b", "b", "-o", "o"]
    assert read_parser(base).ignore_bias_flag is True
    assert read_parser(base + ["--ignore-bias-flag"]).ignore_bias_flag is True
    assert read_parser(base + ["--no-ignore-bias-flag"]).ignore_bias_flag is False
