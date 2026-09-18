"""Tests for the shared pipeline stages."""

from __future__ import annotations

import csv

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
