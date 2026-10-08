"""-ibw/--input-bigwig and `chrombpnet prep bigwig`: a bigwig made once, as
its own step, trains exactly what the reads it came from train."""

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from chrombpnet import parsers
from chrombpnet.options import Options

BIAS_TRAIN = ["bias", "train", "-g", "g.fa", "-c", "c.sizes", "-o", "out",
              "-d", "ATAC", "-p", "p.bed", "-n", "n.bed", "-fl", "f.json",
              "-b", "0.5"]


def test_input_bigwig_takes_the_place_of_reads():
    args = parsers.read_parser(BIAS_TRAIN + ["-ibw", "data.bw"])
    assert args.input_bigwig == "data.bw"
    assert args.input_tagalign_file is None
    for command in (["pipeline"], ["train"], ["bias", "pipeline"]):
        argv = command + BIAS_TRAIN[len(["bias", "train"]):] + ["-ibw", "data.bw"]
        assert parsers.read_parser(argv).input_bigwig == "data.bw"


@pytest.mark.parametrize("inputs", [[], ["-ibw", "data.bw", "-itag", "reads.tagAlign.gz"]])
def test_exactly_one_input_is_required(inputs):
    with pytest.raises(SystemExit):
        parsers.read_parser(BIAS_TRAIN + inputs)


def test_an_omitted_input_bigwig_is_not_recorded():
    """A run from reads records exactly what it did before -ibw existed."""
    args = parsers.read_parser(BIAS_TRAIN + ["-itag", "reads.tagAlign.gz"])
    parsers.drop_absent(args)
    assert not hasattr(args, "input_bigwig")
    args = parsers.read_parser(BIAS_TRAIN + ["-ibw", "data.bw"])
    parsers.drop_absent(args)
    assert args.input_bigwig == "data.bw"


def test_prep_bigwig_command_line():
    args = parsers.read_parser(["prep", "bigwig", "-g", "g.fa", "-c", "c.sizes",
                                "-ibam", "reads.bam", "-op", "out/data",
                                "-d", "DNASE", "-ps", "0", "-ms", "1",
                                "--dnase-convention", "right"])
    assert (args.cmd, args.cmd_prep) == ("prep", "bigwig")
    assert (args.plus_shift, args.minus_shift) == (0, 1)
    assert args.dnase_convention == "right"
    assert not hasattr(args, "data_aug")


def test_a_given_bigwig_skips_preprocessing_but_not_the_shift_qc(monkeypatch, tmp_path):
    import chrombpnet.helpers.preprocessing.analysis.build_pwm_from_bigwig as build_pwm_from_bigwig
    import chrombpnet.helpers.preprocessing.reads_to_bigwig as reads_to_bigwig
    from chrombpnet import stages

    def refuse(_):
        raise AssertionError("reads_to_bigwig ran on a run given -ibw")

    seen = []
    monkeypatch.setattr(reads_to_bigwig, "main", refuse)
    monkeypatch.setattr(build_pwm_from_bigwig, "main", seen.append)
    folds = tmp_path / "fold_0.json"
    folds.write_text(json.dumps({"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}))
    layout = SimpleNamespace(aux=lambda name: f"aux/{name}",
                             evaluation=lambda name: f"evaluation/{name}")
    options = Options({"chr_fold_path": str(folds), "input_bigwig": "given.bw"})

    assert stages.preprocess_and_qc_bigwig(options, layout) == "given.bw"
    assert [ns.bigwig for ns in seen] == ["given.bw"]


@pytest.mark.parametrize("options,warns", [
    (dict(data_type="DNASE"), True),
    (dict(data_type="DNASE", data_aug="revcomp-only"), True),
    (dict(data_type="DNASE", dnase_convention="right"), False),
    (dict(data_type="DNASE", data_aug="forward-only"), False),
    (dict(data_type="ATAC"), False),
])
def test_a_dnase_bigwig_without_its_convention_warns(options, warns):
    import warnings

    from chrombpnet import stages

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stages.warn_if_dnase_convention_assumed(Options(options))
    assert bool([w for w in caught if "--dnase-convention right" in str(w.message)]) is warns


def test_prep_bigwig_then_input_bigwig_trains_what_the_reads_train(tmp_path):
    """`prep bigwig` writes the bigwig a reads run builds for itself, and a
    run given that bigwig with -ibw trains byte-identical models and logs.
    Trains with the default engine, so PyTorch only."""
    pytest.importorskip("torch")
    missing = [tool for tool in ("bedtools", "bedGraphToBigWig")
               if shutil.which(tool) is None]
    if missing:
        pytest.skip(f"needs {', '.join(missing)} on PATH")
    from tests.regression import artifacts, fixture
    from tests.regression.scenarios import GEOMETRY

    paths = fixture.build(tmp_path / "fixture")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0",
               MPLBACKEND="Agg")

    def chrombpnet(*argv):
        subprocess.run([sys.executable, "-m", "chrombpnet.CHROMBPNET", *argv],
                       env=env, check=True, cwd=tmp_path,
                       stdout=subprocess.DEVNULL)

    common = ["-g", str(paths["genome"]), "-c", str(paths["chrom_sizes"]),
              "-d", "ATAC"]
    chrombpnet("prep", "bigwig", *common, "-itag", str(paths["reads"]),
               "-op", str(tmp_path / "prep" / "data"))
    bigwig = tmp_path / "prep" / "data_unstranded.bw"

    train = ["bias", "train", *common, "-p", str(paths["peaks"]),
             "-n", str(paths["nonpeaks"]), "-fl", str(paths["folds"]),
             "-b", "0.5", "-j", "0", "--device", "cpu", *GEOMETRY]
    # Both runs write to the same directory, moved aside after each, because
    # the report PDF embeds the output path.
    out = tmp_path / "out"
    chrombpnet(*train, "-itag", str(paths["reads"]), "-o", str(out))
    out.rename(tmp_path / "reads")
    chrombpnet(*train, "-ibw", str(bigwig), "-o", str(out))
    out.rename(tmp_path / "bigwig")

    assert bigwig.read_bytes() == (
        tmp_path / "reads" / "auxiliary" / "data_unstranded.bw").read_bytes()
    from_reads = artifacts.manifest(tmp_path / "reads")
    from_bigwig = artifacts.manifest(tmp_path / "bigwig")
    # Only the reads run converts reads, and each run records its own input.
    recorded = {"logs/bias.args.json"}
    only_reads = {k for k in from_reads if k.startswith("auxiliary/data")}
    assert set(from_bigwig) == set(from_reads) - only_reads
    differ = sorted(k for k in from_bigwig
                    if k not in recorded and from_bigwig[k] != from_reads[k])
    assert differ == []
    assert any(k.startswith("models/") for k in from_bigwig)

    record_reads = json.loads((tmp_path / "reads/logs/bias.args.json").read_text())
    record_bigwig = json.loads((tmp_path / "bigwig/logs/bias.args.json").read_text())
    assert record_bigwig["input_bigwig"] == record_bigwig["bigwig"] == str(bigwig)
    assert "input_bigwig" not in record_reads
