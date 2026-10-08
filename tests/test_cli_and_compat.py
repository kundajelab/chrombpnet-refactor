import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import _stats_path, softmax
from chrombpnet.training.data_generators.batchgen_generator import subsample_nonpeak_data
from chrombpnet.training.utils.one_hot import dna_to_one_hot
from chrombpnet.evaluation.custom_sequences.viz_sequence import ic_scale
from chrombpnet.evaluation.interpret.interpret import (
    INTERPRET_BATCH_DIVISOR,
    resolve_interpret_batch_size,
)


def test_cli_help_lists_active_command_tree():
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.CHROMBPNET", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    for command in ("pipeline", "train", "qc", "bias", "prep", "pred_bw",
                    "contribs_bw", "modisco_motifs", "footprints"):
        assert command in completed.stdout


def test_training_cli_exposes_the_reproducibility_flags():
    commands = (["pipeline"], ["train"], ["bias", "pipeline"], ["bias", "train"], ["predict"])
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "chrombpnet.CHROMBPNET", *command, "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert "--reproducibility " in completed.stdout.replace("\n", " "), command
        assert "--reproducibility-max" in completed.stdout, command
        # --benchmark is gone; a stale script fails at the parser.
        assert "--benchmark" not in completed.stdout, command


def test_nonpeak_oversampling_is_bounded_by_available_rows():
    seqs = np.zeros((1, 4, 4))
    counts = np.zeros((1, 2))
    coords = np.zeros((1, 4))
    sampled = subsample_nonpeak_data(
        seqs, counts, coords, peak_data_size=10, negative_sampling_ratio=0.9
    )
    assert all(value.shape[0] == 1 for value in sampled)


def test_softmax_is_stable_for_large_logits():
    result = softmax(np.array([[1.0e4, -1.0e4]]))
    np.testing.assert_allclose(result, [[1.0, 0.0]])


def test_single_region_profile_retains_batch_dimension():
    profile = np.zeros((1, 20)).reshape(1, -1)
    assert softmax(profile).shape == (1, 20)


def test_stats_paths_are_model_specific():
    assert len({_stats_path("stats", name) for name in ("bias", "chrombpnet", "chrombpnet_nobias")}) == 3


def test_empty_one_hot_has_typed_empty_result():
    result = dna_to_one_hot([])
    assert result.shape == (0, 0, 4)
    assert result.dtype == np.int8


def test_information_content_scaling_accepts_list_background():
    pwm = np.full((6, 4), 0.25)
    result = ic_scale(pwm, background=[0.25] * 4)
    assert result.shape == pwm.shape
    assert np.isfinite(result).all()


def test_interpret_batch_size_scales_from_the_training_batch_size():
    # Attribution costs about INTERPRET_BATCH_DIVISOR times a training step at
    # the same batch size, so the default tracks -bs rather than matching it.
    for training, expected in ((64, 16), (32, 8), (128, 32), (4, 1), (2, 1)):
        args = SimpleNamespace(batch_size=training, interpret_batch_size=None)
        assert resolve_interpret_batch_size(args) == expected
        assert expected == max(1, training // INTERPRET_BATCH_DIVISOR)
    # An explicit value always wins, and never resolves to zero.
    assert resolve_interpret_batch_size(
        SimpleNamespace(batch_size=128, interpret_batch_size=48)) == 48
    assert resolve_interpret_batch_size(SimpleNamespace(batch_size=1)) == 1
    # Namespaces predating the flag still resolve.
    assert resolve_interpret_batch_size(SimpleNamespace()) == 16


def test_interpret_batch_size_is_exposed_wherever_interpret_runs():
    commands = (
        ["pipeline"], ["train"], ["qc"],
        ["bias", "pipeline"], ["bias", "train"], ["bias", "qc"],
        ["contribs_bw"],
    )
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "chrombpnet.CHROMBPNET", *command, "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert "--interpret-batch-size" in completed.stdout, command


def test_neither_flag_means_the_tensorflow_settings():
    import chrombpnet.parsers as parsers
    from chrombpnet.training.utils.benchmark_utils import (
        BENCHMARK_IMPLICIT_LEVEL,
        benchmark_enables,
        benchmark_level,
    )

    base = ["pipeline", "-ibam", "b", "-d", "ATAC", "-g", "g", "-c", "c",
            "-p", "p", "-n", "n", "-fl", "f", "-b", "b", "-o", "o"]

    def parse(extra):
        args = parsers.read_parser(base + extra)
        parsers.drop_absent(args)
        return args

    assert BENCHMARK_IMPLICIT_LEVEL == "tensorflow"
    # Omitting both flags selects the TensorFlow settings, so this package
    # matches the implementation it is a port of out of the box. Absent flags
    # are dropped, so args.json says only what was typed.
    args = parse([])
    assert not hasattr(args, "reproducibility") and not hasattr(args, "reproducibility_max")
    assert benchmark_level(args) == "tensorflow"
    assert benchmark_enables(args, "keras_adam")
    assert benchmark_enables(args, "keras_initializers")
    assert not benchmark_enables(args, "fixed_example_order")
    assert benchmark_level(parse(["--reproducibility"])) == "medium"
    assert benchmark_level(parse(["--reproducibility-max"])) == "max"
    assert benchmark_level(parse(["--reproducibility", "--reproducibility-max"])) == "max"
    # The old flag and its levels are refused.
    for stale in (["--benchmark"], ["--benchmark", "pytorch"], ["--reproducibility", "max"]):
        with pytest.raises(SystemExit):
            parsers.read_parser(base + stale)


def _one_hot_reference(seqs):
    """The pre-LUT implementation, kept verbatim as the equivalence oracle.

    `dna_to_one_hot` used to recover its base indices with
    `np.unique(..., return_inverse=True)`, i.e. a sort over every one of the
    N*L bytes. The replacement is a table lookup, so this reference is what
    pins "same array, not merely same idea".
    """
    if len(seqs) == 0:
        return np.empty((0, 0, 4), dtype=np.int8)
    seq_len = len(seqs[0])
    seq_concat = "".join(seqs).upper() + "ACGT"
    one_hot_map = np.identity(5)[:, :-1].astype(np.int8)
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85
    _, base_inds = np.unique(base_vals, return_inverse=True)
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))


@pytest.mark.parametrize("seqs", [
    ["ACGT"],
    ["acgt"],
    ["AcGt"],
    ["NNNN"],
    ["AAAA"],
    ["ACGTN", "NTGCA", "aCgTn"],
    ["ACGTRYKMSWBDHVN"],          # every IUPAC ambiguity code -> all zeros
    ["".join("ACGTN"[i % 5] for i in range(37))],
])
def test_one_hot_matches_the_pre_lut_implementation(seqs):
    produced = dna_to_one_hot(seqs)
    expected = _one_hot_reference(seqs)
    assert produced.dtype == expected.dtype
    assert produced.shape == expected.shape
    assert np.array_equal(produced, expected)


def test_one_hot_matches_the_pre_lut_implementation_over_all_ascii():
    """Every single-character input, so no byte is mapped by accident."""
    seqs = [chr(code) for code in range(128)]
    assert np.array_equal(dna_to_one_hot(seqs), _one_hot_reference(seqs))


def test_the_legacy_encoder_still_agrees_with_the_shipping_one():
    """`_dna_to_one_hot_legacy` is unreachable in normal operation, so only a
    test keeps it from rotting into documentation that lies."""
    from chrombpnet.training.utils.one_hot import _dna_to_one_hot_legacy

    seqs = ["ACGTNacgtn" * 3, "NNNNNNNNNN" * 3, "ACGTACGTAC" * 3]
    assert np.array_equal(dna_to_one_hot(seqs),
                          _dna_to_one_hot_legacy(seqs, len(seqs[0])))


def test_one_hot_rejects_ragged_input():
    with pytest.raises(ValueError, match="same length"):
        dna_to_one_hot(["ACGT", "ACG"])


def test_a_missing_tomtom_fails_before_any_work(tmp_path):
    """The motif report looks tomtom up only at the end of a run; the CLI
    checks first and creates nothing."""
    output = tmp_path / "out"
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.CHROMBPNET", "modisco_motifs",
         "-i", str(tmp_path / "scores.h5"), "-op", str(output / "modisco")],
        capture_output=True, text=True, env=dict(os.environ, PATH=str(tmp_path)), cwd=tmp_path)
    assert completed.returncode != 0
    assert "needs MEME's `tomtom`" in completed.stderr
    assert not output.exists()


def test_only_motif_reports_need_tomtom():
    from argparse import Namespace
    from chrombpnet.CHROMBPNET import runs_motif_report

    for cmd, sub in (("pipeline", None), ("qc", None), ("modisco_motifs", None),
                     ("bias", "pipeline"), ("bias", "qc")):
        assert runs_motif_report(Namespace(cmd=cmd, cmd_bias=sub))
    for cmd, sub in (("train", None), ("bias", "train"), ("pred_bw", None),
                     ("contribs_bw", None), ("prep", None)):
        assert not runs_motif_report(Namespace(cmd=cmd, cmd_bias=sub))


_BIAS_QC = ["bias", "qc", "-bw", "obs.bw", "-bm", "bias.pt", "-g", "genome.fa",
            "-c", "chrom.sizes", "-o", "out", "-d", "ATAC", "-p", "peaks.bed",
            "-n", "nonpeaks.bed", "-fl", "fold.json"]
_INTERPRETATION_SETTINGS = ("interpret_subsample", "modisco_max_seqlets", "modisco_window")


def test_interpretation_settings_are_absent_unless_given():
    """Dropped when absent, so <prefix>.args.json is unchanged."""
    import chrombpnet.parsers as parsers

    args = parsers.read_parser(_BIAS_QC)
    parsers.drop_absent(args)
    assert not any(hasattr(args, name) for name in _INTERPRETATION_SETTINGS)
    args = parsers.read_parser(_BIAS_QC + [
        "--interpret-subsample", "100", "--modisco-max-seqlets", "2000",
        "--modisco-window", "400"])
    parsers.drop_absent(args)
    assert [getattr(args, name) for name in _INTERPRETATION_SETTINGS] == [100, 2000, 400]


@pytest.mark.parametrize("flag, value", [
    ("--modisco-window", "401"), ("--modisco-window", "0"),
    ("--modisco-max-seqlets", "-5"), ("--interpret-subsample", "0"),
    ("--interpret-subsample", "ten"),
])
def test_interpretation_settings_are_validated(flag, value, capsys):
    import chrombpnet.parsers as parsers

    with pytest.raises(SystemExit):
        parsers.read_parser(_BIAS_QC + [flag, value])
    assert flag in capsys.readouterr().err


def test_modisco_motifs_keeps_its_own_settings():
    """It has -n/--max-seqlets and -w/--window, and no peaks to subsample."""
    import chrombpnet.parsers as parsers

    with pytest.raises(SystemExit):
        parsers.read_parser(["modisco_motifs", "-i", "s.h5", "-op", "out",
                             "--modisco-window", "400"])


def test_modisco_window_must_fit_the_input_length():
    from argparse import Namespace
    from chrombpnet.CHROMBPNET import check_modisco_window

    with pytest.raises(SystemExit, match="--modisco-window 2200 is wider"):
        check_modisco_window(Namespace(modisco_window=2200, inputlen=2114))
    check_modisco_window(Namespace(modisco_window=2114, inputlen=2114))
    check_modisco_window(Namespace(inputlen=2114))
    # qc takes the input length from the model; run_motifs checks it there
    check_modisco_window(Namespace(modisco_window=2200))
