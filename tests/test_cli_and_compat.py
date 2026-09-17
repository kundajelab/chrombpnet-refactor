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


def test_training_cli_exposes_benchmark_mode():
    commands = (["pipeline"], ["train"], ["bias", "pipeline"], ["bias", "train"])
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "chrombpnet.CHROMBPNET", *command, "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert "--benchmark" in completed.stdout
        # Three levels plus an opt-out, with a bare --benchmark still accepted.
        # Matched without the opening brace so this holds for the TensorFlow
        # help too, which offers no "none".
        assert "lite,medium,max}" in completed.stdout.replace("\n", "")


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


def test_benchmark_defaults_to_lite_when_the_flag_is_omitted():
    import chrombpnet.parsers as parsers
    from chrombpnet.training.utils.benchmark_utils import (
        BENCHMARK_DEFAULT_LEVEL,
        BENCHMARK_IMPLICIT_LEVEL,
        benchmark_enables,
        benchmark_level,
    )

    base = ["pipeline", "-ibam", "b", "-d", "ATAC", "-g", "g", "-c", "c",
            "-p", "p", "-n", "n", "-fl", "f", "-b", "b", "-o", "o"]

    def parse(extra):
        argv = ["chrombpnet"] + base + extra
        original, sys.argv = sys.argv, argv
        try:
            return parsers.read_parser()
        finally:
            sys.argv = original

    assert BENCHMARK_IMPLICIT_LEVEL == "lite"
    # Omitting the flag now selects lite, so the refactor matches the
    # TensorFlow implementation's settings out of the box.
    assert benchmark_level(parse([])) == "lite"
    assert benchmark_enables(parse([]), "keras_adam")
    assert benchmark_enables(parse([]), "keras_initializers")
    assert not benchmark_enables(parse([]), "fixed_example_order")
    # A bare flag still means max, as it did before the default changed.
    assert benchmark_level(parse(["--benchmark"])) == BENCHMARK_DEFAULT_LEVEL == "max"
    # "none" is the escape hatch back to PyTorch's native defaults.
    assert benchmark_level(parse(["--benchmark", "none"])) is None
    assert not benchmark_enables(parse(["--benchmark", "none"]), "keras_adam")
    for level in ("lite", "medium", "max"):
        assert benchmark_level(parse(["--benchmark", level])) == level
