import subprocess
import sys

import numpy as np
import pytest

from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import _stats_path, softmax
from chrombpnet.training.data_generators.batchgen_generator import subsample_nonpeak_data
from chrombpnet.training.utils.one_hot import dna_to_one_hot
from chrombpnet.evaluation.custom_sequences.viz_sequence import ic_scale


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
