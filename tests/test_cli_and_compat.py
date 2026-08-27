import subprocess
import sys

import numpy as np
import pytest

from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import softmax
from chrombpnet.training.data_generators.batchgen_generator import subsample_nonpeak_data


def test_cli_help_lists_active_command_tree():
    completed = subprocess.run(
        [sys.executable, "-m", "chrombpnet.CHROMBPNET", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    for command in ("pipeline", "train", "qc", "bias", "prep", "pred_bw", "contribs_bw", "footprints"):
        assert command in completed.stdout


def test_preserved_nonpeak_oversampling_failure():
    seqs = np.zeros((1, 4, 4))
    counts = np.zeros((1, 2))
    coords = np.zeros((1, 4))
    with pytest.raises(ValueError):
        subsample_nonpeak_data(seqs, counts, coords, peak_data_size=10, negative_sampling_ratio=0.9)


def test_preserved_unstable_softmax_behavior():
    with np.errstate(over="ignore", invalid="ignore"):
        result = softmax(np.array([[1.0e4, -1.0e4]]))
    assert np.isnan(result).any()


def test_preserved_single_region_squeeze_failure_shape():
    squeezed = np.squeeze(np.zeros((1, 20)))
    assert squeezed.ndim == 1
    with pytest.raises(np.AxisError):
        softmax(squeezed)
