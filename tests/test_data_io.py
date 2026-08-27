import h5py
import numpy as np
import pyBigWig

from chrombpnet.evaluation.interpret.interpret import generate_shap_dict
from chrombpnet.evaluation.make_bigwigs.bigwig_helper import write_bigwig
from chrombpnet.training.predict import write_predictions_h5py
from chrombpnet.training.utils import augment
from chrombpnet.training.utils.one_hot import dna_to_one_hot, one_hot_to_dna


def test_one_hot_unknown_and_roundtrip():
    values = dna_to_one_hot(["ACGTN", "tgcan"])
    assert values.shape == (2, 5, 4)
    assert values[:, -1].sum() == 0
    assert one_hot_to_dna(values) == ["ACGTN", "TGCAN"]


def test_crop_and_reverse_complement_semantics():
    seqs = dna_to_one_hot(["AACCGGTT", "ACGTACGT"])
    labels = np.arange(12).reshape(2, 6)
    coords = np.array([["chr1", "10", "f", "1"], ["chr1", "20", "f", "1"]])
    np.random.seed(2)
    cropped, counts, shifted = augment.random_crop(seqs, labels, 6, 4, coords)
    assert cropped.shape == (2, 6, 4)
    assert counts.shape == (2, 4)
    assert np.all(shifted[:, 1].astype(int) >= coords[:, 1].astype(int))
    np.random.seed(1)
    rc_seq, rc_counts, rc_coords = augment.random_rev_comp(
        seqs.copy(), labels.copy(), coords.copy(), frac=1.0
    )
    assert np.array_equal(rc_seq, seqs[:, ::-1, ::-1])
    assert np.array_equal(rc_counts, labels[:, ::-1])
    assert set(rc_coords[:, 2]) == {"r"}


def test_prediction_hdf5_schema(tmp_path):
    prefix = str(tmp_path / "fixture")
    profile = np.full((2, 4), 0.25)
    logcounts = np.array([1.0, 2.0])
    coords = np.array([["chr1", "10", "f", "1"], ["chr2", "20", "f", "0"]])
    write_predictions_h5py(prefix, profile, logcounts, coords)
    with h5py.File(prefix + "_predictions.h5") as handle:
        assert set(handle) == {"coords", "predictions"}
        assert set(handle["coords"]) == {"coords_center", "coords_chrom", "coords_peak"}
        assert set(handle["predictions"]) == {"logcounts", "profs"}
        np.testing.assert_allclose(handle["predictions/profs"], profile)


def test_contribution_hdf5_layout_arrays():
    seqs = dna_to_one_hot(["ACGT"])
    scores = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    values = generate_shap_dict(seqs, scores)
    assert set(values) == {"raw", "shap", "projected_shap"}
    assert values["raw"]["seq"].shape == (1, 4, 4)
    assert values["raw"]["seq"].dtype == np.int8
    assert values["shap"]["seq"].dtype == np.float16
    np.testing.assert_allclose(
        values["projected_shap"]["seq"],
        np.transpose(seqs * scores, (0, 2, 1)),
    )


def test_bigwig_output(tmp_path):
    path = tmp_path / "predictions.bw"
    regions = [["chr1", 10, 14, 12], ["chr1", 20, 24, 22]]
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
    write_bigwig(data, regions, [("chr1", 40)], str(path))
    with pyBigWig.open(str(path)) as handle:
        np.testing.assert_allclose(handle.values("chr1", 10, 14), data[0])
        np.testing.assert_allclose(handle.values("chr1", 20, 24), data[1])
