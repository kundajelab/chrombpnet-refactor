import h5py
import numpy as np
import pytest

from chrombpnet.compression import h5_filters, recompress_h5


def test_filters_by_dtype():
    assert h5_filters(np.zeros((3, 4), np.float16)) == {
        "compression": "gzip", "compression_opts": 9, "shuffle": True}
    assert h5_filters([1, 2, 3], int)["shuffle"] is True
    strings = np.array(["chr1", "chr2"], dtype=h5py.special_dtype(vlen=str))
    assert h5_filters(strings) == {"compression": "gzip", "compression_opts": 9}
    assert h5_filters(np.float32(1.0)) == {}
    assert h5_filters(np.zeros((0, 4))) == {}


def test_recompress_reads_back_identical(tmp_path):
    path = tmp_path / "x.h5"
    rng = np.random.default_rng(0)
    values = rng.standard_normal((200, 4, 100)).astype(np.float16)
    with h5py.File(path, "w") as handle:
        handle.attrs["window_size"] = 8
        group = handle.create_group("g")
        group.create_dataset("values", data=values)
        group.create_dataset("scalar", data=3)
        group["values"].attrs["note"] = "kept"
    before, after = recompress_h5(path)
    assert after < before
    with h5py.File(path, "r") as handle:
        dataset = handle["g/values"]
        assert (dataset.compression, dataset.compression_opts, dataset.shuffle) == ("gzip", 9, True)
        np.testing.assert_array_equal(dataset[()], values)
        assert dataset.attrs["note"] == "kept" and handle.attrs["window_size"] == 8
        assert handle["g/scalar"][()] == 3


def test_recompress_leaves_a_missing_file_alone(tmp_path):
    with pytest.raises(OSError):
        recompress_h5(tmp_path / "missing.h5")
    assert not (tmp_path / "missing.h5.tmp").exists()
