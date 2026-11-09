import argparse
import zlib

import h5py
import numpy as np
import pytest

from chrombpnet import compression
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


def _values(shape, dtype):
    rng = np.random.default_rng(0)
    if np.dtype(dtype).kind == "f":
        return (rng.standard_normal(shape) * 100).astype(dtype)
    return rng.integers(0, 200, size=shape).astype(dtype)


@pytest.mark.parametrize("shape, dtype", [
    ((1003, 37), np.float64),     # edge chunks on both axes
    ((257, 3, 41), np.float16),   # the scores' layout and dtype
    ((5000,), np.int64),
    ((64, 33), np.uint8),         # one-byte values: no shuffle
])
def test_threaded_dataset_is_h5pys_dataset(tmp_path, monkeypatch, shape, dtype):
    monkeypatch.setattr(compression, "THREADED_MIN_BYTES", 0)
    values = _values(shape, dtype)
    with h5py.File(tmp_path / "threaded.h5", "w") as handle:
        compression.create_dataset(handle, "x", values)
    with h5py.File(tmp_path / "h5py.h5", "w") as handle:
        handle.create_dataset("x", data=values, **h5_filters(values))
    with h5py.File(tmp_path / "threaded.h5", "r") as a, h5py.File(tmp_path / "h5py.h5", "r") as b:
        ours, theirs = a["x"], b["x"]
        properties = ("dtype", "shape", "chunks", "compression", "compression_opts", "shuffle", "fillvalue")
        assert [getattr(ours, p) for p in properties] == [getattr(theirs, p) for p in properties]
        np.testing.assert_array_equal(ours[()], values)
        assert ours.id.get_num_chunks() == theirs.id.get_num_chunks()
        for i in range(theirs.id.get_num_chunks()):
            offset = theirs.id.get_chunk_info(i).chunk_offset
            (mask_a, blob_a), (mask_b, blob_b) = ours.id.read_direct_chunk(offset), theirs.id.read_direct_chunk(offset)
            assert mask_a == mask_b == 0
            # The same bytes going into deflate: shuffle order and edge padding match.
            assert zlib.decompress(blob_a) == zlib.decompress(blob_b)


def test_small_and_string_datasets_take_h5pys_path(tmp_path, monkeypatch):
    strings = np.array(["chr1", "chr2"], dtype=h5py.special_dtype(vlen=str))
    with h5py.File(tmp_path / "x.h5", "w") as handle:
        compression.create_dataset(handle, "small", [1, 2, 3], int)
        monkeypatch.setattr(compression, "THREADED_MIN_BYTES", 0)
        compression.create_dataset(handle, "chroms", strings, h5py.special_dtype(vlen=str))
        compression.create_dataset(handle, "scalar", np.float32(1.5))
    with h5py.File(tmp_path / "x.h5", "r") as handle:
        assert handle["small"][()].tolist() == [1, 2, 3] and handle["small"].shuffle
        assert [s.decode() for s in handle["chroms"][()]] == ["chr1", "chr2"]
        assert handle["scalar"][()] == np.float32(1.5)


# --h5-gzip-level and --no-h5-shuffle (compression.configure).

PRED_BW = ["pred_bw", "-cm", "m.pt", "-r", "regions.bed", "-g", "g.fa", "-c", "c.sizes", "-op", "out"]


@pytest.fixture
def settings(monkeypatch):
    """Restore the module's level and shuffle after a test that configures them."""
    monkeypatch.setattr(compression, "GZIP_LEVEL", compression.GZIP_LEVEL)
    monkeypatch.setattr(compression, "SHUFFLE", compression.SHUFFLE)


def test_configure_sets_level_and_shuffle(settings):
    floats = np.zeros((3, 4), np.float32)
    compression.configure(4, False)
    assert h5_filters(floats) == {"compression": "gzip", "compression_opts": 4}
    compression.configure(None, None)  # leaves both as they are
    assert h5_filters(floats) == {"compression": "gzip", "compression_opts": 4}
    compression.configure(6, True)
    assert h5_filters(floats) == {"compression": "gzip", "compression_opts": 6, "shuffle": True}
    for bad in (0, 10, 4.5, True):
        with pytest.raises(ValueError):
            compression.configure(bad)


@pytest.mark.parametrize("level, shuffle", [(4, False), (4, True), (1, False)])
def test_threaded_dataset_follows_the_settings(tmp_path, monkeypatch, settings, level, shuffle):
    monkeypatch.setattr(compression, "THREADED_MIN_BYTES", 0)
    compression.configure(level, shuffle)
    values = _values((1003, 37), np.float32)
    with h5py.File(tmp_path / "threaded.h5", "w") as handle:
        compression.create_dataset(handle, "x", values)
    with h5py.File(tmp_path / "h5py.h5", "w") as handle:
        handle.create_dataset("x", data=values, compression="gzip", compression_opts=level, shuffle=shuffle)
    with h5py.File(tmp_path / "threaded.h5", "r") as a, h5py.File(tmp_path / "h5py.h5", "r") as b:
        ours, theirs = a["x"], b["x"]
        assert (ours.compression_opts, ours.shuffle) == (level, shuffle)
        np.testing.assert_array_equal(ours[()], values)
        for i in range(theirs.id.get_num_chunks()):
            offset = theirs.id.get_chunk_info(i).chunk_offset
            # Each chunk is deflated at the configured level, from the same bytes h5py would.
            blob_a, blob_b = ours.id.read_direct_chunk(offset)[1], theirs.id.read_direct_chunk(offset)[1]
            assert zlib.decompress(blob_a) == zlib.decompress(blob_b)
            assert blob_a == zlib.compress(zlib.decompress(blob_b), level)


def test_cli_options_configure_compression(settings):
    from chrombpnet import parsers
    args = parsers.read_parser(PRED_BW + ["--h5-gzip-level", "4", "--no-h5-shuffle"])
    assert (args.h5_gzip_level, args.h5_shuffle) == (4, False)
    compression.configure_from_args(args)
    assert (compression.GZIP_LEVEL, compression.SHUFFLE) == (4, False)


def test_cli_options_absent_when_omitted(settings):
    from chrombpnet import parsers
    args = parsers.read_parser(PRED_BW)
    parsers.drop_absent(args)
    # args.json records exactly what it did before the options existed.
    assert not hasattr(args, "h5_gzip_level") and not hasattr(args, "h5_shuffle")
    compression.configure_from_args(args)
    assert (compression.GZIP_LEVEL, compression.SHUFFLE) == (9, True)


@pytest.mark.parametrize("level", ["0", "10", "four"])
def test_cli_rejects_a_level_outside_1_to_9(level):
    from chrombpnet import parsers
    with pytest.raises(SystemExit):
        parsers.read_parser(PRED_BW + ["--h5-gzip-level", level])


@pytest.mark.parametrize("command", [
    ["pipeline"], ["train"], ["qc"], ["bias", "pipeline"], ["bias", "qc"], ["pred_bw"],
    ["contribs_bw"], ["footprints"], ["modisco_motifs"], ["predict"]])
def test_every_hdf5_writing_command_takes_the_options(command):
    from chrombpnet import parsers
    parser = parsers.build_parser()
    sub = parser
    for name in command:
        sub = next(a for a in sub._actions if isinstance(a, argparse._SubParsersAction)).choices[name]
    flags = {flag for action in sub._actions for flag in action.option_strings}
    assert {"--h5-gzip-level", "--no-h5-shuffle"} <= flags
