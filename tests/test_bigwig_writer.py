"""bigwig_writer.NumpyBigWig must write exactly the bytes pyBigWig writes for
the span/step call pattern, however often its zoom levels are fed."""

import hashlib

import numpy as np
import pyBigWig
import pytest

from chrombpnet.evaluation.make_bigwigs import bigwig_helper
from chrombpnet.evaluation.make_bigwigs.bigwig_writer import VALS_PER_BLOCK, NumpyBigWig

CHROMS = [("chr1", 3_000_000), ("chr2", 2_000_000), ("chr10", 900_000), ("chrX", 5_000)]


def _calls(seed, regions=600, width=1000, overlap_every=3, chain=0, signed=False):
    """(chrom, start, values) calls in write order, as BigWigWriter makes them:
    gaps, contiguous continuations (overlapping regions end at midpoints), and
    optionally one long contiguous chain spanning several data blocks."""
    rng = np.random.default_rng(seed)
    calls = []
    for chrom, size in CHROMS[:3]:
        position = int(rng.integers(0, 5000))
        for index in range(regions // 3):
            length = int(rng.integers(1, width + 1))
            if position + length >= size:
                break
            values = rng.gamma(0.4, 2.0, size=length).astype(np.float32)
            if signed:
                values -= np.float32(0.8)
            calls.append((chrom, position, values))
            contiguous = index % overlap_every == 0
            position += length if contiguous else length + int(rng.integers(1, 20_000))
    if chain:
        start = 4_000
        values = rng.random(chain).astype(np.float32)
        for piece in np.array_split(values, 7):
            calls.append(("chrX" if chain < 5000 else "chr1", start, piece))
            start += len(piece)
    return calls


def _reference(path, calls):
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(CHROMS)
    for chrom, start, values in calls:
        bw.addEntries(chrom, start, values=values, span=1, step=1)
    bw.close()


def _numpy(path, calls, feed_every):
    bw = NumpyBigWig(str(path), CHROMS)
    for index, (chrom, start, values) in enumerate(calls):
        bw.add_entries(chrom, start, values)
        if feed_every and (index + 1) % feed_every == 0:
            bw.feed_zooms()
    bw.close()


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _chain_calls(seed):
    """A contiguous run longer than two data blocks, on chr1 after others."""
    rng = np.random.default_rng(seed)
    total = 2 * VALS_PER_BLOCK + 1234
    calls, start = [], 10_000
    for piece in np.array_split(rng.random(total).astype(np.float32), 9):
        calls.append(("chr1", start, piece))
        start += len(piece)
    return calls + [("chr2", 50, rng.random(VALS_PER_BLOCK).astype(np.float32))]


CASES = {
    "mixed": lambda: _calls(0),
    "dense": lambda: _calls(1, regions=3000, width=400, overlap_every=1),
    "sparse": lambda: _calls(2, overlap_every=10**9),
    "signed": lambda: _calls(3, signed=True),
    "chain": lambda: _chain_calls(4),
    "one": lambda: [("chr2", 777, np.arange(1000, dtype=np.float32) / 7)],
    "single_base": lambda: [("chr1", 5, np.float32([2.5])), ("chr1", 9, np.float32([0.5]))],
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("feed_every", [0, 1, 37])
def test_matches_pybigwig(tmp_path, case, feed_every):
    calls = CASES[case]()
    _reference(tmp_path / "pybigwig.bw", calls)
    _numpy(tmp_path / "numpy.bw", calls, feed_every)
    assert _digest(tmp_path / "numpy.bw") == _digest(tmp_path / "pybigwig.bw")


def test_reads_back(tmp_path):
    calls = _calls(5)
    _numpy(tmp_path / "numpy.bw", calls, feed_every=50)
    bw = pyBigWig.open(str(tmp_path / "numpy.bw"))
    try:
        for chrom, start, values in calls[::25]:
            np.testing.assert_array_equal(
                np.asarray(bw.values(chrom, start, start + len(values)), dtype=np.float32), values)
        assert bw.header()["nBasesCovered"] == sum(len(v) for _, _, v in calls)
    finally:
        bw.close()


def test_no_chromosomes_fails_up_front(tmp_path):
    with pytest.raises(RuntimeError):
        NumpyBigWig(str(tmp_path / "x.bw"), [])


def test_write_bigwig_matches_pybigwig_end_to_end(tmp_path):
    """write_bigwig's own walk (overlaps end at summit midpoints, sorted
    order) through both writers."""
    rng = np.random.default_rng(6)
    regions = []
    for chrom, size in CHROMS[:3]:
        for summit in np.sort(rng.integers(2000, size - 2000, size=300)):
            regions.append([chrom, int(summit) - 500, int(summit) + 500, int(summit)])
    order = rng.permutation(len(regions))
    regions = [regions[i] for i in order]
    data = rng.gamma(0.5, 1.0, size=(len(regions), 1000)).astype(np.float32)

    bigwig_helper.write_bigwig(data, regions, CHROMS, str(tmp_path / "numpy.bw"))
    calls = []

    class Recorder:
        def add_entries(self, chrom, start, values):
            calls.append((chrom, start, np.array(values)))

        def feed_zooms(self):
            pass

        def close(self):
            pass

    writer = bigwig_helper.BigWigWriter(regions, CHROMS, str(tmp_path / "unused.bw"))
    writer._bw = Recorder()
    writer.add(0, data)
    writer.close()
    _reference(tmp_path / "pybigwig.bw", calls)
    assert _digest(tmp_path / "numpy.bw") == _digest(tmp_path / "pybigwig.bw")


def _plan_rows(plan):
    return [row for start, stop in plan for row in range(start, stop)]


@pytest.mark.parametrize("batch_size,chunk_rows", [(8, 32), (5, 4096), (64, 64)])
def test_stream_plan_keeps_every_batch_whole(batch_size, chunk_rows):
    rng = np.random.default_rng(8)
    regions = [[chrom, int(s), int(s) + 100, int(s) + 50]
               for chrom in ("chr10", "chr2", "chr1") for s in np.sort(rng.integers(0, 10**6, 77))]
    plan = bigwig_helper.stream_plan(regions, CHROMS, batch_size, chunk_rows)
    assert sorted(_plan_rows(plan)) == list(range(len(regions)))
    for start, stop in plan:
        assert start % batch_size == 0
        assert stop % batch_size == 0 or stop == len(regions)
        assert stop - start <= max(batch_size, chunk_rows // batch_size * batch_size)


def test_stream_plan_follows_the_write_order():
    """A file sorted chr10, chr2, chr1 (as `sort -k1,1` leaves chr1, chr10,
    chr2 ... out of the chromosome-sizes order) runs chr1's batches first."""
    regions = ([["chr10", s, s + 100, s + 50] for s in range(0, 4000, 100)]
               + [["chr2", s, s + 100, s + 50] for s in range(0, 4000, 100)]
               + [["chr1", s, s + 100, s + 50] for s in range(0, 4000, 100)])
    plan = bigwig_helper.stream_plan(regions, CHROMS, 10, 20)
    order = bigwig_helper.write_order(regions, CHROMS)
    position = {row: index for index, row in enumerate(order)}
    # Rows arrive in plan order; the writer is never left waiting on a row
    # that a later call holds while an earlier call held rows it could not use.
    arrival = [position[row] for row in _plan_rows(plan)]
    assert arrival == sorted(arrival)


def test_stream_plan_of_a_sorted_file_is_file_order():
    regions = [["chr1", s, s + 100, s + 50] for s in range(0, 10000, 100)]
    assert bigwig_helper.stream_plan(regions, CHROMS, 8, 32) == [
        (start, min(start + 32, 100)) for start in range(0, 100, 32)]
