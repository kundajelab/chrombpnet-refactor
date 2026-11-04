"""bigwig_helper.stream_plan: the forward's batches, whole and in the order
the bigWig writer needs them."""

import numpy as np
import pytest

from chrombpnet.evaluation.make_bigwigs import bigwig_helper

CHROMS = [("chr1", 3_000_000), ("chr2", 2_000_000), ("chr10", 900_000), ("chrX", 5_000)]


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
