from concurrent.futures import ThreadPoolExecutor

import numpy as np

from chrombpnet.evaluation.make_bigwigs.bigwig_writer import NumpyBigWig
from chrombpnet.training.utils.data_utils import one_hot
import pandas as pd
from chrombpnet.regions import NARROWPEAK_SCHEMA, window_bounds
from chrombpnet.sequences import fetch_windows

def read_chrom_sizes(fname):
    with open(fname) as f:
        gs = [x.strip().split('\t') for x in f]
    gs = [(x[0], int(x[1])) for x in gs if len(x)==2]
 
    return gs

def get_seq(peaks_df, genome, width):
    """One-hot windows for the regions whose full window is in bounds.

    The per-region pyfaidx loop this used to run now lives in
    chrombpnet.sequences as the fallback path; the mask semantics are
    unchanged.
    """
    seqs, peaks_used = fetch_windows(peaks_df, genome, width)
    if not peaks_used.any():
        raise ValueError("No regions have a complete in-bounds sequence window")
    return seqs, peaks_used


def get_regions(regions_file, seqlen, regions_used=None):
    # regions file is assumed to be centered at summit (2nd + 10th column)
    # it is adjusted to be of length seqlen centered at summit

    if seqlen % 2:
        raise ValueError("Region sequence length must be even")

    #with open(regions_file) as r:
    #    regions = [x.strip().split('\t') for x in r]

    if isinstance(regions_file, pd.DataFrame):
        regions = regions_file.copy()
        if set(("chr", "start", "summit")) <= set(regions.columns):
            regions = regions[NARROWPEAK_SCHEMA]
    else:
        regions = pd.read_csv(regions_file,sep='\t',header=None)
    #print(regions)
    if regions_used is None:
        regions = [[x[0], int(x[1])+int(x[9])-seqlen//2, int(x[1])+int(x[9])+seqlen//2, int(x[1])+int(x[9])] for x in np.array(regions.values)]
    else:
        regions = [[x[0], int(x[1])+int(x[9])-seqlen//2, int(x[1])+int(x[9])+seqlen//2, int(x[1])+int(x[9])] for x in np.array(regions.values)[regions_used]]

    return regions

def write_bigwig(data, regions, gs, bw_out, debug_chr=None, use_tqdm=False, outstats_file=None):
    """Write ``data`` (one row per region) to ``bw_out``; see BigWigWriter."""
    writer = BigWigWriter(regions, gs, bw_out, debug_chr=debug_chr,
                          use_tqdm=use_tqdm, outstats_file=outstats_file)
    writer.add(0, data)
    writer.close()


def write_order(regions, gs):
    """The order ``write_bigwig`` writes ``regions`` in: by chromosome, in the
    chromosome-sizes order, then by start. Regions may not be sorted."""
    chr_to_idx = {}
    for i,x in enumerate(gs):
        chr_to_idx[x[0]] = i
    return sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))


def stream_plan(regions, gs, batch_size, chunk_rows):
    """Forward calls for streaming ``regions`` into a BigWigWriter: a list of
    ``(start, stop)`` row ranges, each a run of whole batches.

    The batches are the ones a single predict over all rows makes -- rows
    ``k * batch_size`` onward, the last one short -- so every prediction is
    unchanged; only the order the batches run in changes. They run in the order
    the writer needs them (each batch by the earliest write position among its
    rows), so a regions file sorted another way -- ``sort -k1,1`` puts chr10
    before chr2, while the chromosome sizes usually list chr2 first -- still
    streams. Batches adjacent in the file and in that order share a call, up
    to ``chunk_rows`` rows.
    """
    total = len(regions)
    if total == 0:
        return []
    position = np.empty(total, dtype=np.int64)
    position[write_order(regions, gs)] = np.arange(total)
    batches = -(-total // batch_size)
    first_needed = np.minimum.reduceat(position, np.arange(0, total, batch_size))
    per_call = max(1, chunk_rows // batch_size)
    runs = []
    for batch in sorted(range(batches), key=lambda k: first_needed[k]):
        if runs and runs[-1][1] == batch and runs[-1][1] - runs[-1][0] < per_call:
            runs[-1][1] = batch + 1
        else:
            runs.append([batch, batch + 1])
    return [(first * batch_size, min(last * batch_size, total)) for first, last in runs]


class BigWigWriter:
    """``write_bigwig``, fed rows as they are predicted.

    Adapted from the optimization kit's pipelined writer
    (github.com/anthropics/uplifting-biomolecular-modeling, Apache-2.0), which
    writes each forward chunk's regions while the GPU works on the next. The
    kit predicts in sorted order to make that possible; here the forward keeps
    the regions file's order, and the writer goes as far down the sorted order
    as the rows that have arrived allow. A sorted regions file therefore streams
    completely, and an unsorted one waits for its rows -- at worst until the
    last chunk, which is where the whole write used to start.

    The bigWig calls are the same calls in the same order with the same values
    as the one-shot write, so the file is byte-identical however the rows
    arrive. They go to bigwig_writer.NumpyBigWig, which writes what pyBigWig
    writes for them but builds the zoom levels after each arriving batch
    instead of all at close, where libBigWig spends most of a write's time.
    """

    def __init__(self, regions, gs, bw_out, debug_chr=None, use_tqdm=False, outstats_file=None):
        # regions may overlap but as we go in sorted order, at a given position,
        # we will pick the value from the interval whose summit is closest to
        # current position
        self._regions = regions
        self._bw = NumpyBigWig(bw_out, gs)
        self._order = write_order(regions, gs)

        # Hoisted: this set was rebuilt once per region.
        self._debug_chroms = set(np.atleast_1d(debug_chr)) if debug_chr else None

        # Only accumulate the written signal when the stats file will consume it.
        # For 20k regions x 1kb that is 80 MB held plus an 80 MB hstack copy, both
        # of which used to be paid even with outstats_file=None.
        self._outstats_file = outstats_file
        self._all_entries = []
        self._wrote_any = False
        self._cur_chr = ""
        self._cur_end = 0

        self._data = None
        self._arrived = np.zeros(len(regions), dtype=bool)
        self._next = 0                  # position in the sorted order
        self._progress = None
        if use_tqdm:
            from tqdm import tqdm
            self._progress = tqdm(total=len(self._order))

    def add(self, start, rows):
        """Rows ``start`` to ``start + len(rows)`` of the data, in regions order.

        ``rows`` is kept, not copied, until the write is done, so the caller
        must not modify it.
        """
        stop = start + len(rows)
        if start < 0 or stop > len(self._regions):
            raise ValueError(f"rows {start}:{stop} are outside the {len(self._regions)} regions")
        if self._data is None and start == 0 and stop == len(self._regions):
            # The whole array at once (write_bigwig): use it as it is.
            self._data = rows
        else:
            rows = np.asarray(rows)
            if self._data is None:
                self._data = np.empty((len(self._regions),) + rows.shape[1:], dtype=rows.dtype)
            self._data[start:stop] = rows
        self._arrived[start:stop] = True
        self._advance()
        self._bw.feed_zooms()

    def _advance(self):
        order = self._order
        while self._next < len(order) and self._arrived[order[self._next]]:
            self._write(self._next)
            self._next += 1
            if self._progress is not None:
                self._progress.update(1)

    def _write(self, itr):
        regions, order_of_regs, data = self._regions, self._order, self._data
        # subset to chromosome (debugging)
        i = order_of_regs[itr]
        i_chr, i_start, i_end, i_mid = regions[i]
        if self._debug_chroms is not None and i_chr not in self._debug_chroms:
            return

        if i_chr != self._cur_chr:
            self._cur_chr = i_chr
            self._cur_end = 0

        # bring current end to at least start of current region
        if self._cur_end < i_start:
            self._cur_end = i_start

        if regions[i][2] < self._cur_end:
            raise ValueError("Regions must have non-decreasing, valid end coordinates")

        # figure out where to stop for this region, get next region
        # which may partially overlap with this one
        next_end = i_end

        if itr+1 != len(order_of_regs):
            n = order_of_regs[itr+1]
            next_chr, next_start, _, next_mid = regions[n]

            if next_chr == i_chr and next_start < i_end:
                # if next region overlaps with this, end between their midpoints
                next_end = (i_mid+next_mid)//2

        cur_end = self._cur_end
        vals = data[i][cur_end - i_start:next_end - i_start]

        # One fixed-step block per region instead of one bedGraph interval
        # per base. Same values on read-back, ~2.8x faster to write and ~1.5x
        # smaller, but a different on-disk section type -- so the file bytes
        # differ from the interval form the package shipped before.
        # (pyBigWig's addEntries(chrom, start, values=vals, span=1, step=1).)
        if next_end > cur_end:
            self._bw.add_entries(i_chr, int(cur_end), vals)

        self._wrote_any = True
        if self._outstats_file is not None:
            self._all_entries.append(vals)

        self._cur_end = next_end

    def close(self):
        """Finish the file and write the stats; every row must have arrived."""
        if self._next != len(self._order):
            missing = int((~self._arrived).sum())
            raise ValueError(f"{missing} of {len(self._order)} regions never arrived; the bigwig is incomplete")
        if self._progress is not None:
            self._progress.close()

        # Deliberately a region count, not a base count: a region whose slice is
        # empty still counted as "written" under the old `if not all_entries`
        # test, because it appended a zero-length array.
        if not self._wrote_any or self._outstats_file is None:
            self._bw.close()
            if not self._wrote_any:
                raise ValueError("No BigWig entries were written")
            return
        # The stats beside the file's assembly, on a thread: both are mostly
        # NumPy partitions and zlib, which release the GIL, and they write
        # different files.
        with ThreadPoolExecutor(1) as pool:
            stats = pool.submit(self._write_stats)
            self._bw.close()
            stats.result()

    def _write_stats(self):
        # Only the collected case can be stacked -- np.hstack([]) raises.
        all_entries = np.hstack(self._all_entries)
        with open(self._outstats_file, 'w') as f:
            f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
            f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
            f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
            f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
            f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
            f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
            f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
            f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
            f.write("Max\t{:.6f}\n".format(np.max(all_entries)))
