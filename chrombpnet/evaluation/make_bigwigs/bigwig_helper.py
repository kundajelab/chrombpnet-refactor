import pyBigWig
import numpy as np

#: pyBigWig can be built without numpy support, in which case addEntries only
#: accepts Python lists.
_PYBIGWIG_NUMPY = bool(getattr(pyBigWig, "numpy", 0))
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
    # regions may overlap but as we go in sorted order, at a given position,
    # we will pick the value from the interval whose summit is closest to 
    # current position
    
    chr_to_idx = {}
    for i,x in enumerate(gs):
        chr_to_idx[x[0]] = i

    bw = pyBigWig.open(bw_out, 'w')
    bw.addHeader(gs)
    
    # regions may not be sorted, so get their sorted order
    order_of_regs = sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))

    # Hoisted: this set was rebuilt once per region.
    debug_chroms = set(np.atleast_1d(debug_chr)) if debug_chr else None

    # Only accumulate the written signal when the stats file will consume it.
    # For 20k regions x 1kb that is 80 MB held plus an 80 MB hstack copy, both
    # of which used to be paid even with outstats_file=None.
    collect_entries = outstats_file is not None
    all_entries = []
    wrote_any = False
    cur_chr = ""
    cur_end = 0

    iterator = range(len(order_of_regs))
    if use_tqdm:
        from tqdm import tqdm
        iterator = tqdm(iterator)

    for itr in iterator:
        # subset to chromosome (debugging)
        i = order_of_regs[itr]
        i_chr, i_start, i_end, i_mid = regions[i]
        if debug_chroms is not None and i_chr not in debug_chroms:
            continue
    
        if i_chr != cur_chr: 
            cur_chr = i_chr
            cur_end = 0
    
        # bring current end to at least start of current region
        if cur_end < i_start:
            cur_end = i_start
    
        if regions[i][2] < cur_end:
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
       
        vals = data[i][cur_end - i_start:next_end - i_start]

        # numpy arrays rather than three Python lists and a float() per base.
        # pyBigWig's numpy branch only changes how the arguments are parsed
        # before the same bwAddIntervals call, so the file is byte-identical;
        # the list form is kept for builds compiled without numpy support.
        starts = np.arange(cur_end, next_end, dtype=np.int64)
        if _PYBIGWIG_NUMPY:
            bw.addEntries([i_chr] * (next_end - cur_end), starts,
                          ends=starts + 1, values=vals)
        else:
            bw.addEntries([i_chr] * (next_end - cur_end), starts.tolist(),
                          ends=(starts + 1).tolist(), values=vals.tolist())

        wrote_any = True
        if collect_entries:
            all_entries.append(vals)
        
        cur_end = next_end

    bw.close()

    # Deliberately a region count, not a base count: a region whose slice is
    # empty still counted as "written" under the old `if not all_entries`
    # test, because it appended a zero-length array.
    if not wrote_any:
        raise ValueError("No BigWig entries were written")
    if not collect_entries:
        return
    all_entries = np.hstack(all_entries)
    with open(outstats_file, 'w') as f:
        f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
        f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
        f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
        f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
        f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
        f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
        f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
        f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
        f.write("Max\t{:.6f}\n".format(np.max(all_entries)))
