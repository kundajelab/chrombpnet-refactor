"""Stringency by the peak caller's OWN signalValue -- independent of the
bigwig we then measure, so no selection-on-outcome circularity."""
import numpy as np, pandas as pd, pyBigWig
from chrombpnet.regions import NARROWPEAK_SCHEMA

R = "/path/to/workdir/chromatlas/test/chrombpnet-refactor-dnase-rc"
A = "/path/to/lab/projects/chromatin-atlas-2022"
OUT, FACTOR = 1000, 0.8

def read_bed(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=range(10),
                       names=NARROWPEAK_SCHEMA)

def counts_of(bw, df, n=8000):
    if len(df) > n:
        df = df.sample(n=n, random_state=0)
    out = []
    with pyBigWig.open(bw) as h:
        lens = h.chroms()
        for _, r in df.iterrows():
            c = int(r["start"]) + int(r["summit"])
            lo, hi = c - OUT // 2, c + OUT // 2
            if r["chr"] not in lens or lo < 0 or hi > lens[r["chr"]]:
                continue
            out.append(np.nan_to_num(h.values(r["chr"], lo, hi)).sum())
    return np.array(out)

CASES = {
 "K562   ": (f"{R}/rc-only/DNASE/ENCSR000EOT/fold_3/model/auxiliary/data_unstranded.bw",
             f"{A}/DNASE/ENCSR000EOT/preprocessing/downloads/peaks.bed.gz",
             f"{R}/both-full/DNASE/ENCSR000EOT/fold_1/data/output/ENCSR000EOT/negatives.bed.gz"),
 "stomach": (f"{R}/rc-only/DNASE/ENCSR163PKT/fold_1/model/auxiliary/data_unstranded.bw",
             f"{A}/DNASE/ENCSR163PKT/preprocessing/downloads/peaks.bed.gz",
             f"{R}/rc-only/DNASE/ENCSR163PKT/fold_0/data/output/ENCSR163PKT/negatives.bed.gz"),
}

print(f"{'dataset':<8} {'kept (signalValue)':>19} {'peak 1%ile':>11} "
      f"{'threshold':>10} {'retained':>9} {'thr/nonpk med':>14}")
for name, (bw, pk, nb) in CASES.items():
    peaks = read_bed(pk)
    nonpeak_counts = counts_of(bw, read_bed(nb))
    nmed = float(np.median(nonpeak_counts))
    sig = peaks.iloc[:, 6].astype(float)
    for frac in (1.0, 0.5, 0.25, 0.10):
        keep = peaks[sig >= sig.quantile(1 - frac)]
        pc = counts_of(bw, keep)
        p1 = float(np.quantile(pc, 0.01))
        thr = p1 * FACTOR
        retained = float(np.mean(nonpeak_counts < thr))
        print(f"{name:<8} {frac:>18.0%} {p1:>11.1f} {thr:>10.1f} "
              f"{retained:>9.1%} {thr / nmed:>13.2f}x")
    print()
