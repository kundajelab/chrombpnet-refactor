"""Peak vs nonpeak count distributions: is the stomach background hot?"""
import numpy as np, pandas as pd, pyBigWig, sys
from chrombpnet.regions import NARROWPEAK_SCHEMA

R = "/path/to/workdir/chromatlas/test/chrombpnet-refactor-dnase-rc"
OUT = 1000  # outputlen

CASES = {
 "ENCSR000EOT (K562)": (
   f"{R}/rc-only/DNASE/ENCSR000EOT/fold_3/model/auxiliary/data_unstranded.bw",
   f"{R}/rc-only/DNASE/ENCSR000EOT/fold_4/model/auxiliary/filtered.peaks.bed",
   f"{R}/both-full/DNASE/ENCSR000EOT/fold_1/data/output/ENCSR000EOT/negatives.bed.gz"),
 "ENCSR163PKT (stomach)": (
   f"{R}/rc-only/DNASE/ENCSR163PKT/fold_1/model/auxiliary/data_unstranded.bw",
   f"{R}/right/DNASE/ENCSR163PKT/fold_4/model/auxiliary/filtered.peaks.bed",
   f"{R}/rc-only/DNASE/ENCSR163PKT/fold_0/data/output/ENCSR163PKT/negatives.bed.gz"),
}

def counts(bw, bed, n=6000, seed=0):
    df = pd.read_csv(bed, sep="\t", header=None, names=NARROWPEAK_SCHEMA)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    tot, kept = [], 0
    with pyBigWig.open(bw) as h:
        lens = h.chroms()
        for _, r in df.iterrows():
            c = int(r["start"]) + int(r["summit"])
            lo, hi = c - OUT // 2, c + OUT // 2
            if r["chr"] not in lens or lo < 0 or hi > lens[r["chr"]]:
                continue
            tot.append(np.nan_to_num(h.values(r["chr"], lo, hi)).sum()); kept += 1
    return np.array(tot)

print(f"{'dataset':<24} {'set':<9} {'n':>6} {'1%':>8} {'25%':>8} {'median':>9} {'75%':>9} {'99%':>10}")
summary = {}
for name, (bw, pk, np_) in CASES.items():
    for label, bed in (("peaks", pk), ("nonpeaks", np_)):
        v = counts(bw, bed)
        q = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99])
        summary[(name, label)] = (v, q)
        print(f"{name:<24} {label:<9} {v.size:>6} " + " ".join(f"{x:>8.1f}" for x in q[:4]) + f" {q[4]:>10.1f}")

print()
for name in CASES:
    pk, _ = summary[(name, "peaks")]
    nk, _ = summary[(name, "nonpeaks")]
    thr = np.quantile(pk, 0.01) * 0.8          # the DNase bias threshold
    frac = float(np.mean(nk < thr))
    print(f"{name}")
    print(f"   bias threshold = 0.8 x peak 1%tile      = {thr:8.1f}")
    print(f"   fraction of nonpeaks below it           = {frac:8.1%}")
    print(f"   median peak / median nonpeak            = {np.median(pk)/max(np.median(nk),1e-9):8.2f}x")
    print(f"   peak 1%tile / nonpeak median            = {np.quantile(pk,0.01)/max(np.median(nk),1e-9):8.2f}x")
