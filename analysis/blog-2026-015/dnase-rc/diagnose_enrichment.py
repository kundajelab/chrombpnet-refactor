"""Summit-vs-flank enrichment: real localized signal, or diffuse background?

Internally normalized within each 2kb window, so it does not depend on read
depth, peak count, or whether the peak set came from the same alignment.

    python diagnose_enrichment.py --root <run tree>     # DNASE_RC_ROOT
"""
import argparse

import numpy as np, pandas as pd, pyBigWig
from chrombpnet.regions import NARROWPEAK_SCHEMA

W = 2000                      # window; summit = central 200, flank = outer 2x400


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="the dnase-rc run tree (<arm>/<ASSAY>/<ENCID>/fold_<N>/...)")
    R = ap.parse_args(argv).root

    CASES = {
     "ENCSR000EOT (K562)": (
       f"{R}/rc-only/DNASE/ENCSR000EOT/fold_3/model/auxiliary/data_unstranded.bw",
       f"{R}/rc-only/DNASE/ENCSR000EOT/fold_4/model/auxiliary/filtered.peaks.bed"),
     "ENCSR163PKT (stomach)": (
       f"{R}/rc-only/DNASE/ENCSR163PKT/fold_1/model/auxiliary/data_unstranded.bw",
       f"{R}/right/DNASE/ENCSR163PKT/fold_4/model/auxiliary/filtered.peaks.bed"),
    }

    print(f"{'dataset':<24} {'n':>6} {'summit/flank':>13} {'median summit':>14} {'median flank':>13}")
    for name, (bw, pk) in CASES.items():
        df = pd.read_csv(pk, sep="\t", header=None, names=NARROWPEAK_SCHEMA).sample(
            n=4000, random_state=0)
        ratios, summits, flanks = [], [], []
        with pyBigWig.open(bw) as h:
            lens = h.chroms()
            for _, r in df.iterrows():
                c = int(r["start"]) + int(r["summit"])
                lo, hi = c - W // 2, c + W // 2
                if r["chr"] not in lens or lo < 0 or hi > lens[r["chr"]]:
                    continue
                v = np.nan_to_num(h.values(r["chr"], lo, hi))
                s = v[900:1100].mean()                 # central 200 bp
                f = np.concatenate([v[:400], v[-400:]]).mean()
                summits.append(s); flanks.append(f)
                if f > 0:
                    ratios.append(s / f)
        ratios = np.array(ratios)
        print(f"{name:<24} {len(ratios):>6} {np.median(ratios):>13.2f} "
              f"{np.median(summits):>14.3f} {np.median(flanks):>13.3f}")


if __name__ == "__main__":
    main()
