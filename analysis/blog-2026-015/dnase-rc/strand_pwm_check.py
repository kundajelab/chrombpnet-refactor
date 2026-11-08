#!/usr/bin/env python
"""Measure, on real reads, whether an assay records a bond or a base.

This is the training-free half of the DNase reverse-complement evidence, and
the figure that explains it.

DNase I cuts one strand at a time. For a single physical nick, a plus-strand
read and a minus-strand read see the *same* local DNA, one of them backwards.
So the minus-strand cut PWM should be the reverse complement of the plus one
-- and the offset at which the two align says where the symmetry centre sits
relative to the base each is registered on:

    symmetry centre on a bond (DNase 0/+1)  ->  best lag  0  ->  label shift +1
    symmetry centre on a base (ATAC +4/-4)  ->  best lag -1  ->  label shift  0

Generally `best_lag = -1 - 2*mu` for a centre offset `mu`, and the label shift
is `-2*mu`, so **label_shift = best_lag + 1** either way.

`auto_shift_detect.get_pwms` builds both PWMs in plus-strand genomic
orientation, centred on the read `start` (plus) and `end` (minus) -- the raw
5' ends. Those must be moved onto the **deposit** positions before the lag
means anything, because that is what the profile array is indexed by:

    plus  deposit = start + plus_delta
    minus deposit = end   + minus_delta - 1        (genomecov -5 takes end-1)

For DNase 0/+1 the deltas are (0, +1), so both are no-ops and the raw 5' ends
already are the deposits. For ATAC +4/-4 they are (+4, -4), giving start+4
and end-5 -- which for a Tn5 insertion are the *same* base, the centre of the
9 bp stagger. Skipping this step makes ATAC read as lag 0 rather than -1,
i.e. it silently compares raw 5' ends against deposit-registered references.
The deltas come from `conventions.shift_delta_pairs`, so this cannot drift
away from what preprocessing actually does.

    ./strand_pwm_check.py                    # every ENCID in the ids file
    ./strand_pwm_check.py ENCSR000EOT        # just one

Reads are taken from one chromosome through the BAM index. `sample_reads`
shuffles its whole input, so handing it an entire 22 GB BAM means a full pass
over half a billion reads for a 20-column PWM. One chromosome is plenty of
cut sites, and the same restriction applies to every dataset, so the
comparison between them is unaffected.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

IDS = Path("/path/to/workdir/sherlock/ids/chromatlas/"
           "chrombpnet-refactor/ids_chrombpnet_refactor_benchmark_data_aug.txt")
GENOME = Path("/path/to/workdir/chromatlas/reference/hg38/hg38.fa")
OUT = Path("/path/to/workdir/chromatlas/test/chrombpnet-refactor-dnase-rc/analysis") / "strand_pwm_check"
NUM_SAMPLES = 100_000
REGION = "chr1"


def extract_region(bam, region, out_path):
    """`region` of `bam` as a gzipped tagAlign, via the index."""
    with open(out_path, "wb") as handle:
        view = subprocess.Popen(["samtools", "view", "-b", bam, region],
                                stdout=subprocess.PIPE)
        to_bed = subprocess.Popen(["bedtools", "bamtobed", "-i", "stdin"],
                                  stdin=view.stdout, stdout=subprocess.PIPE)
        view.stdout.close()
        gz = subprocess.Popen(["gzip", "-1"], stdin=to_bed.stdout,
                              stdout=handle)
        to_bed.stdout.close()
        codes = [gz.wait(), to_bed.wait(), view.wait()]
    if any(codes):
        raise SystemExit(f"extracting {region} from {bam} failed: {codes}")


def on_deposit_bases(assay, plus_reads, minus_reads):
    """Move both read sets onto the bases the bigwig would record.

    `get_pwms` centres the plus PWM on `start` and the minus on `end`, so
    rewriting those two columns re-registers the PWMs without touching
    `get_pwms` itself.
    """
    from chrombpnet.conventions import resolve, shift_delta_pairs

    convention = resolve(assay, "standard")
    (plus_delta, minus_delta), = shift_delta_pairs(convention, 0, 0)

    plus_reads = plus_reads.copy()
    minus_reads = minus_reads.copy()
    plus_reads["start"] = plus_reads["start"].astype(int) + plus_delta
    minus_reads["end"] = minus_reads["end"].astype(int) + minus_delta - 1
    return plus_reads, minus_reads


def rc(pwm):
    return pwm[::-1, ::-1]


def lag_profile(a, b, lags=range(-4, 5)):
    """corr(a[j], b[j+s]) over the overlap, for each lag s."""
    scores = {}
    n = len(a)
    for s in lags:
        lo, hi = max(0, -s), min(n, n - s)
        if hi - lo < n - 4:
            continue
        x = a[lo:hi].ravel() - a[lo:hi].mean()
        y = b[lo + s:hi + s].ravel() - b[lo + s:hi + s].mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        scores[s] = float(x @ y / denom) if denom else 0.0
    return scores


def datasets(selected):
    rows, seen = [], set()
    for line in IDS.read_text().splitlines()[1:]:
        assay, encid, bam, _peaks, _fold = line.split("\t")
        if encid in seen or (selected and encid not in selected):
            continue
        seen.add(encid)
        rows.append((assay, encid, bam))
    return rows


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chrombpnet.evaluation.custom_sequences import viz_sequence

    n = len(results)
    fig = plt.figure(figsize=(13, 3.1 * n))
    for row, r in enumerate(results):
        # Left: the two logos, minus shown reverse-complemented so a correct
        # convention makes them visibly the same motif.
        for col, (title, pwm) in enumerate(
                ((f"{r['encid']} {r['assay']}  plus-strand cuts", r["plus"]),
                 ("reverse complement of minus-strand cuts", rc(r["minus"])))):
            ax = fig.add_subplot(n, 3, row * 3 + col + 1)
            viz_sequence.plot_weights_given_ax(
                ax, viz_sequence.ic_scale(pwm, background=[.25] * 4),
                height_padding_factor=0.2, length_padding=1.0,
                subticks_frequency=5, highlight={})
            ax.set_title(title, fontsize=9)
            ax.axvline(len(pwm) / 2 - 0.5, color="crimson", lw=1, ls="--")

        # Right: the lag curve. The argmax is the measurement.
        ax = fig.add_subplot(n, 3, row * 3 + 3)
        lags = sorted(r["scores"])
        ax.plot(lags, [r["scores"][s] for s in lags], "o-", color="black")
        ax.axvline(r["best_lag"], color="crimson", lw=1, ls="--")
        ax.set_xlabel("lag of rc(plus) against minus")
        ax.set_ylabel("correlation")
        ax.set_title(f"best lag {r['best_lag']:+d}  ->  label shift "
                     f"{r['best_lag'] + 1:+d}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(str(OUT) + suffix, dpi=160)
    print(f"\nwrote {OUT}.png and {OUT}.pdf")


def main(selected):
    from chrombpnet.helpers.preprocessing.auto_shift_detect import (
        get_pwms, sample_reads)

    results = []
    print(f"{'ENCID':<13} {'assay':<6} {'best lag':>9} {'corr':>7} "
          f"{'runner-up':>10} {'label shift':>12}")
    for assay, encid, bam in datasets(selected):
        if not Path(bam).exists():
            print(f"{encid:<13} {assay:<6}   SKIP (no BAM at {bam})")
            continue
        with tempfile.TemporaryDirectory(dir=OUT.parent) as scratch:
            tagalign = Path(scratch) / f"{encid}.{REGION}.tagAlign.gz"
            print(f"  extracting {REGION} from {Path(bam).name} ...",
                  flush=True)
            extract_region(bam, REGION, tagalign)
            plus_reads, minus_reads = sample_reads(
                None, None, str(tagalign), NUM_SAMPLES, str(GENOME))
        plus_reads, minus_reads = on_deposit_bases(assay, plus_reads, minus_reads)
        plus_pwm, minus_pwm = get_pwms(plus_reads, minus_reads, str(GENOME))
        scores = lag_profile(minus_pwm, rc(plus_pwm))
        best = max(scores, key=scores.get)
        runner = max(v for k, v in scores.items() if k != best)
        print(f"{encid:<13} {assay:<6} {best:>+9d} {scores[best]:>7.3f} "
              f"{runner:>10.3f} {best + 1:>+12d}")
        results.append(dict(assay=assay, encid=encid, plus=plus_pwm,
                            minus=minus_pwm, scores=scores, best_lag=best))

    if not results:
        print("\nNo BAMs found. Run ./00_prepare_bams.sh first.")
        return 1

    print("\nExpected, with both PWMs on their deposit bases:")
    print("  DNASE lag  0  -> the cut sits on a BOND -> label shift +1")
    print("  ATAC  lag -1  -> the cut sits on a BASE -> label shift  0")
    dnase = {r["best_lag"] for r in results if r["assay"] == "DNASE"}
    atac = {r["best_lag"] for r in results if r["assay"] == "ATAC"}
    if dnase and dnase != {0}:
        print(f"  !! DNASE lags are {sorted(dnase)}, not {{0}}")
    if atac and atac != {-1}:
        print(f"  !! ATAC lags are {sorted(atac)}, not {{-1}}")
    plot(results)
    return 0


if __name__ == "__main__":
    sys.exit(main(set(sys.argv[1:])))
