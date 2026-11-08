#!/bin/bash
# Build one chromosome's bigwig under each arm's cut convention, from a real BAM,
# and check the relationships between them hold at genome scale.
#
#   ./check_arms_on_real_reads.sh [ENCID] [CHROM]
#
# `tests/test_preprocessing_conventions.py` already pins the exact deposit
# positions on two synthetic reads. What that cannot show is behaviour at
# scale: whether `genomecov -scale 0.5` is still exact over millions of
# overlapping cuts, and whether the doubled record stream for `both-*`
# survives the sort. This checks those.
#
# Expected:
#   standard == right          byte-identical; `right` changes only augmentation
#   left      = standard shifted one base left
#   both-half  total == standard total      (the conservation gate)
#   both-full  total == 2x standard total

set -euo pipefail

ENCID="${1:-ENCSR000EOT}"
CHROM="${2:-chr1}"

CONDA_ENV="${CONDA_ENV:-/path/to/workdir/software/miniconda3/envs/chrombpnet-pytorch}"
export PATH="${CONDA_ENV}/bin:${PATH}"

OAK="/path/to/workdir"
GENOME="${OAK}/chromatlas/reference/hg38/hg38.fa"
IDS="${OAK}/sherlock/ids/chromatlas/chrombpnet-refactor/ids_chrombpnet_refactor_benchmark_data_aug.txt"

line=$(awk -v e="${ENCID}" '$2==e {print; exit}' "${IDS}")
[[ -n "${line}" ]] || { echo "no such ENCID in ${IDS}: ${ENCID}"; exit 2; }
ASSAY=$(echo "${line}" | cut -f1)
BAM=$(echo "${line}" | cut -f3)
[[ -f "${BAM}" ]] || { echo "missing BAM: ${BAM}"; exit 1; }

WORK="$(dirname "$(readlink -f "$0")")/arm_check/${ENCID}"
mkdir -p "${WORK}"

# One chromosome, through the index, as a tagAlign -- reads_to_bigwig takes
# tagAlign directly, so this avoids a genome-wide pass per arm.
TAG="${WORK}/${CHROM}.tagAlign.gz"
if [[ ! -s "${TAG}" ]]; then
  echo "extracting ${CHROM} from $(basename "${BAM}") ..."
  samtools view -b "${BAM}" "${CHROM}" \
    | bedtools bamtobed -i stdin \
    | gzip -1 > "${TAG}"
fi
echo "${CHROM}: $(zcat "${TAG}" | wc -l) reads"

SIZES="${WORK}/sizes.tsv"
awk -v c="${CHROM}" '$1==c' "${OAK}/chromatlas/reference/hg38/hg38.chrom.sizes" > "${SIZES}"

if [[ "${ASSAY}" == "DNASE" ]]; then
  ARMS=(standard right left both-half both-full)
else
  ARMS=(standard)
fi

for arm in "${ARMS[@]}"; do
  out="${WORK}/${arm}"
  if [[ -s "${out}_unstranded.bw" ]]; then echo "have ${arm}"; continue; fi
  echo "building ${arm} ..."
  # The arm as reads_to_bigwig's --dnase-convention (standard is the old
  # convention, right-deprecated); ATAC takes no convention.
  conv=()
  if [[ "${ASSAY}" == "DNASE" ]]; then
    conv=(--dnase-convention "$([[ "${arm}" == standard ]] && echo right-deprecated || echo "${arm}")")
  fi
  python -m chrombpnet.helpers.preprocessing.reads_to_bigwig \
    -g "${GENOME}" -itag "${TAG}" -c "${SIZES}" -op "${out}" \
    -d "${ASSAY}" -ps 0 -ms 0 ${conv[@]+"${conv[@]}"} > "${out}.log" 2>&1
done

echo
python - "${WORK}" "${CHROM}" "${ARMS[@]}" <<'PY'
import sys
import numpy as np
import pyBigWig

work, chrom, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
tracks = {}
for arm in arms:
    with pyBigWig.open(f"{work}/{arm}_unstranded.bw") as bw:
        length = bw.chroms()[chrom]
        tracks[arm] = np.nan_to_num(np.array(bw.values(chrom, 0, length)))

base = tracks["standard"]
print(f"{'arm':<11} {'total':>16} {'nonzero':>12} {'vs standard':>14}")
for arm in arms:
    v = tracks[arm]
    print(f"{arm:<11} {v.sum():>16,.1f} {int((v != 0).sum()):>12,} "
          f"{v.sum() / base.sum():>14.6f}")

ok = True
def check(label, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(f"  [{'ok' if condition else 'FAIL'}] {label}{detail}")

print()
if "right" in tracks:
    check("right is byte-identical to standard "
          "(it changes augmentation, not preprocessing)",
          np.array_equal(tracks["right"], base))
if "left" in tracks:
    # left names the same bond by its other base, so the whole track moves one left
    shifted = np.concatenate([base[1:], [0.0]])
    same = np.array_equal(tracks["left"][:-1], shifted[:-1])
    check("left is standard shifted one base left", same)
if "both-half" in tracks:
    rel = abs(tracks["both-half"].sum() - base.sum()) / base.sum()
    check("both-half conserves the total", rel < 1e-6, f"  (rel. diff {rel:.2e})")
    check("both-half = (left + right)/2 exactly",
          np.allclose(tracks["both-half"],
                      (np.concatenate([base[1:], [0.0]]) + base) / 2, atol=0))
if "both-full" in tracks:
    rel = abs(tracks["both-full"].sum() - 2 * base.sum()) / (2 * base.sum())
    check("both-full doubles the total", rel < 1e-6, f"  (rel. diff {rel:.2e})")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
PY
