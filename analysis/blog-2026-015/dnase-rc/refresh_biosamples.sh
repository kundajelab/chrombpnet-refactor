#!/bin/bash
# Rebuild biosamples.tsv: ENCID -> short biosample label, from the ENCODE API.
#
#   ./refresh_biosamples.sh          # IDS=... (default: DNASE_RC_IDS), OUT=...
#
# summarize.sh reads the cache rather than the API so it works offline and on
# a compute node with no outbound network. Run this when the ids file gains an
# ENCID, or to pick up a corrected biosample record.
#
# The short label has to stay unambiguous: two donors of the same tissue must
# not collapse to one name, so sex and age are kept when ENCODE reports them
# ("stomach M54" vs "stomach F53"). Cell lines need no qualifier ("K562").

set -euo pipefail

if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
fi
source "${BLOG_SCRIPTS}/_site.sh"

HERE="$(dirname "$(readlink -f "$0")")"
[[ -n "${IDS:-}" ]] || { require_site DNASE_RC_IDS; IDS="${DNASE_RC_IDS}"; }
OUT="${OUT:-${HERE}/biosamples.tsv}"

[[ -n "${CONDA_ENV:-}" ]] || { require_site CONDA_ROOT; CONDA_ENV="${CONDA_ROOT}/envs/chrombpnet-pytorch"; }
export PATH="${CONDA_ENV}/bin:${PATH}"

python - "$IDS" "$OUT" <<'PY'
import json, re, subprocess, sys

ids_path, out_path = sys.argv[1], sys.argv[2]

encids = []
with open(ids_path) as fh:
    next(fh, None)
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1] not in encids:
            encids.append(parts[1])


def short(summary):
    """Compact but unambiguous label from ENCODE's biosample_summary.

    'Homo sapiens K562'                                 -> 'K562'
    'Homo sapiens stomach tissue male adult (54 years)' -> 'stomach M54'
    """
    s = re.sub(r"^Homo sapiens\s+", "", summary or "").strip()
    m = re.match(r"(.+?)\s+(male|female)\b.*?\((\d+)\s*year", s)
    if m:
        tissue = m.group(1).replace(" tissue", "")
        return f"{tissue} {m.group(2)[0].upper()}{m.group(3)}"
    return re.sub(r"\s+tissue$", "", s) or "unknown"


rows = []
for encid in encids:
    result = subprocess.run(
        ["curl", "-sL", "--max-time", "60", "-H", "Accept: application/json",
         f"https://www.encodeproject.org/experiments/{encid}/?frame=object"],
        capture_output=True, text=True)
    try:
        record = json.loads(result.stdout)
        summary = record.get("biosample_summary") or "unknown"
        assay = record.get("assay_title") or "unknown"
    except Exception:
        summary, assay = "unknown", "unknown"
        print(f"  ! lookup failed for {encid}", file=sys.stderr)
    label = short(summary)
    rows.append((encid, label, assay, summary))
    print(f"  {encid:<13} {label:<14} {assay:<11} {summary}")

with open(out_path, "w") as fh:
    fh.write("ENCID\tBIOSAMPLE\tASSAY_TITLE\tBIOSAMPLE_SUMMARY\n")
    for row in rows:
        fh.write("\t".join(row) + "\n")

unknown = [r[0] for r in rows if r[1] == "unknown"]
print(f"\nwrote {out_path} ({len(rows)} ENCIDs)")
if unknown:
    print(f"unresolved: {', '.join(unknown)} -- summarize.sh will fall back "
          f"to the accession for these", file=sys.stderr)
PY
