"""Conventions for the genomic region files the package reads and writes."""

from __future__ import annotations

#: Column names for the 10-column narrowPeak files used throughout the CLI.
#: Only "chr", "start", "end" and "summit" are meaningful; the numbered columns
#: are placeholders kept so the files stay narrowPeak-shaped. "summit" is an
#: offset relative to "start", so a region's centre is ``start + summit``.
#:
#: This was previously duplicated in seven places -- four module-level copies
#: and three inline literals inside ``pd.read_csv`` calls -- which meant a
#: column rename had to be applied by hand in all of them.
NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]


#: Pipelines disable a peak or nonpeak set by passing this string rather than
#: None, because the value travels through an argparse namespace that is dumped
#: to JSON.
DISABLED = "None"


def is_disabled(path):
    """True when a peaks/nonpeaks argument means "do not use this set".

    Compared case-insensitively. The two call sites used to disagree --
    initializers lowercased while predict did not -- so a lowercase "none"
    disabled the set for the generator but not for the metrics that read it.
    """
    return path is None or str(path).lower() == DISABLED.lower()
