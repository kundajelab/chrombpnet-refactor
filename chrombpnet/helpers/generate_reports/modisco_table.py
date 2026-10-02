"""The TF-MoDISco motifs table (``motifs.html``) as the ChromBPNet reports use it.

The table is pandas ``to_html`` output, one ``<tr>`` per pattern, written by
modiscolite's ``report_motifs``. Its TOMTOM statistic columns are ``qval0..2``
with MEME's ``tomtom`` and ``pval0..2`` with TOMTOM-lite (``--tomtom-lite``),
and the report text has to say which.

Adapted from kundajelab/chrombpnet#284's
``chrombpnet/helpers/generate_reports/modisco_table.py``: ``match_stat``,
``stat_name`` and ``stat_description`` are its, and for a q-value table they
reproduce the wording the reports have always used, byte for byte.
``drop_negative_patterns`` takes its name and whole-row approach, but works
line by line so that pandas' layout is kept.
"""

import re

_HEADER_CELL = re.compile(r"<th(?:\s[^>]*)?>(.*?)</th>", re.S)
_CELL = re.compile(r"\s*<t[dh](?:\s[^>]*)?>(.*?)</t[dh]>")


def _is_row_start(line):
	stripped = line.strip()
	return stripped == "<tr>" or stripped.startswith("<tr ")


def _first_cell(row_lines):
	for line in row_lines[1:]:
		match = _CELL.match(line)
		if match:
			return match.group(1).strip()
	return ""


def drop_negative_patterns(html):
	"""Remove the rows of negative patterns (first cell starting ``neg_``).

	Works on pandas' layout one row at a time -- every line from ``<tr>``
	through ``</tr>`` -- so it does not depend on how many columns a row has
	(13 with a motif database and three matches, 4 without one), and it leaves
	the table's structure, ``</tbody>`` included, as pandas wrote it.
	"""
	out, row = [], None
	for line in html.split("\n"):
		if row is None:
			if _is_row_start(line):
				row = [line]
			else:
				out.append(line)
			continue
		row.append(line)
		if line.strip() == "</tr>":
			if not _first_cell(row).startswith("neg_"):
				out.extend(row)
			row = None
	if row is not None:
		out.extend(row)
	return "\n".join(out)


def table_columns(html):
	"""Column names of the table header, in order."""
	return [re.sub(r"<[^>]+>", "", cell).strip() for cell in _HEADER_CELL.findall(html)]


def match_stat(html):
	"""'qval' (MEME tomtom), 'pval' (TOMTOM-lite) or None (no motif database
	was given to the report, so no match columns)."""
	columns = table_columns(html)
	for kind in ("qval", "pval"):
		if any(re.fullmatch(kind + r"\d+", column) for column in columns):
			return kind
	return None


def stat_name(kind):
	return "p-values" if kind == "pval" else "qvals"


def stat_description(kind, n=3):
	"""Wording for the TOMTOM statistic columns of a report table."""
	if kind == "pval":
		return "p-values ({}; TOMTOM-lite, not corrected for multiple testing)".format(
			",".join("pval{}".format(i) for i in range(n)))
	return "qvals ({})".format(",".join("qval{}".format(i) for i in range(n)))
