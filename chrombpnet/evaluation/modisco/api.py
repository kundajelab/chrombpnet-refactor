"""Compatibility layer for the current TF-MoDISco Python API.

ChromBPNet attribution files use channel-first datasets (N, 4, L), whereas
TF-MoDISco consumes channel-last arrays (N, L, 4).  Keeping that conversion
here prevents the command line and the historical helper scripts from
drifting apart again.
"""

from __future__ import annotations

import functools
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from chrombpnet.compression import recompress_h5


PATTERN_GROUPS = ("pos_patterns", "neg_patterns")


def _modisco_modules():
    """Import TF-MoDISco lazily and return ``(modisco, tfmodisco, io, report)``.

    Importing at call time keeps this module importable where ``modiscolite``
    is not installed, which matters because the package is an optional
    ``[reports]`` extra. Tests substitute this function to exercise the adapter
    without the real dependency.
    """
    import modiscolite
    from modiscolite import io, report, tfmodisco

    return modiscolite, tfmodisco, io, report


def _keeps_torch_threads(function):
    """Restore PyTorch's thread count after ``function``.

    TF-MoDISco's first numba ``parallel=True`` call starts numba's OpenMP
    threading layer, which sets the OpenMP thread count to NUMBA_NUM_THREADS
    -- by default every core, whatever OMP_NUM_THREADS says -- in the libgomp
    already loaded into the process: PyTorch's. ``torch.get_num_threads()``
    is that libgomp setting, so PyTorch then splits every CPU op over all
    cores. On a host already busy with other work, each op waits at OpenMP's
    end-of-region barrier for its slowest thread, and CPU-side torch work in
    the same process slows by up to ~60x (measured on a 56-core host at load
    ~80). modisco-lite 2.5.2 makes that numba call when ``modiscolite.util``
    is imported (so ``supports_tomtom_lite`` alone triggers it); 2.6.0 during
    ``TFMoDISco``. The count is read before anything from TF-MoDISco is
    imported and put back afterwards. Without PyTorch in the process
    (``modisco_motifs``, the TensorFlow and JAX engines) this does nothing.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        torch = sys.modules.get("torch")
        threads = torch.get_num_threads() if torch is not None else None
        try:
            return function(*args, **kwargs)
        finally:
            if torch is not None:
                torch.set_num_threads(threads)

    return wrapper


@dataclass(frozen=True)
class Pattern:
    group: str
    name: str
    sequence: np.ndarray
    contrib_scores: np.ndarray
    hypothetical_contribs: np.ndarray
    n_seqlets: int

    @property
    def tag(self) -> str:
        return f"{self.group}.{self.name}"


def centered_slice(length: int, window: int) -> slice:
    """Return the centered even-width slice used by TF-MoDISco's CLI."""
    if window <= 0:
        raise ValueError("TF-MoDISco window must be positive")
    if window % 2:
        raise ValueError("TF-MoDISco window must be even")
    if window > length:
        raise ValueError(f"TF-MoDISco window ({window}) exceeds sequence length ({length})")
    center = length // 2
    return slice(center - window // 2, center + window // 2)


def load_contribution_scores(path, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one-hot sequence and hypothetical attributions as N x L x 4."""
    with h5py.File(path, "r") as handle:
        if "raw" in handle and "shap" in handle:
            seq_ds = handle["raw/seq"]
            attr_ds = handle["shap/seq"]
            if seq_ds.ndim != 3 or seq_ds.shape[1] != 4:
                raise ValueError("Expected raw/seq to have shape (examples, 4, length)")
            if attr_ds.shape != seq_ds.shape:
                raise ValueError("raw/seq and shap/seq must have identical shapes")
            crop = centered_slice(seq_ds.shape[2], window)
            sequences = np.asarray(seq_ds[:, :, crop], dtype=np.float32).transpose(0, 2, 1)
            attributions = np.asarray(attr_ds[:, :, crop], dtype=np.float32).transpose(0, 2, 1)
        elif "input_seqs" in handle and "hyp_scores" in handle:
            seq_ds = handle["input_seqs"]
            attr_ds = handle["hyp_scores"]
            if seq_ds.ndim != 3 or seq_ds.shape[-1] != 4:
                raise ValueError("Expected input_seqs to have shape (examples, length, 4)")
            if attr_ds.shape != seq_ds.shape:
                raise ValueError("input_seqs and hyp_scores must have identical shapes")
            crop = centered_slice(seq_ds.shape[1], window)
            sequences = np.asarray(seq_ds[:, crop, :], dtype=np.float32)
            attributions = np.asarray(attr_ds[:, crop, :], dtype=np.float32)
        else:
            raise ValueError(
                "Unsupported attribution file: expected raw/seq + shap/seq or input_seqs + hyp_scores"
            )

    return sequences, attributions


def run_motifs(input_path, output_path, max_seqlets=50000, window=500, **kwargs):
    """Run TF-MoDISco 2.6 and save its current pattern-oriented HDF5 format."""
    sequences, attributions = load_contribution_scores(input_path, window)
    return run_arrays(sequences, attributions, output_path, max_seqlets, window, **kwargs)


@_keeps_torch_threads
def run_arrays(sequences, attributions, output_path, max_seqlets=50000, window=None, **kwargs):
    """Run TF-MoDISco on channel-last arrays and save current-format results."""
    sequences = np.asarray(sequences, dtype=np.float32)
    attributions = np.asarray(attributions, dtype=np.float32)
    if sequences.ndim != 3 or sequences.shape[-1] != 4:
        raise ValueError("Sequences must have shape (examples, length, 4)")
    if sequences.shape != attributions.shape:
        raise ValueError("Sequences and hypothetical attributions must have identical shapes")
    if window is None:
        window = sequences.shape[1]
    if sequences.shape[1] != window:
        crop = centered_slice(sequences.shape[1], int(window))
        sequences, attributions = sequences[:, crop], attributions[:, crop]
    options = {
        "max_seqlets_per_metacluster": int(max_seqlets),
        "sliding_window_size": 20,
        "flank_size": 5,
        "trim_to_window_size": 30,
        "initial_flank_to_add": 10,
        "final_flank_to_add": 0,
        "target_seqlet_fdr": 0.05,
        "n_leiden_runs": 2,
        "verbose": True,
    }
    options.update(kwargs)
    _, tfmodisco, io, _ = _modisco_modules()
    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        one_hot=sequences,
        hypothetical_contribs=attributions,
        **options,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save_hdf5(str(output_path), pos_patterns, neg_patterns, int(window))
    # TF-MoDISco writes its datasets uncompressed.
    recompress_h5(output_path)
    return pos_patterns, neg_patterns


@_keeps_torch_threads
def supports_tomtom_lite():
    """Whether the installed TF-MoDISco can match motifs with TOMTOM-lite.

    That is ``report_motifs(..., ttl=True)``, in tfmodisco-lite 2.5 and
    later; older releases, such as modisco-lite 2.3.2, do not have it.
    """
    try:
        *_, report = _modisco_modules()
    except ImportError:
        return False
    return "ttl" in inspect.signature(report.report_motifs).parameters


def _has_patterns(path):
    with h5py.File(path, "r") as handle:
        return any(group in handle and len(handle[group]) for group in PATTERN_GROUPS)


@_keeps_torch_threads
def generate_report(input_path, output_dir, motif_file=None, image_suffix="./",
                    tomtom_lite=False):
    """Generate the latest supported compact HTML report (``motifs.html``).

    ``tomtom_lite`` matches the motifs with TOMTOM-lite (memelite) instead of
    MEME's ``tomtom``: no MEME needed, and the table reports uncorrected
    p-values (``pval0..``) instead of q-values. ``ttl`` is passed to
    ``report_motifs`` only then, so the default call is what it always was.
    """
    if tomtom_lite and not supports_tomtom_lite():
        raise RuntimeError(
            "tomtom_lite needs TF-MoDISco 2.5 or later (report_motifs(ttl=...)); "
            "the installed release does not have it")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    *_, report = _modisco_modules()
    options = {}
    # memelite's tomtom fails on an empty pattern list; with no patterns there
    # is nothing to match, and MEME's tomtom is not called either.
    if tomtom_lite and motif_file is not None and _has_patterns(input_path):
        options["ttl"] = True
    report.report_motifs(
        Path(input_path),
        str(output_dir),
        image_suffix,
        motif_file,
        is_writing_tomtom_matrix=False,
        **options,
    )
    return output_dir / "motifs.html"


def iter_patterns(path) -> Iterator[Pattern]:
    """Iterate patterns from the TF-MoDISco 2.6 HDF5 schema."""
    with h5py.File(path, "r") as handle:
        for group_name in PATTERN_GROUPS:
            if group_name not in handle:
                continue
            group = handle[group_name]
            for name in sorted(group, key=lambda value: int(value.rsplit("_", 1)[-1])):
                pattern = group[name]
                required = ("sequence", "contrib_scores", "hypothetical_contribs", "seqlets")
                if any(key not in pattern for key in required):
                    raise ValueError(f"Malformed TF-MoDISco pattern: {group_name}/{name}")
                yield Pattern(
                    group=group_name,
                    name=name,
                    sequence=np.asarray(pattern["sequence"]),
                    contrib_scores=np.asarray(pattern["contrib_scores"]),
                    hypothetical_contribs=np.asarray(pattern["hypothetical_contribs"]),
                    n_seqlets=int(np.asarray(pattern["seqlets/n_seqlets"]).reshape(-1)[0]),
                )


def write_seqlets_fasta(results_path, fasta_path):
    """Write all saved TF-MoDISco seqlets in FASTA format."""
    bases = np.asarray(list("ACGT"))
    with h5py.File(results_path, "r") as handle, open(fasta_path, "w") as output:
        for group_name in PATTERN_GROUPS:
            if group_name not in handle:
                continue
            for pattern_name, pattern in handle[group_name].items():
                seqlets = pattern["seqlets"]
                for index, sequence in enumerate(np.asarray(seqlets["sequence"])):
                    example = int(seqlets["example_idx"][index])
                    start = int(seqlets["start"][index])
                    end = int(seqlets["end"][index])
                    strand = "-" if bool(seqlets["is_revcomp"][index]) else "+"
                    output.write(f">{group_name}.{pattern_name}:example{example}:{start}-{end}({strand})\n")
                    output.write("".join(bases[np.argmax(sequence, axis=1)]) + "\n")


def main(argv=None):
    """Small CLI used by retained workflow assets without the upstream entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ChromBPNet TF-MoDISco 2.6 adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    motifs = commands.add_parser("motifs")
    motifs.add_argument("-i", "--h5py", required=True)
    motifs.add_argument("-o", "--output", required=True)
    motifs.add_argument("-n", "--max-seqlets", type=int, default=50000)
    motifs.add_argument("-w", "--window", type=int, default=500)
    motifs.add_argument("-l", "--n-leiden", type=int, default=2)
    motifs.add_argument("-v", "--verbose", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("-i", "--h5py", required=True)
    report.add_argument("-o", "--output", required=True)
    report.add_argument("-m", "--meme-db", default=None)
    report.add_argument("-s", "--suffix", default="./")
    args = parser.parse_args(argv)
    if args.command == "motifs":
        run_motifs(args.h5py, args.output, args.max_seqlets, args.window,
                   n_leiden_runs=args.n_leiden, verbose=args.verbose)
    else:
        generate_report(args.h5py, args.output, args.meme_db, args.suffix)


if __name__ == "__main__":
    main()
