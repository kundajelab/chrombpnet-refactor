"""Guard the package surface that only ``legacy/`` still consumes.

Nothing in ``chrombpnet/`` imports ``legacy/``, but the dependency runs the
other way: eight archived modules import *into* the live package. That keeps a
handful of functions alive purely for the archive, with nothing enforcing it --
`api._modisco_modules` had in fact already been deleted and its two legacy call
sites left broken before these tests existed.

So this module imports every archived module by file path (they are not
installed, and some of their directories have no ``__init__.py``) and asserts
the specific names they depend on still exist. It is deliberately cheap: it
exists to make a deletion elsewhere fail loudly rather than silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LEGACY_ROOT = Path(__file__).resolve().parents[1] / "legacy"

#: Every archived module, with the third-party packages it needs at import
#: time. Those are optional extras, so a missing one is a skip, not a failure.
LEGACY_MODULES = {
    "legacy/helpers/reformat_chrombpnet_h5.py": (),
    "legacy/modisco/run_modisco.py": ("modiscolite",),
    "legacy/modisco/visualize_motif_matches.py": ("modiscolite",),
    "legacy/modisco/fetch_tomtom.py": ("modiscolite",),
    "legacy/invivo_footprints/run_tfmodisco.py": ("modiscolite",),
    "legacy/invivo_footprints/tf_modiscohits.py": ("modiscolite",),
    "legacy/figure_notebooks/subsampling/viz_sequence.py": (),
}


def _import_by_path(relative: str):
    path = LEGACY_ROOT.parent / relative
    name = relative.replace("/", "_")
    if name.endswith(".py"):  # str.removesuffix is Python 3.9+
        name = name[:-len(".py")]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, relative
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", sorted(LEGACY_MODULES))
def test_archived_module_still_imports(relative):
    for requirement in LEGACY_MODULES[relative]:
        pytest.importorskip(requirement)
    _import_by_path(relative)


def test_names_the_archive_imports_from_the_live_package_still_exist():
    """These are load-bearing for legacy/ only, so nothing else pins them."""
    from chrombpnet.evaluation.modisco import api

    for name in ("_modisco_modules", "iter_patterns", "run_arrays",
                 "run_motifs", "write_seqlets_fasta"):
        assert hasattr(api, name), (
            f"legacy/ imports chrombpnet.evaluation.modisco.api.{name}; "
            "removing it breaks the archive")

    from chrombpnet.evaluation.custom_sequences import viz_sequence

    for name in ("ic_scale", "plot_weights_given_ax", "plot_weights"):
        assert hasattr(viz_sequence, name), (
            f"legacy/ and helpers/preprocessing import viz_sequence.{name}")

    from chrombpnet.model import ChromBPNet, load_model, save_model  # noqa: F401
