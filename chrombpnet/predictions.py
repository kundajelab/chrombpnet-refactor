"""Shared post-processing and file formats for model predictions.

These were duplicated across ``training/predict.py``,
``evaluation/make_bigwigs/predict_to_bigwig.py`` and
``evaluation/marginal_footprints/marginal_footprinting.py``. The copies had
drifted apart in ways that were invisible until compared side by side, so they
live here once.
"""

from __future__ import annotations

import h5py
import numpy as np

from chrombpnet.compression import create_dataset


def softmax(x, temp=1):
    """Row-wise softmax over a ``(batch, positions)`` array of profile logits.

    The row maximum is subtracted before exponentiating. Centring on the *mean*
    instead, as the TensorFlow original did, lets large finite logits overflow
    to infinity and produce NaNs -- recorded as defect 7 in
    ``chrombpnet_bugs.md``. The rank check is part of that fix: a
    ``(positions,)`` array would otherwise be normalized along the wrong axis
    and fail silently.

    Does not modify ``x``. Two of the three previous copies subtracted the
    maximum in place via ``-=``, which wrote through to a caller's float array.
    """
    values = np.asarray(x)
    if values.ndim != 2:
        raise ValueError(
            f"softmax expects a 2-D batch by position array, got {values.shape}")
    scaled = temp * values
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


def write_predictions_h5py(output_prefix, profile, logcts, coords,
                           *, include_peak_flag):
    """Write ``<output_prefix>_predictions.h5``.

    ``include_peak_flag`` adds the ``coords/coords_peak`` dataset carrying the
    peak/nonpeak indicator from column 4 of ``coords``. Only the test-set
    prediction path has that column: ``predict_to_bigwig`` builds its
    coordinates from a regions BED and has nothing to put there. It is
    keyword-only with no default, because the two schemas are not
    interchangeable and a wrong default would silently change the file.

    Datasets are created in a fixed order because HDF5 records creation order,
    so reordering them would change the file on disk without changing its
    contents.
    """
    with h5py.File(f"{output_prefix}_predictions.h5", "w") as h5_file:
        coord_group = h5_file.create_group("coords")
        pred_group = h5_file.create_group("predictions")

        num_examples = len(coords)
        chroms = [str(coords[i][0]) for i in range(num_examples)]
        centers = [int(coords[i][1]) for i in range(num_examples)]

        variable_length_string = h5py.special_dtype(vlen=str)
        chrom_array = np.array(chroms, dtype=variable_length_string)
        create_dataset(coord_group, "coords_chrom", chrom_array, variable_length_string)
        create_dataset(coord_group, "coords_center", centers, int)
        if include_peak_flag:
            peaks = [int(coords[i][3]) for i in range(num_examples)]
            create_dataset(coord_group, "coords_peak", peaks, int)

        create_dataset(pred_group, "profs", profile, float)
        create_dataset(pred_group, "logcounts", logcts, float)
