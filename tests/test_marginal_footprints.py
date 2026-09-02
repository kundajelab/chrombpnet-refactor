import h5py
import numpy as np

from chrombpnet.evaluation.marginal_footprints.marginal_footprinting import (
    save_footprints_h5,
)


def test_save_footprints_preserves_legacy_deepdish_schema(tmp_path):
    path = tmp_path / "footprints.h5"
    expected_profile = np.array([0.1, 0.2, 0.7])
    expected_counts = np.array([12.5])

    save_footprints_h5(
        path,
        {"control": [expected_profile, expected_counts]},
    )

    with h5py.File(path, "r") as h5:
        assert h5.attrs["DEEPDISH_IO_VERSION"] == 12
        assert h5["control"].attrs["TITLE"] == b"list:2"
        np.testing.assert_array_equal(h5["control/i0"][:], expected_profile)
        np.testing.assert_array_equal(h5["control/i1"][:], expected_counts)
        assert h5["control/i0"].attrs["CLASS"] == b"ARRAY"
