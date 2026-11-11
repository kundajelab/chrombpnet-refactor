#!/usr/bin/env python3
"""chrombpnet-refactor's command line with its HDF5 outputs at the original
ChromBPNet's compression: gzip level 4 and no byte-shuffle (the package
writes gzip 9 with shuffle, chrombpnet/compression.py). For a checkout that
predates --h5-gzip-level and --no-h5-shuffle, which do this from the command
line: run_sections34.sh's `refactor4_prev` arm (fa45de5), so `pred_bw` is
timed at the original's compression. Everything else -- the threaded writes
included -- is the package as it is. (The blog's Section 4 runs before those
options existed ran every `refactor4*` arm through this.)

    python refactor_gzip4.py pred_bw <arguments>     # chrombpnet-refactor on PYTHONPATH
"""

import sys

import chrombpnet.compression as compression

compression.GZIP_LEVEL = 4
_package_filters = compression.h5_filters


def h5_filters(data, dtype=None):
    filters = _package_filters(data, dtype)
    filters.pop("shuffle", None)
    return filters


# create_dataset looks both up in the module when it writes a dataset.
compression.h5_filters = h5_filters

from chrombpnet.CHROMBPNET import main  # noqa: E402  (after the patch)

sys.argv = ["chrombpnet"] + sys.argv[1:]
sys.exit(main())
