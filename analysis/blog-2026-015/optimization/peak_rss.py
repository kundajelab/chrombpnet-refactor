#!/usr/bin/env python3
"""Runs COMMAND and writes the peak resident memory of its largest process, in
MB, to OUT: getrusage(RUSAGE_CHILDREN).ru_maxrss, which covers the command and
every descendant it waited for (multiprocessing workers included), on the GPU
node itself. Exits with the command's status. Standard library only, so it runs
in any environment (the kit's Python 3.8 too).

    python -I peak_rss.py OUT COMMAND [ARG ...]
"""

import resource
import subprocess
import sys


def main():
    out, command = sys.argv[1], sys.argv[2:]
    rc = subprocess.call(command)
    peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss   # KB on Linux
    with open(out, "w") as handle:
        handle.write("%.1f\n" % (peak_kb / 1024.0))
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
