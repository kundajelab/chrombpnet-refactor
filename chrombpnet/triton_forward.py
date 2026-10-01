"""Moved to :mod:`chrombpnet.engines.pytorch.triton_forward`.

This name stays importable as an alias for that module -- the same module
object, not a copy -- so existing imports, ``monkeypatch.setattr`` targets and
the Python API behave exactly as before the move.
"""

import sys

from chrombpnet.engines.pytorch import triton_forward as _module

sys.modules[__name__] = _module
