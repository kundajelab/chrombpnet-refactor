"""The finish-stage prefetch must be a pure optimization: no output, no raise."""

import sys

import pytest

from chrombpnet import preimport


def test_start_imports_the_named_modules():
    thread = preimport.start(("json", "base64"))
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert "base64" in sys.modules


def test_a_module_that_cannot_be_imported_is_swallowed():
    """A prefetch must never be the thing that fails a run.

    If a module is genuinely missing, the real import at the point of use
    raises with its own traceback; raising from the background thread would
    only obscure that.
    """
    thread = preimport.start(("chrombpnet_module_that_does_not_exist",))
    thread.join(timeout=30)
    assert not thread.is_alive()


def test_the_default_list_names_the_metrics_module():
    """This is what `compare_with_observed` imports lazily, and the whole
    point of the prefetch -- it drags in scipy.stats and matplotlib."""
    assert "chrombpnet.training.metrics" in preimport.FINISH_STAGE_MODULES


def test_the_thread_is_a_daemon_so_it_cannot_hold_the_process_open():
    thread = preimport.start(("json",))
    assert thread.daemon
    thread.join(timeout=30)
