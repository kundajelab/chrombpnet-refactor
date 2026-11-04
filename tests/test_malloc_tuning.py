"""pred_bw's glibc allocator settings (chrombpnet.malloc_tuning, the optimization kit's malloc_env)."""

import json
import os
import subprocess
import sys

import pytest

from chrombpnet import malloc_tuning

NAMES = tuple(malloc_tuning.SETTINGS)


def _in_subprocess(env_extra=None):
    """apply() in a fresh interpreter (the settings are process-wide): what it applied and what it exported."""
    env = {k: v for k, v in os.environ.items() if k not in NAMES}
    env.update(env_extra or {})
    code = ("import json, os; from chrombpnet import malloc_tuning as m; a = m.apply(); "
            "print(json.dumps({'applied': a, 'env': {k: os.environ.get(k) for k in m.SETTINGS}}))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_kits_three_values():
    assert malloc_tuning.SETTINGS == {
        "MALLOC_MMAP_MAX_": (-4, 0),
        "MALLOC_TRIM_THRESHOLD_": (-1, -1),
        "MALLOC_TOP_PAD_": (-2, 536870912),
    }


@pytest.mark.skipif(malloc_tuning._mallopt() is None, reason="no glibc mallopt here")
def test_apply_sets_and_exports_every_setting():
    result = _in_subprocess()
    assert result["applied"] == {"MALLOC_MMAP_MAX_": 0, "MALLOC_TRIM_THRESHOLD_": -1, "MALLOC_TOP_PAD_": 536870912}
    assert result["env"] == {"MALLOC_MMAP_MAX_": "0", "MALLOC_TRIM_THRESHOLD_": "-1", "MALLOC_TOP_PAD_": "536870912"}


@pytest.mark.skipif(malloc_tuning._mallopt() is None, reason="no glibc mallopt here")
def test_a_setting_already_in_the_environment_is_left_alone():
    result = _in_subprocess({"MALLOC_TOP_PAD_": "1048576"})
    assert "MALLOC_TOP_PAD_" not in result["applied"]
    assert result["env"]["MALLOC_TOP_PAD_"] == "1048576"
    assert set(result["applied"]) == {"MALLOC_MMAP_MAX_", "MALLOC_TRIM_THRESHOLD_"}


def test_without_mallopt_nothing_is_applied(monkeypatch):
    monkeypatch.setattr(malloc_tuning, "_mallopt", lambda: None)
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    assert malloc_tuning.apply() == {}
    assert not any(name in os.environ for name in NAMES)


@pytest.mark.parametrize("flag, expected", [([], 1), (["--no-malloc-env"], 0)])
def test_pred_bw_applies_them_unless_told_not_to(monkeypatch, flag, expected):
    import chrombpnet.CHROMBPNET as cli
    import chrombpnet.evaluation.make_bigwigs.predict_to_bigwig as predict_to_bigwig
    from chrombpnet import engines, fused_kernels

    calls = []
    monkeypatch.setattr(malloc_tuning, "apply", lambda: calls.append(1) or {})
    monkeypatch.setattr(predict_to_bigwig, "main", lambda args: None)
    monkeypatch.setattr(engines, "configure_from_args", lambda args, require=True: None)
    monkeypatch.setattr(fused_kernels, "configure_from_args", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["chrombpnet", "pred_bw", "-cm", "model.pt", "-r", "regions.bed",
                                      "-g", "genome.fa", "-c", "chrom.sizes", "-op", "out/pred"] + flag)
    cli.main()
    assert len(calls) == expected
