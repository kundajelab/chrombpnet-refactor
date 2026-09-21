"""The --optimize phase switch: parsing, grouping, and what each phase owns."""

import pytest

from chrombpnet import optimize
from chrombpnet.parsers import OPTIMIZE_DEFAULT, read_parser


PHASE_1 = ("prefetch", "preimport", "vectorized_metrics", "overlapped_finish",
           "exit_fast")
PHASE_2 = ("triton_forward",)
PHASE_3 = ("shared_inputs", "numpy_writer_args", "lazy_stats", "batched_cast",
           "fixed_step_bigwig")
EVERY_LEVER = PHASE_1 + PHASE_2 + PHASE_3


@pytest.mark.parametrize("value,expected", [
    (None, ()),
    ("", ()),
    ("none", ()),
    ("off", ()),
    ("1", ("1",)),
    ("2", ("2",)),
    ("3", ("3",)),
    ("1,3", ("1", "3")),
    (" 1 , 2 ", ("1", "2")),
    ("3,1", ("1", "3")),
    ("all", ("1", "2", "3")),
    ("1,2,3", ("1", "2", "3")),
])
def test_parsing(value, expected):
    assert optimize.parse(value).active == expected


@pytest.mark.parametrize("value", ["4", "0", "1,4", "phase1", "none,1"])
def test_bad_values_are_rejected(value):
    with pytest.raises(ValueError):
        optimize.parse(value)


@pytest.mark.parametrize("phase,levers", [("1", PHASE_1), ("2", PHASE_2),
                                          ("3", PHASE_3)])
def test_each_phase_enables_exactly_its_own_levers(phase, levers):
    """A lever must belong to one phase, so a run can attribute a difference
    to the phase that caused it."""
    opts = optimize.parse(phase)
    for lever in EVERY_LEVER:
        assert getattr(opts, lever) is (lever in levers), lever


def test_none_enables_nothing_and_all_enables_everything():
    nothing = optimize.parse("none")
    everything = optimize.parse("all")
    for lever in EVERY_LEVER:
        assert getattr(nothing, lever) is False, lever
        assert getattr(everything, lever) is True, lever


def test_phases_compose():
    opts = optimize.parse("1,3")
    assert opts.prefetch and opts.fixed_step_bigwig
    assert not opts.triton_forward


def test_configure_sets_the_process_wide_value(phases):
    phases("2")
    assert optimize.active().triton_forward
    phases("none")
    assert not optimize.active().triton_forward


def test_configure_from_args_leaves_commands_without_the_flag_alone(phases):
    """Adding the flag to a parser should be the only step needed; a
    subcommand that has not got it must not be silently switched off."""
    from types import SimpleNamespace
    phases("1,3")
    before = optimize.active().active
    optimize.configure_from_args(SimpleNamespace())
    assert optimize.active().active == ()
    assert before == ("1", "3")


def test_the_cli_default_is_what_the_parser_says():
    args = read_parser(["pred_bw", "-bm", "b.pt", "-r", "r.bed", "-g", "g.fa",
                        "-c", "c.sizes", "-op", "out"])
    assert args.optimize == OPTIMIZE_DEFAULT
    assert optimize.parse(args.optimize).active == ("1", "3")


@pytest.mark.parametrize("command", ["pred_bw", "contribs_bw", "footprints"])
def test_the_flag_reaches_the_commands_that_have_optimized_paths(command):
    base = {
        "pred_bw": ["-bm", "b.pt", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes",
                    "-op", "out"],
        "contribs_bw": ["-m", "m.pt", "-r", "r.bed", "-g", "g.fa", "-c",
                        "c.sizes", "-op", "out"],
        "footprints": ["-m", "m.pt", "-r", "r.bed", "-g", "g.fa", "-fl",
                       "f.json", "-op", "out", "-pwm_f", "p.tsv"],
    }[command]
    args = read_parser([command, *base, "--optimize", "1,2"])
    assert optimize.parse(args.optimize).active == ("1", "2")
