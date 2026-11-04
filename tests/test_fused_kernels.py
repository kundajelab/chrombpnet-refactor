"""`--predict-fused-kernel-exact` / `-fast`: where the flags live, and what
they switch.

The other optimizations are unconditional now, so what remains testable about
them is that the legacy bodies kept beside them still agree -- those functions
are unreachable in normal operation, and a test is the only thing standing
between "kept for reference" and "kept, and quietly wrong".
"""

import numpy as np
import pytest

from chrombpnet import fused_kernels
from chrombpnet.parsers import read_parser


PREDICTING_COMMANDS = {
    "pred_bw": ["-bm", "b.pt", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes",
                "-op", "out"],
    "footprints": ["-m", "m.pt", "-r", "r.bed", "-g", "g.fa", "-fl", "f.json",
                   "-op", "out", "-pwm_f", "p.tsv"],
}


def test_the_default_is_off():
    """It moves the output, so it cannot be a default."""
    args = read_parser(["pred_bw", *PREDICTING_COMMANDS["pred_bw"]])
    assert args.predict_fused_kernel is False
    assert fused_kernels.configure_from_args(args) is None
    assert not fused_kernels.enabled()


@pytest.mark.parametrize("mode,precision", [("exact", "ieee"), ("fast", "tf32")])
@pytest.mark.parametrize("command", sorted(PREDICTING_COMMANDS))
def test_each_flag_reaches_the_commands_that_predict(command, mode, precision):
    args = read_parser([command, *PREDICTING_COMMANDS[command], f"--predict-fused-kernel-{mode}"])
    assert args.predict_fused_kernel == mode
    assert fused_kernels.configure_from_args(args) == mode
    assert fused_kernels.enabled() and fused_kernels.precision() == precision
    assert fused_kernels.flag() == f"--predict-fused-kernel-{mode}"


@pytest.mark.parametrize("command", sorted(PREDICTING_COMMANDS))
def test_the_two_modes_exclude_each_other(command, capsys):
    with pytest.raises(SystemExit):
        read_parser([command, *PREDICTING_COMMANDS[command],
                     "--predict-fused-kernel-exact", "--predict-fused-kernel-fast"])
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("legacy", ["--predict-fused-kernel", "--predict-fk"])
@pytest.mark.parametrize("command", sorted(PREDICTING_COMMANDS))
def test_the_old_names_still_mean_exact_but_are_not_advertised(command, legacy, capsys):
    """``--predict-fk``, then ``--predict-fused-kernel``, ran the fp32 kernels
    before ``fast`` existed; scripts written for them must keep running, and
    ``--help`` shows only the two current names."""
    args = read_parser([command, *PREDICTING_COMMANDS[command], legacy])
    assert args.predict_fused_kernel == "exact"
    with pytest.raises(SystemExit):
        read_parser([command, "--help"])
    shown = capsys.readouterr().out
    assert "--predict-fused-kernel-exact" in shown and "--predict-fused-kernel-fast" in shown
    assert "--predict-fk" not in shown
    assert not any(word in ("--predict-fused-kernel", "[--predict-fused-kernel]")
                   for word in shown.replace("|", " ").split())


def test_true_means_exact():
    """`configure(True)` was the only way to switch the route on before."""
    assert fused_kernels.configure(True) == "exact"
    assert fused_kernels.precision() == "ieee"


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="fused-kernel mode"):
        fused_kernels.configure("fastest")


def test_contribs_bw_has_no_flag():
    """DeepLIFT needs autograd and hooks on the ReLU modules, so the fused
    route cannot serve attribution. Offering the flag there would promise
    something that could only refuse."""
    args = read_parser(["contribs_bw", "-m", "m.pt", "-r", "r.bed", "-g",
                        "g.fa", "-c", "c.sizes", "-op", "out"])
    assert not hasattr(args, "predict_fused_kernel")
    assert fused_kernels.configure_from_args(args) is None


def test_commands_without_the_flag_leave_it_off():
    """A namespace that never had the attribute must not switch it on."""
    from types import SimpleNamespace
    fused_kernels.configure("fast")
    assert fused_kernels.configure_from_args(SimpleNamespace()) is None


def test_qc_carries_it_so_a_model_can_be_scored_on_the_fused_route():
    args = read_parser(["qc", "-bw", "o.bw", "-cm", "c.pt", "-cmb", "n.pt",
                        "-g", "g.fa", "-c", "c.sizes", "-o", "out",
                        "-d", "DNASE", "-p", "p.bed", "-n", "n.bed",
                        "-fl", "f.json", "--predict-fused-kernel-fast"])
    assert args.predict_fused_kernel == "fast"


def test_the_optimize_flag_is_gone():
    """It existed to attribute a measurement that is finished and recorded."""
    import chrombpnet.parsers as parsers
    assert not hasattr(parsers, "OPTIMIZE_DEFAULT")
    with pytest.raises(SystemExit):
        read_parser(["pred_bw", *PREDICTING_COMMANDS["pred_bw"],
                     "--optimize", "1,3"])


# -- The legacy bodies, kept unreachable but not allowed to rot -------------

def test_the_legacy_one_hot_encoder_agrees():
    from chrombpnet.training.utils.one_hot import (
        _dna_to_one_hot_legacy, dna_to_one_hot)

    seqs = ["ACGTNacgtn" * 4, "NNNNNNNNNN" * 4, "ACGTACGTAC" * 4,
            "ACGTRYKMSW" * 4]
    assert np.array_equal(dna_to_one_hot(seqs),
                          _dna_to_one_hot_legacy(seqs, len(seqs[0])))


def test_the_legacy_profile_metrics_agrees():
    from chrombpnet.training.metrics import (
        _profile_metrics_looped, profile_metrics)

    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(24, 40)).astype(float)
    probs = rng.dirichlet(np.ones(40), size=24)
    counts[3] = 0.0                                  # a skipped region
    produced = profile_metrics(counts, probs)
    expected = _profile_metrics_looped(counts, probs, 1234)
    # Both return the same eight slots; only four carry values.
    for index, name in ((2, "jsd"), (3, "jsd_norm"), (4, "jsd_rnd"),
                        (5, "jsd_rnd_norm")):
        assert np.array_equal(produced[index], expected[index],
                              equal_nan=True), name


def test_the_legacy_predict_agrees():
    pytest.importorskip("torch")  # PyTorch engine only
    from chrombpnet.model import BPNet, ModelConfig, PredictionAdapter

    # input_length - 20 (first conv) - 12 (two dilated) - 74 (profile conv)
    # must be >= output_length, with an even margin for the centre crop.
    config = ModelConfig(filters=4, n_dilation_layers=2, input_length=306,
                         output_length=100)
    adapter = PredictionAdapter(BPNet(config).eval(), device="cpu")
    values = np.zeros((7, 306, 4), dtype=np.int8)
    values[np.arange(7)[:, None], np.arange(306)[None, :],
           np.random.default_rng(1).integers(0, 4, size=(7, 306))] = 1

    profile, counts = adapter.predict(values, batch_size=3)
    legacy_profile, legacy_counts = adapter._predict_legacy(values, 3)
    np.testing.assert_allclose(profile, legacy_profile, rtol=0, atol=1e-6)
    np.testing.assert_allclose(counts, legacy_counts, rtol=0, atol=1e-6)
