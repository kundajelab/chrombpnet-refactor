"""--contribs-profile-stopgrad: the profile contribution target's weights.

By default every engine weights the profile logits as chrombpnet 1.x and bpnet
do (the kundajelab shap fork's midpoint rule, its StopGradient handler being
commented out); the flag honours the stop-gradient. The engines' agreement in
both modes is checked by tests/parity/deepshap_engines.py.
"""

import numpy as np
import pytest

from chrombpnet import parsers

TRAINING = ["-g", "g.fa", "-c", "c.sizes", "-itag", "r.tagAlign", "-o", "out",
            "-d", "ATAC", "-p", "p.bed", "-n", "n.bed", "-fl", "f.json"]
QC = ["-bw", "o.bw", "-g", "g.fa", "-c", "c.sizes", "-o", "out", "-d", "ATAC",
      "-p", "p.bed", "-n", "n.bed", "-fl", "f.json"]
CONTRIBS = ["-m", "m.pt", "-r", "r.bed", "-g", "g.fa", "-c", "c.sizes", "-op", "out"]


@pytest.mark.parametrize("argv", [
    ["pipeline", *TRAINING, "-b", "bias.pt"],
    ["bias", "pipeline", *TRAINING, "-b", "0.5"],
    ["qc", *QC, "-cm", "c.pt", "-cmb", "nb.pt"],
    ["bias", "qc", *QC, "-bm", "bias.pt"],
    ["contribs_bw", *CONTRIBS],
])
def test_every_contribution_scoring_command_takes_the_flag(argv):
    args = parsers.read_parser(argv + ["--contribs-profile-stopgrad"])
    assert args.contribs_profile_stopgrad is True
    # Absent unless given, so a run without it records what it did before.
    args = parsers.read_parser(argv)
    parsers.drop_absent(args)
    assert not hasattr(args, "contribs_profile_stopgrad")


@pytest.mark.parametrize("argv", [["train", *TRAINING, "-b", "bias.pt"],
                                  ["bias", "train", *TRAINING, "-b", "0.5"]])
def test_commands_that_score_no_contributions_refuse_it(argv):
    with pytest.raises(SystemExit):
        parsers.read_parser(argv + ["--contribs-profile-stopgrad"])


def test_interpret_hands_the_flag_to_the_engine(monkeypatch, tmp_path):
    from chrombpnet.evaluation.interpret import interpret as interpret_module

    seen = []

    class Engine:
        def attribute(self, model, seqs, heads, **kwargs):
            seen.append(kwargs["profile_stop_gradient"])
            return iter(())

    seqs = np.eye(4, dtype=np.float32)[np.zeros((1, 8), dtype=int)]
    for flag in (False, True):
        interpret_module.interpret(None, seqs, str(tmp_path / "x"), ["profile"],
                                   device="cpu", engine=Engine(),
                                   profile_stop_gradient=flag)
    assert seen == [False, True]


class _Logits(object):
    """A stand-in model whose profile logits are a given tensor."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, sequence):
        return self.logits, None


def _gradients(logits, stop_gradient):
    torch = pytest.importorskip("torch")
    from chrombpnet.engines.pytorch.attribution import ProfileWrapper

    logits = torch.as_tensor(logits, dtype=torch.float64).requires_grad_()
    wrapper = ProfileWrapper(_Logits(logits), stop_gradient=stop_gradient)
    out = wrapper(None)
    (grad,) = torch.autograd.grad(out.sum(), logits)
    return out.detach().numpy(), grad.numpy()


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_pytorch_profile_weights():
    """d target / d logits is w - mean(w): w = softmax(mn) of each row with the
    stop-gradient; otherwise, for each (input, reference) pair of
    Tangermeme's [inputs; references] batch, the shap fork's midpoint
    0.5 * (softmax(mn_x) + softmax(mn_r)), zero where the logits are equal."""
    rng = np.random.RandomState(0)
    x, r = rng.normal(size=(3, 12)), rng.normal(size=(3, 12))
    # One logit equal after mean-normalizing: r5 - mean(r) = x5 - mean(x).
    r[0, 5] = (12 * (x[0, 5] - x[0].mean()) + r[0].sum() - r[0, 5]) / 11
    logits = np.concatenate([x, r])
    mn = logits - logits.mean(axis=1, keepdims=True)

    _, grad = _gradients(logits, stop_gradient=True)
    w = _softmax(mn)
    np.testing.assert_allclose(grad, w - w.mean(axis=1, keepdims=True), atol=1e-12)

    out, grad = _gradients(logits, stop_gradient=False)
    s_x, s_r = np.split(_softmax(mn), 2)
    mid = np.where(np.abs(mn[:3] - mn[3:]) < 1e-7, 0.0, 0.5 * (s_x + s_r))
    assert mid[0, 5] == 0.0
    w = np.concatenate([mid, mid])
    np.testing.assert_allclose(grad, w - w.mean(axis=1, keepdims=True), atol=1e-12)
    # Both halves are weighted alike, so the input-reference difference of the
    # target is the weighted difference of the logits, which DeepLIFT's
    # attributions sum to.
    np.testing.assert_allclose(out[:3, 0] - out[3:, 0],
                               (mid * (mn[:3] - mn[3:])).sum(axis=1), atol=1e-12)
