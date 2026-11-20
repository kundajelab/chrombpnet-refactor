"""Post-training correction of the enzyme bias (``pipeline`` and ``train``
with ``--post-training``).

ChromBPNet removes the enzyme's sequence bias by co-training: the
accessibility model is fitted on top of a frozen bias model. Post-training
correction separates the two. A plain BPNet is trained on the observed
signal with no bias model, and the frozen bias model is applied to its
predicted profile afterwards. With ``q`` and ``p`` the model's and the bias
model's predicted profiles (the softmax of their logits over the output
window), the corrected profile is proportional to

    plain     q / p                    (the profile logits subtracted)
    shrink    (q + c r) / (p + c)      (the default)

where ``c = eps / L`` for an ``L`` bp window and ``r = (box(q) + c) /
(box(p) + c)`` is the ratio of the two profiles' ``window`` bp running
means (reflected at the ends). Since

    (q + c r) / (p + c) = w q/p + (1 - w) r,      w = p / (p + c),

shrink is the per-base ratio wherever the bias model predicts appreciable
probability, and the neighbourhood's ratio where it predicts almost none.
At such a position ``q / p`` divides by a number the bias model does not
determine, and plain division can amplify prediction error into a spike.

Only the profile is corrected: the log counts are the uncorrected model's.

This module imports nothing from the package, so ``model.py`` can import it.
"""

from __future__ import annotations

import math
import numbers

import torch
from torch import Tensor, nn
from torch.nn import functional as F

#: The pseudo-probability ``c`` is ``SHRINK_EPS / L``: 3% of a uniform
#: profile's value.
SHRINK_EPS = 0.03
#: Width, in bp, of the running means in the neighbourhood ratio ``r``.
SHRINK_WINDOW = 9
RULES = ("shrink", "plain")
#: Gauss-Legendre nodes in shrink's DeepLIFT rule. On synthetic profiles
#: spanning 37 nats, 16 leave 8e-3 of the target's change unassigned and 32
#: leave 1e-5 (1e-6 at 18 nats). A tested default, not a bound.
DEEPLIFT_NODES = 32


def check_settings(rule: str, eps: float, window: int, length: int) -> None:
    """Refuse a rule, ``eps`` or ``window`` a ``length`` bp profile cannot take."""
    if rule not in RULES:
        raise ValueError(f"Unknown post-training rule {rule!r}; expected one of {', '.join(RULES)}")
    if rule == "plain":
        return
    if not (isinstance(eps, numbers.Real) and math.isfinite(eps) and eps > 0):
        raise ValueError(f"shrink needs a finite eps > 0, got {eps!r}")
    if not (isinstance(window, numbers.Integral) and window >= 1 and window % 2 == 1
            and window // 2 < length):
        raise ValueError(
            f"shrink needs an odd integer window whose half-width is below the "
            f"profile length {length}, got {window!r}")


def _running_mean(values: Tensor, window: int) -> Tensor:
    half = window // 2
    # Reflection (position -1 is position 1), from slices: reflection
    # padding's CUDA backward has no deterministic kernel, which
    # --reproducibility-max requires of contribution scoring.
    padded = torch.cat(
        (values[:, 1:half + 1].flip(-1), values, values[:, -half - 1:-1].flip(-1)), dim=-1)
    return F.avg_pool1d(padded.unsqueeze(1), window, stride=1).squeeze(1)


def correct_logits(logits: Tensor, bias_logits: Tensor, rule: str = "shrink",
                   eps: float = SHRINK_EPS, window: int = SHRINK_WINDOW) -> Tensor:
    """Logits of the bias-corrected profile.

    ``logits`` and ``bias_logits`` are ``(batch, length)`` profile logits of
    the uncorrected model and of the bias model over the same window. The
    softmax of the result is the corrected profile.
    """
    if logits.ndim != 2 or logits.shape != bias_logits.shape:
        raise ValueError(
            "Model and bias profile logits must both be (batch, length), "
            f"got {tuple(logits.shape)} and {tuple(bias_logits.shape)}")
    length = logits.shape[-1]
    check_settings(rule, eps, window, length)
    if rule == "plain":
        return logits - bias_logits
    if logits.device.type == "mps":
        raise RuntimeError(
            "The shrink rule is computed in float64, which MPS does not have; "
            "use --device cpu or cuda, or the plain rule")
    # float64: c is 3e-5 for a 1,000 bp profile, and a hole's p is far below it.
    q = torch.softmax(logits.double(), dim=-1)
    p = torch.softmax(bias_logits.double(), dim=-1)
    c = eps / length
    ratio = (_running_mean(q, window) + c) / (_running_mean(p, window) + c)
    return torch.log((q + c * ratio) / (p + c)).to(logits.dtype)


class Correction(nn.Module):
    """The rule as a module, which DeepLIFT can be given a rule for
    (:meth:`PostTrainingChromBPNet.deep_lift_ops`). Its one input holds the
    model's and the bias model's profile logits: ``(batch, 2, length)``."""

    def __init__(self, rule: str, eps: float, window: int) -> None:
        super().__init__()
        self.rule, self.eps, self.window = rule, eps, window

    def forward(self, logits: Tensor) -> Tensor:
        return correct_logits(logits[:, 0], logits[:, 1], self.rule, self.eps, self.window)


class PostTrainingChromBPNet(nn.Module):
    """A plain BPNet with its profile bias-corrected after training.

    ``uncorrected_model`` was trained on the observed signal with no bias
    model; ``bias_model`` is the bias model it is corrected with. The
    output is ``(corrected profile logits, uncorrected logcounts)``, so a
    checkpoint of this model is read wherever a ``chrombpnet_nobias`` one is.

    For inference and attribution only: both models are frozen. Gradients
    still reach the input through both, as attribution needs.
    """

    def __init__(self, uncorrected_model: nn.Module, bias_model: nn.Module,
                 rule: str = "shrink", eps: float = SHRINK_EPS,
                 window: int = SHRINK_WINDOW) -> None:
        super().__init__()
        config, bias_config = uncorrected_model.config, bias_model.config
        if not config.counts_head:
            raise ValueError("The uncorrected model must have a counts head")
        if ((bias_config.input_length, bias_config.output_length)
                != (config.input_length, config.output_length)):
            raise ValueError("Bias and uncorrected model input/output lengths differ")
        check_settings(rule, eps, window, config.output_length)
        self.config = config
        self.uncorrected_model = uncorrected_model
        self.bias_model = bias_model
        self.rule, self.eps, self.window = rule, float(eps), int(window)
        self.correction = Correction(rule, self.eps, self.window)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @property
    def input_shape(self):
        return self.uncorrected_model.input_shape

    @property
    def output_shape(self):
        return self.uncorrected_model.output_shape

    def forward(self, sequence: Tensor):
        profile, logcounts = self.uncorrected_model(sequence)
        bias_profile, _ = self.bias_model(sequence)
        return self.correction(torch.stack((profile, bias_profile), dim=1)), logcounts

    def deep_lift_ops(self):
        """The DeepLIFT rule the correction needs, as tangermeme's
        ``additional_nonlinear_ops``. None for plain, which is linear. Shrink
        is not, and gets its Jacobian integrated from the reference's logits
        to the input's (tangermeme's ``integrated_gradients_op``), so that
        profile contribution scores sum to the change in their target up to
        the quadrature's error. Call this on the model that is attributed:
        a wrapper around it, such as DataParallel, would hide the rule.
        """
        if self.rule == "plain":
            return None
        if self.correction._backward_hooks:
            # Tangermeme leaves a module that already has one unhooked, and
            # the correction would then be attributed as if it were linear.
            raise RuntimeError("The correction already has a backward hook, so "
                               "DeepLIFT cannot attach its rule to it")
        try:
            from tangermeme.deep_lift_shap import integrated_gradients_op
        except ImportError as error:
            raise RuntimeError(
                "Contribution scores through the shrink correction need "
                "tangermeme 1.6 or later (envs/chrombpnet-pytorch.yml)") from error
        return {Correction: integrated_gradients_op(DEEPLIFT_NODES)}
