"""PyTorch architecture hook for ``pipeline`` / ``train --posthoc``.

The model that is trained is a plain BPNet, on the observed signal, with no
bias model. Once it is trained, the bias model is applied to its predicted
profile (:mod:`chrombpnet.engines.pytorch.posthoc`) and the result is saved
as the run's no-bias model, which the rest of the pipeline evaluates as it
evaluates a co-trained one.
"""

import copy

from chrombpnet.engines.pytorch.model import (
    ModelConfig,
    UncorrectedBPNet,
    initialize_for_benchmark,
    load_model,
    save_model,
)
from chrombpnet.engines.pytorch.posthoc import (
    SHRINK_EPS,
    SHRINK_WINDOW,
    PostHocChromBPNet,
    check_settings,
)


def build_model(args, model_params):
    if "bias_model_path" not in model_params:
        raise ValueError("bias_model_path is required for post hoc correction")
    model = UncorrectedBPNet(ModelConfig.from_mapping(model_params))
    # The rule is --posthoc's; shrink when this file is named by -a instead.
    rule = getattr(args, "posthoc", None) or "shrink"
    # Before training, not after it: the rule must suit the profile length.
    check_settings(rule, SHRINK_EPS, SHRINK_WINDOW, model.config.output_length)
    if rule == "shrink" and str(getattr(args, "device", None) or "").startswith("mps"):
        raise ValueError("--posthoc shrink is computed in float64, which MPS does not "
                         "have; use --device cpu or cuda, or --posthoc plain")
    initialize_for_benchmark(model, args)
    # save_model_without_bias is given only the model and the output prefix.
    model.posthoc = {"bias_model_path": model_params["bias_model_path"], "rule": rule}
    return model


def save_model_without_bias(model, output_prefix):
    settings = getattr(model, "posthoc", None)
    if settings is None:
        return
    device = next(model.parameters()).device
    bias_model = load_model(settings["bias_model_path"], map_location=device)
    # A copy, because the corrected model freezes what it holds.
    corrected = PostHocChromBPNet(copy.deepcopy(model), bias_model, rule=settings["rule"])
    save_model(
        corrected,
        output_prefix + "_nobias.pt",
        provenance={"posthoc_bias_model": str(settings["bias_model_path"])},
    )
