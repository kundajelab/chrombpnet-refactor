"""PyTorch architecture hook for ``pipeline`` / ``train --post-training``.

The model that is trained is a plain BPNet, on the observed signal, with no
bias model. Once it is trained, the bias model is applied to its predicted
profile (:mod:`chrombpnet.engines.pytorch.post_training`) and the result is
saved as the run's no-bias model, which the rest of the pipeline evaluates as
it evaluates a co-trained one.
"""

import copy

from chrombpnet.engines.pytorch.model import (
    ModelConfig,
    UncorrectedBPNet,
    initialize_for_benchmark,
    load_model,
    save_model,
)
from chrombpnet.engines.pytorch.post_training import (
    SHRINK_EPS,
    SHRINK_WINDOW,
    PostTrainingChromBPNet,
    check_settings,
)


def build_model(args, model_params):
    if "bias_model_path" not in model_params:
        raise ValueError("bias_model_path is required for post-training correction")
    model = UncorrectedBPNet(ModelConfig.from_mapping(model_params))
    # The rule is --post-training's; shrink when -a names this file instead.
    rule = getattr(args, "post_training", None) or "shrink"
    # Before training, not after it: the rule must suit the profile length.
    check_settings(rule, SHRINK_EPS, SHRINK_WINDOW, model.config.output_length)
    if rule == "shrink" and str(getattr(args, "device", None) or "").startswith("mps"):
        raise ValueError("--post-training shrink is computed in float64, which MPS does not "
                         "have; use --device cpu or cuda, or --post-training plain")
    initialize_for_benchmark(model, args)
    # save_model_without_bias is given only the model and the output prefix.
    model.post_training = {"bias_model_path": model_params["bias_model_path"], "rule": rule}
    return model


def save_model_without_bias(model, output_prefix):
    settings = getattr(model, "post_training", None)
    if settings is None:
        return
    device = next(model.parameters()).device
    bias_model = load_model(settings["bias_model_path"], map_location=device)
    # A copy, because the corrected model freezes what it holds.
    corrected = PostTrainingChromBPNet(copy.deepcopy(model), bias_model, rule=settings["rule"])
    save_model(
        corrected,
        output_prefix + "_nobias.pt",
        provenance={"post_training_bias_model": str(settings["bias_model_path"])},
    )
