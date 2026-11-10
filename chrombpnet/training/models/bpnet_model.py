"""Default PyTorch bias-model architecture hook."""

import dataclasses

from chrombpnet.engines.pytorch.model import BiasModel, ModelConfig, initialize_for_benchmark


def getModelGivenModelOptionsAndWeightInits(args, model_params):
    config = ModelConfig.from_mapping(model_params)
    if getattr(args, "bias_profile_only", None) or getattr(args, "pretrained_bias", None):
        # --bias-profile-only, which --pretrained-bias implies: no counts
        # head; trained on the profile NLL.
        config = dataclasses.replace(config, counts_head=False)
    model = BiasModel(config)
    initialize_for_benchmark(model, args)
    return model


def build_model(args, model_params):
    return getModelGivenModelOptionsAndWeightInits(args, model_params)


def save_model_without_bias(model, output_prefix):
    return None
