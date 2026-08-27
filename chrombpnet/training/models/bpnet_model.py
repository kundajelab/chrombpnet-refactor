"""Default PyTorch bias-model architecture hook."""

from chrombpnet.model import BiasModel, ModelConfig


def getModelGivenModelOptionsAndWeightInits(args, model_params):
    return BiasModel(ModelConfig.from_mapping(model_params))


def build_model(args, model_params):
    return getModelGivenModelOptionsAndWeightInits(args, model_params)


def save_model_without_bias(model, output_prefix):
    return None
