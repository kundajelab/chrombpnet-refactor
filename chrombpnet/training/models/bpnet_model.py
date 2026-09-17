"""Default PyTorch bias-model architecture hook."""

from chrombpnet.model import BiasModel, ModelConfig, initialize_for_benchmark


def getModelGivenModelOptionsAndWeightInits(args, model_params):
    model = BiasModel(ModelConfig.from_mapping(model_params))
    initialize_for_benchmark(model, args)
    return model


def build_model(args, model_params):
    return getModelGivenModelOptionsAndWeightInits(args, model_params)


def save_model_without_bias(model, output_prefix):
    return None
