"""Student architecture hook for ``chrombpnet distill``: a bias-free BPNet."""

from chrombpnet.engines.pytorch.model import BPNet, ModelConfig, initialize_for_benchmark


def build_model(args, model_params):
    model = BPNet(ModelConfig.from_mapping(model_params))
    initialize_for_benchmark(model, args)
    return model


def save_model_without_bias(model, output_prefix):
    # The student is already the bias-free model.
    return None
