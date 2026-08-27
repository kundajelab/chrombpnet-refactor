"""Default bias-factorized PyTorch ChromBPNet architecture hook."""

from chrombpnet.model import ChromBPNet, ModelConfig, load_model, save_model


def getModelGivenModelOptionsAndWeightInits(args, model_params):
    if "bias_model_path" not in model_params:
        raise ValueError("bias_model_path is required for ChromBPNet")
    config = ModelConfig.from_mapping(model_params)
    bias_model = load_model(model_params["bias_model_path"], map_location="cpu")
    return ChromBPNet(config, bias_model)


def build_model(args, model_params):
    return getModelGivenModelOptionsAndWeightInits(args, model_params)


def save_model_without_bias(model, output_prefix):
    if not isinstance(model, ChromBPNet):
        return
    save_model(
        model.tf_model,
        output_prefix + "_nobias.pt",
        kind="chrombpnet_nobias",
    )
