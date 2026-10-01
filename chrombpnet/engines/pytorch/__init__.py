"""The PyTorch engine: the package's original, and default, implementation."""

from __future__ import annotations

import numpy as np

from chrombpnet.engines.base import Engine


def adjust_bias_model_logcounts(bias_model, seqs, cts):
    """
    Given a bias model, sequences and associated counts, the function adds a
    constant to the output of the bias_model's logcounts that minimises squared
    error between predicted logcounts and observed logcounts (infered from
    cts). This simply reduces to adding the average difference between observed
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's last layer is a dense layer that outputs logcounts.
    This would change if you change the model.
    """
    import torch
    from chrombpnet.model import PredictionAdapter

    # safeguards to prevent misuse
    print("Predicting within adjust counts")
    adapter = PredictionAdapter(bias_model, device="cpu")
    _, pred_logcts = adapter.predict(seqs, batch_size=64, verbose=True)
    delta = np.mean(np.log(1+cts) - pred_logcts.ravel())
    with torch.no_grad():
        bias_model.count_dense.bias.add_(float(delta))
    return bias_model


class PyTorchEngine(Engine):
    name = "pytorch"
    checkpoint_suffix = ".pt"
    environment = "chrombpnet-pytorch"

    def require(self):
        try:
            import torch  # noqa: F401
        except ImportError as error:
            raise self._unavailable(error) from error

    # -- training ----------------------------------------------------------
    def default_architecture(self, kind):
        # These two paths are written into <prefix>.args.json, so the files
        # stay where they have always been rather than moving under engines/.
        if kind == "bias":
            import chrombpnet.training.models.bpnet_model as architecture
        elif kind == "chrombpnet":
            import chrombpnet.training.models.chrombpnet_with_bias_model as architecture
        else:
            raise ValueError(f"Unknown model kind {kind!r}")
        return architecture.__file__

    def get_model(self, args, parameters):
        from chrombpnet.training import train
        return train.get_model(args, parameters)

    def fit(self, model, train_gen, valid_gen, args, parameters, architecture_module):
        from chrombpnet.training import train
        return train.fit_and_evaluate(
            model, train_gen, valid_gen, args, architecture_module, parameters)

    # -- inference ---------------------------------------------------------
    def resolve_device(self, value=None):
        from chrombpnet.model import resolve_device
        return resolve_device(value)

    def load_predictor(self, path, device=None, *, announce=True, summary=True):
        from chrombpnet.model import load_model_wrapper
        return load_model_wrapper(path, device, announce=announce, summary=summary)

    def rescale_bias_logcounts(self, bias_path, output_path, seqs, cts):
        from chrombpnet.model import load_model, save_model
        bias_model = load_model(bias_path, map_location="cpu")
        bias_model_scaled = adjust_bias_model_logcounts(bias_model, seqs, cts)
        save_model(bias_model_scaled, output_path, kind="bias")

    # -- attribution -------------------------------------------------------
    def load_attribution_model(self, path, device):
        # The module itself, not the prediction adapter: DeepLIFT needs
        # autograd and hooks on the nn.ReLU modules.
        from chrombpnet.model import load_model
        return load_model(path, map_location=device)

    def input_length(self, model):
        return model.config.input_length

    def attribute(self, model, seqs, profile_or_counts, *, device, batch_size,
                  num_shuffles, seed):
        from chrombpnet.engines.pytorch import attribution
        return attribution.attribute(
            model, seqs, profile_or_counts, device=device,
            batch_size=batch_size, num_shuffles=num_shuffles, seed=seed)


ENGINE = PyTorchEngine()
