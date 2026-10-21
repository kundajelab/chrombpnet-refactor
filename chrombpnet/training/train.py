"""Model training, shared by every engine.

The parameter table, the architecture loader, the generators and the
``<prefix>.args.json`` record are framework-neutral and live here. Seeding,
building and fitting the model belong to the engine; the PyTorch loop is in
:mod:`chrombpnet.engines.pytorch.train`, and its names are re-exported from
this module for existing callers.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict

import chrombpnet.training.data_generators.initializers as initializers


os.environ["PYTHONHASHSEED"] = "0"


def get_model_param_dict(args) -> Dict[str, str]:
    params: Dict[str, str] = {}
    with open(args.params) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            key, value = line.split("\t", 1)
            params[key] = value
    required = {
        "counts_loss_weight",
        "filters",
        "n_dil_layers",
        "inputlen",
        "outputlen",
        "negative_sampling_ratio",
        "max_jitter",
        "chr_fold_path",
    }
    missing = sorted(required - set(params))
    if missing:
        raise ValueError("Missing model parameters: " + ", ".join(missing))
    if args.chr_fold_path != params["chr_fold_path"]:
        raise ValueError("Parameter file was generated with a different fold file")
    if int(params["inputlen"]) % 2 or int(params["outputlen"]) % 2:
        raise ValueError("Input and output lengths must be even")
    return params


def _load_architecture(path: str):
    spec = importlib.util.spec_from_file_location("chrombpnet_user_architecture", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load architecture file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PYTORCH_LOOP = frozenset({
    "get_model", "fit_and_evaluate", "_run_epoch", "_build_optimizer",
    "_tensor_batch",
})


def __getattr__(name):
    if name in _PYTORCH_LOOP:
        from chrombpnet.engines.pytorch import train as _loop
        return getattr(_loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(args):
    """Train one model with the configured engine.

    Shared by every engine: the parameter table, the generators and the
    ``<prefix>.args.json`` record. Building and fitting the model are the
    engine's.
    """
    from chrombpnet import engines

    engine = engines.current()
    parameters = get_model_param_dict(args)
    model, architecture_module = engine.get_model(args, parameters)
    train_generator = initializers.initialize_generators(
        args, "train", parameters, return_coords=False
    )
    # --model-selection pearsonr-jsd scores the validation peaks, so the
    # validation batches carry the coordinates' peak flag. Only for it: the
    # Keras engines would read a third element as sample weights.
    valid_generator = initializers.initialize_generators(
        args, "valid", parameters,
        return_coords=getattr(args, "model_selection", None) == "pearsonr-jsd",
    )
    engine.fit(
        model, train_generator, valid_generator, args, parameters, architecture_module
    )
    with open(args.output_prefix + ".args.json", "w") as handle:
        json.dump(vars(args), handle, indent=4)
    return model
