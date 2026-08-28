"""Training loop for PyTorch ChromBPNet models."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

import chrombpnet.training.data_generators.initializers as initializers
from chrombpnet.model import ChromBPNet, resolve_device, save_model
from chrombpnet.training.utils.losses import chrombpnet_loss


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


def get_model(args, parameters):
    architecture = _load_architecture(args.architecture_from_file)
    builder = getattr(
        architecture,
        "build_model",
        getattr(architecture, "getModelGivenModelOptionsAndWeightInits", None),
    )
    if builder is None:
        raise AttributeError("Architecture must define build_model(args, parameters)")

    # Seed before construction so --seed controls every trainable parameter.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = builder(args, parameters)
    return model, architecture


def _tensor_batch(batch, device):
    sequences, labels = batch[:2]
    sequence_tensor = torch.as_tensor(sequences, dtype=torch.float32, device=device)
    if sequence_tensor.shape[-1] == 4:
        sequence_tensor = sequence_tensor.permute(0, 2, 1)
    profile = torch.as_tensor(labels[0], dtype=torch.float32, device=device)
    logcounts = torch.as_tensor(labels[1], dtype=torch.float32, device=device)
    return sequence_tensor, profile, logcounts


def _run_epoch(model, generator, device, counts_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    batch_rows = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index in range(len(generator)):
            sequence, profile, logcounts = _tensor_batch(generator[batch_index], device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            profile_prediction, count_prediction = model(sequence)
            total, profile_loss, count_loss = chrombpnet_loss(
                profile_prediction,
                count_prediction,
                profile,
                logcounts,
                counts_weight,
            )
            if training:
                total.backward()
                optimizer.step()
            values = np.array(
                [total.item(), profile_loss.item(), count_loss.item()], dtype=np.float64
            )
            totals += values
            batch_rows.append((batch_index, *values.tolist()))
    denominator = max(len(generator), 1)
    return totals / denominator, batch_rows


def fit_and_evaluate(model, train_gen, valid_gen, args, architecture_module, parameters):
    device = resolve_device(getattr(args, "device", None))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    counts_weight = float(parameters["counts_loss_weight"])
    checkpoint_path = args.output_prefix + ".pt"
    epoch_log_path = args.output_prefix + ".log"
    batch_log_path = epoch_log_path + ".batch"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epoch_rows = []
    batch_rows_all = []

    for epoch in range(args.epochs):
        train_values, train_batches = _run_epoch(
            model, train_gen, device, counts_weight, optimizer=optimizer
        )
        valid_values, _ = _run_epoch(model, valid_gen, device, counts_weight)
        epoch_rows.append((epoch, *train_values.tolist(), *valid_values.tolist()))
        batch_rows_all.extend((epoch, *row) for row in train_batches)

        validation_loss = valid_values[0]
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_model(
                model,
                checkpoint_path,
                provenance={"seed": args.seed, "best_epoch": epoch},
            )
        else:
            epochs_without_improvement += 1
        train_gen.on_epoch_end()
        if epochs_without_improvement >= args.early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_model(
        model,
        checkpoint_path,
        provenance={"seed": args.seed, "epochs_completed": len(epoch_rows)},
    )
    architecture_module.save_model_without_bias(model, args.output_prefix)

    with open(epoch_log_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "loss",
                "logits_profile_predictions_loss",
                "logcount_predictions_loss",
                "val_loss",
                "val_logits_profile_predictions_loss",
                "val_logcount_predictions_loss",
            ]
        )
        writer.writerows(epoch_rows)
    with open(batch_log_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Epoch", "Batch", "loss", "profile_loss", "counts_loss"])
        writer.writerows(batch_rows_all)


def main(args):
    parameters = get_model_param_dict(args)
    model, architecture_module = get_model(args, parameters)
    train_generator = initializers.initialize_generators(
        args, "train", parameters, return_coords=False
    )
    valid_generator = initializers.initialize_generators(
        args, "valid", parameters, return_coords=False
    )
    fit_and_evaluate(
        model, train_generator, valid_generator, args, architecture_module, parameters
    )
    with open(args.output_prefix + ".args.json", "w") as handle:
        json.dump(vars(args), handle, indent=4)
    return model


if __name__ == "__main__":
    import chrombpnet.training.utils.argmanager as argmanager

    main(argmanager.fetch_train_args())
