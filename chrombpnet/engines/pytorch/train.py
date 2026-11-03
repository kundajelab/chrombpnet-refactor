"""Training loop for the PyTorch engine.

Moved verbatim from ``chrombpnet/training/train.py``, which keeps the parts
every engine shares: the parameter table, the architecture loader, the
generators and the ``<prefix>.args.json`` record.
"""

from __future__ import annotations

import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch

from chrombpnet.engines.pytorch.losses import chrombpnet_loss
from chrombpnet.engines.pytorch.model import resolve_device, save_model
from chrombpnet.training.train import _load_architecture
from chrombpnet.training.utils.benchmark_utils import (
    benchmark_enables,
    benchmark_level,
    use_keras_batch_order_shuffle,
)


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
    if benchmark_enables(args, "deterministic_kernels"):
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
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


def _run_epoch(
    model,
    generator,
    device,
    counts_weight,
    optimizer=None,
    sample_weighted=False,
    batch_order=None,
    target_fn=None,
):
    """One pass over ``generator``; returns the mean (total, profile, counts)
    losses and the per-batch rows.

    ``target_fn(sequence)``, when given, supplies the profile and log-count
    targets in place of the observed ones: distillation's teacher targets,
    computed on the same augmented batch.
    """
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    total_weight = 0
    batch_rows = []
    order = range(len(generator)) if batch_order is None else batch_order
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index in order:
            sequence, profile, logcounts = _tensor_batch(generator[batch_index], device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            profile_prediction, count_prediction = model(sequence)
            if target_fn is not None:
                profile, logcounts = target_fn(sequence)
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
            weight = int(sequence.shape[0]) if sample_weighted else 1
            totals += values * weight
            total_weight += weight
            batch_rows.append((batch_index, *values.tolist()))
    denominator = max(total_weight, 1)
    return totals / denominator, batch_rows


def _build_optimizer(model, args):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    options = {"lr": args.learning_rate}
    if benchmark_enables(args, "keras_adam"):
        # Keras Adam defaults to epsilon 1e-7; PyTorch defaults to 1e-8.
        options.update({"eps": 1e-7, "betas": (0.9, 0.999), "amsgrad": False})
    if benchmark_enables(args, "scalar_adam_dispatch"):
        # Foreach/fused kernels change the reduction order of the update.
        options.update({"foreach": False, "fused": False})
    return torch.optim.Adam(parameters, **options)


def fit_and_evaluate(model, train_gen, valid_gen, args, architecture_module, parameters,
                     target_fn=None, provenance_extra=None):
    """Train, keep the weights of the epoch with the lowest validation loss
    (as chrombpnet 1.x's checkpoint callback does), and write the logs.

    ``target_fn`` replaces the observed targets with ``target_fn(sequence)``
    in training and validation (``chrombpnet distill``'s teacher targets).
    With no ``valid_gen`` every one of ``--epochs`` epochs runs and the last
    weights are kept: distillation's all-chromosome student, which has no
    chromosomes left to validate on. ``provenance_extra`` is recorded in the
    checkpoint alongside the seed and epoch.
    """
    device = resolve_device(getattr(args, "device", None))
    model.to(device)
    benchmark = benchmark_level(args)
    sample_weighted = benchmark_enables(args, "keras_loss_aggregation")
    shuffle_batch_order = use_keras_batch_order_shuffle(args)
    optimizer = _build_optimizer(model, args)
    counts_weight = float(parameters["counts_loss_weight"])
    checkpoint_path = args.output_prefix + ".pt"
    epoch_log_path = args.output_prefix + ".log"
    batch_log_path = epoch_log_path + ".batch"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0
    epoch_rows = []
    batch_rows_all = []

    for epoch in range(args.epochs):
        # Keras shuffles Sequence batch indices unless fit(shuffle=False).
        batch_order = (
            np.random.permutation(len(train_gen)) if shuffle_batch_order else None
        )
        train_values, train_batches = _run_epoch(
            model, train_gen, device, counts_weight, optimizer=optimizer,
            sample_weighted=sample_weighted, batch_order=batch_order,
            target_fn=target_fn,
        )
        batch_rows_all.extend((epoch, *batch_row) for batch_row in train_batches)
        if valid_gen is None:
            # Nothing to validate on: keep training, keep the latest weights.
            epoch_rows.append((epoch, *train_values.tolist(), *[float("nan")] * 3))
            best_state = copy.deepcopy(model.state_dict())
            train_gen.on_epoch_end()
            continue
        valid_values, _ = _run_epoch(
            model, valid_gen, device, counts_weight, sample_weighted=sample_weighted,
            target_fn=target_fn,
        )
        epoch_rows.append((epoch, *train_values.tolist(), *valid_values.tolist()))

        validation_loss = valid_values[0]
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
            provenance = {"seed": args.seed, "best_epoch": epoch,
                          "benchmark": benchmark}
            provenance.update(provenance_extra or {})
            save_model(model, checkpoint_path, provenance=provenance)
        else:
            epochs_without_improvement += 1
        train_gen.on_epoch_end()
        if epochs_without_improvement >= args.early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_provenance = {"seed": args.seed, "epochs_completed": len(epoch_rows),
                        "benchmark": benchmark}
    if provenance_extra or valid_gen is None:
        # Only the new modes record more, so default checkpoints are unchanged.
        final_provenance["best_epoch"] = (len(epoch_rows) - 1 if valid_gen is None
                                          else best_epoch)
        final_provenance.update(provenance_extra or {})
    save_model(model, checkpoint_path, provenance=final_provenance)
    architecture_module.save_model_without_bias(model, args.output_prefix)

    with open(epoch_log_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "epoch",
            "loss",
            "logits_profile_predictions_loss",
            "logcount_predictions_loss",
            "val_loss",
            "val_logits_profile_predictions_loss",
            "val_logcount_predictions_loss",
        ]
        writer.writerow(header)
        writer.writerows(epoch_rows)
    with open(batch_log_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Epoch", "Batch", "loss", "profile_loss", "counts_loss"])
        writer.writerows(batch_rows_all)
