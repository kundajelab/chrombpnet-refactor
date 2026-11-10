"""``chrombpnet bias pretrain`` and ``bias pipeline --pretrained-bias``: a
profile-only bias model whose convolutions are learned from other datasets
of the same assay (PyTorch engine).

From the chrombpnet-bias experiment's ``pre-finetune`` mode:

- :func:`main` trains one bias trunk -- the bias architecture's first and
  dilated convolutions -- shared by several datasets, each with its own
  profile head, on their filtered nonpeaks with the profile loss. Every
  optimizer step takes one batch from every dataset (shorter ones cycle), so
  each dataset weighs the same whatever its size, and the validation loss is
  the mean of the datasets' mean losses.
- :func:`adapt` starts a new profile-only bias model from that trunk,
  frozen, under a zero profile head: the only part ``bias pipeline`` then
  trains, on the target dataset's nonpeaks. ``pipeline`` trains on the
  result as on any profile-only bias, adding it to the profile only.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from chrombpnet.engines.pytorch.losses import multinomial_nll
from chrombpnet.engines.pytorch.model import (
    BiasModel,
    ModelConfig,
    MultitaskBias,
    initialize_keras_default,
    load_model,
    resolve_device,
    save_model,
)

#: The manifest's columns: one row per pretraining dataset.
MANIFEST_COLUMNS = ("dataset", "bigwig", "peaks", "nonpeaks")


def read_manifest(path):
    """The datasets of a ``bias pretrain`` manifest, paths made absolute
    (relative ones are relative to the manifest)."""
    path = Path(path).resolve()
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or any(column not in rows[0] for column in MANIFEST_COLUMNS):
        raise ValueError(f"{path}: a header with the columns {', '.join(MANIFEST_COLUMNS)} "
                         "and one row per dataset is required")
    datasets = []
    for row in rows:
        name = (row["dataset"] or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"{path}: dataset names may contain only letters, digits, "
                             f"- and _, not {name!r}")
        entry = {"dataset": name}
        for column in MANIFEST_COLUMNS[1:]:
            value = Path((row[column] or "").strip())
            value = value if value.is_absolute() else path.parent / value
            if not value.is_file():
                raise ValueError(f"{path}: {name}'s {column} {value} does not exist")
            entry[column] = str(value)
        datasets.append(entry)
    names = [entry["dataset"] for entry in datasets]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(f"{path}: two or more datasets with distinct names are required")
    return datasets


def _read_fold(path):
    with open(path) as handle:
        return json.load(handle)


def _dataset_generators(args, dataset, directory):
    """One dataset's training and validation generators, on its nonpeaks
    filtered as ``bias pipeline`` filters them, into ``directory``."""
    from chrombpnet.helpers.hyperparameters import find_bias_hyperparams
    from chrombpnet.training.data_generators import initializers
    from chrombpnet.training.train import get_model_param_dict

    prefix = str(directory) + "/"
    os.makedirs(prefix, exist_ok=True)
    options = args.step(bigwig=dataset["bigwig"], peaks=dataset["peaks"],
                        nonpeaks=dataset["nonpeaks"], output_prefix=prefix)
    np.random.seed(args.seed)
    find_bias_hyperparams.main(options)
    parameters = get_model_param_dict(options.step(params=prefix + "bias_model_params.tsv"))
    training = options.step(replacing={"peaks": "None",
                                       "nonpeaks": prefix + "filtered.bias_nonpeaks.bed"})
    np.random.seed(args.seed)
    generators = tuple(initializers.initialize_generators(training, mode, parameters,
                                                          return_coords=False)
                       for mode in ("train", "valid"))
    for mode, generator in zip(("training", "validation"), generators):
        if len(generator) == 0:
            raise ValueError(f"{dataset['dataset']}: no {mode} nonpeaks after filtering")
    return generators, parameters


def _mean_losses(model, generators, device):
    """Each dataset's mean validation loss."""
    model.eval()
    losses = {}
    with torch.no_grad():
        for task, generator in generators.items():
            values = []
            for index in range(len(generator)):
                sequence, profile, _ = _tensor_batch(generator[index], device)
                values.append(multinomial_nll(profile, model(sequence, task)).item())
            losses[task] = float(np.mean(values))
    return losses


def _tensor_batch(batch, device):
    from chrombpnet.engines.pytorch.train import _tensor_batch as convert
    return convert(batch, device)


def fit(model, generators, args, output_prefix, provenance):
    """Train ``model`` on every dataset's batches, keep the epoch with the
    lowest mean validation loss, and write ``<prefix>.pt`` and
    ``<prefix>.log``."""
    from chrombpnet.engines.pytorch.train import _build_optimizer
    from chrombpnet.training.utils.benchmark_utils import use_keras_batch_order_shuffle

    device = resolve_device(getattr(args, "device", None))
    model.to(device)
    optimizer = _build_optimizer(model, args)
    shuffle = use_keras_batch_order_shuffle(args)
    tasks = list(generators)
    training = {task: generators[task][0] for task in tasks}
    validation = {task: generators[task][1] for task in tasks}
    checkpoint_path = output_prefix + ".pt"
    best_loss, best_state, best_epoch, stale = float("inf"), None, None, 0
    rows = []
    for epoch in range(args.epochs):
        model.train()
        orders = {task: (np.random.permutation(len(training[task])) if shuffle
                         else np.arange(len(training[task]))) for task in tasks}
        sums = {task: [0.0, 0] for task in tasks}
        for step in range(max(len(generator) for generator in training.values())):
            optimizer.zero_grad(set_to_none=True)
            for task in tasks:
                order = orders[task]
                sequence, profile, _ = _tensor_batch(
                    training[task][int(order[step % len(order)])], device)
                loss = multinomial_nll(profile, model(sequence, task))
                (loss / len(tasks)).backward()
                sums[task][0] += loss.item()
                sums[task][1] += 1
            optimizer.step()
        train_losses = {task: total / count for task, (total, count) in sums.items()}
        valid_losses = _mean_losses(model, validation, device)
        validation_loss = float(np.mean(list(valid_losses.values())))
        rows.append([epoch, float(np.mean(list(train_losses.values()))), validation_loss,
                     *[train_losses[task] for task in tasks],
                     *[valid_losses[task] for task in tasks]])
        print(f"epoch {epoch}: mean validation loss {validation_loss:.6g}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            save_model(model, checkpoint_path,
                       provenance={**provenance, "best_epoch": epoch})
        else:
            stale += 1
        for generator in training.values():
            generator.on_epoch_end()
        if stale >= args.early_stop:
            break
    if best_state is None:
        raise ValueError("bias pretrain needs at least one epoch (-e)")
    model.load_state_dict(best_state)
    save_model(model, checkpoint_path,
               provenance={**provenance, "best_epoch": best_epoch,
                           "epochs_completed": len(rows)})
    with open(output_prefix + ".log", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "val_loss", *[f"loss_{task}" for task in tasks],
                         *[f"val_loss_{task}" for task in tasks]])
        writer.writerows(rows)
    return model


def main(args):
    """``chrombpnet bias pretrain``."""
    from chrombpnet.engines.pytorch.train import seed_for_construction

    datasets = read_manifest(args.manifest)
    output = Path(args.output_dir)
    for directory in ("models", "logs", "auxiliary"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    generators, geometry = {}, None
    for dataset in datasets:
        print(f"=== {dataset['dataset']}: filtering nonpeaks", flush=True)
        generators[dataset["dataset"]], parameters = _dataset_generators(
            args, dataset, output / "auxiliary" / dataset["dataset"])
        geometry = geometry or parameters

    seed_for_construction(args)
    model = MultitaskBias(ModelConfig.from_mapping(geometry), list(generators),
                          args.data_type, _read_fold(args.chr_fold_path))
    initialize_keras_default(model, args.seed)
    fit(model, generators, args, str(output / "models" / "multitask_bias"),
        provenance={"seed": args.seed, "datasets": datasets,
                    "manifest": str(Path(args.manifest).resolve())})
    os.replace(output / "models" / "multitask_bias.log",
               output / "logs" / "multitask_bias.log")
    with open(output / "logs" / "multitask_bias.args.json", "w") as handle:
        json.dump(vars(args), handle, indent=4)
    print(f"Wrote {output / 'models' / 'multitask_bias.pt'}", flush=True)
    return model


def adapt(model, path, args):
    """Copy the trunk of the ``bias pretrain`` checkpoint ``path`` into
    ``model``, a new profile-only bias model, freeze it, and zero the
    profile head, which is then all that trains."""
    pretrained = load_model(path)
    if not isinstance(pretrained, MultitaskBias):
        raise ValueError(f"--pretrained-bias {path} is not a `chrombpnet bias pretrain` "
                         "checkpoint")
    if not isinstance(model, BiasModel) or model.config.counts_head:
        raise ValueError("--pretrained-bias needs a profile-only bias model; a custom "
                         "architecture (-a) must build one")
    if pretrained.assay != args.data_type:
        raise ValueError(f"--pretrained-bias {path} was pretrained on {pretrained.assay}, "
                         f"not {args.data_type}")
    if pretrained.fold != _read_fold(args.chr_fold_path):
        raise ValueError(
            f"--pretrained-bias {path} was pretrained on another fold file, so its trunk "
            "may have trained on this run's validation or test chromosomes; pretrain with "
            "this run's -fl")
    for field, flag in (("input_length", "-il"), ("output_length", "-ol"),
                        ("filters", "-fil"), ("n_dilation_layers", "-dil")):
        expected, given = getattr(pretrained.config, field), getattr(model.config, field)
        if expected != given:
            raise ValueError(f"--pretrained-bias {path} has {field} {expected}, this run "
                             f"{given}: pass {flag} {expected}")
    result = model.load_state_dict(pretrained.trunk.state_dict(), strict=False)
    if result.unexpected_keys or any(not key.startswith("profile_conv.")
                                     for key in result.missing_keys):
        raise RuntimeError(f"Cannot copy the trunk of {path}: {result}")
    with torch.no_grad():
        model.profile_conv.weight.zero_()
        model.profile_conv.bias.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("profile_conv."))
    print(f"bias model: the frozen trunk of {path} (pretrained on "
          f"{', '.join(pretrained.tasks)}) under a new profile head", flush=True)
    return model
