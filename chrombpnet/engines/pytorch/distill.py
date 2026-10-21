"""``chrombpnet distill``: one bias-free student from several fold models.

The student is a fresh accessibility BPNet trained on its teachers' averaged
predictions instead of the observed counts. On every batch -- after jitter
and reverse complementation, so the teachers see exactly what the student
sees -- the teachers' bias-free models run in eval mode, and the targets are

    profile:    mean_t softmax(logits_t) * expm1(mean_t logcounts_t)
    log counts: mean_t logcounts_t

The loss is ChromBPNet's own, multinomial NLL on the profile (with the
pseudo-counts above, which ``multinomial_nll`` accepts) plus
``counts_loss_weight`` times the log-count MSE, so a region weighs in the
profile term by its predicted total as an observed one does by its read
count. Epochs are kept and stopped on the same loss over the validation
chromosomes, against the teachers' targets there.

The student is saved as a ``chrombpnet_nobias`` checkpoint, so ``predict``,
``contribs_bw`` and ``footprints`` read it as they read a trained model's.

With ``--all-chromosomes`` every chromosome in the fold file is trained on,
there are none left to validate on, and exactly ``--epochs`` epochs run.
"""

from __future__ import annotations

import json
import os
import statistics

import torch

from chrombpnet.engines.pytorch.model import BPNet, ChromBPNet, load_model, resolve_device


def load_teachers(paths, device):
    """The teachers' bias-free models, frozen and in eval mode.

    ``chrombpnet_nobias.pt`` checkpoints, or ``chrombpnet.pt`` ones, whose
    accessibility submodel is used.
    """
    teachers = []
    for path in paths:
        if not str(path).endswith(".pt"):
            raise ValueError(f"distill reads PyTorch .pt teachers, not {path!r}")
        model = load_model(path, map_location=device)
        if isinstance(model, ChromBPNet):
            model = model.tf_model
        elif type(model) is not BPNet:
            raise ValueError(f"{path} is a bias model, not a ChromBPNet teacher")
        if model.count_dense is None:
            raise ValueError(f"{path} has no counts head to distill")
        for parameter in model.parameters():
            parameter.requires_grad = False
        teachers.append(model.eval())
    if not teachers:
        raise ValueError("distill needs at least one teacher")
    geometry = {(t.config.input_length, t.config.output_length) for t in teachers}
    if len(geometry) > 1:
        raise ValueError(f"teachers differ in input/output length: {sorted(geometry)}")
    return teachers


class TeacherTargets:
    """``target_fn`` for the training loop: the teachers' averaged profile
    pseudo-counts and log counts for a batch of sequences."""

    def __init__(self, teachers):
        self.teachers = list(teachers)

    @torch.no_grad()
    def __call__(self, sequence):
        probabilities = None
        logcounts = None
        for teacher in self.teachers:
            teacher.eval()
            profile, counts = teacher(sequence)
            p = torch.softmax(profile.float(), dim=-1)
            probabilities = p if probabilities is None else probabilities + p
            logcounts = counts.float() if logcounts is None else logcounts + counts.float()
        probabilities = probabilities / len(self.teachers)
        logcounts = logcounts / len(self.teachers)
        total = torch.expm1(logcounts).clamp_min(0.0)
        return probabilities * total, logcounts


def all_chromosome_folds(fold_path, output_path):
    """Write a fold file whose train split is every chromosome of
    ``fold_path`` and whose valid and test splits are empty."""
    with open(fold_path) as handle:
        splits = json.load(handle)
    chromosomes = []
    for split in ("train", "valid", "test"):
        for chromosome in splits.get(split, []):
            if chromosome not in chromosomes:
                chromosomes.append(chromosome)
    with open(output_path, "w") as handle:
        json.dump({"train": chromosomes, "valid": [], "test": []}, handle, indent=4)
    return output_path


def main(args):
    from chrombpnet.engines.pytorch import train
    import chrombpnet.training.data_generators.initializers as initializers
    import chrombpnet.training.models.distill_student_model as student_architecture

    device = resolve_device(getattr(args, "device", None))
    teachers = load_teachers(args.teachers, device)
    config = teachers[0].config

    weight = args.counts_loss_weight
    if weight is None:
        weight = float(statistics.median(t.config.counts_loss_weight for t in teachers))
        print(f"counts loss weight: {weight} (the teachers' median)")

    directory = os.path.dirname(args.output_prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fold_path = args.chr_fold_path
    if args.all_chromosomes:
        fold_path = all_chromosome_folds(fold_path, args.output_prefix + ".folds.json")

    parameters = {
        "inputlen": config.input_length,
        "outputlen": config.output_length,
        "filters": args.filters or config.filters,
        "n_dil_layers": args.n_dilation_layers or config.n_dilation_layers,
        "counts_loss_weight": weight,
        "max_jitter": args.max_jitter,
        "negative_sampling_ratio": args.negative_sampling_ratio,
        "chr_fold_path": fold_path,
    }
    step = args.step(architecture_from_file=student_architecture.__file__,
                     replacing={"chr_fold_path": fold_path})
    model, architecture = train.get_model(step, parameters)
    train_generator = initializers.initialize_generators(
        step, "train", parameters, return_coords=False)
    valid_generator = None
    if not args.all_chromosomes:
        valid_generator = initializers.initialize_generators(
            step, "valid", parameters, return_coords=False)
    train.fit_and_evaluate(
        model, train_generator, valid_generator, step, architecture, parameters,
        target_fn=TeacherTargets(teachers),
        provenance_extra={
            "distilled_from": [str(path) for path in args.teachers],
            "counts_loss_weight": weight,
            "all_chromosomes": bool(args.all_chromosomes),
        },
    )
    with open(args.output_prefix + ".args.json", "w") as handle:
        json.dump(vars(step), handle, indent=4)
    return model
