"""Sequence contribution scores with Tangermeme DeepLIFT-SHAP."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import pandas as pd
import pyfaidx
import torch
from torch import nn

from chrombpnet.evaluation.interpret import input_utils
from chrombpnet.model import load_model, resolve_device


from chrombpnet.regions import NARROWPEAK_SCHEMA

# Contribution scoring needs roughly four times the GPU memory of a training
# step at the same batch size, for two compounding reasons: DeepLIFT pushes the
# examples and their references through the model together (twice the rows), and
# Tangermeme's hooks cache a clone of the input and the output of every
# non-linear module on top of the activations autograd already saves (about
# twice again). Dividing the training batch size by this factor makes the
# interpret step cost about as much GPU memory as the training step it follows:
# measured on an L40S for the 512-filter/8-dilation model at 2114 bp, training
# at batch size 64 peaks at 4.54 GB and attribution at batch size 16 at 4.63 GB.
INTERPRET_BATCH_DIVISOR = 4
DEFAULT_TRAINING_BATCH_SIZE = 64


def resolve_interpret_batch_size(args, default=DEFAULT_TRAINING_BATCH_SIZE):
    """Batch size for contribution scoring, scaled from the training one.

    An explicit ``--interpret-batch-size`` always wins. Otherwise the training
    batch size is divided by :data:`INTERPRET_BATCH_DIVISOR`, so raising or
    lowering ``-bs`` moves both steps together.
    """
    explicit = getattr(args, "interpret_batch_size", None)
    if explicit:
        return max(1, int(explicit))
    training = getattr(args, "batch_size", None) or default
    return max(1, int(training) // INTERPRET_BATCH_DIVISOR)


class CountsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sequence):
        return self.model(sequence)[1]


class ProfileWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sequence):
        logits = self.model(sequence)[0]
        mean_normalized = logits - logits.mean(dim=1, keepdim=True)
        weights = torch.softmax(mean_normalized.detach(), dim=1)
        return (weights * mean_normalized).sum(dim=1, keepdim=True)


def fetch_interpret_args():
    parser = argparse.ArgumentParser(description="Get sequence contribution scores")
    parser.add_argument("-g", "--genome", required=True)
    parser.add_argument("-r", "--regions", required=True)
    parser.add_argument("-m", "--model-h5", dest="model_h5", required=True)
    parser.add_argument("-o", "--output-prefix", required=True)
    parser.add_argument("-d", "--debug-chr", nargs="+", default=None)
    parser.add_argument(
        "-p",
        "--profile-or-counts",
        nargs="+",
        default=["counts", "profile"],
        choices=["counts", "profile"],
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument(
        "-ibs",
        "--interpret-batch-size",
        type=int,
        default=None,
        help="Batch size for contribution scoring (default: batch size / {})".format(
            INTERPRET_BATCH_DIVISOR
        ),
    )
    parser.add_argument("--num-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def generate_shap_dict(seqs, scores):
    if seqs.shape != scores.shape or seqs.shape[2] != 4:
        raise ValueError("Sequences and hypothetical scores must both be NLC")
    return {
        "raw": {"seq": np.transpose(seqs, (0, 2, 1)).astype(np.int8)},
        "shap": {"seq": np.transpose(scores, (0, 2, 1)).astype(np.float16)},
        "projected_shap": {
            "seq": np.transpose(seqs * scores, (0, 2, 1)).astype(np.float16)
        },
    }


def _save_scores(path, values):
    with h5py.File(path, "w") as handle:
        for group_name, datasets in values.items():
            group = handle.create_group(group_name)
            for name, data in datasets.items():
                group.create_dataset(name, data=data, compression="gzip")


def interpret(
    model,
    seqs,
    output_prefix,
    profile_or_counts,
    *,
    device,
    batch_size=64,
    num_shuffles=20,
    seed=1234,
):
    seqs = np.asarray(seqs)
    if seqs.ndim != 3 or seqs.shape[2] != 4:
        raise ValueError(f"Sequences must have shape (examples, length, 4), got {seqs.shape}")
    invalid_examples = ~np.all(np.isclose(seqs.sum(axis=2), 1.0), axis=1)
    if np.any(invalid_examples):
        raise ValueError(
            f"{int(invalid_examples.sum())} sequence(s) contain unknown or invalid bases; "
            "contribution scoring requires ACGT-only sequences"
        )
    # Tangermeme is intentionally lazy-loaded so prediction/preprocessing users
    # do not pay its import and numba compilation cost.
    from tangermeme.deep_lift_shap import deep_lift_shap

    sequence = torch.as_tensor(seqs, dtype=torch.float32).permute(0, 2, 1)
    targets = []
    if "counts" in profile_or_counts:
        targets.append(("counts", CountsWrapper(model)))
    if "profile" in profile_or_counts:
        targets.append(("profile", ProfileWrapper(model)))

    for name, wrapper in targets:
        wrapper.to(device).eval()
        scores = deep_lift_shap(
            wrapper,
            sequence,
            hypothetical=True,
            n_shuffles=num_shuffles,
            random_state=seed,
            batch_size=batch_size,
            device=device,
            verbose=True,
        )
        scores_nlc = scores.detach().cpu().permute(0, 2, 1).numpy()
        _save_scores(
            f"{output_prefix}.{name}_scores.h5",
            generate_shap_dict(seqs, scores_nlc),
        )


def main(args):
    with open(f"{args.output_prefix}.interpret.args.json", "w") as handle:
        json.dump(vars(args), handle, indent=4)
    regions_df = pd.read_csv(args.regions, sep="\t", names=NARROWPEAK_SCHEMA)
    if getattr(args, "debug_chr", None):
        regions_df = regions_df[regions_df["chr"].isin(args.debug_chr)]
    device = resolve_device(getattr(args, "device", None))
    model = load_model(args.model_h5, map_location=device)
    input_length = model.config.input_length
    genome = pyfaidx.Fasta(args.genome)
    seqs, regions_used = input_utils.get_seq(regions_df, genome, input_length)
    genome.close()
    skipped_regions = int((~regions_used).sum())
    if skipped_regions:
        print(
            f"Skipping {skipped_regions} region(s) that are truncated or contain "
            "non-ACGT reference bases"
        )
    regions_df[regions_used].to_csv(
        f"{args.output_prefix}.interpreted_regions.bed",
        header=False,
        index=False,
        sep="\t",
    )
    batch_size = resolve_interpret_batch_size(args)
    print(f"Contribution scoring with batch size {batch_size}")
    interpret(
        model,
        seqs,
        args.output_prefix,
        args.profile_or_counts,
        device=device,
        batch_size=batch_size,
        num_shuffles=getattr(args, "num_shuffles", 20),
        seed=getattr(args, "seed", 1234),
    )


if __name__ == "__main__":
    main(fetch_interpret_args())
