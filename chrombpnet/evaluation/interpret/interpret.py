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


NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]


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
    parser.add_argument("--batch-size", type=int, default=64)
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
    regions_df[regions_used].to_csv(
        f"{args.output_prefix}.interpreted_regions.bed",
        header=False,
        index=False,
        sep="\t",
    )
    interpret(
        model,
        seqs,
        args.output_prefix,
        args.profile_or_counts,
        device=device,
        batch_size=getattr(args, "batch_size", 64),
        num_shuffles=getattr(args, "num_shuffles", 20),
        seed=getattr(args, "seed", 1234),
    )


if __name__ == "__main__":
    main(fetch_interpret_args())
