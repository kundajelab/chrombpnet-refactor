"""Run TF-MoDISco 2.6 on ChromBPNet contribution-score files."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from chrombpnet.evaluation.custom_sequences import viz_sequence
from chrombpnet.evaluation.modisco.api import iter_patterns, run_motifs, write_seqlets_fasta


def fetch_modisco_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scores_prefix", required=True,
                        help="Prefix to .profile_scores.h5/.counts_scores.h5")
    parser.add_argument("-p", "--profile_or_counts", required=True,
                        choices=("profile", "counts"))
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-c", "--crop", type=int, default=1000)
    parser.add_argument("-m", "--max_seqlets", type=int, default=50000)
    return parser.parse_args()


def save_plot(weights, destination):
    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(
        ax=ax, array=weights, height_padding_factor=0.2,
        length_padding=1.0, subticks_frequency=1.0, highlight={}, ylabel="")
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def main():
    args = fetch_modisco_args()
    os.makedirs(args.output_dir, exist_ok=True)
    score_type = args.profile_or_counts
    scores_path = f"{args.scores_prefix}.{score_type}_scores.h5"
    if not os.path.exists(scores_path):
        raise FileNotFoundError(scores_path)

    results_path = os.path.join(
        args.output_dir, f"modisco_results_allChroms_{score_type}.hdf5")
    seqlet_path = os.path.join(args.output_dir, f"seqlets_{score_type}.txt")
    logo_dir = os.path.join(args.output_dir, f"untrimmed_logos_{score_type}")
    os.makedirs(logo_dir, exist_ok=True)

    run_motifs(scores_path, results_path, max_seqlets=args.max_seqlets, window=args.crop)
    write_seqlets_fasta(results_path, seqlet_path)
    for pattern in iter_patterns(results_path):
        filename = pattern.tag.replace(".", "_")
        save_plot(pattern.contrib_scores, os.path.join(logo_dir, f"contrib_{filename}.png"))
        save_plot(pattern.sequence, os.path.join(logo_dir, f"sequence_{filename}.png"))
    print(f"Saved TF-MoDISco results to {results_path}")


if __name__ == "__main__":
    main()
