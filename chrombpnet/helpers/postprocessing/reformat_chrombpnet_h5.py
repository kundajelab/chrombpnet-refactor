"""Combine no-bias and bias PyTorch checkpoints.

The filename is retained for compatibility with old workflow imports. The
result is a structured ``.pt`` checkpoint; Keras HDF5 models are not accepted.
"""

import argparse
from pathlib import Path

from chrombpnet.model import ChromBPNet, load_model, save_model


def main(chrombpnet_nb, bias_model_scaled, output_dir, device=None):
    no_bias = load_model(chrombpnet_nb, map_location=device or "cpu")
    bias = load_model(bias_model_scaled, map_location=device or "cpu")
    model = ChromBPNet(no_bias.config, bias, tf_model=no_bias)
    output = Path(output_dir) / "chrombpnet_recompiled.pt"
    save_model(model, output, kind="chrombpnet")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-cnb", "--chrombpnet-nb", required=True)
    parser.add_argument("-bm", "--bias-model-scaled", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args.chrombpnet_nb, args.bias_model_scaled, args.output_dir, args.device)
