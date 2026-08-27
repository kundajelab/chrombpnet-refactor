# ChromBPNet Refactor (PyTorch)

This repository is a PyTorch replica of Kundaje Lab ChromBPNet, based on
TensorFlow source commit `09938fdb4397ec0006510e5251e48920a505d4de`.
It retains the original preprocessing, model-training, prediction, contribution
scoring, marginal-footprinting, and reporting workflows while using native
PyTorch models and `.pt` checkpoints.

The original README is retained as `README.tensorflow.md`. Confirmed issues in
the reference implementation and their compatibility status are recorded in
`chrombpnet_bugs.md`.

## Installation

```bash
conda env create -f environment.yml
conda activate chrombpnet-refactor
pip install -e .
```

The environment installs a CUDA-enabled PyTorch build which automatically falls
back to CPU. Commands accept `--device cpu`, `--device cuda`, or a specific
device such as `--device cuda:1`.

## Python API

```python
import torch
from chrombpnet import BiasModel, ChromBPNet, ModelConfig, load_model, save_model

config = ModelConfig(
    input_length=2114,
    output_length=1000,
    filters=128,
    n_dilation_layers=4,
    counts_loss_weight=1.0,
)
bias = BiasModel(config)
model = ChromBPNet(config, bias)

# Native tensor layout is NCL: batch, A/C/G/T channels, sequence length.
profile_logits, logcounts = model(torch.zeros(2, 4, 2114))
save_model(model, "chrombpnet.pt")
model = load_model("chrombpnet.pt")
```

Structured checkpoints contain a format version, model type, architecture
configuration, source provenance, and a standard PyTorch `state_dict`.
TensorFlow/Keras `.h5` models are intentionally not accepted by the production
API.

## Command line

The active command hierarchy is retained:

```text
chrombpnet pipeline
chrombpnet train
chrombpnet qc
chrombpnet bias pipeline
chrombpnet bias train
chrombpnet bias qc
chrombpnet prep nonpeaks
chrombpnet prep splits
chrombpnet pred_bw
chrombpnet contribs_bw
chrombpnet footprints
```

Model arguments now point to `.pt` files. Run `chrombpnet <command> --help` for
the complete set of genomics inputs and options.

## Cross-framework parity

Run the self-contained comparison from this repository:

```bash
conda run -n chrombpnet-refactor python test.py
```

`test.py` runs TensorFlow reference calculations in the separate `chrombpnet`
environment, synchronizes deterministic weights, and compares bias, no-bias,
and combined outputs and losses at `atol=1e-6`, `rtol=1e-5`.

Independent training runs are not expected to produce elementwise-identical
weights because framework random-number generators and GPU kernels differ.

## Citation and license

See `CITATION.cff` for the ChromBPNet citation. This refactor retains the
original MIT license.
