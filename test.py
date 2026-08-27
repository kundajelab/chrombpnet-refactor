#!/usr/bin/env python3
"""Cross-framework ChromBPNet parity harness.

TensorFlow is executed only in the original ``chrombpnet`` environment and
PyTorch only in the refactor environment. Both workers consume identical
canonical Keras-layout weights and deterministic fixtures.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


DEFAULT_ORIGINAL_REPO = "/path/to/workdir/software/chrombpnet"
DEFAULT_ORIGINAL_PYTHON = (
    "/path/to/workdir/software/miniconda3/envs/chrombpnet/bin/python"
)
ATOL = 1e-6
RTOL = 1e-5


def _canonical_weights(rng, prefix, filters=8):
    values = {
        prefix + "first_kernel": rng.normal(0, 0.04, (21, 4, filters)).astype("float32"),
        prefix + "first_bias": rng.normal(0, 0.02, filters).astype("float32"),
        prefix + "profile_kernel": rng.normal(0, 0.04, (75, filters, 1)).astype("float32"),
        prefix + "profile_bias": rng.normal(0, 0.02, 1).astype("float32"),
        prefix + "count_kernel": rng.normal(0, 0.04, (filters, 1)).astype("float32"),
        prefix + "count_bias": rng.normal(0, 0.02, 1).astype("float32"),
    }
    for layer in (1, 2):
        values[prefix + "dil{}_kernel".format(layer)] = rng.normal(
            0, 0.04, (3, filters, filters)
        ).astype("float32")
        values[prefix + "dil{}_bias".format(layer)] = rng.normal(
            0, 0.02, filters
        ).astype("float32")
    return values


def create_fixture(path):
    rng = np.random.RandomState(2026)
    bases = rng.randint(0, 4, size=(4, 128))
    sequences = np.eye(4, dtype="float32")[bases]
    true_profile = rng.poisson(1.5, size=(4, 20)).astype("float32")
    true_logcounts = np.log1p(true_profile.sum(axis=1, keepdims=True)).astype("float32")
    values = {
        "sequences": sequences,
        "true_profile": true_profile,
        "true_logcounts": true_logcounts,
    }
    values.update(_canonical_weights(rng, "bias_"))
    values.update(_canonical_weights(rng, "tf_"))
    np.savez(path, **values)


def _keras_set_weights(model, values, prefix, no_bias=False):
    root = "wo_bias_bpnet_" if no_bias else "bpnet_"
    model.get_layer(root + "1st_conv").set_weights(
        [values[prefix + "first_kernel"], values[prefix + "first_bias"]]
    )
    for layer in (1, 2):
        model.get_layer(root + "{}conv".format(layer)).set_weights(
            [values[prefix + "dil{}_kernel".format(layer)], values[prefix + "dil{}_bias".format(layer)]]
        )
    profile_name = root + "prof_out_precrop" if no_bias else "prof_out_precrop"
    count_name = root + "logcount_predictions" if no_bias else "logcount_predictions"
    model.get_layer(profile_name).set_weights(
        [values[prefix + "profile_kernel"], values[prefix + "profile_bias"]]
    )
    model.get_layer(count_name).set_weights(
        [values[prefix + "count_kernel"], values[prefix + "count_bias"]]
    )


def reference_worker(args):
    script_dir = str(Path(__file__).resolve().parent)
    sys.path = [args.original_repo] + [
        item for item in sys.path if item and str(Path(item).resolve()) != script_dir
    ]
    import tensorflow as tf
    from chrombpnet.training.models import bpnet_model
    from chrombpnet.training.models import chrombpnet_with_bias_model
    from chrombpnet.training.utils.losses import multinomial_nll
    from chrombpnet.training.utils.one_hot import dna_to_one_hot
    from chrombpnet.training.utils import augment
    from chrombpnet.training.predict import write_predictions_h5py

    values = np.load(args.fixture)
    params = {
        "filters": "8",
        "n_dil_layers": "2",
        "counts_loss_weight": "3.0",
        "inputlen": "128",
        "outputlen": "20",
    }
    model_args = SimpleNamespace(seed=7, learning_rate=0.001)
    bias = bpnet_model.getModelGivenModelOptionsAndWeightInits(model_args, params)
    tf_branch = chrombpnet_with_bias_model.bpnet_model(8, 2, 128, 20)
    _keras_set_weights(bias, values, "bias_", no_bias=False)
    _keras_set_weights(tf_branch, values, "tf_", no_bias=True)
    bias_profile, bias_counts = bias.predict(values["sequences"], verbose=0)
    tf_profile, tf_counts = tf_branch.predict(values["sequences"], verbose=0)
    combined_profile = bias_profile + tf_profile
    combined_counts = tf.reduce_logsumexp(
        tf.concat([tf_counts, bias_counts], axis=-1), axis=-1, keepdims=True
    ).numpy()
    profile_loss = multinomial_nll(values["true_profile"], combined_profile).numpy()
    count_loss = tf.reduce_mean(
        tf.square(values["true_logcounts"] - combined_counts)
    ).numpy()

    np.random.seed(31)
    seq = dna_to_one_hot(["AACCGGTT", "ACGTNNNN"])
    labels = np.arange(12).reshape(2, 6)
    coords = np.array([["chr1", "10", "f", "1"], ["chr2", "20", "f", "0"]])
    cropped = augment.random_crop(seq, labels, 6, 4, coords)
    output_prefix = str(Path(args.output).with_suffix("")) + "_prediction"
    write_predictions_h5py(
        output_prefix,
        np.exp(combined_profile) / np.exp(combined_profile).sum(axis=1, keepdims=True),
        combined_counts[:, 0],
        coords,
    )
    np.savez(
        args.output,
        bias_profile=bias_profile,
        bias_counts=bias_counts,
        tf_profile=tf_profile,
        tf_counts=tf_counts,
        combined_profile=combined_profile,
        combined_counts=combined_counts,
        profile_loss=profile_loss,
        count_loss=count_loss,
        total_loss=profile_loss + 3.0 * count_loss,
        one_hot=seq,
        cropped_seq=cropped[0],
        cropped_labels=cropped[1],
        cropped_coords=cropped[2],
        squeeze_ndim=np.array(np.squeeze(combined_profile[:1]).ndim),
        exp_expm1_gap=np.array(np.exp(combined_counts[0, 0]) - np.expm1(combined_counts[0, 0])),
    )


def _torch_assign(model, values, prefix):
    import torch

    with torch.no_grad():
        model.first_conv.weight.copy_(torch.from_numpy(values[prefix + "first_kernel"]).permute(2, 1, 0))
        model.first_conv.bias.copy_(torch.from_numpy(values[prefix + "first_bias"]))
        for index, conv in enumerate(model.dilated_convs, start=1):
            conv.weight.copy_(
                torch.from_numpy(values[prefix + "dil{}_kernel".format(index)]).permute(2, 1, 0)
            )
            conv.bias.copy_(torch.from_numpy(values[prefix + "dil{}_bias".format(index)]))
        model.profile_conv.weight.copy_(
            torch.from_numpy(values[prefix + "profile_kernel"]).permute(2, 1, 0)
        )
        model.profile_conv.bias.copy_(torch.from_numpy(values[prefix + "profile_bias"]))
        model.count_dense.weight.copy_(torch.from_numpy(values[prefix + "count_kernel"]).T)
        model.count_dense.bias.copy_(torch.from_numpy(values[prefix + "count_bias"]))


def torch_worker(args):
    sys.path.insert(0, args.refactor_repo)
    import torch
    from chrombpnet.model import BPNet, BiasModel, ChromBPNet, ModelConfig
    from chrombpnet.training.utils.losses import chrombpnet_loss
    from chrombpnet.training.utils.one_hot import dna_to_one_hot
    from chrombpnet.training.utils import augment
    from chrombpnet.training.predict import write_predictions_h5py

    values = np.load(args.fixture)
    config = ModelConfig(128, 20, 8, 2, 3.0)
    bias = BiasModel(config)
    tf_branch = BPNet(config)
    _torch_assign(bias, values, "bias_")
    _torch_assign(tf_branch, values, "tf_")
    model = ChromBPNet(config, bias, tf_branch)
    x = torch.from_numpy(values["sequences"]).permute(0, 2, 1)
    with torch.no_grad():
        bias_profile, bias_counts = bias(x)
        tf_profile, tf_counts = tf_branch(x)
        combined_profile, combined_counts = model(x)
        total, profile_loss, count_loss = chrombpnet_loss(
            combined_profile,
            combined_counts,
            torch.from_numpy(values["true_profile"]),
            torch.from_numpy(values["true_logcounts"]),
            3.0,
        )
    arrays = [item.numpy() for item in (bias_profile, bias_counts, tf_profile, tf_counts, combined_profile, combined_counts)]
    np.random.seed(31)
    seq = dna_to_one_hot(["AACCGGTT", "ACGTNNNN"])
    labels = np.arange(12).reshape(2, 6)
    coords = np.array([["chr1", "10", "f", "1"], ["chr2", "20", "f", "0"]])
    cropped = augment.random_crop(seq, labels, 6, 4, coords)
    probability = np.exp(arrays[4]) / np.exp(arrays[4]).sum(axis=1, keepdims=True)
    output_prefix = str(Path(args.output).with_suffix("")) + "_prediction"
    write_predictions_h5py(output_prefix, probability, arrays[5][:, 0], coords)
    np.savez(
        args.output,
        bias_profile=arrays[0], bias_counts=arrays[1],
        tf_profile=arrays[2], tf_counts=arrays[3],
        combined_profile=arrays[4], combined_counts=arrays[5],
        profile_loss=np.array(profile_loss), count_loss=np.array(count_loss), total_loss=np.array(total),
        one_hot=seq, cropped_seq=cropped[0], cropped_labels=cropped[1], cropped_coords=cropped[2],
        squeeze_ndim=np.array(np.squeeze(arrays[4][:1]).ndim),
        exp_expm1_gap=np.array(np.exp(arrays[5][0, 0]) - np.expm1(arrays[5][0, 0])),
    )


def _run_worker(python, script, mode, common, env):
    command = [python, script, mode]
    for name, value in common.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    subprocess.run(command, check=True, env=env)


def _compare_h5(left, right, atol, rtol):
    import h5py

    with h5py.File(left) as a, h5py.File(right) as b:
        assert set(a) == set(b)
        for group in a:
            assert set(a[group]) == set(b[group])
            for dataset in a[group]:
                av = a[group][dataset][()]
                bv = b[group][dataset][()]
                if av.dtype.kind in "SUO":
                    av = av.astype("S")
                    bv = bv.astype("S")
                    if not np.array_equal(av, bv):
                        raise AssertionError("HDF5 mismatch: {}/{}".format(group, dataset))
                else:
                    np.testing.assert_allclose(av, bv, atol=atol, rtol=rtol)


def main(args):
    original_python = Path(args.original_python)
    original_repo = Path(args.original_repo).resolve()
    refactor_repo = Path(args.refactor_repo or Path(__file__).resolve().parent).resolve()
    if not original_python.is_file():
        raise SystemExit("Original interpreter not found: {}".format(original_python))
    if not (original_repo / "chrombpnet").is_dir():
        raise SystemExit("Original repository not found: {}".format(original_repo))

    temporary = Path(tempfile.mkdtemp(prefix="chrombpnet-parity-"))
    try:
        fixture = temporary / "fixture.npz"
        reference = temporary / "reference.npz"
        torch_result = temporary / "torch.npz"
        create_fixture(fixture)
        base_env = os.environ.copy()
        base_env["CUDA_VISIBLE_DEVICES"] = ""
        reference_env = base_env.copy()
        reference_env["PYTHONPATH"] = str(original_repo)
        torch_env = base_env.copy()
        torch_env["PYTHONPATH"] = str(refactor_repo)
        script = str(Path(__file__).resolve())
        _run_worker(
            str(original_python), script, "--reference-worker",
            {"fixture": fixture, "output": reference, "original_repo": original_repo},
            reference_env,
        )
        _run_worker(
            args.torch_python or sys.executable, script, "--torch-worker",
            {"fixture": fixture, "output": torch_result, "refactor_repo": refactor_repo},
            torch_env,
        )

        expected = np.load(reference)
        actual = np.load(torch_result)
        names = [
            "bias_profile", "bias_counts", "tf_profile", "tf_counts",
            "combined_profile", "combined_counts", "profile_loss", "count_loss",
            "total_loss", "one_hot", "cropped_seq", "cropped_labels", "cropped_coords",
            "squeeze_ndim", "exp_expm1_gap",
        ]
        failures = []
        print("component                         max_abs_error  result")
        print("-------------------------------- ------------- -------")
        for name in names:
            try:
                if expected[name].dtype.kind in "SUO":
                    np.testing.assert_array_equal(expected[name], actual[name])
                    error = 0.0
                else:
                    error = float(np.max(np.abs(expected[name] - actual[name])))
                    np.testing.assert_allclose(expected[name], actual[name], atol=args.atol, rtol=args.rtol)
                status = "PASS"
            except AssertionError as exc:
                failures.append((name, str(exc)))
                status = "FAIL"
                error = float("nan")
            print("{:<32} {:>13.3g} {}".format(name, error, status))
        try:
            _compare_h5(
                str(reference.with_suffix("")) + "_prediction_predictions.h5",
                str(torch_result.with_suffix("")) + "_prediction_predictions.h5",
                args.atol,
                args.rtol,
            )
            print("{:<32} {:>13.3g} {}".format("prediction_hdf5", 0.0, "PASS"))
        except AssertionError as exc:
            failures.append(("prediction_hdf5", str(exc)))
            print("{:<32} {:>13} {}".format("prediction_hdf5", "nan", "FAIL"))
        if failures:
            for name, detail in failures:
                print("\n{}: {}".format(name, detail), file=sys.stderr)
            return 1
        print("\nAll TensorFlow/PyTorch parity checks passed (atol={}, rtol={}).".format(args.atol, args.rtol))
        return 0
    finally:
        if args.keep_temp:
            print("Fixtures retained at {}".format(temporary))
        else:
            shutil.rmtree(str(temporary))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--reference-worker", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--torch-worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--fixture")
    result.add_argument("--output")
    result.add_argument("--original-repo", default=DEFAULT_ORIGINAL_REPO)
    result.add_argument("--original-python", default=DEFAULT_ORIGINAL_PYTHON)
    result.add_argument("--refactor-repo", default=None)
    result.add_argument("--torch-python", default=None)
    result.add_argument("--atol", type=float, default=ATOL)
    result.add_argument("--rtol", type=float, default=RTOL)
    result.add_argument("--keep-temp", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.reference_worker:
        reference_worker(arguments)
    elif arguments.torch_worker:
        torch_worker(arguments)
    else:
        raise SystemExit(main(arguments))
