#!/usr/bin/env python3
"""DeepSHAP agreement of the three engines, with PyTorch as the reference.

The PyTorch engine's profile target weights the mean-normalized logits by
their softmax under ``detach`` -- the target's stop-gradient -- and
tangermeme's DeepLIFT treats those weights as constants. The TensorFlow engine
(the kundajelab shap fork, with the ``StopGradient`` handler it has commented
out registered again) and the JAX engine (``profile_weighting="softmax_x"``)
must compute the same scores. Before that fix both weighted the logits by the
midpoint of the input's and the reference's softmax, as the original
ChromBPNet does.

One bias-architecture BPNet with fixed Keras-layout weights, including
non-zero biases so that ReLUs switch between a sequence and its references;
random sequences; one reference set. Three interpreters, files in between:

``tensorflow`` this package's TensorFlow engine, in ``chrombpnet-tensorflow``:
               builds the model and saves it as a 1.x ``.h5``, records the
               engine's reference stream (``shap_utils.shuffle_several_times``
               on one ``RandomState(seed)``), and scores both heads through
               ``engine.attribute``, the path the CLI takes.
``jax``        the JAX engine, in ``chrombpnet-jax``: loads that ``.h5`` and
               scores both heads through ``engine.attribute``. As a control it
               also scores the profile head with the midpoint weighting
               (``profile_weighting="chrombpnet"``).
``pytorch``    the PyTorch engine's target wrappers with tangermeme, in
               ``chrombpnet-pytorch``: the same weights
               (``assign_keras_layout_weights``) and the TensorFlow engine's
               references, passed explicitly, because ``engine.attribute``
               draws tangermeme's own.

Checks:

* the JAX engine's references equal the TensorFlow engine's bitwise;
* the three models' predictions agree within 1e-4;
* TensorFlow and JAX each agree with PyTorch within relative L2 1e-4, both
  heads;
* the midpoint weighting does not (relative L2 above
  ``tf_engine.MIN_PROFILE_DEPARTURE``), so the comparison tells the two
  weightings apart.

Usage::

    python tests/parity/deepshap_engines.py [--tf-python P] [--jax-python P] [--torch-python P] [--keep]
    CHROMBPNET_DEEPSHAP_ENGINES=1 pytest tests/parity/test_deepshap_engines.py

On a machine where the PyTorch environment has another name, pass it with
``--torch-python``.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OPT_IN = "CHROMBPNET_DEEPSHAP_ENGINES"
SEED = 1234
NUM_SEQS = 4
NUM_SHUFFLES = 20
INPUTLEN = 600
OUTPUTLEN = 200
FILTERS = 8
#: tf_engine._set_bpnet_weights assigns tf_engine.N_DIL_LAYERS layers.
N_DIL_LAYERS = 2
HEADS = ("counts", "profile")
TOLERANCE = 1e-4


def _parity_module(name):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import importlib
    return importlib.import_module("tests.parity." + name)


def _import_this_package():
    """Import chrombpnet from this checkout, and fail if an editable install
    elsewhere answered instead."""
    sys.path.insert(0, str(REPO_ROOT))
    import chrombpnet
    if not str(Path(chrombpnet.__file__).resolve()).startswith(str(REPO_ROOT)):
        raise RuntimeError("worker imported chrombpnet from {}, not {}".format(
            chrombpnet.__file__, REPO_ROOT))


def model_weights():
    """Keras-layout BPNet weights: the canonical Glorot kernels, and biases
    drawn uniformly in [-0.1, 0.1) instead of zeros."""
    from chrombpnet.training.utils.benchmark_utils import canonical_bpnet_weights

    values = canonical_bpnet_weights(FILTERS, N_DIL_LAYERS, SEED)
    rng = np.random.RandomState(SEED + 1)
    for key in sorted(values):
        if key.endswith("_bias"):
            values[key] = rng.uniform(-0.1, 0.1, values[key].shape).astype(np.float32)
    return values


def sequences():
    from chrombpnet.training.utils.one_hot import dna_to_one_hot

    rng = np.random.RandomState(SEED + 2)
    return dna_to_one_hot(["".join(rng.choice(list("ACGT"), INPUTLEN)) for _ in range(NUM_SEQS)])


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------

def tensorflow_worker(workdir):
    _import_this_package()
    tf_engine = _parity_module("tf_engine")
    tf_engine._pin_tensorflow()
    from chrombpnet import engines
    from chrombpnet.engines.tensorflow import shap_utils

    engine = engines.configure("tensorflow")
    values, seqs = model_weights(), sequences()
    args = types.SimpleNamespace(seed=SEED, learning_rate=1e-3,
                                 architecture_from_file=engine.default_architecture("bias"))
    parameters = {"filters": FILTERS, "n_dil_layers": N_DIL_LAYERS, "counts_loss_weight": 1.0,
                  "inputlen": INPUTLEN, "outputlen": OUTPUTLEN}
    model, _ = engine.get_model(args, parameters)
    tf_engine._set_bpnet_weights(model, values, "bpnet_", "")
    path = os.path.join(workdir, "bias.h5")
    model.save(path)
    profile, counts = model.predict(seqs, batch_size=NUM_SEQS)

    # The stream engine.attribute draws, one RandomState(seed) per head.
    rng = np.random.RandomState(SEED)
    references = np.stack([shap_utils.shuffle_several_times([seq], NUM_SHUFFLES, rng=rng)[0]
                           for seq in seqs])
    np.savez(os.path.join(workdir, "inputs.npz"), seqs=seqs, references=references,
             **{"w_" + key: value for key, value in values.items()})

    handle = engine.load_attribution_model(path, None)
    scores = dict(engine.attribute(handle, seqs, list(HEADS), device=None, batch_size=1,
                                   num_shuffles=NUM_SHUFFLES, seed=SEED))
    np.savez(os.path.join(workdir, "tensorflow.npz"), predicted_profile=profile,
             predicted_counts=counts, **scores)


def jax_worker(workdir):
    _import_this_package()
    from chrombpnet import engines
    from chrombpnet.engines.jax.attribution import SequentialReferences
    from chrombpnet.engines.jax.attribution.explainer import DeepLiftShap

    engine = engines.configure("jax")
    seqs = np.load(os.path.join(workdir, "inputs.npz"))["seqs"]
    model = engine.load_attribution_model(os.path.join(workdir, "bias.h5"), "cpu")
    profile, counts = model.predict(seqs, batch_size=NUM_SEQS, verbose=0)
    scores = dict(engine.attribute(model, seqs, list(HEADS), device="cpu",
                                   batch_size=NUM_SHUFFLES, num_shuffles=NUM_SHUFFLES, seed=SEED))
    scores["references"] = SequentialReferences(NUM_SHUFFLES, SEED)(seqs, 0, len(seqs))
    midpoint = DeepLiftShap(model, heads=("profile",), profile_weighting="chrombpnet",
                            num_shuffles=NUM_SHUFFLES)
    scores["profile_midpoint"] = midpoint.explain(
        seqs, references=SequentialReferences(NUM_SHUFFLES, SEED))["profile"]
    np.savez(os.path.join(workdir, "jax.npz"), predicted_profile=profile,
             predicted_counts=counts, **scores)


def pytorch_worker(workdir):
    _import_this_package()
    import torch
    from tangermeme.deep_lift_shap import deep_lift_shap

    from chrombpnet.engines.pytorch.attribution import CountsWrapper, ProfileWrapper
    from chrombpnet.engines.pytorch.model import BPNet, ModelConfig, assign_keras_layout_weights

    torch.set_num_threads(1)
    inputs = np.load(os.path.join(workdir, "inputs.npz"))
    values = {key[2:]: inputs[key] for key in inputs.files if key.startswith("w_")}
    model = BPNet(ModelConfig(input_length=INPUTLEN, output_length=OUTPUTLEN,
                              filters=FILTERS, n_dilation_layers=N_DIL_LAYERS))
    assign_keras_layout_weights(model, values)
    model.eval()
    # NLC -> NCL, as the engine's attribute does; references (N, K, L, 4) -> (N, K, 4, L)
    sequence = torch.as_tensor(inputs["seqs"], dtype=torch.float32).permute(0, 2, 1)
    references = torch.as_tensor(inputs["references"], dtype=torch.float32).permute(0, 1, 3, 2)
    with torch.no_grad():
        profile, counts = model(sequence)
    results = {"predicted_profile": profile.numpy(), "predicted_counts": counts.numpy()}
    for head, wrapper in (("counts", CountsWrapper(model)), ("profile", ProfileWrapper(model))):
        wrapper.eval()
        scores = deep_lift_shap(wrapper, sequence, references=references, hypothetical=True,
                                batch_size=NUM_SHUFFLES, device="cpu")
        results[head] = scores.detach().cpu().permute(0, 2, 1).numpy()
    np.savez(os.path.join(workdir, "pytorch.npz"), **results)


WORKERS = {"tensorflow": tensorflow_worker, "jax": jax_worker, "pytorch": pytorch_worker}


# --------------------------------------------------------------------------
# Comparison and driver
# --------------------------------------------------------------------------

def compare(workdir, report):
    tf_engine = _parity_module("tf_engine")
    inputs, tf, jx, pt = (np.load(os.path.join(workdir, name + ".npz"))
                          for name in ("inputs", "tensorflow", "jax", "pytorch"))
    report.equal("references: JAX engine = TensorFlow engine",
                 jx["references"].astype(np.int8), inputs["references"].astype(np.int8))
    for output in ("profile", "counts"):
        key = "predicted_" + output
        for name, side in (("TensorFlow", tf), ("JAX", jx)):
            report.close("predictions: {} {} vs PyTorch".format(name, output),
                         np.reshape(side[key], pt[key].shape), pt[key], 1e-4)
    for head in HEADS:
        for name, side in (("TensorFlow", tf), ("JAX", jx)):
            report.relative_l2("shap {}: {} vs PyTorch".format(head, name),
                               side[head], pt[head], TOLERANCE)
    distance = tf_engine.relative_l2(jx["profile_midpoint"], pt["profile"])
    report.check("control: midpoint weighting differs from PyTorch",
                 distance > tf_engine.MIN_PROFILE_DEPARTURE,
                 "relative L2 {:.2e} (> {:g})".format(distance, tf_engine.MIN_PROFILE_DEPARTURE))


def _sibling_python(env_name):
    """The python of a sibling conda environment of the one running this."""
    return str(Path(sys.executable).parents[2] / env_name / "bin" / "python")


def _run(python, worker, workdir):
    command = [python, str(Path(__file__).resolve()), "--worker", worker, "--workdir", workdir]
    # CPU only, single-threaded, full float32 in JAX (JAX_PLATFORMS=cpu)
    env = dict(os.environ, **_parity_module("jax_engine").PINNED_ENV)
    env.pop("PYTHONPATH", None)
    log = os.path.join(workdir, "logs", worker + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as handle:
        completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = Path(log).read_text().splitlines()[-30:]
        raise RuntimeError("{} worker failed:\n{}".format(worker, "\n".join(tail)))


def main(arguments):
    workdir = tempfile.mkdtemp(prefix="chrombpnet-deepshap-engines-")
    print("workdir:", workdir)
    try:
        for worker, python in (("tensorflow", arguments.tf_python), ("jax", arguments.jax_python),
                               ("pytorch", arguments.torch_python)):
            print("running the {} engine ...".format(worker))
            _run(python, worker, workdir)
        report = _parity_module("jax_engine").Report()
        compare(workdir, report)
        if report.failed:
            print("\n{} of {} checks failed.".format(len(report.failed), len(report.rows)))
            return 1
        print("\nAll {} checks passed: the TensorFlow and JAX engines' DeepSHAP scores agree "
              "with the PyTorch engine's.".format(len(report.rows)))
        return 0
    finally:
        if arguments.keep:
            print("kept", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument("--tf-python", default=_sibling_python("chrombpnet-tensorflow"))
    result.add_argument("--jax-python", default=_sibling_python("chrombpnet-jax"))
    result.add_argument("--torch-python", default=_sibling_python("chrombpnet-pytorch"))
    result.add_argument("--keep", action="store_true", help="keep the work directory")
    result.add_argument("--worker", choices=sorted(WORKERS), help=argparse.SUPPRESS)
    result.add_argument("--workdir", help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.worker:
        WORKERS[arguments.worker](arguments.workdir)
    else:
        raise SystemExit(main(arguments))
