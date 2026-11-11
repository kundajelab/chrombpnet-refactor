#!/usr/bin/env python3
"""Agreement of the TensorFlow engine with the original Keras ChromBPNet.

The tensorflow-old engine (``chrombpnet/engines/tensorflow_old``) claims to reproduce
the original implementation (kundajelab/chrombpnet@09938fd), given the same
inputs. "The same inputs" is deliberately narrow: the data pipeline,
preprocessing and seeding are this package's and may differ, so this harness
feeds both implementations identical batches, identical initial weights and
identical DeepSHAP references, and then compares every output.

The two run different TensorFlows: the original its own pinned stack (TF 2.8
on Python 3.8, its own conda environment, ``--reference-python``), the engine
TF 2.21 with tf-keras on Python 3.12 (this environment). Their CPU kernels
round differently, so outputs are compared within ``TOLERANCE`` -- the
relative L2 distance of each array -- and every check says whether it came
out bitwise equal or by how much it differs. Model configurations, which the
two Keras releases write with different fields, are compared by structure:
each layer's class, name and inputs.

DeepSHAP is compared in both of the engine's modes. By default it scores as
the original does, whose shap fork has the ``StopGradient`` handler commented
out (``deep_tf.py:595``). With ``--contribs-profile-stopgrad`` it registers
that handler for the profile head; the harness registers it on the reference
side too, and the two must still agree. The handler must change the profile
head and leave the counts head alone.

Each side runs in its own processes, because both code bases are the package
``chrombpnet``:

``prepare``   this package: builds the regression fixture, records a batch
              stream from the shared generator and draws initial weights.
``bias_init`` the original: writes the bias model the bias-factorized model
              starts from, as a 1.x checkpoint the engine must load.
``reference`` the original repository on ``sys.path``, nothing of this one,
              in the original's environment.
``engine``    this package's TensorFlow engine.

Each training run is its own process, as it is on the command line, so that
TensorFlow's process-global state (seeds, layer-name counters, graph mode)
cannot leak between cases. The harness -- not either implementation -- pins
TensorFlow to one thread, for both sides alike. It does not set
``TF_DETERMINISTIC_OPS``: under it TF refuses any unseeded random op, and the
original builds its bias-factorized model before seeding. Single-threaded CPU
execution is deterministic instead, which the self-check confirms.

What is compared:

* ``build``   model structure, predictions of the bias, combined and no-bias
              models, and ``test_on_batch`` losses;
* ``train``   checkpoint weights and Adam state, the ``.log`` and
              ``.log.batch`` losses, and the no-bias model, for a bias model
              and a bias-factorized model where early stopping fires, and a
              bias-factorized model that runs out of epochs. The profile
              heads' output biases are left out (``UNIDENTIFIED_WEIGHTS``):
              the loss cannot see them, so their values are rounding noise;
* ``rescale`` the hyperparameter finder's bias count-head rescale;
* ``shap``    DeepSHAP hypothetical scores for both heads as the original
              computes them, and with the StopGradient handler on both sides
              (``--contribs-profile-stopgrad``), which must leave the counts
              head unchanged and change the profile head;
* ``interop`` each side loading and predicting with the other side's
              models: the engine reads 1.x's Python 3.8 count head, the
              original reads the one the engine writes back in that form, and
              loads the engine's bias model compiled, as its ``chrombpnet
              train -b`` does; and the two sides' trained models predicting
              alike (profiles as log-probabilities).

The reference is first checked against itself, exactly, since an agreement
claim means nothing if the reference does not reproduce its own output; so
are the engine's in-process and CLI DeepSHAP paths.

Usage::

    python tests/parity/tf_engine.py [--original-repo PATH] [--keep]
    CHROMBPNET_TF_ENGINE_PARITY=1 pytest tests/parity
"""

import argparse
import contextlib
import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
#: A checkout of the original (kundajelab/chrombpnet, ORIGINAL_COMMIT): beside this
#: repository by default. Override with --original-repo or CHROMBPNET_ORIGINAL_REPO.
DEFAULT_ORIGINAL_REPO = os.environ.get("CHROMBPNET_ORIGINAL_REPO", str(REPO_ROOT.parent / "chrombpnet"))
#: The original's interpreter: its own conda environment, ``chrombpnet``, beside
#: this one (Python 3.8, TF 2.8, the kundajelab shap fork). Override with
#: --reference-python or CHROMBPNET_ORIGINAL_PYTHON.
DEFAULT_REFERENCE_PYTHON = os.environ.get(
    "CHROMBPNET_ORIGINAL_PYTHON",
    str(Path(sys.prefix).parent / "chrombpnet" / "bin" / "python"))
ORIGINAL_COMMIT = "09938fdb4397ec0006510e5251e48920a505d4de"
PARITY_OPT_IN = "CHROMBPNET_TF_ENGINE_PARITY"

SEED = 1234
SHAP_SEED = 1234
EPOCHS = 6
BATCH_SIZE = 8
FILTERS = 8
N_DIL_LAYERS = 2
HEADS = ("counts", "profile")
TRACKABLES = [
    "logcount_predictions_loss", "loss", "logits_profile_predictions_loss",
    "val_logcount_predictions_loss", "val_loss",
    "val_logits_profile_predictions_loss",
]

#: (case, model kind, epochs, early_stop, learning_rate)
TRAINING_CASES = (
    ("bias_A", "bias", EPOCHS, 1, 0.01),
    ("chrombpnet_A", "chrombpnet", EPOCHS, 1, 0.1),
    # Runs out of epochs with its best validation epoch before the last, so
    # restoring the best epoch is observable.
    ("chrombpnet_B", "chrombpnet", 5, 10, 0.1),
)

#: Environment both sides run in. Thread and determinism pins are the
#: harness's, applied identically to both implementations.
PINNED_ENV = {
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "",
    # Also set through tf.config in each worker; the environment variables are
    # what reach the attribution subprocess the engine spawns.
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
}


#: Largest relative L2 distance (||engine - original|| / ||original||) each
#: kind of output may show. Measured with TF 2.21 against TF 2.8 on one CPU
#: thread (each run prints its own): predictions of untrained models, the
#: bias rescale and each side's loading of the other's models come out
#: bitwise equal or within 1e-7; DeepSHAP scores within 2.1e-6; five epochs
#: of training within 1.2e-7 (losses), 1.1e-6 (Adam state) and 4.6e-6
#: (weights). The limits leave a factor of 20 or more, and stay well below
#: what a real departure shows: a different batch order moved the trained
#: bias model's weights by 0.3.
TOLERANCE = {
    "build": 1e-5,
    "train": 1e-4,
    "rescale": 1e-5,
    "shap": 1e-4,
    "interop": 1e-5,
}

#: Relative L2 distance between the profile scores with the StopGradient
#: honoured and without it (the original's midpoint weights) above which the
#: two weightings count as told apart. Measured: 1.05 here and in jax_engine.py, 0.64 in
#: deepshap_engines.py; float32 agreement is 1e-6 or better.
MIN_PROFILE_DEPARTURE = 1e-3

#: Weights the loss cannot see, left out of the weight and optimizer-state
#: checks: the bias of a profile head's output convolution adds one constant
#: to every logit of a profile, which the multinomial loss (a softmax) does
#: not depend on. Its gradient is rounding noise, which Adam scales up to
#: full-sized steps, so each implementation walks it at random from the same
#: start. Profile predictions are compared as log-probabilities, which it does
#: not move.
UNIDENTIFIED_WEIGHTS = ("prof_out_precrop/bias",)

#: Models each side loads from both sides' training runs: (key, case, file,
#: compiled). The bias model is loaded compiled, as chrombpnet 1.x's
#: ``chrombpnet train -b`` and hyperparameter finder load it, so its saved
#: optimizer must be one Keras 2.8 can rebuild.
INTEROP_MODELS = (
    ("chrombpnet", "chrombpnet_A", "chrombpnet.h5", False),
    ("bias", "bias_A", "bias.h5", True),
)


# --------------------------------------------------------------------------
# Shared by both workers: they import this file, never each other's package.
# --------------------------------------------------------------------------

@contextlib.contextmanager
def honour_stop_gradient():
    """Register the shap fork's ``break_dependence`` for ``StopGradient`` --
    the line it has commented out, ``deep_tf.py:595`` -- and restore the
    previous entry on exit. The harness's own copy of what the engine does
    (``engines/tensorflow_old/attribution.honour_stop_gradient``), for the
    reference worker, which must not import this package.
    """
    from shap.explainers.deep import deep_tf

    missing = object()
    previous = deep_tf.op_handlers.get("StopGradient", missing)
    deep_tf.op_handlers["StopGradient"] = deep_tf.break_dependence
    try:
        yield
    finally:
        if previous is missing:
            del deep_tf.op_handlers["StopGradient"]
        else:
            deep_tf.op_handlers["StopGradient"] = previous


def relative_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _pin_tensorflow():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def _parameters(kind, workdir):
    parameters = {
        "filters": str(FILTERS),
        "n_dil_layers": str(N_DIL_LAYERS),
        "counts_loss_weight": "2.0",
        "inputlen": None,
        "outputlen": None,
    }
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    parameters["inputlen"] = str(int(fixture["inputlen"]))
    parameters["outputlen"] = str(int(fixture["outputlen"]))
    if kind == "chrombpnet":
        parameters["bias_model_path"] = os.path.join(workdir, "bias_init.h5")
    return parameters


def _set_bpnet_weights(model, values, root, head):
    """Assign Keras-layout BPNet arrays by layer name."""
    model.get_layer(root + "1st_conv").set_weights(
        [values["first_kernel"], values["first_bias"]])
    for layer in range(1, N_DIL_LAYERS + 1):
        model.get_layer("{}{}conv".format(root, layer)).set_weights(
            [values["dilated_{}_kernel".format(layer)],
             values["dilated_{}_bias".format(layer)]])
    model.get_layer(head + "prof_out_precrop").set_weights(
        [values["profile_kernel"], values["profile_bias"]])
    model.get_layer(head + "logcount_predictions").set_weights(
        [values["count_kernel"], values["count_bias"]])


def _initial_weights(fixture, prefix):
    return {key[len(prefix):]: fixture[key]
            for key in fixture.files if key.startswith(prefix)}


def _inject(model, kind, fixture):
    """Identical starting weights for the trainable BPNet, on both sides."""
    if kind == "bias":
        _set_bpnet_weights(model, _initial_weights(fixture, "bias_w_"), "bpnet_", "")
    else:
        _set_bpnet_weights(model.get_layer("model_wo_bias"),
                           _initial_weights(fixture, "tf_w_"),
                           "wo_bias_bpnet_", "wo_bias_bpnet_")


def _replay_class():
    from tensorflow import keras

    class Replay(keras.utils.Sequence):
        """Replays a recorded batch stream, one recorded epoch per epoch."""

        def __init__(self, fixture, case, split):
            self.fixture = fixture
            self.prefix = "{}_{}".format(case, split)
            self.n_batches = int(fixture[self.prefix + "_n_batches"])
            self.n_epochs = int(fixture[self.prefix + "_n_epochs"])
            self.epoch = 0

        def __len__(self):
            return self.n_batches

        def __getitem__(self, index):
            key = "{}_e{}_b{}_".format(self.prefix, self.epoch, index)
            return (self.fixture[key + "x"],
                    [self.fixture[key + "p"], self.fixture[key + "c"]])

        def on_epoch_end(self):
            self.epoch = min(self.epoch + 1, self.n_epochs - 1)

    return Replay


def _training_args(case, epochs, early_stop, learning_rate, out, architecture):
    return types.SimpleNamespace(
        seed=SEED, learning_rate=learning_rate, epochs=epochs,
        early_stop=early_stop, trackables=list(TRACKABLES),
        output_prefix=os.path.join(out, "chrombpnet" if "chrombpnet" in case else "bias"),
        architecture_from_file=architecture, batch_size=BATCH_SIZE)


# --------------------------------------------------------------------------
# prepare: this package, shared generator
# --------------------------------------------------------------------------

def prepare(workdir):
    sys.path.insert(0, str(REPO_ROOT))
    import chrombpnet  # noqa: F401  (libstdc++ preload)
    from tests.regression import fixture as regression_fixture
    from chrombpnet.training.data_generators import initializers
    from chrombpnet.training.utils.benchmark_utils import canonical_bpnet_weights

    paths = regression_fixture.build(Path(workdir) / "fixture")
    inputlen, outputlen = regression_fixture.INPUTLEN, regression_fixture.OUTPUTLEN
    arrays = {"inputlen": np.array(inputlen), "outputlen": np.array(outputlen)}

    def record(case, peaks, nonpeaks, max_jitter, ratio):
        args = types.SimpleNamespace(
            data_type="ATAC", data_aug=None, chr_fold_path=str(paths["folds"]),
            peaks=peaks, nonpeaks=nonpeaks, genome=str(paths["genome"]),
            bigwig=str(paths["bigwig"]), batch_size=BATCH_SIZE, seed=SEED,
            inputlen=inputlen, outputlen=outputlen)
        parameters = {"inputlen": inputlen, "outputlen": outputlen,
                      "max_jitter": max_jitter, "negative_sampling_ratio": ratio}
        np.random.seed(SEED)
        for split, epochs in (("train", EPOCHS), ("valid", 1)):
            generator = initializers.initialize_generators(
                args, split, parameters, return_coords=False)
            prefix = "{}_{}".format(case, split)
            arrays[prefix + "_n_batches"] = np.array(len(generator))
            arrays[prefix + "_n_epochs"] = np.array(epochs)
            for epoch in range(epochs):
                if len(generator) != int(arrays[prefix + "_n_batches"]):
                    raise RuntimeError("batch count changed between epochs")
                for index in range(len(generator)):
                    x, (profile, logcounts) = generator[index]
                    key = "{}_e{}_b{}_".format(prefix, epoch, index)
                    arrays[key + "x"] = np.asarray(x, dtype=np.float32)
                    arrays[key + "p"] = np.asarray(profile, dtype=np.float32)
                    arrays[key + "c"] = np.asarray(logcounts, dtype=np.float32)
                generator.on_epoch_end()

    record("bias", "None", str(paths["nonpeaks"]), 0, 1.0)
    record("chrombpnet", str(paths["peaks"]), str(paths["nonpeaks"]),
           regression_fixture.MAX_JITTER, 0.1)
    for prefix, seed in (("bias_w_", SEED), ("tf_w_", SEED + 1)):
        for key, value in canonical_bpnet_weights(FILTERS, N_DIL_LAYERS, seed).items():
            arrays[prefix + key] = value

    # Inputs for prediction, rescale and DeepSHAP: validation sequences, which
    # are ACGT-only (the fixture genome has no N).
    valid = [arrays["chrombpnet_valid_e0_b{}_x".format(i)]
             for i in range(int(arrays["chrombpnet_valid_n_batches"]))]
    valid = np.concatenate(valid)
    arrays["predict_x"] = valid[:16]
    arrays["shap_x"] = valid[:3]
    bias_valid = np.concatenate(
        [arrays["bias_valid_e0_b{}_x".format(i)]
         for i in range(int(arrays["bias_valid_n_batches"]))])
    bias_counts = np.concatenate(
        [arrays["bias_valid_e0_b{}_p".format(i)].sum(-1)
         for i in range(int(arrays["bias_valid_n_batches"]))])
    arrays["rescale_x"], arrays["rescale_cts"] = bias_valid, bias_counts
    np.savez(os.path.join(workdir, "fixture.npz"), **arrays)


def task_bias_init(side, workdir):
    """The starting bias model for the bias-factorized cases: the original's
    bias architecture with the shared weights, saved by the original -- a 1.x
    checkpoint, which the engine must read as it reads any."""
    _pin_tensorflow()
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    args = _training_args("bias", 1, 1, 0.001, workdir, side.architecture("bias"))
    model, _ = side.get_model(args, _parameters("bias", workdir))
    _set_bpnet_weights(model, _initial_weights(fixture, "tf_w_"), "bpnet_", "")
    model.save(os.path.join(workdir, "bias_init.h5"))


# --------------------------------------------------------------------------
# The two implementations behind one small interface
# --------------------------------------------------------------------------

class Reference:
    """kundajelab/chrombpnet@09938fd, as its own CLI would call it."""

    name = "reference"

    def __init__(self, original_repo):
        excluded = str(REPO_ROOT)
        sys.path[:] = [original_repo] + [
            item for item in sys.path
            if item and not str(Path(item).resolve()).startswith(excluded)]
        self.repo = original_repo
        import chrombpnet
        if not str(Path(chrombpnet.__file__).resolve()).startswith(str(Path(original_repo).resolve())):
            raise RuntimeError("reference worker imported the wrong chrombpnet: "
                               + chrombpnet.__file__)

    def architecture(self, kind):
        name = "bpnet_model.py" if kind == "bias" else "chrombpnet_with_bias_model.py"
        return os.path.join(self.repo, "chrombpnet", "training", "models", name)

    def get_model(self, args, parameters):
        from chrombpnet.training import train
        np.random.seed(args.seed)  # train.main, immediately before get_model
        return train.get_model(args, parameters)

    def fit(self, model, train_gen, valid_gen, args, parameters, architecture):
        from chrombpnet.training import train
        train.fit_and_evaluate(model, train_gen, valid_gen, args, architecture)

    def load_predictor(self, path):
        from chrombpnet.training import predict
        return predict.load_model_wrapper(types.SimpleNamespace(model_h5=path))

    def load_compiled(self, path):
        from chrombpnet.helpers.hyperparameters import param_utils
        return param_utils.load_model_wrapper(path)

    def rescale(self, bias_path, out, seqs, cts):
        from chrombpnet.helpers.hyperparameters import find_chrombpnet_hyperparams as finder
        from chrombpnet.helpers.hyperparameters import param_utils
        bias_model = param_utils.load_model_wrapper(bias_path)
        finder.adjust_bias_model_logcounts(bias_model, seqs, cts).save(out)

    def shap(self, model_path, seqs, head, seed, stop_gradient=False):
        """``stop_gradient``: with the harness's StopGradient handler, as the
        engine runs under --contribs-profile-stopgrad; without it, the
        original exactly."""
        # Importing interpret disables eager execution, as the original's CLI
        # did at startup; it must happen before the model is loaded.
        from chrombpnet.evaluation.interpret import interpret, input_utils, shap_utils
        import deeplift.dinuc_shuffle

        _pin_tensorflow()
        model = input_utils.load_model_wrapper(types.SimpleNamespace(model_h5=model_path))
        captured = {}
        original = interpret.generate_shap_dict

        def capture(seqs_, scores):
            captured["scores"] = np.array(scores)
            return original(seqs_, scores)

        interpret.generate_shap_dict = capture
        # The original draws its references from an unseeded RandomState;
        # give it the stream the engine seeds, fresh for each head.
        shap_utils.dinuc_shuffle = functools.partial(
            deeplift.dinuc_shuffle.dinuc_shuffle, rng=np.random.RandomState(seed))
        out = tempfile.mkdtemp()
        with honour_stop_gradient() if stop_gradient else contextlib.nullcontext():
            interpret.interpret(model, seqs, os.path.join(out, "scores"), [head])
        shutil.rmtree(out)
        return captured["scores"]


class EngineSide:
    """This package's TensorFlow engine."""

    name = "engine"

    def __init__(self):
        sys.path.insert(0, str(REPO_ROOT))
        import chrombpnet  # noqa: F401
        from chrombpnet import engines
        self.engine = engines.configure("tensorflow-old")

    def architecture(self, kind):
        return self.engine.default_architecture(kind)

    def get_model(self, args, parameters):
        return self.engine.get_model(args, parameters)

    def fit(self, model, train_gen, valid_gen, args, parameters, architecture):
        self.engine.fit(model, train_gen, valid_gen, args, parameters, architecture)

    def load_predictor(self, path):
        return self.engine.load_predictor(path)

    def load_compiled(self, path):
        from chrombpnet.engines.tensorflow_old.predictor import load_model
        return load_model(path, compile=True)

    def rescale(self, bias_path, out, seqs, cts):
        self.engine.rescale_bias_logcounts(bias_path, out, seqs, cts)

    def shap(self, model_path, seqs, head, seed, stop_gradient=False):
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        _pin_tensorflow()
        from chrombpnet.engines.tensorflow_old import attribution
        from chrombpnet.engines.tensorflow_old.predictor import load_model

        model = load_model(model_path, compile=False)
        # The engine seeds its own references, a fresh RandomState(seed) per
        # head; the reference worker is given the same stream.
        return attribution.explain(model, seqs, [head], 20, seed,
                                   stop_gradient=stop_gradient)[head]

    def shap_end_to_end(self, model_path, seqs, seed, stop_gradient=False):
        """The CLI's path: a spawned graph-mode child, both heads in one go."""
        handle = self.engine.load_attribution_model(model_path, None)
        return dict(self.engine.attribute(
            handle, seqs, list(HEADS), device=None, batch_size=1,
            num_shuffles=20, seed=seed, profile_stop_gradient=stop_gradient))


# --------------------------------------------------------------------------
# Tasks: each runs in its own process, for one side
# --------------------------------------------------------------------------

def task_build(side, workdir, out):
    tf = _pin_tensorflow()
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    results, configs = {}, {}
    for kind in ("bias", "chrombpnet"):
        args = _training_args(kind, 1, 1, 0.001, out, side.architecture(kind))
        model, architecture = side.get_model(args, _parameters(kind, workdir))
        _inject(model, kind, fixture)
        configs[kind] = model.to_json()
        profile, counts = model.predict(fixture["predict_x"], batch_size=BATCH_SIZE)
        results[kind + "_profile"], results[kind + "_counts"] = profile, counts
        batch = "{}_train_e0_b0_".format(kind)
        losses = model.test_on_batch(
            fixture[batch + "x"], [fixture[batch + "p"], fixture[batch + "c"]])
        results[kind + "_losses"] = np.array(losses)
        if kind == "chrombpnet":
            nobias = model.get_layer("model_wo_bias")
            profile, counts = nobias.predict(fixture["predict_x"], batch_size=BATCH_SIZE)
            results["nobias_profile"], results["nobias_counts"] = profile, counts
        tf.keras.backend.clear_session()
    np.savez(os.path.join(out, "build.npz"), **results)
    Path(out, "configs.json").write_text(json.dumps(configs, sort_keys=True))


def task_train(side, workdir, out, case):
    _pin_tensorflow()
    _, kind, epochs, early_stop, learning_rate = next(c for c in TRAINING_CASES if c[0] == case)
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    os.makedirs(out, exist_ok=True)
    args = _training_args(case, epochs, early_stop, learning_rate, out, side.architecture(kind))
    parameters = _parameters(kind, workdir)
    model, architecture = side.get_model(args, parameters)
    _inject(model, kind, fixture)
    if side.name == "reference":
        # Keep the best-epoch checkpoint that fit_and_evaluate's final
        # model.save overwrites. ModelCheckpoint passes keyword arguments;
        # the final save does not.
        save = model.save

        def save_and_snapshot(path, *a, **k):
            if not a and not k and os.path.exists(path):
                shutil.copyfile(path, args.output_prefix + ".best.h5")
            return save(path, *a, **k)

        model.save = save_and_snapshot
    Replay = _replay_class()
    side.fit(model, Replay(fixture, kind, "train"), Replay(fixture, kind, "valid"),
             args, parameters, architecture)


def task_rescale(side, workdir, out):
    _pin_tensorflow()
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    side.rescale(os.path.join(workdir, "bias_init.h5"),
                 os.path.join(out, "bias_model_scaled.h5"),
                 fixture["rescale_x"], fixture["rescale_cts"])


def task_shap(side, workdir, out):
    """Both heads without the StopGradient handler (the default, the original)
    and with it (``stopgrad_``, --contribs-profile-stopgrad)."""
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    model = os.path.join(workdir, "bias_init.h5")
    scores = {}
    for head in HEADS:
        scores[head] = side.shap(model, fixture["shap_x"], head, SHAP_SEED)
        scores["stopgrad_" + head] = side.shap(model, fixture["shap_x"], head, SHAP_SEED,
                                               stop_gradient=True)
    if side.name == "engine":
        for prefix, stop_gradient in (("", False), ("stopgrad_", True)):
            end_to_end = side.shap_end_to_end(model, fixture["shap_x"], SHAP_SEED, stop_gradient)
            for head in HEADS:
                scores["end_to_end_" + prefix + head] = end_to_end[head]
    np.savez(os.path.join(out, "shap.npz"), **scores)


def task_interop(side, workdir, out, sides):
    _pin_tensorflow()
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    results = {}
    for name in sides:
        for key, case, filename, compiled in INTEROP_MODELS:
            path = os.path.join(workdir, name, case, filename)
            model = side.load_compiled(path) if compiled else side.load_predictor(path)
            x = fixture["predict_x"] if key == "chrombpnet" else fixture["rescale_x"][:16]
            profile, counts = model.predict(x, batch_size=BATCH_SIZE)
            results["{}_{}_profile".format(name, key)] = profile
            results["{}_{}_counts".format(name, key)] = counts
    np.savez(os.path.join(out, "interop.npz"), **results)


def worker(arguments):
    side = Reference(arguments.original_repo) if arguments.side == "reference" else EngineSide()
    workdir, out = arguments.workdir, arguments.out
    os.makedirs(out, exist_ok=True)
    if arguments.task == "bias_init":
        task_bias_init(side, workdir)
    elif arguments.task == "build":
        task_build(side, workdir, out)
    elif arguments.task == "train":
        task_train(side, workdir, os.path.join(out, arguments.case), arguments.case)
    elif arguments.task == "rescale":
        task_rescale(side, workdir, out)
    elif arguments.task == "shap":
        task_shap(side, workdir, out)
    elif arguments.task == "interop":
        task_interop(side, workdir, out, arguments.sides)
    else:
        raise ValueError(arguments.task)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _structure(model_json):
    """[(class, name, input layers)] of every layer, nested models included:
    what must match between two Keras releases, whose configs otherwise carry
    their own fields. A Lambda is reduced to whether it is the logsumexp count
    head, recognised without unmarshalling it -- its bytecode is the Python
    that wrote it, which need not be the one running this."""
    sys.path.insert(0, str(REPO_ROOT))
    from chrombpnet.legacy_lambda import is_logsumexp_lambda

    rows = []

    def walk(model):
        for layer in model["config"]["layers"]:
            inputs = sorted({inbound[0] for node in layer.get("inbound_nodes", [])
                             for inbound in (node if isinstance(node, list) else [])
                             if isinstance(inbound, list) and inbound})
            kind = layer["class_name"]
            if kind == "Lambda":
                kind += ":logsumexp" if is_logsumexp_lambda(layer["config"]) else ":other"
            rows.append((kind, layer["config"].get("name"), inputs))
            if layer["class_name"] in ("Functional", "Model"):
                walk(layer)

    walk(json.loads(model_json))
    return rows


def _attr_value(key, value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    return repr(value)


def h5_contents(path, root="/", key=None):
    """{name: (dtype, shape, bytes or attrs)} for every group and dataset."""
    import h5py

    contents = {}
    with h5py.File(path, "r") as handle:
        top = handle[root]

        def visit(name, node):
            label = key(name) if key else name
            attrs = tuple((k, _attr_value(k, node.attrs[k])) for k in sorted(node.attrs))
            if isinstance(node, h5py.Dataset):
                contents[label] = (str(node.dtype), node.shape, node[()].tobytes(), attrs)
            elif not key:
                contents[label] = ("group", attrs)

        if not key:
            contents["/"] = ("group", tuple((k, _attr_value(k, top.attrs[k]))
                                            for k in sorted(top.attrs)))
        top.visititems(visit)
    return contents


def _weight_arrays(path, root="model_weights", depth=2):
    """Arrays under ``root`` keyed by the last ``depth`` parts of their path
    (``layer/variable`` whatever the nesting), or the whole path if ``depth``
    is None."""
    import h5py

    arrays = {}
    with h5py.File(path, "r") as handle:
        if root not in handle:
            return arrays

        def visit(name, node):
            if isinstance(node, h5py.Dataset):
                arrays["/".join(name.split("/")[-depth:]) if depth else name] = node[()]

        handle[root].visititems(visit)
    return arrays


def _table(path, sep):
    """(header, rows as floats) of a CSVLogger log or a LossHistory batch log."""
    lines = Path(path).read_text().strip().splitlines()
    header = lines[0].split(sep)
    rows = [[float("nan") if value in ("None", "") else float(value)
             for value in line.split(sep)] for line in lines[1:]]
    return header, np.array(rows, dtype=np.float64)


class Report:
    def __init__(self):
        self.rows = []

    def check(self, label, ok, detail="", always=False):
        self.rows.append((label, bool(ok), detail))
        print("{:<62} {}{}".format(label, "PASS" if ok else "FAIL",
                                   "  " + detail if detail and (always or not ok) else ""))
        sys.stdout.flush()

    def exact(self, label, a, b):
        """Bitwise equal: for comparisons within one environment."""
        a, b = np.asarray(a), np.asarray(b)
        self.check(label, a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())

    def close(self, label, a, b, tolerance):
        """Within ``tolerance`` relative L2 of the original (``b``); the report
        says whether the two came out bitwise equal, or how far apart."""
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return self.check(label, False, "shapes {} and {}".format(a.shape, b.shape))
        if a.dtype == b.dtype and a.tobytes() == b.tobytes():
            return self.check(label, True, "bitwise", always=True)
        # NaN is a value neither side has (a None in a log): it must be
        # missing from both at the same places, and the rest is compared.
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        missing = np.isnan(b)
        if not np.array_equal(np.isnan(a), missing):
            return self.check(label, False, "missing values at different places")
        a, b = a[~missing], b[~missing]
        distance = relative_l2(a, b)
        self.check(label, distance <= tolerance,
                   "relative L2 {:.2g}, max |delta| {:.2g} (limit {:g})".format(
                       distance, float(np.max(np.abs(a - b), initial=0.0)), tolerance), always=True)

    def files(self, label, a, b):
        self.check(label, Path(a).read_bytes() == Path(b).read_bytes())

    def h5(self, label, a, b, exclude=()):
        """Every group, dataset and attribute equal: within one environment."""
        left, right = h5_contents(a), h5_contents(b)
        for prefix in exclude:
            left = {k: v for k, v in left.items() if not k.startswith(prefix)}
            right = {k: v for k, v in right.items() if not k.startswith(prefix)}
        differing = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        self.check(label, not differing, "differs at " + ", ".join(differing[:4]))

    def weights_close(self, label, engine, original, tolerance, root="model_weights", subset=False,
                      depth=2):
        """The same-named weights of two checkpoints, all together, within
        ``tolerance``, but for UNIDENTIFIED_WEIGHTS; with ``subset`` only
        those ``engine`` has."""
        left, right = _weight_arrays(engine, root, depth), _weight_arrays(original, root, depth)
        keys = sorted(left) if subset else sorted(set(left) | set(right))
        keys = [k for k in keys if not any(part in k for part in UNIDENTIFIED_WEIGHTS)]
        missing = [k for k in keys if k not in left or k not in right]
        if not keys or missing:
            return self.check(label, False, "missing " + ", ".join(missing[:4]) if missing else "no weights")
        self.close(label + " ({} arrays)".format(len(keys)),
                   np.concatenate([np.ravel(left[k]).astype(np.float64) for k in keys]),
                   np.concatenate([np.ravel(right[k]).astype(np.float64) for k in keys]), tolerance)

    def table_close(self, label, engine, original, sep, tolerance):
        """The losses two logs share, within ``tolerance``."""
        (left_head, left), (right_head, right) = _table(engine, sep), _table(original, sep)
        shared = [c for c in right_head if c in left_head]
        extra = sorted(set(left_head) ^ set(right_head))
        if left.shape[0] != right.shape[0] or not shared:
            return self.check(label, False, "{} rows and {} rows".format(left.shape[0], right.shape[0]))
        self.close(label + (" (not in both: {})".format(", ".join(extra)) if extra else ""),
                   left[:, [left_head.index(c) for c in shared]],
                   right[:, [right_head.index(c) for c in shared]], tolerance)

    @property
    def failed(self):
        return [label for label, ok, _ in self.rows if not ok]


def _log_probabilities(logits):
    """Profile logits as log-probabilities: what the multinomial loss sees."""
    logits = np.asarray(logits, np.float64).reshape(len(logits), -1)
    top = logits.max(axis=1, keepdims=True)
    return logits - top - np.log(np.exp(logits - top).sum(axis=1, keepdims=True))


def _epochs_run(log_path):
    return len(Path(log_path).read_text().strip().splitlines()) - 1


def _best_and_last_epoch(log_path):
    """(index of the lowest val_loss, index of the last epoch) in a CSVLogger log."""
    lines = Path(log_path).read_text().strip().splitlines()
    column = lines[0].split(",").index("val_loss")
    losses = [float(line.split(",")[column]) for line in lines[1:]]
    return int(np.argmin(losses)), len(losses) - 1


def compare(workdir, report):
    ref, ref2, eng = (os.path.join(workdir, n) for n in ("reference", "reference2", "engine"))
    tol = TOLERANCE

    # 0. The reference must reproduce itself, exactly (one environment).
    for case, kind, *_ in TRAINING_CASES:
        prefix = "chrombpnet" if kind == "chrombpnet" else "bias"
        report.h5("self: reference {} checkpoint".format(case),
                  os.path.join(ref, case, prefix + ".h5"), os.path.join(ref2, case, prefix + ".h5"))
        report.files("self: reference {} .log.batch".format(case),
                     os.path.join(ref, case, prefix + ".log.batch"),
                     os.path.join(ref2, case, prefix + ".log.batch"))

    # 1. Models before training.
    configs_ref = json.loads(Path(ref, "configs.json").read_text())
    configs_eng = json.loads(Path(eng, "configs.json").read_text())
    for kind in ("bias", "chrombpnet"):
        report.check("build: {} model structure".format(kind),
                     _structure(configs_ref[kind]) == _structure(configs_eng[kind]))
    build_ref, build_eng = np.load(os.path.join(ref, "build.npz")), np.load(os.path.join(eng, "build.npz"))
    for key in sorted(build_ref.files):
        report.close("build: " + key, build_eng[key], build_ref[key], tol["build"])

    # 2. Training. Where early stopping fires (case A) the engine must match
    # the original's checkpoint. Where training runs out of epochs (case B)
    # the engine restores the best validation epoch and the original saved the
    # last one, so the engine's weights must match the original's best-epoch
    # checkpoint instead -- which the reference worker snapshots before
    # overwriting it.
    for case, kind, epochs, early_stop, _ in TRAINING_CASES:
        prefix = "chrombpnet" if kind == "chrombpnet" else "bias"
        r, e = os.path.join(ref, case, prefix), os.path.join(eng, case, prefix)
        ran, ran_engine = _epochs_run(r + ".log"), _epochs_run(e + ".log")
        stopped_early = ran < epochs
        expect = case.endswith("_A")
        report.check("train {}: early stopping {} (ran {} of {} epochs; engine {})".format(
            case, "fired" if expect else "did not fire", ran, epochs, ran_engine),
            stopped_early == expect and ran_engine == ran)
        report.table_close("train {}: .log losses".format(case), e + ".log", r + ".log", ",", tol["train"])
        report.table_close("train {}: .log.batch losses".format(case),
                           e + ".log.batch", r + ".log.batch", "\t", tol["train"])
        # Optimizer state is the last epoch's on both sides (restoring the
        # best epoch restores only the weights); its variables are named
        # alike only because the engine trains with the legacy Adam, as 1.x.
        report.weights_close("train {}: optimizer state".format(case), e + ".h5", r + ".h5",
                             tol["train"], root="optimizer_weights", depth=None)
        if expect:
            report.weights_close("train {}: checkpoint weights".format(case), e + ".h5", r + ".h5", tol["train"])
            if kind == "chrombpnet":
                report.weights_close("train {}: no-bias model".format(case),
                                     e + "_nobias.h5", r + "_nobias.h5", tol["train"])
            continue
        best, last = _best_and_last_epoch(r + ".log")
        report.check("train {}: best epoch ({}) is not the last ({})".format(case, best, last),
                     best != last)
        report.weights_close("train {}: weights = reference's best epoch".format(case),
                             e + ".h5", r + ".best.h5", tol["train"])
        if kind == "chrombpnet":
            report.weights_close("train {}: no-bias model = reference's best epoch".format(case),
                                 e + "_nobias.h5", r + ".best.h5", tol["train"], subset=True)

    # 3. Rescale, DeepSHAP, interop.
    report.weights_close("rescale: bias_model_scaled.h5", os.path.join(eng, "bias_model_scaled.h5"),
                         os.path.join(ref, "bias_model_scaled.h5"), tol["rescale"])
    shap_ref, shap_eng = np.load(os.path.join(ref, "shap.npz")), np.load(os.path.join(eng, "shap.npz"))
    for prefix, mode in (("", ""), ("stopgrad_", " with --contribs-profile-stopgrad")):
        for head in HEADS:
            report.close("shap: {} hypothetical scores{}".format(head, mode),
                         shap_eng[prefix + head], shap_ref[prefix + head], tol["shap"])
            report.exact("shap: {} scores{} via the CLI's subprocess path".format(head, mode),
                         shap_eng[prefix + head], shap_eng["end_to_end_" + prefix + head])
    report.exact("shap: counts unchanged by the StopGradient handler",
                 shap_ref["stopgrad_counts"], shap_ref["counts"])
    report.exact("shap: counts unchanged by the StopGradient handler (engine)",
                 shap_eng["stopgrad_counts"], shap_eng["counts"])
    departure = relative_l2(shap_eng["stopgrad_profile"], shap_ref["profile"])
    report.check("shap: the StopGradient handler changes the profile head",
                 departure > MIN_PROFILE_DEPARTURE,
                 "relative L2 {:.3g} <= {:g}".format(departure, MIN_PROFILE_DEPARTURE))
    print("    (relative L2 between the two profile weightings: {:.3g})".format(departure))
    interop = [np.load(os.path.join(d, "interop.npz")) for d in (ref, eng)]
    for kind, case, _, compiled in INTEROP_MODELS:
        how = " (compiled)" if compiled else ""
        for model in ("reference", "engine"):
            for output in ("profile", "counts"):
                key = "{}_{}_{}".format(model, kind, output)
                report.close("interop: {} {} model {}, engine vs original loading it{}".format(
                    model, kind, output, how), interop[1][key], interop[0][key], tol["interop"])
        for output in ("profile", "counts"):
            engine_out = interop[0]["engine_{}_{}".format(kind, output)]
            reference_out = interop[0]["reference_{}_{}".format(kind, output)]
            if output == "profile":
                engine_out, reference_out = _log_probabilities(engine_out), _log_probabilities(reference_out)
            report.close("interop: both sides' {} models predict alike ({}{})".format(
                kind, output, " as log-probabilities" if output == "profile" else ""),
                engine_out, reference_out, tol["train"])


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

#: The reference side's interpreter; set by main from --reference-python.
REFERENCE_PYTHON = [DEFAULT_REFERENCE_PYTHON]


def _run(task, side, workdir, out, original_repo, case=None, sides=None):
    python = REFERENCE_PYTHON[0] if side == "reference" else sys.executable
    command = [python, str(Path(__file__).resolve()), "--worker",
               "--side", side, "--task", task, "--workdir", workdir, "--out", out,
               "--original-repo", original_repo]
    if case:
        command += ["--case", case]
    if sides:
        command += ["--sides"] + list(sides)
    env = dict(os.environ, **PINNED_ENV)
    env.pop("PYTHONPATH", None)
    log = os.path.join(workdir, "logs", "{}-{}{}.log".format(side, task, "-" + case if case else ""))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as handle:
        completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = Path(log).read_text().splitlines()[-25:]
        raise RuntimeError("{} {} {} failed:\n{}".format(side, task, case or "", "\n".join(tail)))


def _check_original_repo(original_repo):
    if not (Path(original_repo) / "chrombpnet" / "training" / "train.py").is_file():
        raise SystemExit("Original repository not found: {}".format(original_repo))
    try:
        head = subprocess.run(["git", "-C", original_repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    if head and head != ORIGINAL_COMMIT:
        print("warning: original repository is at {}, not {}".format(head, ORIGINAL_COMMIT))


def main(arguments):
    _check_original_repo(arguments.original_repo)
    if not Path(arguments.reference_python).is_file():
        raise SystemExit("The original's interpreter was not found: {} (--reference-python, "
                         "or CHROMBPNET_ORIGINAL_PYTHON)".format(arguments.reference_python))
    REFERENCE_PYTHON[0] = arguments.reference_python
    workdir = tempfile.mkdtemp(prefix="chrombpnet-tf-parity-")
    print("workdir:", workdir)
    print("original:", arguments.reference_python, "| engine:", sys.executable)
    repo = arguments.original_repo
    try:
        print("preparing the fixture, batch stream and initial weights ...")
        _run("prepare", "engine", workdir, workdir, repo)
        _run("bias_init", "reference", workdir, workdir, repo)
        for side, out in (("reference", "reference"), ("reference", "reference2"), ("engine", "engine")):
            outdir = os.path.join(workdir, out)
            print("running", out, "...")
            for case, *_ in TRAINING_CASES:
                _run("train", side, workdir, outdir, repo, case=case)
            if out == "reference2":
                continue
            _run("build", side, workdir, outdir, repo)
            _run("rescale", side, workdir, outdir, repo)
            _run("shap", side, workdir, outdir, repo)
        for side in ("reference", "engine"):
            _run("interop", side, workdir, os.path.join(workdir, side), repo,
                 sides=("reference", "engine"))
        report = Report()
        compare(workdir, report)
        if report.failed:
            print("\n{} of {} checks failed.".format(len(report.failed), len(report.rows)))
            return 1
        bitwise = sum(1 for _, ok, detail in report.rows if ok and detail == "bitwise")
        print("\nAll {} checks passed ({} of the engine-vs-original comparisons bitwise): the "
              "TensorFlow engine reproduces the original implementation within TOLERANCE, "
              "departing only where it fixes a documented defect.".format(len(report.rows), bitwise))
        return 0
    finally:
        if arguments.keep:
            print("kept", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument("--original-repo", default=DEFAULT_ORIGINAL_REPO)
    result.add_argument("--reference-python", default=DEFAULT_REFERENCE_PYTHON,
                        help="the original's interpreter (default: the chrombpnet env beside this one)")
    result.add_argument("--keep", action="store_true", help="keep the work directory")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--side", help=argparse.SUPPRESS)
    result.add_argument("--task", help=argparse.SUPPRESS)
    result.add_argument("--workdir", help=argparse.SUPPRESS)
    result.add_argument("--out", help=argparse.SUPPRESS)
    result.add_argument("--case", help=argparse.SUPPRESS)
    result.add_argument("--sides", nargs="*", help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.worker:
        if arguments.task == "prepare":
            prepare(arguments.workdir)
        else:
            worker(arguments)
    else:
        raise SystemExit(main(arguments))
