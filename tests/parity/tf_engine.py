#!/usr/bin/env python3
"""Byte-identity of the TensorFlow engine with the original Keras ChromBPNet.

The TensorFlow engine (``chrombpnet/engines/tensorflow``) claims to reproduce
the original implementation (kundajelab/chrombpnet@09938fd) exactly, given
the same inputs. "The same inputs" is deliberately narrow: the data pipeline,
preprocessing and seeding are this package's and may differ, so this harness
feeds both implementations identical batches, identical initial weights and
identical DeepSHAP references, and then requires every output to be equal --
not close, equal.

Everything runs in the ``chrombpnet-tensorflow`` environment, in separate
processes, because both code bases are the package ``chrombpnet``:

``prepare``   this package: builds the regression fixture, records a batch
              stream from the shared generator, draws initial weights, and
              writes a bias model to start the bias-factorized model from.
``reference`` the original repository on ``sys.path``, nothing of this one.
``engine``    this package's TensorFlow engine.

Each training run is its own process, as it is on the command line, so that
TensorFlow's process-global state (seeds, layer-name counters, graph mode)
cannot leak between cases. The harness -- not either implementation -- pins
TensorFlow to one thread, for both sides alike. It does not set
``TF_DETERMINISTIC_OPS``: under it TF 2.8 refuses any unseeded random op, and
the original builds its bias-factorized model before seeding. Single-threaded
CPU execution is deterministic instead, which the self-check confirms.

What is compared, exactly:

* ``build``   model configuration, predictions of the bias, combined and
              no-bias models, and ``test_on_batch`` losses;
* ``train``   checkpoints (every weight, the optimizer state, the Keras
              attributes), ``.log`` and ``.log.batch``, and the no-bias model,
              for a bias model and a bias-factorized model where early stopping
              fires, and a bias-factorized model that runs out of epochs;
* ``rescale`` the hyperparameter finder's bias count-head rescale;
* ``shap``    DeepSHAP hypothetical scores for both heads;
* ``interop`` each side loading and predicting with the other side's model.

The reference is first checked against itself, since a byte-identity claim
means nothing if the reference does not reproduce its own output.

Keras serializes the bias-factorized model's ``Lambda`` count head as
marshalled bytecode, which embeds the defining file's path and line number.
Those necessarily differ between the two code bases, so the comparison
decodes the bytecode and compares the instructions, constants and names.

Usage::

    python tests/parity/tf_engine.py [--original-repo PATH] [--keep]
    CHROMBPNET_TF_ENGINE_PARITY=1 pytest tests/parity
"""

import argparse
import base64
import functools
import json
import marshal
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_REPO = "/path/to/workdir/software/chrombpnet"
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
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
}


# --------------------------------------------------------------------------
# Shared by both workers: they import this file, never each other's package.
# --------------------------------------------------------------------------

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

    # The starting bias model for the bias-factorized cases: the engine's
    # bias architecture with the shared weights, saved as a Keras checkpoint.
    _pin_tensorflow()
    from chrombpnet import engines
    engine = engines.get("tensorflow")
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    args = _training_args("bias", 1, 1, 0.001, workdir, engine.default_architecture("bias"))
    model, _ = engine.get_model(args, _parameters("bias", workdir))
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

    def rescale(self, bias_path, out, seqs, cts):
        from chrombpnet.helpers.hyperparameters import find_chrombpnet_hyperparams as finder
        from chrombpnet.helpers.hyperparameters import param_utils
        bias_model = param_utils.load_model_wrapper(bias_path)
        finder.adjust_bias_model_logcounts(bias_model, seqs, cts).save(out)

    def shap(self, model_path, seqs, head, seed):
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
        self.engine = engines.configure("tensorflow")

    def architecture(self, kind):
        return self.engine.default_architecture(kind)

    def get_model(self, args, parameters):
        return self.engine.get_model(args, parameters)

    def fit(self, model, train_gen, valid_gen, args, parameters, architecture):
        self.engine.fit(model, train_gen, valid_gen, args, parameters, architecture)

    def load_predictor(self, path):
        return self.engine.load_predictor(path)

    def rescale(self, bias_path, out, seqs, cts):
        self.engine.rescale_bias_logcounts(bias_path, out, seqs, cts)

    def shap(self, model_path, seqs, head, seed):
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        _pin_tensorflow()
        from chrombpnet.engines.tensorflow import attribution
        from chrombpnet.engines.tensorflow.predictor import load_model

        model = load_model(model_path, compile=False)
        # The engine seeds its own references, a fresh RandomState(seed) per
        # head; the reference worker is given the same stream.
        return attribution.explain(model, seqs, [head], 20, seed)[head]

    def shap_end_to_end(self, model_path, seqs, seed):
        """The CLI's path: a spawned graph-mode child, both heads in one go."""
        handle = self.engine.load_attribution_model(model_path, None)
        return dict(self.engine.attribute(
            handle, seqs, list(HEADS), device=None, batch_size=1,
            num_shuffles=20, seed=seed))


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
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    scores = {}
    for head in HEADS:
        scores[head] = side.shap(os.path.join(workdir, "bias_init.h5"),
                                 fixture["shap_x"], head, SHAP_SEED)
    if side.name == "engine":
        end_to_end = side.shap_end_to_end(os.path.join(workdir, "bias_init.h5"),
                                          fixture["shap_x"], SHAP_SEED)
        for head in HEADS:
            scores["end_to_end_" + head] = end_to_end[head]
    np.savez(os.path.join(out, "shap.npz"), **scores)


def task_interop(side, workdir, out, sides):
    _pin_tensorflow()
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    results = {}
    for name in sides:
        path = os.path.join(workdir, name, "chrombpnet_A", "chrombpnet.h5")
        model = side.load_predictor(path)
        profile, counts = model.predict(fixture["predict_x"], batch_size=BATCH_SIZE)
        results[name + "_profile"], results[name + "_counts"] = profile, counts
    np.savez(os.path.join(out, "interop.npz"), **results)


def worker(arguments):
    side = Reference(arguments.original_repo) if arguments.side == "reference" else EngineSide()
    workdir, out = arguments.workdir, arguments.out
    os.makedirs(out, exist_ok=True)
    if arguments.task == "build":
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

def _normalize_code(code):
    """A code object's behaviour, without where it was defined."""
    consts = tuple(_normalize_code(c) if isinstance(c, types.CodeType) else repr(c)
                   for c in code.co_consts)
    return (code.co_code.hex(), consts, code.co_names, code.co_varnames,
            code.co_argcount, code.co_freevars, code.co_cellvars)


def _normalize_keras_json(text):
    """Keras model JSON with each Lambda's bytecode reduced to its behaviour."""
    def walk(node):
        if isinstance(node, dict):
            if node.get("class_name") == "Lambda":
                config = dict(node["config"])
                function = config.get("function")
                if isinstance(function, list) and function and isinstance(function[0], str):
                    code = marshal.loads(base64.decodebytes(function[0].encode("ascii")))
                    config["function"] = [repr(_normalize_code(code))] + list(function[1:])
                node = dict(node, config=config)
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node
    return json.dumps(walk(json.loads(text)), sort_keys=True)


def _attr_value(key, value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if key == "model_config" and isinstance(value, str):
        return _normalize_keras_json(value)
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


def _weights_by_layer(path, root="model_weights"):
    """Weight arrays keyed by ``layer/variable``, whatever the nesting."""
    return h5_contents(path, root, key=lambda name: "/".join(name.split("/")[-2:]))


class Report:
    def __init__(self):
        self.rows = []

    def check(self, label, equal, detail=""):
        self.rows.append((label, bool(equal), detail))
        print("{:<58} {}{}".format(label, "PASS" if equal else "FAIL",
                                   "  " + detail if detail and not equal else ""))
        sys.stdout.flush()

    def arrays(self, label, a, b):
        a, b = np.asarray(a), np.asarray(b)
        same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        detail = ""
        if not same and a.shape == b.shape and a.dtype.kind == "f":
            detail = "max |delta| = {:.3g}".format(float(np.max(np.abs(a - b))))
        self.check(label, same, detail)

    def files(self, label, a, b):
        self.check(label, Path(a).read_bytes() == Path(b).read_bytes())

    def h5(self, label, a, b, exclude=()):
        left, right = h5_contents(a), h5_contents(b)
        for prefix in exclude:
            left = {k: v for k, v in left.items() if not k.startswith(prefix)}
            right = {k: v for k, v in right.items() if not k.startswith(prefix)}
        differing = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        self.check(label, not differing, "differs at " + ", ".join(differing[:4]))

    def weights(self, label, a, b):
        left, right = _weights_by_layer(a), _weights_by_layer(b)
        differing = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        self.check(label, bool(left) and not differing, "differs at " + ", ".join(differing[:4]))

    def weights_within(self, label, part, whole):
        """Every weight in ``part`` equals the same-named weight in ``whole``."""
        small, big = _weights_by_layer(part), _weights_by_layer(whole)
        bad = sorted(k for k in small if small[k] != big.get(k))
        self.check(label, bool(small) and not bad, "differs at " + ", ".join(bad[:4]))

    @property
    def failed(self):
        return [label for label, ok, _ in self.rows if not ok]


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

    # 0. The reference must reproduce itself.
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
        report.check("build: {} model configuration".format(kind),
                     _normalize_keras_json(configs_ref[kind]) == _normalize_keras_json(configs_eng[kind]))
    build_ref, build_eng = np.load(os.path.join(ref, "build.npz")), np.load(os.path.join(eng, "build.npz"))
    for key in sorted(build_ref.files):
        report.arrays("build: " + key, build_ref[key], build_eng[key])

    # 2. Training. Where early stopping fires (case A) everything is
    # identical. Where training runs out of epochs (case B) the engine
    # restores the best validation epoch and the original saved the last one,
    # so the engine's weights must equal the original's best-epoch checkpoint
    # instead -- which the reference worker snapshots before overwriting it.
    for case, kind, epochs, early_stop, _ in TRAINING_CASES:
        prefix = "chrombpnet" if kind == "chrombpnet" else "bias"
        r, e = os.path.join(ref, case, prefix), os.path.join(eng, case, prefix)
        ran = _epochs_run(r + ".log")
        stopped_early = ran < epochs
        expect = case.endswith("_A")
        report.check("train {}: early stopping {} (ran {} of {} epochs)".format(
            case, "fired" if expect else "did not fire", ran, epochs), stopped_early == expect)
        report.files("train {}: .log".format(case), r + ".log", e + ".log")
        report.files("train {}: .log.batch".format(case), r + ".log.batch", e + ".log.batch")
        if expect:
            report.h5("train {}: checkpoint".format(case), r + ".h5", e + ".h5")
            if kind == "chrombpnet":
                report.h5("train {}: no-bias model".format(case), r + "_nobias.h5", e + "_nobias.h5")
            continue
        best, last = _best_and_last_epoch(r + ".log")
        report.check("train {}: best epoch ({}) is not the last ({})".format(case, best, last),
                     best != last)
        report.weights("train {}: weights = reference's best epoch".format(case),
                       r + ".best.h5", e + ".h5")
        report.h5("train {}: optimizer state and attributes".format(case),
                  r + ".h5", e + ".h5", exclude=("model_weights",))
        if kind == "chrombpnet":
            report.weights_within("train {}: no-bias model = reference's best epoch".format(case),
                                  e + "_nobias.h5", r + ".best.h5")

    # 3. Rescale, DeepSHAP, interop.
    report.h5("rescale: bias_model_scaled.h5", os.path.join(ref, "bias_model_scaled.h5"),
              os.path.join(eng, "bias_model_scaled.h5"))
    shap_ref, shap_eng = np.load(os.path.join(ref, "shap.npz")), np.load(os.path.join(eng, "shap.npz"))
    for head in HEADS:
        report.arrays("shap: {} hypothetical scores".format(head), shap_ref[head], shap_eng[head])
        report.arrays("shap: {} scores via the CLI's subprocess path".format(head),
                      shap_eng[head], shap_eng["end_to_end_" + head])
    interop = [np.load(os.path.join(d, "interop.npz")) for d in (ref, eng)]
    for model in ("reference", "engine"):
        for output in ("profile", "counts"):
            key = "{}_{}".format(model, output)
            report.arrays("interop: {} model {}, loaded by both".format(model, output),
                          interop[0][key], interop[1][key])
    report.arrays("interop: both models predict alike",
                  interop[0]["reference_profile"], interop[0]["engine_profile"])


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _run(task, side, workdir, out, original_repo, case=None, sides=None):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
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
    workdir = tempfile.mkdtemp(prefix="chrombpnet-tf-parity-")
    print("workdir:", workdir)
    repo = arguments.original_repo
    try:
        print("preparing the fixture, batch stream and initial weights ...")
        _run("prepare", "engine", workdir, workdir, repo)
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
        print("\nAll {} checks passed: the TensorFlow engine reproduces the original "
              "implementation byte for byte, departing only where it fixes a "
              "documented defect.".format(len(report.rows)))
        return 0
    finally:
        if arguments.keep:
            print("kept", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument("--original-repo", default=DEFAULT_ORIGINAL_REPO)
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
