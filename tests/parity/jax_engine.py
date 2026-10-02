#!/usr/bin/env python3
"""Agreement of the JAX engine with the original Keras ChromBPNet.

The JAX engine (``chrombpnet/engines/jax``) is kundajelab/chrombpnet#284's
Keras 3 / JAX port of the original implementation (kundajelab/chrombpnet
@09938fd). Unlike the TensorFlow engine it cannot be byte-identical -- XLA and
TensorFlow 2.8 schedule float32 arithmetic differently -- so this harness
checks agreement to the tolerances the PR validated against the original
(``tests/parity`` in the PR), on identical inputs: the recorded batch stream,
initial weights and fixture of ``tests/parity/tf_engine.py``.

Two interpreters, files in between, like ``tf_engine.py``:

``prepare``  this package, in ``chrombpnet-tensorflow``: the fixture, batch
             stream, initial weights and a 1.x bias model (``tf_engine.prepare``).
``golden``   the **original** repository, in ``chrombpnet-tensorflow``: model
             outputs, ``test_on_batch`` losses, a 20-step Adam trace, the bias
             rescale and DeepSHAP; its models are saved as 1.x ``.h5`` files.
``jax``      this package's JAX engine, in ``chrombpnet-jax``, CPU, full float32.

Checks (tolerances as in the PR, CPU, full float32):

* predictions of the bias, bias-factorized and no-bias models: max abs <= 1e-4;
* the full model equals its no-bias and bias parts combined: max abs <= 1e-5;
* ``test_on_batch`` losses: relative <= 1e-5;
* a 20-step Adam trace from identical weights and batches: loss relative
  <= 1e-5 at the first step and <= 1e-3 at the last; final weights relative
  L2 <= 5e-3 overall and <= 5e-2 per array (excluding the profile head's
  scalar bias, whose norm is ~0);
* the bias count-head rescale: predictions max abs <= 1e-4;
* DeepSHAP references: **bitwise** equal; hypothetical scores, both heads,
  for the initial bias model and the original's trained no-bias model,
  through ``engine.attribute``: relative L2 <= 1e-4. The golden scores are the
  original's with the shap fork's StopGradient handler registered
  (``tf_engine.honour_stop_gradient``), because the engine honours the profile
  target's ``stop_gradient``; without it the original weights the profile
  logits by a midpoint, and the engine's profile scores must differ from
  those by more than ``tf_engine.MIN_PROFILE_DEPARTURE``;
* the JAX engine loads the original's 1.x models (including the full
  bias-factorized model with its bytecode ``Lambda``) and predicts within 1e-4;
* a ``.keras`` save/load round trip is exact;
* ``export --legacy-h5``: the JAX engine's bias, bias-factorized and no-bias
  models, exported to the 1.x ``.h5`` layout, load in this package's
  TensorFlow engine and in the original implementation, and predict within
  1e-4 of the JAX engine.

The golden worker is checked against itself first.

Usage::

    python tests/parity/jax_engine.py [--original-repo PATH] [--tf-python P] [--jax-python P] [--keep]
    CHROMBPNET_JAX_ENGINE_PARITY=1 pytest tests/parity/test_jax_engine.py
"""

import argparse
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
PARITY_OPT_IN = "CHROMBPNET_JAX_ENGINE_PARITY"
SHAP_SEED = 1234
NUM_SHUFFLES = 20
TRACE_STEPS = 20
TRACE_LEARNING_RATE = 1e-3

#: Thread and platform pins, the harness's own, for both sides.
PINNED_ENV = {
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
    "KERAS_BACKEND": "jax",
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def _sibling_python(env_name):
    """The python of a sibling conda environment of the one running this."""
    envs = Path(sys.executable).parents[2]
    return str(envs / env_name / "bin" / "python")


def _tf_engine():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tests.parity import tf_engine
    return tf_engine


def _import_this_package():
    """Import chrombpnet from this checkout, and fail if an editable install
    elsewhere answered instead."""
    sys.path.insert(0, str(REPO_ROOT))
    import chrombpnet
    if not str(Path(chrombpnet.__file__).resolve()).startswith(str(REPO_ROOT)):
        raise RuntimeError("worker imported chrombpnet from {}, not {}".format(
            chrombpnet.__file__, REPO_ROOT))


# --------------------------------------------------------------------------
# Shared by both workers
# --------------------------------------------------------------------------

def _batches(fixture, kind, steps):
    """The recorded training batches, epoch by epoch, for ``steps`` steps."""
    prefix = "{}_train".format(kind)
    n_batches = int(fixture[prefix + "_n_batches"])
    n_epochs = int(fixture[prefix + "_n_epochs"])
    out = []
    for step in range(steps):
        epoch, index = divmod(step, n_batches)
        key = "{}_e{}_b{}_".format(prefix, min(epoch, n_epochs - 1), index)
        out.append((fixture[key + "x"], fixture[key + "p"], fixture[key + "c"]))
    return out


def _layer_weights(model, prefix=""):
    """{layer/index: array} for every layer, recursing into nested models."""
    weights = {}
    for layer in model.layers:
        if hasattr(layer, "layers") and layer.layers:
            weights.update(_layer_weights(layer, prefix + layer.name + "/"))
            continue
        for index, value in enumerate(layer.get_weights()):
            weights["{}{}/{}".format(prefix, layer.name, index)] = np.asarray(value)
    return weights


def _named_losses(names, values):
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    return dict(zip(names, values))


def _save_npz(path, mapping):
    np.savez(path, **{key.replace("/", "|"): value for key, value in mapping.items()})


def _load_npz(path):
    data = np.load(path)
    return {key.replace("|", "/"): data[key] for key in data.files}


# --------------------------------------------------------------------------
# golden: the original implementation, in the TensorFlow environment
# --------------------------------------------------------------------------

def golden(workdir, out, original_repo):
    tf_engine = _tf_engine()
    side = tf_engine.Reference(original_repo)
    tf = tf_engine._pin_tensorflow()
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    os.makedirs(out, exist_ok=True)
    results = {}
    for kind in ("bias", "chrombpnet"):
        args = tf_engine._training_args(
            kind, 1, 1, TRACE_LEARNING_RATE, out, side.architecture(kind))
        model, architecture = side.get_model(args, tf_engine._parameters(kind, workdir))
        tf_engine._inject(model, kind, fixture)
        model.save(os.path.join(out, "{}_built.h5".format(kind)))
        if kind == "chrombpnet":
            architecture.save_model_without_bias(model, os.path.join(out, "chrombpnet_built"))
            nobias = model.get_layer("model_wo_bias")
            profile, counts = nobias.predict(fixture["predict_x"], batch_size=8)
            results["nobias_profile"], results["nobias_counts"] = profile, counts
        profile, counts = model.predict(fixture["predict_x"], batch_size=8)
        results[kind + "_profile"], results[kind + "_counts"] = profile, counts
        batch = "{}_train_e0_b0_".format(kind)
        losses = model.test_on_batch(fixture[batch + "x"], [fixture[batch + "p"], fixture[batch + "c"]])
        for name, value in _named_losses(model.metrics_names, losses).items():
            results["{}_test|{}".format(kind, name)] = np.float64(value)
        trace = []
        for x, p, c in _batches(fixture, kind, TRACE_STEPS):
            model.reset_metrics()
            values = _named_losses(model.metrics_names, model.train_on_batch(x, [p, c]))
            trace.append([values["loss"], values["logits_profile_predictions_loss"],
                          values["logcount_predictions_loss"]])
        results[kind + "_trace"] = np.array(trace)
        model.save(os.path.join(out, "{}_trace.h5".format(kind)))
        if kind == "chrombpnet":
            # a trained no-bias model, for DeepSHAP with weights unlike the bias model's
            architecture.save_model_without_bias(model, os.path.join(out, "chrombpnet_trace"))
        profile, counts = model.predict(fixture["predict_x"], batch_size=8)
        results[kind + "_trace_profile"], results[kind + "_trace_counts"] = profile, counts
        _save_npz(os.path.join(out, "{}_trace_weights.npz".format(kind)), _layer_weights(model))
        tf.keras.backend.clear_session()
    np.savez(os.path.join(out, "golden.npz"), **results)

    side.rescale(os.path.join(workdir, "bias_init.h5"), os.path.join(out, "bias_model_scaled.h5"),
                 fixture["rescale_x"], fixture["rescale_cts"])
    scaled = side.load_predictor(os.path.join(out, "bias_model_scaled.h5"))
    profile, counts = scaled.predict(fixture["predict_x"], batch_size=8)
    np.savez(os.path.join(out, "rescale.npz"), profile=profile, counts=counts)
    tf.keras.backend.clear_session()


def golden_shap(workdir, out, original_repo):
    """DeepSHAP with the original code; its own process (graph mode)."""
    tf_engine = _tf_engine()
    side = tf_engine.Reference(original_repo)
    import deeplift.dinuc_shuffle

    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    seqs = fixture["shap_x"]
    rng = np.random.RandomState(SHAP_SEED)
    references = np.stack([
        np.stack([deeplift.dinuc_shuffle.dinuc_shuffle(seq, rng=rng) for _ in range(NUM_SHUFFLES)])
        for seq in seqs]).astype(np.int8)
    scores = {"references": references}
    for name, path in (("bias", os.path.join(workdir, "bias_init.h5")),
                       ("nobias", os.path.join(out, "chrombpnet_trace_nobias.h5"))):
        for head in ("counts", "profile"):
            scores["{}_{}".format(name, head)] = side.shap(path, seqs, head, SHAP_SEED,
                                                            stop_gradient=True)
        scores["{}_profile_midpoint".format(name)] = side.shap(path, seqs, "profile", SHAP_SEED,
                                                                stop_gradient=False)
    np.savez(os.path.join(out, "shap.npz"), **scores)


# --------------------------------------------------------------------------
# jax: this package's JAX engine, in the JAX environment
# --------------------------------------------------------------------------

def jax_side(workdir, golden_dir, out):
    _import_this_package()
    from chrombpnet import engines
    tf_engine = _tf_engine()

    engine = engines.configure("jax")
    engine.require()
    import keras

    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    os.makedirs(out, exist_ok=True)
    results = {}
    for kind in ("bias", "chrombpnet"):
        args = tf_engine._training_args(
            kind, 1, 1, TRACE_LEARNING_RATE, out, engine.default_architecture(kind))
        args.precision = "highest"
        args.device = "cpu"
        model, architecture = engine.get_model(args, tf_engine._parameters(kind, workdir))
        tf_engine._inject(model, kind, fixture)
        if kind == "chrombpnet":
            architecture.save_model_without_bias(model, os.path.join(out, "chrombpnet_built"))
            nobias = model.get_layer("model_wo_bias")
            profile, counts = nobias.predict(fixture["predict_x"], batch_size=8, verbose=0)
            results["nobias_profile"], results["nobias_counts"] = profile, counts
        model.save(os.path.join(out, "{}_built.keras".format(kind)))
        profile, counts = model.predict(fixture["predict_x"], batch_size=8, verbose=0)
        results[kind + "_profile"], results[kind + "_counts"] = profile, counts
        batch = "{}_train_e0_b0_".format(kind)
        losses = model.test_on_batch(fixture[batch + "x"], (fixture[batch + "p"], fixture[batch + "c"]),
                                     return_dict=True)
        for name, value in losses.items():
            results["{}_test|{}".format(kind, name)] = np.float64(value)
        trace = []
        for x, p, c in _batches(fixture, kind, TRACE_STEPS):
            model.reset_metrics()
            values = model.train_on_batch(x, (p, c), return_dict=True)
            trace.append([values["loss"], values["logits_profile_predictions_loss"],
                          values["logcount_predictions_loss"]])
        results[kind + "_trace"] = np.array(trace, dtype=np.float64)
        profile, counts = model.predict(fixture["predict_x"], batch_size=8, verbose=0)
        results[kind + "_trace_profile"], results[kind + "_trace_counts"] = profile, counts
        _save_npz(os.path.join(out, "{}_trace_weights.npz".format(kind)), _layer_weights(model))
        keras.backend.clear_session()

    # Round trip, and the full model against its parts.
    built = engine.load_predictor(os.path.join(out, "chrombpnet_built.keras"), announce=False, summary=False)
    profile, counts = built.predict(fixture["predict_x"], batch_size=8, verbose=0)
    results["roundtrip_profile"], results["roundtrip_counts"] = profile, counts
    nobias = engine.load_predictor(os.path.join(out, "chrombpnet_built_nobias.keras"), announce=False, summary=False)
    bias = engine.load_predictor(os.path.join(workdir, "bias_init.h5"), announce=False, summary=False)
    p_nb, c_nb = nobias.predict(fixture["predict_x"], batch_size=8, verbose=0)
    p_b, c_b = bias.predict(fixture["predict_x"], batch_size=8, verbose=0)
    results["parts_profile"] = p_nb + p_b
    results["parts_counts"] = np.logaddexp(c_nb, c_b)

    # The original's 1.x models, loaded by the JAX engine.
    for kind in ("bias", "chrombpnet"):
        model = engine.load_predictor(os.path.join(golden_dir, "{}_trace.h5".format(kind)),
                                      announce=False, summary=False)
        profile, counts = model.predict(fixture["predict_x"], batch_size=8, verbose=0)
        results["interop_{}_profile".format(kind)] = profile
        results["interop_{}_counts".format(kind)] = counts
    np.savez(os.path.join(out, "jax.npz"), **results)

    # export --legacy-h5, for the TensorFlow-side workers to load.
    from chrombpnet.engines.jax.export_legacy_h5 import export_legacy_h5

    for name, source in EXPORTS.items():
        export_legacy_h5(os.path.join(out, source), os.path.join(out, "export_{}.h5".format(name)))

    # Bias rescale, through the engine.
    scaled_path = os.path.join(out, "bias_model_scaled.keras")
    engine.rescale_bias_logcounts(os.path.join(workdir, "bias_init.h5"), scaled_path,
                                  fixture["rescale_x"], fixture["rescale_cts"])
    scaled = engine.load_predictor(scaled_path, announce=False, summary=False)
    profile, counts = scaled.predict(fixture["predict_x"], batch_size=8, verbose=0)
    np.savez(os.path.join(out, "rescale.npz"), profile=profile, counts=counts)

    # DeepSHAP, through the engine, as contribs_bw calls it.
    from chrombpnet.engines.jax.attribution import SequentialReferences

    seqs = fixture["shap_x"]
    scores = {"references": SequentialReferences(NUM_SHUFFLES, SHAP_SEED)(seqs, 0, len(seqs))}
    for name, path in (("bias", os.path.join(workdir, "bias_init.h5")),
                       ("nobias", os.path.join(golden_dir, "chrombpnet_trace_nobias.h5"))):
        model = engine.load_attribution_model(path, "cpu")
        for head, values in engine.attribute(model, seqs, ["counts", "profile"], device="cpu",
                                             batch_size=NUM_SHUFFLES, num_shuffles=NUM_SHUFFLES,
                                             seed=SHAP_SEED):
            scores["{}_{}".format(name, head)] = values
    np.savez(os.path.join(out, "shap.npz"), **scores)


#: Exported model -> the JAX-engine checkpoint it is exported from. Their
#: JAX predictions are jax.npz's "<name>_profile"/"<name>_counts".
EXPORTS = {
    "bias": "bias_built.keras",
    "chrombpnet": "chrombpnet_built.keras",
    "nobias": "chrombpnet_built_nobias.keras",
}


def _predict_exports(load, workdir, jax_dir, name):
    fixture = np.load(os.path.join(workdir, "fixture.npz"))
    results = {}
    for model_name in EXPORTS:
        model = load(os.path.join(jax_dir, "export_{}.h5".format(model_name)))
        profile, counts = model.predict(fixture["predict_x"], batch_size=8)
        results[model_name + "_profile"], results[model_name + "_counts"] = profile, counts
    np.savez(os.path.join(jax_dir, name + ".npz"), **results)


def tf_export(workdir, jax_dir):
    """This package's TensorFlow engine loads the exports."""
    _import_this_package()
    from chrombpnet import engines
    _tf_engine()._pin_tensorflow()
    engine = engines.get("tensorflow")
    _predict_exports(lambda path: engine.load_predictor(path, announce=False, summary=False),
                     workdir, jax_dir, "tf_export")


def golden_export(workdir, jax_dir, original_repo):
    """The original implementation loads the exports."""
    tf_engine = _tf_engine()
    side = tf_engine.Reference(original_repo)
    tf_engine._pin_tensorflow()
    _predict_exports(side.load_predictor, workdir, jax_dir, "golden_export")


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.rows = []

    def check(self, label, ok, detail=""):
        self.rows.append((label, bool(ok)))
        print("{:<64} {}  {}".format(label, "PASS" if ok else "FAIL", detail))
        sys.stdout.flush()

    def close(self, label, a, b, atol):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.check(label, a.shape == b.shape and err <= atol,
                   "max |delta| {:.2e} (<= {:g})".format(err, atol))

    def relative(self, label, a, b, rtol):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        err = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
        self.check(label, a.shape == b.shape and err <= rtol,
                   "relative {:.2e} (<= {:g})".format(err, rtol))

    def relative_l2(self, label, a, b, rtol):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        err = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        self.check(label, a.shape == b.shape and err <= rtol,
                   "relative L2 {:.2e} (<= {:g})".format(err, rtol))

    def equal(self, label, a, b):
        a, b = np.asarray(a), np.asarray(b)
        self.check(label, a.shape == b.shape and np.array_equal(a, b), "bitwise")

    @property
    def failed(self):
        return [label for label, ok in self.rows if not ok]


def _weights_agree(report, label, golden_weights, jax_weights):
    names = sorted(golden_weights)
    missing = sorted(set(names) ^ set(jax_weights))
    report.check(label + ": same weights", not missing, ", ".join(missing[:4]))
    if missing:
        return
    a = np.concatenate([golden_weights[n].ravel() for n in names]).astype(np.float64)
    b = np.concatenate([jax_weights[n].ravel() for n in names]).astype(np.float64)
    report.check(label + ": final weights, overall",
                 np.linalg.norm(a - b) / np.linalg.norm(a) <= 5e-3,
                 "relative L2 {:.2e} (<= 5e-3)".format(np.linalg.norm(a - b) / np.linalg.norm(a)))
    worst, worst_name = 0.0, ""
    for name in names:
        if name.endswith("prof_out_precrop/1"):
            continue
        g, j = golden_weights[name].astype(np.float64), jax_weights[name].astype(np.float64)
        err = np.linalg.norm(g - j) / max(np.linalg.norm(g), 1e-12)
        if err > worst:
            worst, worst_name = err, name
    report.check(label + ": final weights, per array", worst <= 5e-2,
                 "worst relative L2 {:.2e} at {} (<= 5e-2)".format(worst, worst_name))


def compare(golden_dir, golden_dir2, jax_dir, report):
    g, g2, j = (np.load(os.path.join(d, "golden.npz" if d != jax_dir else "jax.npz"))
                for d in (golden_dir, golden_dir2, jax_dir))
    for key in sorted(g.files):
        report.equal("self: golden " + key, g[key], g2[key])

    for key in ("bias_profile", "bias_counts", "chrombpnet_profile", "chrombpnet_counts",
                "nobias_profile", "nobias_counts"):
        report.close("predict: " + key, j[key], g[key], 1e-4)
    report.close("full model = parts: profile", j["chrombpnet_profile"], j["parts_profile"], 1e-5)
    report.close("full model = parts: counts", j["chrombpnet_counts"], j["parts_counts"], 1e-5)
    report.equal(".keras round trip: profile", j["roundtrip_profile"], j["chrombpnet_profile"])
    report.equal(".keras round trip: counts", j["roundtrip_counts"], j["chrombpnet_counts"])

    for kind in ("bias", "chrombpnet"):
        for name in ("loss", "logits_profile_predictions_loss", "logcount_predictions_loss"):
            key = "{}_test|{}".format(kind, name)
            report.relative("test_on_batch {}: {}".format(kind, name), j[key], g[key], 1e-5)
        trace_g, trace_j = g[kind + "_trace"], j[kind + "_trace"]
        report.relative("trace {}: losses at step 1".format(kind), trace_j[0], trace_g[0], 1e-5)
        report.relative("trace {}: losses at step {}".format(kind, TRACE_STEPS),
                        trace_j[-1], trace_g[-1], 1e-3)
        _weights_agree(report, "trace " + kind,
                       _load_npz(os.path.join(golden_dir, kind + "_trace_weights.npz")),
                       _load_npz(os.path.join(jax_dir, kind + "_trace_weights.npz")))
        report.close("interop: JAX loads the original's {} model: profile".format(kind),
                     j["interop_{}_profile".format(kind)], g[kind + "_trace_profile"], 1e-4)
        report.close("interop: JAX loads the original's {} model: counts".format(kind),
                     j["interop_{}_counts".format(kind)], g[kind + "_trace_counts"], 1e-4)

    rg, rj = (np.load(os.path.join(d, "rescale.npz")) for d in (golden_dir, jax_dir))
    for key in ("profile", "counts"):
        report.close("rescale: " + key, rj[key], rg[key], 1e-4)

    for reader in ("tf_export", "golden_export"):
        loaded = np.load(os.path.join(jax_dir, reader + ".npz"))
        who = "TensorFlow engine" if reader == "tf_export" else "original"
        for name in EXPORTS:
            for output in ("profile", "counts"):
                key = "{}_{}".format(name, output)
                report.close("export --legacy-h5: {} {} loaded by the {}".format(name, output, who),
                             loaded[key], j[key], 1e-4)

    sg, sj = (np.load(os.path.join(d, "shap.npz")) for d in (golden_dir, jax_dir))
    report.equal("shap: references", sj["references"], sg["references"])
    for name in ("bias", "nobias"):
        for head in ("counts", "profile"):
            key = "{}_{}".format(name, head)
            report.relative_l2("shap: {} hypothetical scores".format(key), sj[key], sg[key], 1e-4)
        departure = _tf_engine().relative_l2(sj[name + "_profile"], sg[name + "_profile_midpoint"])
        minimum = _tf_engine().MIN_PROFILE_DEPARTURE
        report.check("shap: {}_profile departs from the original's midpoint".format(name),
                     departure > minimum, "relative L2 {:.2e} (> {:g})".format(departure, minimum))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _run(python, mode, workdir, out, arguments, log_name):
    command = [python, str(Path(__file__).resolve()), "--worker", mode,
               "--workdir", workdir, "--out", out, "--original-repo", arguments.original_repo]
    env = dict(os.environ, **PINNED_ENV)
    env.pop("PYTHONPATH", None)
    log = os.path.join(workdir, "logs", log_name + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as handle:
        completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = Path(log).read_text().splitlines()[-30:]
        raise RuntimeError("{} failed:\n{}".format(log_name, "\n".join(tail)))


def main(arguments):
    tf_engine = _tf_engine()
    tf_engine._check_original_repo(arguments.original_repo)
    workdir = tempfile.mkdtemp(prefix="chrombpnet-jax-parity-")
    print("workdir:", workdir)
    golden_dir, golden_dir2, jax_dir = (os.path.join(workdir, n) for n in ("golden", "golden2", "jax"))
    try:
        print("preparing the fixture, batch stream and initial weights ...")
        command = [arguments.tf_python, str(Path(tf_engine.__file__).resolve()), "--worker",
                   "--task", "prepare", "--workdir", workdir, "--out", workdir,
                   "--original-repo", arguments.original_repo]
        env = dict(os.environ, **tf_engine.PINNED_ENV)
        env.pop("PYTHONPATH", None)
        subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
        for out in (golden_dir, golden_dir2):
            print("running the original implementation ({}) ...".format(os.path.basename(out)))
            _run(arguments.tf_python, "golden", workdir, out, arguments, os.path.basename(out))
        _run(arguments.tf_python, "golden_shap", workdir, golden_dir, arguments, "golden-shap")
        print("running the JAX engine ...")
        golden_link = os.path.join(jax_dir, ".golden")
        os.makedirs(jax_dir)
        os.symlink(golden_dir, golden_link)
        _run(arguments.jax_python, "jax", workdir, jax_dir, arguments, "jax")
        _run(arguments.tf_python, "tf_export", workdir, jax_dir, arguments, "tf-export")
        _run(arguments.tf_python, "golden_export", workdir, jax_dir, arguments, "golden-export")
        report = Report()
        compare(golden_dir, golden_dir2, jax_dir, report)
        if report.failed:
            print("\n{} of {} checks failed.".format(len(report.failed), len(report.rows)))
            return 1
        print("\nAll {} checks passed: the JAX engine agrees with the original implementation "
              "within the tolerances above.".format(len(report.rows)))
        return 0
    finally:
        if arguments.keep:
            print("kept", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def parser():
    tf_engine = _tf_engine()
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument("--original-repo", default=tf_engine.DEFAULT_ORIGINAL_REPO)
    result.add_argument("--tf-python", default=_sibling_python("chrombpnet-tensorflow"))
    result.add_argument("--jax-python", default=_sibling_python("chrombpnet-jax"))
    result.add_argument("--keep", action="store_true", help="keep the work directory")
    result.add_argument("--worker", help=argparse.SUPPRESS)
    result.add_argument("--workdir", help=argparse.SUPPRESS)
    result.add_argument("--out", help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.worker == "golden":
        golden(arguments.workdir, arguments.out, arguments.original_repo)
    elif arguments.worker == "golden_shap":
        golden_shap(arguments.workdir, arguments.out, arguments.original_repo)
    elif arguments.worker == "jax":
        jax_side(arguments.workdir, os.path.join(arguments.out, ".golden"), arguments.out)
    elif arguments.worker == "tf_export":
        tf_export(arguments.workdir, arguments.out)
    elif arguments.worker == "golden_export":
        golden_export(arguments.workdir, arguments.out, arguments.original_repo)
    elif arguments.worker:
        raise SystemExit("unknown worker {}".format(arguments.worker))
    else:
        raise SystemExit(main(arguments))
