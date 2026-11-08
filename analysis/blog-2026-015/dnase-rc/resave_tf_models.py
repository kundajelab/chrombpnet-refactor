#!/usr/bin/env python
"""Re-save the old TensorFlow ChromBPNet models that Python 3.8 cannot load.

The chromatin-atlas `chrombpnet_model_feb15*` runs were saved by Keras 2.4
under Python 3.7. Keras stores a Lambda layer as marshalled Python bytecode,
and bytecode only loads in the Python that wrote it, so the `chrombpnet` env
(Python 3.8, the original package run_compare.sh scores them with) fails on
them with "ValueError: bad marshal data". Their only Lambda is the count
head, `logcount_predictions`:

    lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)

So each model is rebuilt from its own saved architecture with that one
function re-marshalled under Python 3.8, its weights loaded unchanged, and
saved as a copy. The originals are never written to.

    <root>/standard_tensorflow/<ASSAY>/<ENCID>/fold_<N>/
        chrombpnet.h5               the re-saved model (run_compare.sh uses
                                    it in place of the original)
        reference_outputs_py37.npz  the ORIGINAL model's outputs on fixed
                                    random sequences, under Python 3.7
        resave.json                 source, checks, and their results
        own_test_metrics.json       the copy scored on the run's own test
                                    set (verify), to set against the run's
                                    recorded chrombpnet_metrics.json

Three steps, each in the interpreter that can do it (resave_tf_models.sh
runs all three):

    reference  Python 3.7 (env bpnet): load each original, record outputs
    resave     Python 3.8 (env chrombpnet): rebuild, check every weight array
               is identical, save, reload the way the original package does,
               and match the Python 3.7 outputs
    verify     Python 3.8 + GPU: score the copy on the run's own test set with
               the original package, against its recorded metrics

Which models: every TENSORFLOW_DIR in the ids table whose Lambda bytecode
loads under Python 3.7 and not under 3.8. Each step decides by trying to
load it in its own interpreter, so nothing is listed by hand.

Written to run under both Python 3.7 and 3.8.
"""

import argparse
import base64
import hashlib
import json
import marshal
import os
import subprocess
import sys
import time

ROOT = "/path/to/workdir/chromatlas/test/chrombpnet-refactor-dnase-rc"
IDS = ("/path/to/workdir/sherlock/ids/chromatlas/"
       "chrombpnet-refactor/ids_chrombpnet_refactor_benchmark_data_aug.txt")
ATLAS_DIR = "/path/to/lab/projects/chromatin-atlas-2022"
REFERENCE_DIR = "/path/to/workdir/chromatlas/reference"
SUBDIR = "standard_tensorflow"

# The one Lambda these models have, and what it computes (disassembled from
# the Python 3.7 bytecode: anusri's chrombpnet_with_bias_model.py line 128).
COUNT_HEAD = "logcount_predictions"
N_REFERENCE, SEED = 64, 0
# Outputs from TF 2.4 and TF 2.8 on CPU agree to float32 rounding; a rebuilt
# head that computed something else would be off by orders of magnitude more.
TOLERANCE = 1e-3


def first_file(*paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def runs(ids_path):
    """(assay, encid, fold, tf_dir, model .h5) for each ids row with a TF model."""
    with open(ids_path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cols = [header.index(c) for c in ("ASSAY", "ENCID", "FOLD_NUM", "TENSORFLOW_DIR")]
        out = []
        for line in fh:
            f = line.rstrip("\r\n").split("\t")
            if len(f) <= max(cols) or not f[cols[3]]:
                continue
            assay, encid, fold, tf_dir = (f[i] for i in cols)
            tf_dir = tf_dir.rstrip("/")
            h5 = first_file(os.path.join(tf_dir, "models", "chrombpnet.h5"),
                            os.path.join(tf_dir, "chrombpnet.h5"))
            if h5:
                out.append((assay, encid, fold, tf_dir, h5))
    return out


def out_dir(root, assay, encid, fold):
    return os.path.join(root, SUBDIR, assay, encid, "fold_" + fold)


def model_config(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        return json.loads(f.attrs["model_config"])


def lambda_layers(config):
    """Every Lambda layer config in a (possibly nested) model config."""
    found = []

    def walk(c):
        if isinstance(c, dict):
            if c.get("class_name") == "Lambda":
                found.append(c["config"])
            for v in c.values():
                walk(v)
        elif isinstance(c, list):
            for v in c:
                walk(v)
    walk(config)
    return found


def bytecode_loads_here(h5_path):
    for lam in lambda_layers(model_config(h5_path)):
        try:
            marshal.loads(base64.b64decode(lam["function"][0]))
        except (ValueError, EOFError, TypeError):
            return False
    return True


def targets(args, loads_here):
    """The models to act on: those whose bytecode does (3.7) / does not (3.8) load."""
    picked = []
    for assay, encid, fold, tf_dir, h5 in runs(args.ids):
        if bytecode_loads_here(h5) == loads_here:
            picked.append((assay, encid, fold, tf_dir, h5))
    return picked


def reference_inputs(width):
    import numpy as np
    seq = np.random.RandomState(SEED).randint(0, 4, (N_REFERENCE, width))
    return np.eye(4, dtype="float32")[seq]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def step_reference(args):
    import numpy as np
    import tensorflow as tf
    for assay, encid, fold, tf_dir, h5 in targets(args, loads_here=True):
        d = out_dir(args.root, assay, encid, fold)
        dest = os.path.join(d, "reference_outputs_py37.npz")
        if os.path.exists(dest) and not args.force:
            print("exists, skipped: " + dest)
            continue
        os.makedirs(d, exist_ok=True)
        model = tf.keras.models.load_model(h5, compile=False, custom_objects={"tf": tf})
        x = reference_inputs(model.input_shape[1])
        profile, counts = model.predict(x, batch_size=16)
        np.savez(dest, profile=profile, counts=counts, source=h5,
                 python=sys.version.split()[0], tensorflow=tf.__version__)
        print("reference outputs: %s %s fold %s -> %s" % (assay, encid, fold, dest))


def step_resave(args):
    import h5py
    import numpy as np
    import tensorflow as tf
    from keras.utils.generic_utils import func_dump
    from tensorflow.keras.utils import get_custom_objects

    # Compiled against a globals dict so `tf` is a global lookup, as in the
    # original bytecode (LOAD_GLOBAL tf). Written inline here, `tf` would be
    # captured as a closure cell holding the module, which Keras cannot store.
    count_head = eval("lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)",
                      {"tf": tf})

    def digests(arrays):
        return sorted(hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
                      + str(a.shape) for a in arrays)

    for assay, encid, fold, tf_dir, h5 in targets(args, loads_here=False):
        d = out_dir(args.root, assay, encid, fold)
        dest = os.path.join(d, "chrombpnet.h5")
        if os.path.exists(dest) and not args.force:
            print("exists, skipped: " + dest)
            continue
        ref_path = os.path.join(d, "reference_outputs_py37.npz")
        if not os.path.exists(ref_path):
            sys.exit("no %s -- run the reference step (Python 3.7) first" % ref_path)

        config = model_config(h5)
        lams = lambda_layers(config)
        names = [lam["name"] for lam in lams]
        if names != [COUNT_HEAD]:
            sys.exit("%s: Lambda layers %s, expected only %s; not rebuilding a "
                     "function this script has not checked" % (h5, names, COUNT_HEAD))
        lams[0]["function"] = func_dump(count_head)
        lams[0]["function_type"] = "lambda"

        model = tf.keras.models.model_from_json(json.dumps(config),
                                                custom_objects={"tf": tf})
        model.load_weights(h5)

        # Every saved array, matched by content against what was loaded.
        saved = []
        with h5py.File(h5, "r") as f:
            f["model_weights"].visititems(
                lambda _n, o: saved.append(o[()]) if isinstance(o, h5py.Dataset) else None)
        weights_identical = digests(saved) == digests(model.get_weights())
        if not weights_identical:
            sys.exit("%s: loaded weights differ from the saved ones" % h5)

        os.makedirs(d, exist_ok=True)
        model.save(dest, include_optimizer=False)

        # Reload the copy the way the original package's predict.py does.
        get_custom_objects().update({"tf": tf})
        reloaded = tf.keras.models.load_model(dest, compile=False)
        ref = np.load(ref_path)
        with tf.device("/CPU:0"):
            profile, counts = reloaded.predict(reference_inputs(reloaded.input_shape[1]),
                                               batch_size=16)
        diff_profile = float(np.max(np.abs(profile - ref["profile"])))
        diff_counts = float(np.max(np.abs(counts - ref["counts"])))
        outputs_match = max(diff_profile, diff_counts) < TOLERANCE

        with h5py.File(h5, "r") as f:
            keras_version = f.attrs.get("keras_version")
        if isinstance(keras_version, bytes):
            keras_version = keras_version.decode()
        write_json(os.path.join(d, "resave.json"), {
            "source": h5,
            "source_md5": md5(h5),
            "source_keras_version": str(keras_version),
            "resaved": dest,
            "resaved_with": {"python": sys.version.split()[0],
                             "tensorflow": tf.__version__},
            "rebuilt_lambda": COUNT_HEAD + ": tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)",
            "weights": {"arrays": len(saved), "identical": weights_identical},
            "outputs_vs_original_python37": {
                "sequences": N_REFERENCE, "seed": SEED,
                "reference": {"python": str(ref["python"]), "tensorflow": str(ref["tensorflow"])},
                "max_abs_diff_profile_logits": diff_profile,
                "max_abs_diff_logcounts": diff_counts,
                "tolerance": TOLERANCE, "match": outputs_match},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print("%s %s fold %s: %d weight arrays identical; outputs vs Python 3.7 "
              "original: profile %.2e, counts %.2e -> %s"
              % (assay, encid, fold, len(saved), diff_profile, diff_counts,
                 "OK" if outputs_match else "MISMATCH"))
        if not outputs_match:
            sys.exit("outputs differ from the original beyond %g" % TOLERANCE)


def step_verify(args):
    import chrombpnet
    where = os.path.dirname(os.path.dirname(os.path.realpath(chrombpnet.__file__)))
    print("scoring with the chrombpnet package in " + where)
    for assay, encid, fold, tf_dir, h5 in runs(args.ids):
        d = out_dir(args.root, assay, encid, fold)
        model = os.path.join(d, "chrombpnet.h5")
        record = os.path.join(d, "resave.json")
        if not (os.path.exists(model) and os.path.exists(record)):
            continue
        with open(record) as fh:
            info = json.load(fh)
        if "own_test" in info and not args.force:
            print("exists, skipped: own-test check of " + model)
            continue
        peaks = first_file(*(os.path.join(tf_dir, p) for p in (
            "auxiliary/filtered.peaks.bed", "auxiliary/filtered.peaks.bed.gz",
            "filtered.peaks.bed", "filtered.peaks.bed.gz")))
        bigwig = os.path.join(ATLAS_DIR, assay, encid, "preprocessing", "bigWigs",
                              encid + ".bigWig")
        recorded_path = first_file(os.path.join(tf_dir, "evaluation", "chrombpnet_metrics.json"),
                                   os.path.join(tf_dir, "chrombpnet_metrics.json"))
        prefix = os.path.join(d, "own_test")
        subprocess.run([sys.executable, "-m", "chrombpnet.training.predict",
                        "-m", model, "-g", os.path.join(REFERENCE_DIR, "hg38", "hg38.fa"),
                        "-b", bigwig, "-p", peaks, "-n", "None",
                        "-fl", os.path.join(REFERENCE_DIR, "folds", "fold_%s.json" % fold),
                        "-bs", str(args.batch_size), "-o", prefix], check=True)
        if os.path.exists(prefix + "_predictions.h5"):
            os.remove(prefix + "_predictions.h5")
        with open(prefix + "_metrics.json") as fh:
            got = json.load(fh)
        with open(recorded_path) as fh:
            recorded = json.load(fh)
        rows = {}
        for group, key in (("counts_metrics", "pearsonr"), ("profile_metrics", "median_jsd")):
            a, b = got[group]["peaks"][key], recorded[group]["peaks"][key]
            rows[key] = {"resaved": a, "recorded": b, "diff": a - b}
        info["own_test"] = {"peaks": peaks, "bigwig": bigwig, "recorded": recorded_path,
                            "batch_size": args.batch_size, "package": where,
                            "metrics": rows}
        write_json(record, info)
        print("%s %s fold %s own test: pearsonr %.6f (recorded %.6f), median_jsd %.6f "
              "(recorded %.6f)" % (assay, encid, fold, rows["pearsonr"]["resaved"],
                                   rows["pearsonr"]["recorded"], rows["median_jsd"]["resaved"],
                                   rows["median_jsd"]["recorded"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["reference", "resave", "verify"])
    ap.add_argument("--ids", default=IDS)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="verify: prediction batch size (as run_compare.sh)")
    ap.add_argument("--force", action="store_true", help="redo work that exists")
    args = ap.parse_args(argv)
    {"reference": step_reference, "resave": step_resave, "verify": step_verify}[args.step](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
