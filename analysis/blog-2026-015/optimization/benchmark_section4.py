#!/usr/bin/env python3
"""One measurement of one of the blog's Section 4 changes, old or new form, in
this process. run_section4.sh runs every measurement in a fresh process, the
forms in a different order each round, and puts the right checkout on
PYTHONPATH for each.

    python -I benchmark_section4.py STAGE VARIANT WORKDIR

Prints one JSON line: stage, variant, seconds (the stage alone), rss_peak_mb
(the process), rss_added_mb (peak minus resident before the stage),
out_bytes (what the stage wrote, where it writes).

Stages; WORKDIR/inputs.json names their inputs (run_section4.sh writes it):
  gc      old | new             prep nonpeaks' GC matching on the fold's
                                candidates and foreground: the original
                                package's get_gc_matched_negatives.main (loaded
                                by path) | this package's
  shared  unshared | shared     pred_bw's per-model inputs for three models
                                (regions, sequences, chrom sizes, output
                                regions, observed counts): computed per model
                                (a fresh _SharedInputs each, as before) | once
  bigwig  list | numpy | fixed  one model's bigWig over the 20,000 regions, at
                                693d162 (PYTHONPATH): pyBigWig given Python
                                lists, as the original | NumPy arrays |
                                fixed-step sections (main's form)
  cast    legacy | batched      the forward over the 20,000 regions:
                                PredictionAdapter._predict_legacy (the whole
                                input cast to float32 up front) | predict (cast
                                per batch into a preallocated output)
  h5      gzip4 | gzip9 | gzip9threads
                                one model's *_predictions.h5: h5py's default
                                gzip, as the original and 693d162 (that file,
                                by path) | gzip 9 + shuffle on one thread
                                (faee27f) | the same filters on threads
"""

import argparse
import importlib.util
import json
import os
import resource
import sys
import time


def rss_now_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def rss_peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predictions(inputs):
    import h5py
    import numpy as np
    with h5py.File(inputs["predictions_h5"], "r") as handle:
        profs = handle["predictions/profs"][()]
        logcts = handle["predictions/logcounts"][()]
        chroms = [c.decode() if isinstance(c, bytes) else c for c in handle["coords/coords_chrom"][()]]
        centers = handle["coords/coords_center"][()]
    return profs, np.asarray(logcts), [[c, int(x)] for c, x in zip(chroms, centers)]


def stage_gc(variant, inputs, work):
    if variant == "old":
        module = load_by_path("original_gc_matching", inputs["original_gc_script"])
    else:
        from chrombpnet.helpers.make_gc_matched_negatives import get_gc_matched_negatives as module
    prefix = os.path.join(work, "gc_" + variant)
    args = argparse.Namespace(candidate_negatives=inputs["candidates"], foreground_gc_bed=inputs["foreground_gc"],
                              output_prefix=prefix, chr_fold_path=inputs["fold_json"],
                              # 1.x's split, 2/2/1 per peak (since 4f91f65 main reads each
                              # fold's ratio; 1.x's script reads only the train one)
                              neg_to_pos_ratio_train=2, neg_to_pos_ratio_val=2, neg_to_pos_ratio_test=1,
                              seed=1234)
    return (lambda: module.main(args)), (lambda: os.path.getsize(prefix + ".bed"))


def stage_shared(variant, inputs, work):
    import pyfaidx
    from chrombpnet.evaluation.make_bigwigs import predict_to_bigwig as p2b
    args = argparse.Namespace(regions=inputs["regions"], chrom_sizes=inputs["chrom_sizes"],
                              bigwig=inputs["observed_bigwig"], debug_chr=None)
    genome = pyfaidx.Fasta(inputs["genome"])

    def one_pass(shared):
        frame = shared.regions_df()
        _, used = shared.sequences(2114)
        shared.chrom_sizes()
        shared.regions(2114, 1000)
        shared.observed_counts(frame.loc[used], 2114, 1000)

    def run():
        if variant == "shared":
            shared = p2b._SharedInputs(args, genome)
            for _ in range(3):
                one_pass(shared)
        else:
            for _ in range(3):
                shared = p2b._SharedInputs(args, genome)
                one_pass(shared)
                del shared
    return run, (lambda: 0)


def stage_bigwig(variant, inputs, work):
    import numpy as np
    from chrombpnet import optimize
    from chrombpnet.evaluation.make_bigwigs import bigwig_helper
    if not hasattr(optimize, "_ACTIVE"):
        sys.exit("benchmark_section4: the bigwig stage runs at 693d162 (PYTHONPATH=code-693d162)")
    if variant == "numpy" and not bigwig_helper._PYBIGWIG_NUMPY:
        sys.exit("benchmark_section4: this pyBigWig has no NumPy support; the numpy form would fall back to lists")

    class Forms:
        lazy_stats = True                      # the same for every form: only the write differs
        fixed_step_bigwig = variant == "fixed"
        numpy_writer_args = variant == "numpy"
    optimize._ACTIVE = Forms()
    profs, logcts, _ = predictions(inputs)
    data = profs * np.expm1(logcts.reshape(-1))[:, None]
    regions = bigwig_helper.get_regions(inputs["preds_bed"], profs.shape[1])
    sizes = bigwig_helper.read_chrom_sizes(inputs["chrom_sizes"])
    out = os.path.join(work, "bigwig_%s.bw" % variant)
    return (lambda: bigwig_helper.write_bigwig(data, regions, sizes, out)), (lambda: os.path.getsize(out))


def stage_cast(variant, inputs, work):
    import pyfaidx
    import torch
    from chrombpnet.engines.pytorch.model import load_prediction_model
    from chrombpnet.evaluation.make_bigwigs import predict_to_bigwig as p2b
    args = argparse.Namespace(regions=inputs["regions"], chrom_sizes=inputs["chrom_sizes"],
                              bigwig=None, debug_chr=None)
    seqs, _ = p2b._SharedInputs(args, pyfaidx.Fasta(inputs["genome"])).sequences(2114)
    model = load_prediction_model(inputs["chrombpnet_pt"], device="cuda")
    model.predict(seqs[:256], batch_size=256)   # CUDA context and kernels, outside the timing
    torch.cuda.synchronize()

    def run():
        if variant == "batched":
            model.predict(seqs, batch_size=256)
        else:
            model._predict_legacy(seqs, 256)
        torch.cuda.synchronize()
    return run, (lambda: 0)


def stage_h5(variant, inputs, work):
    if variant == "gzip4":
        module = load_by_path("predictions_693d162", inputs["old_predictions_py"])
    else:
        from chrombpnet import compression, predictions as module
        if variant == "gzip9":
            if hasattr(compression, "THREADED_MIN_BYTES"):
                compression.THREADED_MIN_BYTES = 1 << 62     # faee27f's single-threaded writer
        elif not hasattr(compression, "create_dataset"):
            sys.exit("benchmark_section4: this checkout has no threaded HDF5 writer")
    profs, logcts, coords = predictions(inputs)
    prefix = os.path.join(work, "h5_" + variant)
    return ((lambda: module.write_predictions_h5py(prefix, profs, logcts, coords, include_peak_flag=False)),
            (lambda: os.path.getsize(prefix + "_predictions.h5")))


STAGES = {"gc": stage_gc, "shared": stage_shared, "bigwig": stage_bigwig, "cast": stage_cast, "h5": stage_h5}


def main():
    stage, variant, work = sys.argv[1:4]
    inputs = json.load(open(os.path.join(work, "inputs.json")))
    out = os.path.join(work, "out")
    os.makedirs(out, exist_ok=True)
    run, size = STAGES[stage](variant, inputs, out)
    before = rss_now_mb()
    start = time.perf_counter()
    run()
    seconds = time.perf_counter() - start
    print(json.dumps({"stage": stage, "variant": variant, "seconds": round(seconds, 3),
                      "rss_peak_mb": round(rss_peak_mb(), 1), "rss_added_mb": round(rss_peak_mb() - before, 1),
                      "out_bytes": size()}), flush=True)


if __name__ == "__main__":
    main()
