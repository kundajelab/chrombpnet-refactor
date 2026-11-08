#!/usr/bin/env python3
"""chrombpnet-refactor's forward alone, on one route: blog 2026-015 Section 4's
inference row (run_sections34.sh's forward_eager / forward_fast / forward_exact
arms).

    python forward_timer.py OUT_JSON MODEL_PT REGIONS GENOME CHROM_SIZES MODE [BATCH] [REPEATS]

MODE: eager  PredictionAdapter's default forward (cuDNN, TF32 on: PyTorch's default)
      fast   --predict-fused-kernel-fast: the optimization kit's TF32 Triton kernels
      exact  --predict-fused-kernel-exact: the kit's fp32 Triton kernels

The regions are featurized once (as pred_bw does, _SharedInputs), the route is
built and one batch is predicted outside the timing (CUDA context, kernel
compilation, the fast route's build-time check), then the whole set is
predicted REPEATS times, each timed with a CUDA synchronize -- the same
measurement as benchmark_section4.py's `cast` stage. Writes the seconds per
repeat, their median, and the versions and route details.
"""

import argparse
import json
import os
import statistics
import sys
import time


def main():
    out, model_pt, regions, genome, chrom_sizes, mode = sys.argv[1:7]
    batch = int(sys.argv[7]) if len(sys.argv) > 7 else 256
    repeats = int(sys.argv[8]) if len(sys.argv) > 8 else 3
    if mode not in ("eager", "fast", "exact"):
        sys.exit("forward_timer: MODE must be eager, fast or exact")

    import pyfaidx
    import torch
    from chrombpnet import fused_kernels
    from chrombpnet.engines.pytorch.model import load_prediction_model
    from chrombpnet.evaluation.make_bigwigs import predict_to_bigwig as p2b

    namespace = argparse.Namespace(regions=regions, chrom_sizes=chrom_sizes, bigwig=None, debug_chr=None)
    seqs, _ = p2b._SharedInputs(namespace, pyfaidx.Fasta(genome)).sequences(2114)

    # FORWARD_DEVICE=cpu: a smoke test of this script, not a measurement.
    device = os.environ.get("FORWARD_DEVICE", "cuda")
    synchronize = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    fused_kernels.configure(None if mode == "eager" else mode)
    started = time.perf_counter()
    model = load_prediction_model(model_pt, device=device)
    model.predict(seqs[:batch], batch_size=batch)
    synchronize()
    setup = time.perf_counter() - started

    seconds = []
    for _ in range(repeats):
        begin = time.perf_counter()
        model.predict(seqs, batch_size=batch)
        synchronize()
        seconds.append(round(time.perf_counter() - begin, 3))

    forward = model._forward
    record = {
        "mode": mode, "regions": int(len(seqs)), "batch_size": batch, "seconds": seconds,
        "median_s": statistics.median(seconds), "setup_s": round(setup, 3),
        "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else device, "route": {
            key: getattr(forward, key) for key in ("arch", "tile", "precision", "tc_tile", "measured")
            if hasattr(forward, key)},
    }
    try:
        import triton
        record["triton"] = triton.__version__
    except Exception:
        record["triton"] = None
    with open(out, "w") as handle:
        json.dump(record, handle, indent=1)
    print("forward_timer:", json.dumps(record))


if __name__ == "__main__":
    main()
