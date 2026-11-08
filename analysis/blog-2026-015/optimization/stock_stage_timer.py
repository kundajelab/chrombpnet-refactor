"""Stock chrombpnet's pred_bw, unchanged, with each stage the optimization kit
replaces timed: for the blog's "Measured effect (in original chrombpnet)"
column, one row per kit lever (summarize_kit_levers.py pairs these with the
kit's own stage timings from its run record).

    python -I stock_stage_timer.py STAGES_JSON pred_bw <stock pred_bw arguments>

Wraps, by module attribute, the functions stock's pred_bw calls (its code and
call order are untouched):

    load_model     predict_to_bigwig.load_model_wrapper
    featurise      bigwig_helper.get_seq (pyfaidx + one-hot)       -> prefetch
    forward        the loaded model's predict                     -> forward_route
    bigwig_write   bigwig_helper.write_bigwig                     -> bigwig_writer
    h5_write       predict_to_bigwig.write_predictions_h5py       -> h5_writer
    metrics        compare_with_observed minus h5_write           -> metrics_stage
    finish_imports chrombpnet.training.metrics (SciPy, Matplotlib),
                   imported where compare_with_observed would      -> preimport
and records when main() returned; the caller's process-exit time minus that is
the interpreter and CUDA teardown (-> exit_fast). Refuses to report if any kit
module was imported, so the numbers are stock's.
"""

import json
import os
import sys
import time

STAGES = {"process_start_epoch": time.time()}


def _timed(name, function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            STAGES[name] = STAGES.get(name, 0.0) + time.perf_counter() - start
    return wrapper


def main():
    out_path, argv = sys.argv[1], sys.argv[2:]
    start = time.perf_counter()
    import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
    import chrombpnet.evaluation.make_bigwigs.predict_to_bigwig as predict_to_bigwig  # imports TensorFlow
    from chrombpnet.CHROMBPNET import main as stock_main
    STAGES["imports"] = time.perf_counter() - start

    bigwig_helper.get_seq = _timed("featurise", bigwig_helper.get_seq)
    bigwig_helper.write_bigwig = _timed("bigwig_write", bigwig_helper.write_bigwig)
    predict_to_bigwig.write_predictions_h5py = _timed("h5_write", predict_to_bigwig.write_predictions_h5py)

    compare = predict_to_bigwig.compare_with_observed
    def compare_with_observed(*args, **kwargs):
        begin = time.perf_counter()
        import chrombpnet.training.metrics  # noqa: F401 -- the first call's import, timed apart
        STAGES["finish_imports"] = STAGES.get("finish_imports", 0.0) + time.perf_counter() - begin
        return _timed("compare_with_observed", compare)(*args, **kwargs)
    predict_to_bigwig.compare_with_observed = compare_with_observed

    loader = predict_to_bigwig.load_model_wrapper
    def load_model_wrapper(model_hdf5):
        model = _timed("load_model", loader)(model_hdf5)
        model.predict = _timed("forward", model.predict)
        return model
    predict_to_bigwig.load_model_wrapper = load_model_wrapper

    sys.argv = ["chrombpnet"] + argv
    begin = time.perf_counter()
    stock_main()
    STAGES["main"] = time.perf_counter() - begin
    STAGES["metrics"] = STAGES.get("compare_with_observed", 0.0) - STAGES.get("h5_write", 0.0)
    # The kit env's .pth imports chrombpnet_opt._autoload into every process;
    # with CHROMBPNET_OPT unset it installs nothing (its docstring), as in the
    # kit's own `--mode off` stock child. Anything else of the kit's is refused.
    inert = {"chrombpnet_opt", "chrombpnet_opt._autoload"}
    kit = sorted(m for m in sys.modules
                 if m.split(".")[0] in ("chrombpnet_opt", "chrombpnet_fastkit", "chrombpnet_k1") and m not in inert)
    mode = (os.environ.get("CHROMBPNET_OPT") or "").strip().lower()
    if kit or mode not in ("", "off"):
        sys.exit("stock_stage_timer: kit code was active (modules %s, CHROMBPNET_OPT=%r); these would not be stock's timings"
                 % (", ".join(kit[:5]) or "none", mode))
    STAGES["main_return_epoch"] = time.time()
    with open(out_path, "w") as handle:
        json.dump({k: (round(v, 3) if isinstance(v, float) and not k.endswith("_epoch") else v)
                   for k, v in STAGES.items()}, handle, indent=1)


if __name__ == "__main__":
    main()
