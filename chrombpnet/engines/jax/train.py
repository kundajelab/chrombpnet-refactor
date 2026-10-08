"""Training for the JAX engine.

``get_model`` and ``fit_and_evaluate`` are adapted from kundajelab/chrombpnet#284
(NNFC-GMD/chrombpnet@14ca773), chrombpnet/training/train.py: its architecture
loader, callbacks (in the 1.x order, with ``SwapEMAWeights`` first under
``--ema`` and a float32 checkpoint under ``--precision bf16``), ``model.fit``
call and saves. The PR does part of this in its ``train.main``; here the shared
``train.main`` builds the model before the generators and writes
``<prefix>.args.json`` itself, so the engine does the rest. What differs:

* device, precision and seeding (the PR's ``train.main``) happen in
  ``get_model``, and the precision it sets is restored when ``fit`` ends;
* batches come from the shared generator through
  :class:`~chrombpnet.engines.jax.sequence_adapter.KerasPyDataset`, and
  ``fit(shuffle=...)`` follows ``--reproducibility`` (True without it, as in
  the PR);
* ``--lr-schedule cosine`` needs the number of training steps, which the
  architecture file cannot know when it builds the optimizer; the PR built the
  training generator first, and here the model is recompiled with the schedule
  once ``fit`` has the generator. The argument namespace is not modified;
* checkpoints are ``.keras`` files, this engine's format;
* ``--reproducibility-max`` assigns the shared Keras-layout initial weights
  and asks XLA for deterministic GPU kernels.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
import sys

import keras

from chrombpnet.engines.jax import callbacks, model_io, runtime
from chrombpnet.engines.jax import DETERMINISTIC_XLA_FLAG as _DETERMINISTIC_FLAG
from chrombpnet.engines.jax.optimizers import make_optimizer
from chrombpnet.engines.jax.sequence_adapter import KerasPyDataset
from chrombpnet.engines.keras_weights import assign_canonical_weights
from chrombpnet.training.utils.benchmark_utils import (
    benchmark_enables,
    use_keras_batch_order_shuffle,
)

#: The CLI's ``--trackables`` default, for callers whose namespace lacks it.
DEFAULT_TRACKABLES = [
    "logcount_predictions_loss", "loss", "logits_profile_predictions_loss",
    "val_logcount_predictions_loss", "val_loss",
    "val_logits_profile_predictions_loss",
]

#: Checkpoint suffix this engine writes.
CHECKPOINT_SUFFIX = ".keras"

#: (dtype policy, matmul precision) to restore when training ends.
_SAVED_PRECISION = []


def _load_architecture(path):
    # load under a real module name (not ''), so objects defined in the file get a usable __module__
    stem = re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
    spec = importlib.util.spec_from_file_location("chrombpnet_architecture_" + stem, path)
    architecture_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = architecture_module
    spec.loader.exec_module(architecture_module)
    return architecture_module


def _enable_deterministic_kernels():
    """XLA reads XLA_FLAGS when it creates its GPU client, so set it before.

    From the command line ``JaxEngine.prepare`` already has; this covers
    library callers that reach ``get_model`` directly."""
    import jax

    flags = os.environ.get("XLA_FLAGS", "")
    if _DETERMINISTIC_FLAG in flags:
        return
    if runtime.backends_initialized():
        if jax.default_backend() != "cpu":
            raise RuntimeError(
                "--reproducibility-max needs deterministic GPU kernels, which JAX can only be given before it "
                "starts; set XLA_FLAGS={} in the environment".format(_DETERMINISTIC_FLAG))
        return
    os.environ["XLA_FLAGS"] = (flags + " " + _DETERMINISTIC_FLAG).strip()


def _save_precision():
    import jax

    _SAVED_PRECISION.append(
        (keras.config.dtype_policy(), jax.config.jax_default_matmul_precision))


def _restore_precision():
    import jax

    while _SAVED_PRECISION:
        policy, matmul_precision = _SAVED_PRECISION.pop()
        keras.config.set_dtype_policy(policy)
        jax.config.update("jax_default_matmul_precision", matmul_precision)


def get_model(args, parameters):
    model_io.require_jax_backend()
    runtime.configure_device(getattr(args, "device", None))
    if benchmark_enables(args, "deterministic_kernels"):
        _enable_deterministic_kernels()
    # --precision bf16 / highest set a process-wide dtype policy / matmul
    # precision; fit restores them, so later pipeline steps (predict,
    # interpret) build, load and run models as before
    _save_precision()
    try:
        runtime.configure_precision(getattr(args, "precision", None) or "default")
        # seeds python random (initial weights, batch order), numpy and keras
        keras.utils.set_random_seed(args.seed)
        architecture_module = _load_architecture(args.architecture_from_file)
        builder = getattr(architecture_module, "getModelGivenModelOptionsAndWeightInits", None)
        if builder is None:
            raise AttributeError(
                "Architecture must define getModelGivenModelOptionsAndWeightInits(args, parameters)")
        build_args = args
        if getattr(args, "lr_schedule", None) == "cosine" and not getattr(args, "total_steps", None):
            # The architecture file builds the optimizer, and a cosine
            # schedule needs the step count, which only fit knows; build with
            # a constant rate here, and fit compiles the schedule in.
            build_args = copy.copy(args)
            build_args.lr_schedule = "constant"
        model = builder(build_args, parameters)
        print("got the model")
        if benchmark_enables(args, "shared_initial_weights"):
            assign_canonical_weights(model, parameters, args.seed)
    except BaseException:
        _restore_precision()
        raise
    return model, architecture_module


def _recompile(model, optimizer):
    """Compile again with a new optimizer and every other argument unchanged."""
    config = dict(model._compile_config.config)
    config["optimizer"] = optimizer
    model.compile(**config)


class Float32ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint that writes the model with float32 policies (used under --precision bf16), so that the
    best-so-far model left by an interrupted run loads and runs in float32 like the final one. The layers go back
    to their training policies right after each save."""

    def _save_model(self, epoch, batch, logs):
        # every ModelCheckpoint save goes through _save_model (tests/test_training.py checks the interrupted case)
        with runtime.float32_policy(self.model):
            super()._save_model(epoch, batch, logs)


def fit_and_evaluate(model, train_gen, valid_gen, args, architecture_module):
    try:
        if getattr(args, "lr_schedule", None) == "cosine" and not getattr(args, "total_steps", None):
            # the schedule is part of the optimizer: give it the step budget
            schedule_args = copy.copy(args)
            schedule_args.total_steps = len(train_gen) * args.epochs
            _recompile(model, make_optimizer(schedule_args))

        model_output_path_name = args.output_prefix + CHECKPOINT_SUFFIX
        model_output_path_logs_name = args.output_prefix + ".log"

        checkpoint_class = Float32ModelCheckpoint if runtime.bf16_active() else keras.callbacks.ModelCheckpoint
        checkpointer = checkpoint_class(filepath=model_output_path_name, monitor="val_loss", mode="min",  verbose=1, save_best_only=True)
        # Keras 3 restores the best weights at the end of training even when --epochs is reached without an early stop
        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=args.early_stop, verbose=1, restore_best_weights=True)
        history = callbacks.LossHistory(model_output_path_logs_name + ".batch", getattr(args, "trackables", DEFAULT_TRACKABLES))
        csvlogger = keras.callbacks.CSVLogger(model_output_path_logs_name, append=False)
        cur_callbacks = [checkpointer, earlystopper, csvlogger, history]
        if getattr(args, "ema", False):
            # validate on the EMA weights and put them in the model at every epoch end, before the checkpoint and the
            # early stopping callbacks see the model, so the saved models hold EMA weights
            cur_callbacks.insert(0, keras.callbacks.SwapEMAWeights(swap_on_epoch=True))

        train_data, valid_data = KerasPyDataset(train_gen), KerasPyDataset(valid_gen)
        # Build the loss, metrics and optimizer from one batch now. Otherwise
        # fit starts a pass over the training data just to look at its first
        # batch -- shuffling the batch order once more and ending that pass
        # with an on_epoch_end; see sequence_adapter.
        model._symbolic_build(data_batch=train_data[0])

        model.fit(train_data,
                  validation_data=valid_data,
                  epochs=args.epochs,
                  verbose=1,
                  callbacks=cur_callbacks,
                  shuffle=use_keras_batch_order_shuffle(args))

        if runtime.bf16_active():
            # trained with mixed_bfloat16: save float32 models (the weights are float32 already) so that predict,
            # footprints and interpret run them in full precision
            runtime.set_float32_policy(model)

        print('save model')
        model.save(model_output_path_name)

        architecture_module.save_model_without_bias(model, args.output_prefix)
    finally:
        _restore_precision()
