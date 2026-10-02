"""Training for the TensorFlow engine.

``get_model`` and ``fit_and_evaluate`` are adapted from kundajelab/chrombpnet
@09938fd, chrombpnet/training/train.py. The callbacks, their arguments and
their order, the ``model.fit`` call and the two saves are the original's. What
differs:

* batches come from the shared generator through
  :class:`~chrombpnet.engines.tensorflow.sequence_adapter.KerasSequence`;
* the architecture file is loaded exactly as the original loads it,
  ``SourceFileLoader('', path).load_module()``, not with the shared loader:
  Keras records a ``Lambda`` layer's defining module name in the saved model,
  and the original's is the empty string;
* ``--benchmark`` controls that apply to Keras are honoured. At the default
  level none of them changes anything: ``fit`` shuffles batches exactly as the
  original's default ``shuffle=True`` does;
* fixed defects of the original: ``--seed`` is applied before the model is
  built, and the best validation checkpoint is restored even when training
  runs out of epochs.
"""

from __future__ import annotations

import importlib.machinery
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as tfcallbacks

from chrombpnet.engines.tensorflow import callbacks
from chrombpnet.engines.tensorflow.predictor import configure_device
from chrombpnet.engines.tensorflow.sequence_adapter import KerasSequence
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


def _enable_deterministic_kernels():
    import tensorflow as tf

    enable = getattr(tf.config.experimental, "enable_op_determinism", None)
    if enable is None:
        raise RuntimeError(
            "--benchmark max needs tf.config.experimental.enable_op_determinism")
    enable()


def get_model(args, parameters):
    configure_device(getattr(args, "device", None))
    if benchmark_enables(args, "deterministic_kernels"):
        _enable_deterministic_kernels()
    # Seed everything before the model is built. The original seeded only
    # NumPy here; its bias-factorized builder seeds TensorFlow and `random`
    # only after loading the bias model and building the trainable tower, so
    # --seed never reached that tower's initial weights (defect 6 in
    # chrombpnet_bugs.md). The builders still re-seed where they always did,
    # so everything drawn after construction is unchanged.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    architecture_module = importlib.machinery.SourceFileLoader(
        '', args.architecture_from_file).load_module()
    builder = getattr(
        architecture_module,
        "build_model",
        getattr(architecture_module, "getModelGivenModelOptionsAndWeightInits", None),
    )
    if builder is None:
        raise AttributeError(
            "Architecture must define getModelGivenModelOptionsAndWeightInits(args, parameters)")
    model = builder(args, parameters)
    print("got the model")
    if benchmark_enables(args, "shared_initial_weights"):
        assign_canonical_weights(model, parameters, args.seed)
    return model, architecture_module


def fit_and_evaluate(model, train_gen, valid_gen, args, architecture_module):
    model_output_path_h5_name=args.output_prefix+".h5"
    model_output_path_logs_name=args.output_prefix+".log"

    checkpointer = tfcallbacks.ModelCheckpoint(filepath=model_output_path_h5_name, monitor="val_loss", mode="min",  verbose=1, save_best_only=True)
    earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=args.early_stop, verbose=1, restore_best_weights=True)
    history= callbacks.LossHistory(model_output_path_logs_name+".batch",getattr(args, "trackables", DEFAULT_TRACKABLES))
    csvlogger = tfcallbacks.CSVLogger(model_output_path_logs_name, append=False)
    #reduce_lr = tfcallbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.4, patience=args.early_stop-2, min_lr=0.00000001)
    cur_callbacks=[checkpointer,earlystopper,csvlogger,history]

    model.fit(KerasSequence(train_gen),
              validation_data=KerasSequence(valid_gen),
              epochs=args.epochs,
              verbose=1,
              callbacks=cur_callbacks,
              # True unless --benchmark medium/max pins the example order;
              # True is also Keras's default, which the original relied on.
              shuffle=use_keras_batch_order_shuffle(args))

    if (benchmark_enables(args, "restore_best_checkpoint")
            and os.path.exists(model_output_path_h5_name)):
        # EarlyStopping(restore_best_weights=True) restores the best epoch
        # only when it stops training, so the original saved the *last*
        # epoch's weights over the best checkpoint whenever a run used all of
        # --epochs. ModelCheckpoint(save_best_only=True) has just written the
        # best epoch to this file; take its weights back before saving. When
        # early stopping did fire these are the weights it restored, so that
        # case is unchanged.
        model.load_weights(model_output_path_h5_name)

    print('save model')
    model.save(model_output_path_h5_name)

    architecture_module.save_model_without_bias(model, args.output_prefix)
