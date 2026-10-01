"""The bias model's count-head rescale, for the TensorFlow engine.

``adjust_bias_model_logcounts`` is verbatim from kundajelab/chrombpnet@09938fd,
chrombpnet/helpers/hyperparameters/find_chrombpnet_hyperparams.py;
``rescale_bias_logcounts`` is that file's load / adjust / save sequence.
"""

from __future__ import annotations

import numpy as np
import tensorflow.keras as keras

from chrombpnet.engines.tensorflow.predictor import load_model


def adjust_bias_model_logcounts(bias_model, seqs, cts):
    """
    Given a bias model, sequences and associated counts, the function adds a
    constant to the output of the bias_model's logcounts that minimises squared
    error between predicted logcounts and observed logcounts (infered from
    cts). This simply reduces to adding the average difference between observed
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's last layer is a dense layer that outputs logcounts.
    This would change if you change the model.
    """

    # safeguards to prevent misuse
    #assert(bias_model.layers[-1].name == "logcount_predictions")
    assert(bias_model.layers[-1].name == "logcounts" or bias_model.layers[-1].name == "logcount_predictions")
    assert(bias_model.layers[-1].output_shape==(None,1))
    assert(isinstance(bias_model.layers[-1], keras.layers.Dense))

    print("Predicting within adjust counts")
    _, pred_logcts = bias_model.predict(seqs, verbose=True)
    delta = np.mean(np.log(1+cts) - pred_logcts.ravel())

    dw, db = bias_model.layers[-1].get_weights()
    bias_model.layers[-1].set_weights([dw, db+delta])
    return bias_model


def rescale_bias_logcounts(bias_path, output_path, seqs, cts):
    # param_utils.load_model_wrapper: compiled, so the saved model keeps its
    # optimizer state and training configuration.
    bias_model = load_model(bias_path, compile=True)
    print("got the model")
    bias_model.summary()
    bias_model_scaled = adjust_bias_model_logcounts(bias_model, seqs, cts)
    bias_model_scaled.save(output_path)
