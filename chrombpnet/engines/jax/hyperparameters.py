"""The bias model's count-head rescale, for the JAX engine.

``get_logcount_layer`` and ``adjust_bias_model_logcounts`` are from
kundajelab/chrombpnet#284 (NNFC-GMD/chrombpnet@14ca773),
chrombpnet/helpers/hyperparameters/find_chrombpnet_hyperparams.py, verbatim;
``rescale_bias_logcounts`` is that file's load / adjust / save sequence, with
the result saved in this engine's ``.keras`` format.
"""

from __future__ import annotations

import keras
import numpy as np

from chrombpnet.engines.jax import model_io


def get_logcount_layer(bias_model):
    layer_names = [layer.name for layer in bias_model.layers]
    for name in ["logcount_predictions", "logcounts"]:
        if name in layer_names:
            return bias_model.get_layer(name)
    return bias_model.layers[-1]

def adjust_bias_model_logcounts(bias_model, seqs, cts):
    """
    Given a bias model, sequences and associated counts, the function adds a
    constant to the output of the bias_model's logcounts that minimises squared
    error between predicted logcounts and observed logcounts (infered from
    cts). This simply reduces to adding the average difference between observed
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's logcounts output comes from a dense layer named logcount_predictions (or logcounts).
    This would change if you change the model.
    """

    # safeguards to prevent misuse
    layer = get_logcount_layer(bias_model)
    assert(layer.name == "logcounts" or layer.name == "logcount_predictions")
    assert(tuple(layer.output.shape)==(None,1))
    assert(isinstance(layer, keras.layers.Dense))
    assert(any(layer.output is output for output in bias_model.outputs)) # no layer after the dense layer

    print("Predicting within adjust counts")
    _, pred_logcts = bias_model.predict(seqs, verbose=True)
    delta = np.mean(np.log(1+cts) - pred_logcts.ravel())

    dw, db = layer.get_weights()
    layer.set_weights([dw, (db+delta).astype(db.dtype)])
    return bias_model


def rescale_bias_logcounts(bias_path, output_path, seqs, cts):
    # param_utils.load_model_wrapper
    bias_model = model_io.load_model(bias_path)
    print("got the model")
    bias_model.summary()
    bias_model_scaled = adjust_bias_model_logcounts(bias_model, seqs, cts)
    bias_model_scaled.save(output_path)
