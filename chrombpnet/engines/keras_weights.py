"""Keras-layout BPNet weights shared by the Keras-based engines.

``--benchmark max`` starts every engine from the same arrays, drawn from one
framework-neutral NumPy stream (``canonical_bpnet_weights``). TensorFlow 2.8's
tf.keras and Keras 3 address layers and their weights identically, so the
TensorFlow and JAX engines assign them with the same function.
"""

from __future__ import annotations

from chrombpnet.training.utils.benchmark_utils import canonical_bpnet_weights


def assign_canonical_weights(model, parameters, seed):
    """Give the trainable BPNet the shared Keras-layout initial arrays.

    ``--benchmark max``: the same arrays the PyTorch engine assigns through
    ``initialize_keras_compatible``, drawn from one framework-neutral NumPy
    stream. For a bias-factorized model only the no-bias tower is assigned;
    the bias model is pretrained and frozen.
    """
    filters = int(parameters["filters"])
    n_dilation_layers = int(parameters["n_dil_layers"])
    values = canonical_bpnet_weights(filters, n_dilation_layers, seed)
    if any(layer.name == "model_wo_bias" for layer in model.layers):
        target, root, head = model.get_layer("model_wo_bias"), "wo_bias_bpnet_", "wo_bias_bpnet_"
    else:
        target, root, head = model, "bpnet_", ""
    target.get_layer(root + "1st_conv").set_weights(
        [values["first_kernel"], values["first_bias"]])
    for layer in range(1, n_dilation_layers + 1):
        target.get_layer("{}{}conv".format(root, layer)).set_weights(
            [values[f"dilated_{layer}_kernel"], values[f"dilated_{layer}_bias"]])
    target.get_layer(head + "prof_out_precrop").set_weights(
        [values["profile_kernel"], values["profile_bias"]])
    target.get_layer(head + "logcount_predictions").set_weights(
        [values["count_kernel"], values["count_bias"]])
