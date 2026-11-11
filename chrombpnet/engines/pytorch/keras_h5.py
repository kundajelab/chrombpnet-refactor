"""chrombpnet 1.x models (TF-Keras 2 ``.h5``) as this package's PyTorch models.

``load_model`` reads a ``.h5`` through :func:`load_keras_h5`, so the PyTorch
engine takes the original ChromBPNet's ``bias.h5`` / ``bias_scaled.h5``
(-> :class:`BiasModel`), ``chrombpnet_nobias.h5`` (-> :class:`BPNet`) and the
full ``chrombpnet.h5`` (-> :class:`ChromBPNet`: its nested ``model`` and
``model_wo_bias`` towers) wherever it takes a ``.pt``, with ``pred_bw``, ``contribs_bw``, ``footprints``, ``qc``, ``predict``,
``distill`` and the training commands' ``-b``. Files ``chrombpnet export``
writes in that layout load the same way.

The weights map onto the PyTorch layers as ``assign_keras_layout_weights``
maps them for ``--reproducibility-max`` (convolution kernels ``(K, in, out)``
-> ``(out, in, K)``, the dense kernel transposed), with the layer names
chrombpnet 1.x gives them: ``[wo_bias_]bpnet_1st_conv``,
``[wo_bias_]bpnet_<k>conv``, and the heads ``prof_out_precrop`` and
``logcount_predictions``, prefixed ``wo_bias_bpnet_`` in the accessibility
tower only. bpnet-lite's ``BPNet.from_chrombpnet`` (github.com/jmschrei/
bpnet-lite, MIT) reads the same names into its own classes; this module
gives the same arrays (tests/test_keras_h5.py) and also reads the full
model, the input and output lengths from the file's model config, and
refuses an architecture it does not recognise instead of loading part of it.
"""

from __future__ import annotations

import json
import re

import numpy as np

#: The kernel widths of chrombpnet 1.x's BPNet towers.
FIRST_WIDTH, DILATED_WIDTH, PROFILE_WIDTH = 21, 3, 75


class KerasH5Error(ValueError):
    """The .h5 is not a chrombpnet 1.x model this loader understands."""


def _attr_json(handle, name):
    value = handle.attrs.get(name)
    if value is None:
        raise KerasH5Error(f"{handle.file.filename} has no {name!r}: not a Keras full-model .h5")
    return json.loads(value.decode() if isinstance(value, bytes) else value)


def _input_length(layers):
    for layer in layers:
        if layer["class_name"] == "InputLayer":
            shape = layer["config"].get("batch_input_shape") or layer["config"].get("batch_shape")
            if shape and len(shape) == 3 and shape[2] == 4 and shape[1]:
                return int(shape[1])
    raise KerasH5Error("no (length, 4) sequence input layer")


def _tower(layers, input_length):
    """Geometry of one BPNet tower from its layer configs: (prefix, head
    prefix, filters, dilation layers, output length)."""
    convs = {layer["config"]["name"]: layer for layer in layers if layer["class_name"] == "Conv1D"}
    first = [name for name in convs if name.endswith("bpnet_1st_conv")]
    if len(first) != 1:
        raise KerasH5Error(f"expected one *bpnet_1st_conv layer, found {first or 'none'}")
    prefix = first[0][: -len("1st_conv")]                    # "bpnet_" or "wo_bias_bpnet_"
    head = prefix if prefix.startswith("wo_bias_") else ""   # 1.x names the bias model's heads bare
    dilated = sorted(int(m.group(1)) for name in convs
                     for m in [re.fullmatch(re.escape(prefix) + r"(\d+)conv", name)] if m)
    if not dilated or dilated != list(range(1, len(dilated) + 1)):
        raise KerasH5Error(f"dilated convolutions {prefix}<k>conv are not 1..n: {dilated}")
    expected = {first[0]: (FIRST_WIDTH, 1), head + "prof_out_precrop": (PROFILE_WIDTH, 1)}
    expected.update({f"{prefix}{k}conv": (DILATED_WIDTH, 2 ** k) for k in dilated})
    filters = None
    for name, (width, dilation) in expected.items():
        if name not in convs:
            raise KerasH5Error(f"no {name} layer")
        config = convs[name]["config"]
        if (list(config["kernel_size"]) != [width] or list(config["dilation_rate"]) != [dilation]
                or list(config.get("strides", [1])) != [1] or config.get("padding") != "valid"):
            raise KerasH5Error(
                f"{name} is width {config['kernel_size']}, dilation {config['dilation_rate']}, "
                f"padding {config.get('padding')!r}; chrombpnet 1.x's is width {width}, "
                f"dilation {dilation}, 'valid'")
        if name == first[0]:
            filters = int(config["filters"])
    # The profile is cropped to the output length by the Cropping1D that
    # takes the profile convolution's output.
    profile_length = input_length - (FIRST_WIDTH - 1) - sum(
        2 * 2 ** k for k in dilated) - (PROFILE_WIDTH - 1)
    crops = [layer for layer in layers if layer["class_name"] == "Cropping1D"
             and _inbound(layer) == head + "prof_out_precrop"]
    if len(crops) != 1:
        raise KerasH5Error("no Cropping1D on the profile convolution's output")
    left, right = (int(c) for c in crops[0]["config"]["cropping"])
    if left != right:
        raise KerasH5Error(f"the profile crop is not symmetric ({left}, {right})")
    return prefix, head, filters, len(dilated), profile_length - left - right


def _inbound(layer):
    """The name of the one layer feeding ``layer`` (TF-Keras 2 and Keras 3 forms)."""
    nodes = layer.get("inbound_nodes") or []
    if not nodes:
        return None
    node = nodes[0]
    if isinstance(node, list):                      # TF-Keras 2: [[name, node, tensor, kwargs]]
        return node[0][0] if node and isinstance(node[0], list) else None
    args = node.get("args", []) if isinstance(node, dict) else []
    history = args[0].get("config", {}).get("keras_history") if args and isinstance(args[0], dict) else None
    return history[0] if history else None


def _weights(group, layer):
    """[kernel, bias] of ``layer`` under ``group``, by the layer's own
    ``weight_names`` where it records them."""
    if layer not in group:
        raise KerasH5Error(f"no weights for {layer}")
    node = group[layer]
    names = node.attrs.get("weight_names")
    if names is not None and len(names):
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        values = [np.asarray(node[name]) for name in names]
    else:
        values = []
        for weight in ("kernel", "bias"):
            for path in (f"{layer}/{weight}:0", f"{weight}:0", weight):
                if path in node:
                    values.append(np.asarray(node[path]))
                    break
    if len(values) != 2:
        raise KerasH5Error(f"{layer} has {len(values)} weight arrays, not a kernel and a bias")
    kernel, bias = values if values[0].ndim > values[1].ndim else values[::-1]
    return kernel, bias


def _read(group, layers, input_length):
    """(ModelConfig, Keras-layout arrays) of one tower."""
    from chrombpnet.engines.pytorch.model import ModelConfig

    prefix, head, filters, n_dilation_layers, output_length = _tower(layers, input_length)
    values = {}
    values["first_kernel"], values["first_bias"] = _weights(group, prefix + "1st_conv")
    for k in range(1, n_dilation_layers + 1):
        values[f"dilated_{k}_kernel"], values[f"dilated_{k}_bias"] = _weights(group, f"{prefix}{k}conv")
    values["profile_kernel"], values["profile_bias"] = _weights(group, head + "prof_out_precrop")
    values["count_kernel"], values["count_bias"] = _weights(group, head + "logcount_predictions")
    config = ModelConfig(input_length, output_length, filters=filters, n_dilation_layers=n_dilation_layers)
    return config, {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in values.items()}


def load_keras_h5(path):
    """The chrombpnet 1.x model in ``path`` as a BiasModel, BPNet or ChromBPNet (on the CPU, in eval mode)."""
    import h5py

    from chrombpnet.engines.pytorch.model import BiasModel, BPNet, ChromBPNet, assign_keras_layout_weights

    with h5py.File(path, "r") as handle:
        config = _attr_json(handle, "model_config")
        weights = handle["model_weights"]
        layers = config["config"]["layers"]
        input_length = _input_length(layers)
        nested = {layer["config"]["name"]: layer["config"]["layers"] for layer in layers
                  if layer["class_name"] in ("Functional", "Model") and "layers" in layer["config"]}
        if nested:
            towers = {}
            for name, inner in nested.items():
                is_bias = not any(l["config"].get("name", "").startswith("wo_bias_") for l in inner)
                towers["bias" if is_bias else "accessibility"] = (name, inner)
            if set(towers) != {"bias", "accessibility"} or len(nested) != 2:
                raise KerasH5Error(f"expected a bias and a no-bias tower, found {sorted(nested)}")
            bias_config, bias_values = _read(weights[towers["bias"][0]], towers["bias"][1], input_length)
            config, values = _read(weights[towers["accessibility"][0]], towers["accessibility"][1],
                                   input_length)
            model = ChromBPNet(config, BiasModel(bias_config))
            assign_keras_layout_weights(model.tf_model, values)
            assign_keras_layout_weights(model.bias_model, bias_values)
        else:
            is_bias = not any(l["config"].get("name", "").startswith("wo_bias_") for l in layers)
            config, values = _read(weights, layers, input_length)
            model = (BiasModel if is_bias else BPNet)(config)
            assign_keras_layout_weights(model, values)
    return model.eval()
