"""chrombpnet 1.x TF-Keras .h5 models loaded as PyTorch models (engines/pytorch/keras_h5).

The files are written here in chrombpnet 1.x's layout -- its layer names, its
model config, its ``model_weights`` groups, the full model's nested towers --
from a PyTorch model's own weights, so the round trip must give the same
arrays and the same predictions. (Measured on ENCODE's ENCSR868FGK fold 0
models: identical tensors to bpnet-lite's ``BPNet.from_chrombpnet``, and
predictions within ~4e-5 of the maximum signal of ``--engine tensorflow-old``.)
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from chrombpnet import engines  # noqa: E402
from chrombpnet.engines.pytorch import keras_h5  # noqa: E402
from chrombpnet.engines.pytorch.model import (  # noqa: E402
    BiasModel, BPNet, ChromBPNet, ModelConfig, load_model)


def _conv(name, inbound, filters, width, dilation=1, padding="valid"):
    return {"class_name": "Conv1D", "inbound_nodes": [[[inbound, 0, 0, {}]]],
            "config": {"name": name, "filters": filters, "kernel_size": [width],
                       "dilation_rate": [dilation], "strides": [1], "padding": padding}}


def _layers(model, prefix, head, padding="valid"):
    """The layer configs chrombpnet 1.x writes for one BPNet tower."""
    c = model.config
    layers = [{"class_name": "InputLayer", "inbound_nodes": [],
               "config": {"name": "sequence", "batch_input_shape": [None, c.input_length, 4]}},
              _conv(prefix + "1st_conv", "sequence", c.filters, 21)]
    previous = prefix + "1st_conv"
    for k in range(1, c.n_dilation_layers + 1):
        layers.append(_conv(f"{prefix}{k}conv", previous, c.filters, 3, 2 ** k, padding))
        previous = f"{prefix}{k}conv"
    layers.append(_conv(head + "prof_out_precrop", previous, 1, 75))
    length = c.input_length - 20 - sum(2 * 2 ** k for k in range(1, c.n_dilation_layers + 1)) - 74
    crop = (length - c.output_length) // 2
    layers.append({"class_name": "Cropping1D", "inbound_nodes": [[[head + "prof_out_precrop", 0, 0, {}]]],
                   "config": {"name": head + "logits_profile_predictions_preflatten", "cropping": [crop, crop]}})
    layers.append({"class_name": "Dense", "inbound_nodes": [[["gap", 0, 0, {}]]],
                   "config": {"name": head + "logcount_predictions"}})
    return layers


def _weights(group, model, prefix, head, nested):
    def put(layer, module):
        kernel = module.weight.detach().numpy()
        kernel = kernel.T if kernel.ndim == 2 else kernel.transpose(2, 1, 0)
        names = [f"{layer}/kernel:0", f"{layer}/bias:0"]
        if nested:      # the full model: model_weights/<tower>/<layer>/<weight>:0
            group.create_dataset(names[0], data=kernel)
            group.create_dataset(names[1], data=module.bias.detach().numpy())
        else:           # a single tower: model_weights/<layer>/<layer>/<weight>:0, named in weight_names
            node = group.create_group(layer)
            node.attrs["weight_names"] = [n.encode() for n in names]
            node.create_dataset(names[0], data=kernel)
            node.create_dataset(names[1], data=module.bias.detach().numpy())

    put(prefix + "1st_conv", model.first_conv)
    for k, conv in enumerate(model.dilated_convs, start=1):
        put(f"{prefix}{k}conv", conv)
    put(head + "prof_out_precrop", model.profile_conv)
    put(head + "logcount_predictions", model.count_dense)


def _write(path, model):
    with h5py.File(path, "w") as handle:
        handle.attrs["keras_version"] = b"2.4.0"
        weights = handle.create_group("model_weights")
        if isinstance(model, ChromBPNet):
            towers = [("model_wo_bias", model.tf_model, "wo_bias_bpnet_", "wo_bias_bpnet_"),
                      ("model", model.bias_model, "bpnet_", "")]
            layers = [{"class_name": "InputLayer", "inbound_nodes": [],
                       "config": {"name": "sequence",
                                  "batch_input_shape": [None, model.config.input_length, 4]}}]
            for name, tower, prefix, head in towers:
                layers.append({"class_name": "Functional", "inbound_nodes": [[["sequence", 0, 0, {}]]],
                               "config": {"name": name, "layers": _layers(tower, prefix, head)}})
                _weights(weights.create_group(name), tower, prefix, head, nested=True)
        else:
            prefix, head = ("bpnet_", "") if isinstance(model, BiasModel) else ("wo_bias_bpnet_",) * 2
            layers = _layers(model, prefix, head)
            _weights(weights, model, prefix, head, nested=False)
        handle.attrs["model_config"] = json.dumps(
            {"class_name": "Functional", "config": {"name": "model", "layers": layers}}).encode()


def _models():
    torch.manual_seed(0)
    bias = BiasModel(ModelConfig(206, 100, filters=8, n_dilation_layers=2))
    accessibility = BPNet(ModelConfig(206, 100, filters=16, n_dilation_layers=2))
    full = ChromBPNet(accessibility.config, BiasModel(bias.config))
    full.tf_model.load_state_dict(accessibility.state_dict())
    full.bias_model.load_state_dict(bias.state_dict())
    return {"bias": bias, "chrombpnet_nobias": accessibility, "chrombpnet": full}


@pytest.mark.parametrize("kind, cls", [("bias", BiasModel), ("chrombpnet_nobias", BPNet),
                                       ("chrombpnet", ChromBPNet)])
def test_a_1x_model_loads_as_the_same_pytorch_model(tmp_path, kind, cls):
    original = _models()[kind].eval()
    path = tmp_path / f"{kind}.h5"
    _write(path, original)
    loaded = load_model(path)
    assert type(loaded) is cls
    assert loaded.config.input_length == 206 and loaded.config.output_length == 100
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (3, 206)), 4).permute(0, 2, 1).float()
    with torch.no_grad():
        for got, want in zip(loaded(x), original(x)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_an_unknown_architecture_is_refused_not_half_loaded(tmp_path):
    model = _models()["chrombpnet_nobias"]
    path = tmp_path / "same_padding.h5"
    _write(path, model)
    with h5py.File(path, "r+") as handle:
        config = json.loads(handle.attrs["model_config"])
        for layer in config["config"]["layers"]:
            if layer["config"]["name"] == "wo_bias_bpnet_2conv":
                layer["config"]["padding"] = "same"
        handle.attrs["model_config"] = json.dumps(config).encode()
    with pytest.raises(keras_h5.KerasH5Error, match="'same'"):
        load_model(path)


def test_a_file_that_is_not_a_keras_model_is_refused(tmp_path):
    path = tmp_path / "scores.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.zeros(3))
    with pytest.raises(keras_h5.KerasH5Error, match="model_config"):
        load_model(path)


def test_the_pytorch_engine_reads_h5():
    assert engines.get("pytorch").can_read("model.chrombpnet.fold_0.h5")
    assert engines.get("pytorch").can_read("bias.HDF5")
    # A plain .h5 still goes to tensorflow-old unless --engine says otherwise.
    assert engines.infer_from_path("bias.h5") == "tensorflow-old"


def test_bpnet_lite_reads_the_same_arrays(tmp_path):
    """bpnet-lite's own 1.x loader, where it is installed."""
    lite = pytest.importorskip("bpnetlite.bpnet")
    model = _models()["chrombpnet_nobias"]
    path = tmp_path / "chrombpnet_nobias.h5"
    _write(path, model)
    ours, theirs = load_model(path), lite.BPNet.from_chrombpnet(str(path))
    pairs = [("first_conv", "iconv"), ("profile_conv", "fconv"), ("count_dense", "linear")] + [
        (f"dilated_convs.{i}", f"rconvs.{i}") for i in range(2)]
    for a, b in pairs:
        for weight in ("weight", "bias"):
            torch.testing.assert_close(getattr(ours.get_submodule(a), weight).detach(),
                                       getattr(theirs.get_submodule(b), weight).detach().float(),
                                       rtol=0, atol=0)
