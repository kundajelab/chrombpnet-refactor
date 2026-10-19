"""The count head of the bias-factorized model, loadable on any Python.

chrombpnet 1.x builds it as ``Lambda(lambda x: tf.math.reduce_logsumexp(x,
axis=-1, keepdims=True))``, and Keras 2 saves a lambda as marshalled bytecode
of the Python that wrote it: 1.x files hold Python 3.8 bytecode, which
Python 3.11 and newer cannot unmarshal. Here the head is a Lambda of a named
function, ``chrombpnet_logsumexp`` -- the same computation -- and

* ``CountHeadLambda``, in this engine's custom objects as ``Lambda``, loads a
  1.x bytecode head (recognised without unmarshalling it, see
  chrombpnet.legacy_lambda) or a named one as that Lambda, and every other
  Lambda as Keras does;
* ``make_original_readable`` rewrites a saved model so that chrombpnet 1.x's
  Keras 2.8, and the tools that read its models, load what this engine
  writes: the head in 1.x's exact bytecode form, and without the two config
  fields tf-keras added since, at their defaults.
"""

from __future__ import annotations

import json

import tensorflow as tf
from tensorflow.keras.layers import Lambda

from chrombpnet import legacy_lambda

COUNT_HEAD_NAME = "logcount_predictions"


def chrombpnet_logsumexp(x):
    """The count head: log(exp(no-bias counts) + exp(bias counts))."""
    return tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)


def count_head():
    """The count-head layer the architecture builds."""
    return Lambda(chrombpnet_logsumexp, name=COUNT_HEAD_NAME)


class CountHeadLambda(Lambda):
    """``Lambda`` as this engine loads it (see the module docstring)."""

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if config.get("name") == COUNT_HEAD_NAME and legacy_lambda.is_logsumexp_lambda(config):
            config = {key: value for key, value in config.items()
                      if key not in legacy_lambda.LAMBDA_ONLY_KEYS}
            return Lambda(chrombpnet_logsumexp, **config)
        return Lambda.from_config(config, custom_objects=custom_objects)


def make_original_readable(path):
    """Rewrite the Keras 2 ``.h5`` model at ``path`` so that chrombpnet 1.x
    (Keras 2.8) loads it; returns whether the file changed.

    * The count head becomes 1.x's bytecode Lambda (legacy_lambda).
    * ``InputLayer`` loses ``optional`` and every model config ``trainable``:
      fields tf-keras writes and Keras 2.8 refuses (``Unrecognized keyword
      arguments: ['optional']``), dropped only at their defaults (``False``,
      ``True``), where both releases read the same thing without them.

    Weights, optimizer state and every other field are left alone, and
    tf-keras reads the result as it reads any 1.x file.
    """
    import h5py

    with h5py.File(path, "r+") as handle:
        raw = handle.attrs["model_config"]
        model_config = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        changed = False

        def walk(model):
            nonlocal changed
            config = model.get("config", {})
            if config.get("trainable") is True:
                del config["trainable"]
                changed = True
            for layer in config.get("layers", []):
                layer_config = layer.get("config", {})
                kind = layer.get("class_name")
                if (kind == "Lambda" and layer_config.get("name") == COUNT_HEAD_NAME
                        and legacy_lambda.is_logsumexp_lambda(layer_config)):
                    fields = legacy_lambda.logsumexp_lambda_fields("bytecode")
                    if any(layer_config.get(key) != value for key, value in fields.items()):
                        layer_config.update(fields)
                        changed = True
                elif kind == "InputLayer" and layer_config.get("optional") is False:
                    del layer_config["optional"]
                    changed = True
                elif kind in ("Functional", "Model"):
                    walk(layer)

        walk(model_config)
        if changed:
            text = json.dumps(model_config)
            handle.attrs["model_config"] = text.encode("utf-8") if isinstance(raw, bytes) else text
    return changed
