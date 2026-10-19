"""The count-head Lambda of chrombpnet 1.x ``chrombpnet.h5`` files, without a framework.

The full bias-factorized model of chrombpnet 1.x combines the bias and no-bias
log-count heads with ``Lambda(lambda x: tf.math.reduce_logsumexp(x, axis=-1,
keepdims=True))``, which Keras 2 saves as Python 3.8 marshalled bytecode.
Python 3.8 - 3.10 can unmarshal it; 3.11 and newer cannot. The JAX and
TensorFlow engines both recognise it here -- by the name of the function its
bytecode calls, without unmarshalling it -- and rebuild the head as their own
layer, and both can write it back verbatim (``logsumexp_lambda_fields``) so
that chrombpnet 1.x readers load the file. Imports nothing but the standard
library, so either engine can use it.
"""

import base64

# Name of the function the count-head Lambda of `chrombpnet export --legacy-h5 --count-head named` files refers to
# (function_type "function"): TF-Keras 2.x resolves it through custom_objects, chrombpnet maps it to its own layer.
LOGSUMEXP_LAMBDA_FUNCTION = "chrombpnet_logsumexp"

# The count-head Lambda of the chrombpnet 1.x chrombpnet.h5 files (TF-Keras 2.12 on Python 3.8), verbatim:
# `lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)` as base64 marshalled Python 3.8 bytecode
LEGACY_LOGSUMEXP_BYTECODE = (
    "4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMSAAAAdABqAWoCfABkAWQCZAONA1MAKQRO6f////9U\n"
    "KQLaBGF4aXPaCGtlZXBkaW1zKQPaAnRm2gRtYXRo2hByZWR1Y2VfbG9nc3VtZXhwKQHaAXipAHII\n"
    "AAAA+kgvb3B0L2Nocm9tYnBuZXQvY2hyb21icG5ldC90cmFpbmluZy9tb2RlbHMvY2hyb21icG5l\n"
    "dF93aXRoX2JpYXNfbW9kZWwucHnaCDxsYW1iZGE+gAAAAPMAAAAA\n")
LEGACY_LOGSUMEXP_MODULE = "chrombpnet.training.models.chrombpnet_with_bias_model"

COUNT_HEADS = ("bytecode", "named")   # the default first

# Config keys only a Lambda has (dropped when it is rebuilt as a plain layer)
LAMBDA_ONLY_KEYS = ("function", "output_shape", "mask", "arguments", "module", "function_type",
                    "output_shape_type", "output_shape_module")


def lambda_code_bytes(function):
    """Raw marshalled code of a serialized Lambda function (list or dict form), or b"" if unknown."""
    code = None
    if isinstance(function, (list, tuple)) and function:
        code = function[0]
    elif isinstance(function, dict):
        code = function.get("config", {}).get("code", function.get("code"))
    elif isinstance(function, str):
        code = function
    if not isinstance(code, str):
        return b""
    try:
        return base64.decodebytes(code.encode("ascii"))
    except (ValueError, TypeError):
        return b""


def is_logsumexp_lambda(config):
    """The 1.x bytecode Lambda (`tf.math.reduce_logsumexp`) or the named-function Lambda of legacy exports.

    Keras 3's legacy loader drops `function_type` / `module` before from_config, so a named function arrives as
    just the name.
    """
    function = config.get("function")
    if config.get("function_type", "function") == "function" and function == LOGSUMEXP_LAMBDA_FUNCTION:
        return True
    return b"reduce_logsumexp" in lambda_code_bytes(function)


def logsumexp_lambda_fields(count_head="bytecode"):
    """The Lambda config keys of the count head after name / trainable / dtype, in TF-Keras 2.12's order: the
    1.x bytecode Lambda ("bytecode") or the named-function one ("named")."""
    if count_head == "bytecode":
        function, function_type, module = [LEGACY_LOGSUMEXP_BYTECODE, None, None], "lambda", LEGACY_LOGSUMEXP_MODULE
    elif count_head == "named":
        function, function_type, module = LOGSUMEXP_LAMBDA_FUNCTION, "function", None
    else:
        raise ValueError("count_head must be one of {}, got {!r}.".format(", ".join(COUNT_HEADS), count_head))
    return {"function": function, "function_type": function_type, "module": module, "output_shape": None,
            "output_shape_type": "raw", "output_shape_module": None, "arguments": {}}
