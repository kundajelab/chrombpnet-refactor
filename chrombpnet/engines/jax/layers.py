# JAX engine. From kundajelab/chrombpnet#284 (NNFC-GMD/chrombpnet@14ca773),
# chrombpnet/training/utils/layers.py,
# except that an error message names this module, and that the Lambda
# recognition, which the TensorFlow engine shares, is in chrombpnet.legacy_lambda.
import keras
from keras import ops

from chrombpnet.legacy_lambda import LAMBDA_ONLY_KEYS as _LAMBDA_ONLY_KEYS
from chrombpnet.legacy_lambda import LOGSUMEXP_LAMBDA_FUNCTION  # noqa: F401  (re-exported)
from chrombpnet.legacy_lambda import is_logsumexp_lambda as _is_logsumexp_lambda
from chrombpnet.legacy_lambda import lambda_code_bytes as _lambda_code_bytes  # noqa: F401


@keras.saving.register_keras_serializable(package="chrombpnet")
class LogSumExp(keras.layers.Layer):
    """Combine the bias and no-bias log-count heads: logsumexp over the last axis, keepdims.

    Replaces the `Lambda(lambda x: tf.math.reduce_logsumexp(...))` of chrombpnet 1.x, which cannot be
    serialized portably (marshalled Python bytecode) and cannot run on the JAX backend.
    """

    def call(self, inputs):
        return ops.logsumexp(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)


class LogSumExpCompat(LogSumExp):
    """Stand-in for the logsumexp Lambda inside chrombpnet 1.x `chrombpnet.h5` files.

    Passed to load_model as custom_objects={"Lambda": LogSumExpCompat}. It never unmarshals the stored bytecode
    (which only loads on the Python version that wrote it). `chrombpnet export --legacy-h5` files carry the same
    Lambda, or with `--count-head named` one that names the function (function_type "function", function
    LOGSUMEXP_LAMBDA_FUNCTION) instead of storing bytecode, which is accepted too. Any other Lambda is refused
    rather than silently replaced. from_config returns a plain LogSumExp, so a loaded 1.x model re-saves as
    `chrombpnet>LogSumExp` and loads again.
    """

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        name = config.get("name")
        if name != "logcount_predictions" or not _is_logsumexp_lambda(config):
            raise ValueError(
                "Cannot load Lambda layer {!r}: only the logsumexp count head of chrombpnet 1.x models "
                "(`logcount_predictions`) is supported. Rebuild custom architectures without Lambda layers "
                "(see chrombpnet.engines.jax.layers).".format(name))
        for key in _LAMBDA_ONLY_KEYS:
            config.pop(key, None)
        return LogSumExp(**config)
