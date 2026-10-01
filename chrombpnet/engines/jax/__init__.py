"""The JAX engine: a placeholder.

``--engine jax`` is accepted on the command line so that the selection path,
the environment file (``envs/chrombpnet-jax.yml``) and the tests exist ahead of
an implementation. Selecting it fails in ``CHROMBPNET.main`` before any output
directory is created.

An implementation fills in every method of :class:`chrombpnet.engines.base.Engine`.
Two constraints carry over from the other engines and should shape it:

* the predictor must present the Keras-shaped surface the shared code uses
  (NLC input, ``(profile_logits, logcounts)`` output, ``input_shape`` and
  ``output_shape``), so that nothing outside this package changes; and
* ``chrombpnet.training.utils.benchmark_utils.canonical_bpnet_weights`` already
  produces Keras-layout initial weights from a framework-neutral NumPy stream,
  which is how a JAX model can start from exactly the arrays the PyTorch and
  TensorFlow engines use at ``--benchmark max``.
"""

from __future__ import annotations

from chrombpnet.engines.base import Engine, EngineNotImplemented


class JaxEngine(Engine):
    name = "jax"
    #: Not chosen yet; a JAX checkpoint format (e.g. Orbax or flax msgpack)
    #: comes with the implementation. Until then no suffix maps to this engine.
    checkpoint_suffix = None
    environment = "chrombpnet-jax"

    def _not_implemented(self, *args, **kwargs):
        raise EngineNotImplemented(
            "The JAX engine is a placeholder and is not implemented yet. "
            "Use --engine pytorch (the default) or --engine tensorflow.")

    require = _not_implemented
    default_architecture = _not_implemented
    get_model = _not_implemented
    fit = _not_implemented
    resolve_device = _not_implemented
    load_predictor = _not_implemented
    rescale_bias_logcounts = _not_implemented
    load_attribution_model = _not_implemented
    input_length = _not_implemented
    attribute = _not_implemented


ENGINE = JaxEngine()
