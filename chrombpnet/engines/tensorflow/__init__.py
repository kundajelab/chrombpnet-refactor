"""The TensorFlow engine. Registered ahead of its implementation."""

from __future__ import annotations

from chrombpnet.engines.base import Engine, EngineNotImplemented


class TensorFlowEngine(Engine):
    name = "tensorflow"
    checkpoint_suffix = ".h5"
    environment = "chrombpnet-tensorflow"

    def require(self):
        raise EngineNotImplemented(
            "The TensorFlow engine is not available in this revision.")


ENGINE = TensorFlowEngine()
