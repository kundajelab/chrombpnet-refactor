"""Loading models for prediction in the JAX engine.

A Keras 3 model already has the surface the shared code uses (``predict``,
``predict_on_batch``, ``input_shape``, ``output_shape``, ``summary``) and
returns ``[profile_logits, logcounts]``. The one thing this wrapper adds is from
kundajelab/chrombpnet#284's ``training/predict.py``: JAX compiles a function
once per input shape, so a short final batch is zero-padded to the batch size
already compiled and the padded rows are dropped from the outputs.
"""

from __future__ import annotations

import numpy as np

from chrombpnet.engines.jax import model_io
from chrombpnet.engines.jax.runtime import configure_device


class KerasPredictor:
    def __init__(self, model):
        self.model = model
        self._batch_size = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict_on_batch(self, x):
        x = np.asarray(x)
        rows = x.shape[0]
        if self._batch_size is None or rows > self._batch_size:
            self._batch_size = rows
        if rows < self._batch_size:
            pad = np.zeros((self._batch_size - rows,) + x.shape[1:], dtype=x.dtype)
            x = np.concatenate([x, pad])
        outputs = self.model.predict_on_batch(x)
        return [np.asarray(output)[:rows] for output in outputs]

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x, *args, **kwargs)


def load_model_wrapper(path, device=None, *, announce=True, summary=True):
    configure_device(device)
    model = model_io.load_model(path, compile=False)
    if announce:
        print("got the model")
    if summary:
        model.summary()
    return KerasPredictor(model)
