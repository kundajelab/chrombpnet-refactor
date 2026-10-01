"""Present the shared batch generator to Keras as a ``keras.utils.Sequence``.

The original implementation's ``ChromBPNetBatchGenerator`` subclassed
``keras.utils.Sequence`` itself. The shared generator cannot, because it also
serves the PyTorch engine, so this thin adapter restores the type Keras's
data adapter dispatches on. Keras then drives it exactly as it drove the
original: one ``__getitem__`` per batch, in an order Keras shuffles with
Python's ``random`` when ``fit(shuffle=True)``, and ``on_epoch_end`` after
every pass -- including over the validation generator.
"""

from __future__ import annotations

from tensorflow import keras


class KerasSequence(keras.utils.Sequence):
    def __init__(self, generator):
        self.generator = generator

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        return self.generator[index]

    def on_epoch_end(self):
        self.generator.on_epoch_end()
