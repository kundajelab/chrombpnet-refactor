"""Present the shared batch generator to Keras 3 as a ``keras.utils.PyDataset``.

Keras 3's data adapter dispatches on ``PyDataset``. The shared generator cannot
subclass it, because it also serves the PyTorch engine, so this adapter does --
the counterpart of the TensorFlow engine's ``KerasSequence``. With one worker
and no multiprocessing, Keras calls it synchronously: one ``__getitem__`` per
batch, in an order it shuffles with Python's ``random`` when
``fit(shuffle=True)``.

The generator must see one ``on_epoch_end`` per completed pass, as it does under
TF-Keras and the PyTorch engine: each call re-draws its jitter, reverse
complements and negatives, and with ``--reproducibility-max`` epoch ``k``'s draws
are derived from ``(seed, mode, k)`` so that every engine trains on the same
schedule. Keras 3 also calls ``on_epoch_end`` from ``EpochIterator.reset()``,
before the first training epoch and before every validation pass, when no pass
has begun. Those calls are not forwarded: only an ``on_epoch_end`` that follows
an ``on_epoch_begin`` reaches the generator. (Keras also starts a pass to peek
at the first batch when the model's loss and metrics are not yet built; the
engine builds them before ``fit``, so that does not happen.)

Keras 3 reads a third element of a batch tuple as sample weights and a list of
targets as one structure, so batches are passed as ``(x, (profile, logcounts))``
and coordinates, when a generator returns them, are dropped.
"""

from __future__ import annotations

import keras


class KerasPyDataset(keras.utils.PyDataset):
    def __init__(self, generator):
        super().__init__(workers=1, use_multiprocessing=False)
        self.generator = generator
        self._in_pass = False

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        batch = self.generator[index]
        return batch[0], tuple(batch[1])

    def on_epoch_begin(self):
        self._in_pass = True

    def on_epoch_end(self):
        if self._in_pass:
            self._in_pass = False
            self.generator.on_epoch_end()
