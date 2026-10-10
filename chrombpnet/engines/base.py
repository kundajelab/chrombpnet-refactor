"""The interface every deep-learning engine implements.

Everything in ChromBPNet that needs a deep-learning framework goes through
these methods; everything else -- preprocessing, the data generators, metrics,
bigwig writing, footprinting, motif discovery and the reports -- is shared,
framework-neutral code that sees a model only through them.

The prediction contract is deliberately the small Keras-shaped surface the
shared code already used before there was a choice of engine:
``predictor.input_shape == (None, inputlen, 4)``,
``predictor.output_shape == [(None, outputlen), (None, 1)]``, and
``predict``/``predict_on_batch`` taking NLC one-hot arrays and returning
``(profile_logits, logcounts)`` as float32 NumPy arrays. A Keras model already
is one; the PyTorch engine supplies ``chrombpnet.model.PredictionAdapter``.
"""

from __future__ import annotations


class EngineUnavailable(RuntimeError):
    """The selected engine's framework is not installed in this environment."""


class EngineNotImplemented(NotImplementedError):
    """The selected engine exists only as a placeholder."""


class Engine:
    """One deep-learning backend. Subclasses override every method.

    Framework imports happen inside methods, never at module import time, so
    any engine object can be constructed -- and asked for its name or
    checkpoint suffix -- in an environment that lacks its framework.
    """

    #: The ``--engine`` value that selects this engine.
    name = None
    #: Suffix of the checkpoints this engine writes, e.g. ``".pt"``.
    checkpoint_suffix = None
    #: Suffixes of the checkpoints this engine can load, if more than the one
    #: it writes.
    readable_suffixes = None
    #: The conda environment that provides this engine.
    environment = None

    def can_read(self, path):
        """Whether this engine can load the checkpoint at ``path``."""
        import os

        if not path or not isinstance(path, str):
            return False
        suffixes = self.readable_suffixes or (self.checkpoint_suffix,)
        return os.path.splitext(path)[1].lower() in suffixes

    # -- availability ------------------------------------------------------
    def prepare(self, args):
        """Settings that must be in place before the framework is imported or
        starts (environment variables). Called with the parsed command line,
        before :meth:`require`. Nothing by default."""

    def require(self):
        """Import the framework, or raise :class:`EngineUnavailable`."""
        raise NotImplementedError

    def _unavailable(self, error):
        return EngineUnavailable(
            f"--engine {self.name} needs its framework, which this Python "
            f"environment cannot import ({error}). Activate the "
            f"{self.environment} conda environment (envs/{self.environment}.yml)."
        )

    # -- training ----------------------------------------------------------
    def default_architecture(self, kind):
        """Path of the built-in architecture file for ``kind``.

        ``kind`` is ``"bias"`` or ``"chrombpnet"``. The path is loaded with
        ``-a/--architecture-from-file`` semantics, and it is recorded in
        ``<prefix>.args.json``.
        """
        raise NotImplementedError

    def get_model(self, args, parameters):
        """Seed, then build the model from ``args.architecture_from_file``.

        Returns ``(model, architecture_module)``.
        """
        raise NotImplementedError

    def fit(self, model, train_gen, valid_gen, args, parameters, architecture_module):
        """Train, writing ``<prefix><suffix>``, ``<prefix>.log`` and
        ``<prefix>.log.batch``; the architecture module writes the no-bias model.
        """
        raise NotImplementedError

    # -- inference ---------------------------------------------------------
    def resolve_device(self, value=None):
        """Turn a ``--device`` value into whatever this engine places work on."""
        raise NotImplementedError

    def load_predictor(self, path, device=None, *, announce=True, summary=True):
        """Load a checkpoint for prediction (see the module docstring)."""
        raise NotImplementedError

    def rescale_bias_logcounts(self, bias_path, output_path, seqs, cts):
        """Shift the bias model's count head so it fits ``cts``, and save it.

        Adds ``mean(log(1 + cts) - predicted_logcounts)`` to the bias term of
        the count head's final dense layer and writes the result to
        ``output_path``.
        """
        raise NotImplementedError

    # -- attribution -------------------------------------------------------
    def load_attribution_model(self, path, device):
        """Load a checkpoint for DeepLIFT/DeepSHAP on an already-resolved device."""
        raise NotImplementedError

    def input_length(self, model):
        """Sequence length an attribution model expects."""
        raise NotImplementedError

    def attribute(self, model, seqs, profile_or_counts, *, device, batch_size,
                  num_shuffles, seed, profile_stop_gradient=False):
        """Yield ``(head, hypothetical_scores)`` for each requested head.

        ``seqs`` is an ``(N, L, 4)`` ACGT-only one-hot array and the scores
        have the same shape. Heads are yielded in the order counts, profile,
        so a caller can write each one before the next is computed.

        The profile target is ``sum(softmax(stop_gradient(mn)) * mn)`` over
        the mean-normalized logits ``mn``. By default every engine weights
        the logits as chrombpnet 1.x and bpnet do, whose kundajelab shap fork
        leaves the ``StopGradient`` handler off: for each (input, reference)
        pair, by the midpoint ``0.5 * (softmax(mn_x) + softmax(mn_r))``.
        ``profile_stop_gradient`` (``--contribs-profile-stopgrad``) honours the
        stop-gradient instead: the input's own ``softmax(mn_x)``. The counts
        head is the same either way.
        """
        raise NotImplementedError
