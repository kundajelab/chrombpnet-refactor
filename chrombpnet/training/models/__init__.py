"""The PyTorch engine's built-in architecture files.

They stay here rather than under ``chrombpnet/engines/pytorch`` because their
paths are recorded in ``<prefix>.args.json``. The model classes are re-exported
lazily, so importing this package does not import torch.
"""

__all__ = ["BPNet", "BiasModel", "ChromBPNet", "ModelConfig"]


def __getattr__(name):
    if name in __all__:
        from chrombpnet.engines.pytorch import model as _model
        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
