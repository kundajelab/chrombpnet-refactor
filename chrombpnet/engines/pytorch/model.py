"""PyTorch implementations of the ChromBPNet model family.

The public modules use PyTorch's native NCL convention: batch, channels, length.
File/data adapters at the edge of the package accept the historical NLC layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from chrombpnet import fused_kernels


SOURCE_COMMIT = "09938fdb4397ec0006510e5251e48920a505d4de"
CHECKPOINT_VERSION = 1


def _as_bool(value: Any) -> bool:
    """A config flag that may arrive as a bool or as parameter-table text."""
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "")
    return bool(value)


@dataclass(frozen=True)
class ModelConfig:
    input_length: int = 2114
    output_length: int = 1000
    filters: int = 512
    n_dilation_layers: int = 8
    counts_loss_weight: float = 1.0
    #: False for a profile-only bias model (``bias pipeline
    #: --bias-profile-only``): no counts head, and a ChromBPNet built on it
    #: takes its counts from the accessibility model alone. Checkpoints
    #: written before the field existed have one, hence the default.
    counts_head: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        return cls(
            input_length=int(values.get("input_length", values.get("inputlen", 2114))),
            output_length=int(values.get("output_length", values.get("outputlen", 1000))),
            filters=int(values.get("filters", 512)),
            n_dilation_layers=int(
                values.get("n_dilation_layers", values.get("n_dil_layers", 8))
            ),
            counts_loss_weight=float(values.get("counts_loss_weight", 1.0)),
            counts_head=_as_bool(values.get("counts_head", True)),
        )

    def validate(self) -> None:
        if self.input_length % 2 or self.output_length % 2:
            raise ValueError("input_length and output_length must be even")
        if self.input_length < self.output_length:
            raise ValueError("input_length must be at least output_length")
        if self.filters < 1 or self.n_dilation_layers < 0:
            raise ValueError("filters must be positive and dilation layers non-negative")


def _center_crop(x: Tensor, target_length: int) -> Tensor:
    if x.shape[-1] == target_length:
        return x
    difference = x.shape[-1] - target_length
    if difference < 0 or difference % 2:
        raise ValueError(
            f"Cannot symmetrically crop length {x.shape[-1]} to {target_length}"
        )
    left = difference // 2
    return x[..., left : left + target_length]


class BPNet(nn.Module):
    """Single-task BPNet used as either the bias or TF model."""

    def __init__(self, config: Union[ModelConfig, Mapping[str, Any]]) -> None:
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig.from_mapping(config)
        self.config.validate()

        channels = self.config.filters
        self.first_conv = nn.Conv1d(4, channels, kernel_size=21, padding=0)
        self.first_activation = nn.ReLU()
        self.dilated_convs = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                dilation=2**layer,
                padding=0,
            )
            for layer in range(1, self.config.n_dilation_layers + 1)
        )
        self.dilated_activations = nn.ModuleList(
            nn.ReLU() for _ in range(self.config.n_dilation_layers)
        )
        self.profile_conv = nn.Conv1d(channels, 1, kernel_size=75, padding=0)
        self.count_dense = nn.Linear(channels, 1) if self.config.counts_head else None

        with torch.no_grad():
            probe = torch.zeros(1, 4, self.config.input_length)
            feature = self._features(probe)
            profile_length = self.profile_conv(feature).shape[-1]
        if profile_length < self.config.output_length:
            raise ValueError(
                "Architecture receptive field leaves profile shorter than output_length: "
                f"{profile_length} < {self.config.output_length}"
            )
        if (profile_length - self.config.output_length) % 2:
            raise ValueError("Profile crop must be symmetric")

    @property
    def input_shape(self) -> Tuple[None, int, int]:
        return (None, 4, self.config.input_length)

    @property
    def output_shape(self) -> Tuple[Tuple[None, int], Optional[Tuple[None, int]]]:
        """The profile and counts output shapes; counts is None without a
        counts head."""
        counts = (None, 1) if self.config.counts_head else None
        return ((None, self.config.output_length), counts)

    def _features(self, sequence: Tensor) -> Tensor:
        x = self.first_activation(self.first_conv(sequence))
        for convolution, activation in zip(self.dilated_convs, self.dilated_activations):
            convolved = activation(convolution(x))
            x = convolved + _center_crop(x, convolved.shape[-1])
        return x

    def forward(self, sequence: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """``(profile logits, logcounts)``; logcounts is None without a counts
        head."""
        if sequence.ndim != 3 or sequence.shape[1] != 4:
            raise ValueError("sequence must have shape (batch, 4, length)")
        if sequence.shape[-1] != self.config.input_length:
            raise ValueError(
                f"Expected sequence length {self.config.input_length}, got {sequence.shape[-1]}"
            )
        x = self._features(sequence.float())
        profile = _center_crop(self.profile_conv(x), self.config.output_length)[:, 0, :]
        if self.count_dense is None:
            return profile, None
        logcounts = self.count_dense(x.mean(dim=-1))
        return profile, logcounts


class BiasModel(BPNet):
    """Semantic name for the background enzyme-bias model."""


def assign_keras_layout_weights(model: BPNet, values: Mapping[str, np.ndarray]) -> None:
    """Assign Keras-layout BPNet arrays to the corresponding PyTorch layers."""
    with torch.no_grad():
        model.first_conv.weight.copy_(
            torch.from_numpy(values["first_kernel"]).permute(2, 1, 0)
        )
        model.first_conv.bias.copy_(torch.from_numpy(values["first_bias"]))
        for layer, convolution in enumerate(model.dilated_convs, start=1):
            convolution.weight.copy_(
                torch.from_numpy(values[f"dilated_{layer}_kernel"]).permute(2, 1, 0)
            )
            convolution.bias.copy_(
                torch.from_numpy(values[f"dilated_{layer}_bias"])
            )
        model.profile_conv.weight.copy_(
            torch.from_numpy(values["profile_kernel"]).permute(2, 1, 0)
        )
        model.profile_conv.bias.copy_(torch.from_numpy(values["profile_bias"]))
        if model.count_dense is not None:
            model.count_dense.weight.copy_(torch.from_numpy(values["count_kernel"]).T)
            model.count_dense.bias.copy_(torch.from_numpy(values["count_bias"]))


def initialize_keras_compatible(model: BPNet, seed: int) -> None:
    """Assign canonical Keras-layout Glorot weights to a PyTorch BPNet."""
    from chrombpnet.training.utils.benchmark_utils import canonical_bpnet_weights

    values = canonical_bpnet_weights(
        model.config.filters, model.config.n_dilation_layers, seed
    )
    assign_keras_layout_weights(model, values)


def initialize_keras_default(model: BPNet, seed: int) -> None:
    """Re-initialize a BPNet the way Keras initializes Conv1D/Dense layers.

    Keras defaults to Glorot/Xavier-uniform kernels with zero biases, while
    PyTorch defaults to Kaiming-uniform kernels with uniform biases. Unlike
    :func:`initialize_keras_compatible` this only matches the initializer
    family, so the two frameworks draw different concrete values. Draws come
    from a dedicated generator so they do not depend on how many values the
    native constructor already consumed from the global RNG.
    """
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, (nn.Conv1d, nn.Linear)):
                continue
            weight = module.weight
            # Conv1d weight is (out, in, kernel); Linear weight is (out, in).
            receptive_field = int(np.prod(weight.shape[2:])) if weight.ndim > 2 else 1
            fan_in = weight.shape[1] * receptive_field
            fan_out = weight.shape[0] * receptive_field
            limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
            weight.uniform_(-limit, limit, generator=generator)
            if module.bias is not None:
                module.bias.zero_()


def initialize_for_benchmark(model: BPNet, args) -> str:
    """Apply the initializer for the run's reproducibility level.

    ``--reproducibility-max`` shares the exact arrays with the other engines;
    otherwise the weights come from Keras's default initializer family
    (Glorot/Xavier-uniform, zero biases), as in the TensorFlow implementation.
    PyTorch's own default initialization is never used.
    """
    from chrombpnet.training.utils.benchmark_utils import benchmark_enables

    if benchmark_enables(args, "shared_initial_weights"):
        initialize_keras_compatible(model, args.seed)
        return "shared_canonical_arrays"
    initialize_keras_default(model, args.seed)
    return "keras_default_glorot"


class ChromBPNet(nn.Module):
    """Bias-factorized ChromBPNet with a frozen bias submodel."""

    def __init__(
        self,
        config: Union[ModelConfig, Mapping[str, Any]],
        bias_model: BPNet,
        tf_model: Optional[BPNet] = None,
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig.from_mapping(config)
        self.config.validate()
        if bias_model.config.input_length != self.config.input_length:
            raise ValueError("Bias and TF input lengths differ")
        if bias_model.config.output_length != self.config.output_length:
            raise ValueError("Bias and TF output lengths differ")
        self.bias_model = bias_model
        self.tf_model = tf_model or BPNet(self.config)
        for parameter in self.bias_model.parameters():
            parameter.requires_grad = False
        self.bias_model.eval()

    @property
    def input_shape(self) -> Tuple[None, int, int]:
        return self.tf_model.input_shape

    @property
    def output_shape(self) -> Tuple[Tuple[None, int], Tuple[None, int]]:
        return self.tf_model.output_shape

    def train(self, mode: bool = True) -> "ChromBPNet":
        super().train(mode)
        self.bias_model.eval()
        return self

    def forward(self, sequence: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            bias_profile, bias_logcounts = self.bias_model(sequence)
        tf_profile, tf_logcounts = self.tf_model(sequence)
        profile = tf_profile + bias_profile
        if bias_logcounts is None:
            # A profile-only bias joins the profile head alone; the counts
            # are the accessibility model's.
            return profile, tf_logcounts
        logcounts = torch.logsumexp(
            torch.cat((tf_logcounts, bias_logcounts), dim=-1), dim=-1, keepdim=True
        )
        return profile, logcounts


def model_type(model: nn.Module) -> str:
    if isinstance(model, ChromBPNet):
        return "chrombpnet"
    if isinstance(model, BiasModel):
        return "bias"
    if isinstance(model, BPNet):
        return "chrombpnet_nobias"
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def checkpoint_payload(
    model: nn.Module,
    *,
    kind: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("Model does not expose a ModelConfig")
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "model_type": kind or model_type(model),
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "provenance": {
            "source_repository_commit": SOURCE_COMMIT,
            **dict(provenance or {}),
        },
    }
    if isinstance(model, ChromBPNet):
        # Bias and TF branches normally have different filter/dilation counts.
        payload["bias_config"] = asdict(model.bias_model.config)
    return payload


def save_model(
    model: nn.Module,
    path: Union[str, Path],
    *,
    kind: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    torch.save(checkpoint_payload(model, kind=kind, provenance=provenance), str(path))


def _load_payload(path: Union[str, Path], map_location: Any) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.6
        return torch.load(str(path), map_location=map_location)


def load_model(
    path: Union[str, Path], map_location: Any = "cpu"
) -> nn.Module:
    payload = _load_payload(path, map_location)
    if int(payload.get("format_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    config = ModelConfig.from_mapping(payload["config"])
    kind = payload["model_type"]
    if kind == "bias":
        model: nn.Module = BiasModel(config)
    elif kind == "chrombpnet_nobias":
        model = BPNet(config)
    elif kind == "chrombpnet":
        bias = BiasModel(ModelConfig.from_mapping(payload.get("bias_config", payload["config"])))
        model = ChromBPNet(config, bias)
    else:
        raise ValueError(f"Unknown model_type {kind!r}")
    model.load_state_dict(payload["state_dict"])
    # map_location is a torch.load argument, which also accepts a callable or a
    # dict; only forward it to .to() when it really is a device.
    if isinstance(map_location, (str, torch.device)):
        model.to(map_location)
    model.eval()
    return model


class PredictionAdapter:
    """Small Keras-like adapter used by the retained genomics utilities."""

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        config = model.config
        self.input_shape = (None, config.input_length, 4)
        # No counts output for a profile-only bias model; predict() then
        # returns None for the logcounts.
        self.output_shape = [(None, config.output_length),
                             (None, 1) if config.counts_head else None]
        # --predict-fused-kernel-exact / -fast. Built after .to(device): the kernels snapshot and
        # transpose the weights at construction, so the module has to be on
        # its final device first. Refusals propagate -- the route is opt-in
        # and silently falling back would hide that it did not engage.
        self._forward = self.model
        if fused_kernels.enabled():
            from chrombpnet.engines.pytorch import triton_forward
            self._forward = triton_forward.build(
                self.model, self.device, precision=fused_kernels.precision())

    def summary(self) -> None:
        print(self.model)

    def predict(
        self,
        values: Union[np.ndarray, Tensor, list],
        batch_size: int = 64,
        verbose: Union[bool, int] = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(values, list):
            values = values[0]
        if not isinstance(values, np.ndarray):
            values = np.asarray(values.detach().cpu() if torch.is_tensor(values) else values)
        if values.ndim != 3:
            raise ValueError("Input must be three-dimensional")
        if values.shape[-1] == 4:
            channel_last = True
        elif values.shape[1] == 4:
            channel_last = False
        else:
            raise ValueError("Input must be NLC or NCL with four sequence channels")

        total = values.shape[0]
        if total == 0:
            # torch.cat over no batches raised here before the outputs were
            # preallocated; keep that rather than quietly returning empties.
            raise RuntimeError("Cannot predict on an empty input")

        batch_size = int(batch_size)
        profiles = None
        logcounts = None
        self.model.eval()
        # inference_mode rather than no_grad: same arithmetic, less bookkeeping.
        # Results are copied into ordinary NumPy buffers below, so no inference
        # tensor escapes to a caller that might later mutate it.
        with torch.inference_mode():
            for start in range(0, total, batch_size):
                chunk = values[start:start + batch_size]
                # One contiguous NCL float32 buffer per batch, instead of
                # casting the whole input up front (16x the one-hot memory:
                # 676 MB for 20,000 x 2,114) and slicing a permuted,
                # non-contiguous view that had to be transposed at every
                # transfer. ascontiguousarray also absorbs the negative-stride
                # reverse-complement views (x[:, ::-1, ::-1]) that torch
                # cannot wrap.
                if channel_last:
                    chunk = chunk.transpose(0, 2, 1)
                chunk = np.ascontiguousarray(chunk, dtype=np.float32)
                profile, counts = self._forward(torch.from_numpy(chunk).to(self.device))
                if profiles is None:
                    profiles = np.empty((total,) + tuple(profile.shape[1:]), dtype=np.float32)
                    if counts is not None:
                        logcounts = np.empty((total,) + tuple(counts.shape[1:]), dtype=np.float32)
                stop = start + profile.shape[0]
                profiles[start:stop] = profile.cpu().numpy()
                if counts is not None:
                    logcounts[start:stop] = counts.cpu().numpy()
        return profiles, logcounts

    def _predict_legacy(self, values, batch_size):
        """Legacy: the pre-batching predict. Unreachable in normal operation.

        It casts the whole input to one float32 CPU tensor up front and
        slices a permuted, non-contiguous view.
        """
        if isinstance(values, np.ndarray) and any(stride < 0 for stride in values.strides):
            values = np.ascontiguousarray(values)
        array = torch.as_tensor(values, dtype=torch.float32)
        if array.ndim != 3:
            raise ValueError("Input must be three-dimensional")
        if array.shape[-1] == 4:
            array = array.permute(0, 2, 1)
        elif array.shape[1] != 4:
            raise ValueError("Input must be NLC or NCL with four sequence channels")
        profiles, logcounts = [], []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, array.shape[0], int(batch_size)):
                profile, counts = self._forward(
                    array[start:start + int(batch_size)].to(self.device))
                profiles.append(profile.cpu())
                if counts is not None:
                    logcounts.append(counts.cpu())
        return (torch.cat(profiles).numpy(),
                torch.cat(logcounts).numpy() if logcounts else None)

    def predict_on_batch(self, values: Union[np.ndarray, Tensor, list]):
        size = len(values[0] if isinstance(values, list) else values)
        return self.predict(values, batch_size=max(size, 1))


def load_prediction_model(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> PredictionAdapter:
    return PredictionAdapter(load_model(path, map_location=device), device=device)


def load_model_wrapper(
    path: Union[str, Path],
    device: Optional[str] = None,
    *,
    announce: bool = True,
    summary: bool = True,
) -> PredictionAdapter:
    """Load a checkpoint for prediction, with the CLI's console output.

    This replaces five near-identical copies that differed only in what they
    printed: the two nested in ``pipelines.py`` printed no banner, and
    ``training/predict.py`` had its ``summary()`` call commented out. The flags
    keep each call site's stdout exactly as it was.
    """
    model = load_prediction_model(path, device=resolve_device(device))
    if announce:
        print("got the model")
    if summary:
        model.summary()
    return model


def resolve_device(value: Optional[str] = None) -> torch.device:
    if value:
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
