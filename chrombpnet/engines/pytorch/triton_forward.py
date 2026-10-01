"""The ChromBPNet forward on the kit's Triton kernels (optimization phase 2).

`chrombpnet/engines/pytorch/_vendor/chrombpnet_k1/kernels.py` is the ChromBPNet forward
re-expressed as Triton kernels, taken byte-identical from the ChromBPNet
optimization kit at github.com/anthropics/uplifting-biomolecular-modeling
(Apache-2.0). This module is the local half: it presents a `BPNet` to those
kernels, picks the per-architecture tile, and refuses when any of the
kernels' assumptions does not hold.

**Numerics.** The kernels are fp32-exact -- each convolution output is one
serial FMA chain, the order cuDNN's IMPLICIT_GEMM uses. The eager path is
not: `torch.backends.cudnn.allow_tf32` defaults to True, so ordinary
convolutions here run in TF32, whose error against an fp64 reference is
about 1.1e-3. Measured against the eager path with TF32 disabled, a single
dilated convolution is *bitwise* equal and the full model's log-counts are
bitwise equal at every batch shape tested; the profile head agrees to ~2e-5,
because `head_serial` sums the 75-tap kernel in TensorFlow's order rather
than cuDNN's. So phase 2 makes the forward more accurate, not less -- but it
does move the output, which is why it is off by default.

**Refusals, not fallbacks.** Every unmet precondition raises
`TritonRouteRefused` naming the check. Two of the kernels' constraints fail
*silently* if ignored rather than erroring -- see `filters` below -- so a
quiet fallback would be worse than useless here.
"""

from __future__ import annotations

import json
import os

_ARCH_TILES_PATH = os.path.join(
    os.path.dirname(__file__), "_vendor", "chrombpnet_k1", "arch_tiles.json")

#: The counts head's dense kernel splits C into 16 lanes (`S = C // 16`) and
#: the butterfly combine halves its group size repeatedly. At C=8 that gives
#: S=0 and the kernel writes zeros; at a non-power-of-two group the butterfly
#: drops elements. Neither raises. Production uses exactly these two widths.
SUPPORTED_FILTERS = (128, 512)

#: Triton reworked its fp32 `tl.dot` lowering across releases, and the whole
#: point of `input_precision="ieee"` is that it stays a sequential FMA chain.
#: Outside this window the kernels may still run but the numerics claim is
#: unverified, so refuse rather than quietly produce something else.
SUPPORTED_TRITON = ((3, 0), (4, 0))     # [min, max)

#: Kernel indices are int32.
MAX_ELEMENTS = 2 ** 31


class TritonRouteRefused(RuntimeError):
    """The Triton forward cannot serve this model, device or stack."""


def _load_arch_tiles():
    with open(_ARCH_TILES_PATH) as handle:
        return json.load(handle)


def _triton_version_tuple(module):
    parts = []
    for piece in str(module.__version__).split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


def describe(device=None):
    """`(arch, entry)` from the kit's table for this device, or refuse."""
    import torch

    if not torch.cuda.is_available():
        raise TritonRouteRefused(
            "the Triton forward needs a CUDA device; none is available "
            "(use --optimize without phase 2)")
    index = None
    if device is not None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise TritonRouteRefused(
                f"the Triton forward needs a CUDA device; asked for "
                f"'{resolved}' (use --optimize without phase 2)")
        index = resolved.index
    major, minor = torch.cuda.get_device_capability(index)
    arch = f"cuda-{major}{minor}"
    entries = _load_arch_tiles()["entries"]
    if arch not in entries:
        raise TritonRouteRefused(
            f"no measured tile for {arch} ({torch.cuda.get_device_name(index)}); "
            f"the kit's table covers {sorted(entries)}")
    entry = entries[arch]
    route = entry.get("default_route", {}).get("det", {})
    if route.get("route") != "k1":
        raise TritonRouteRefused(
            f"{arch} has no bitwise Triton route. The kit's own basis: "
            f"{route.get('basis', 'not stated')}")
    return arch, entry


def require(model, device=None):
    """Raise unless the Triton forward can serve `model` on `device`."""
    from chrombpnet.engines.pytorch.model import BPNet, ChromBPNet

    try:
        import triton
    except Exception as exc:
        raise TritonRouteRefused(f"triton is not importable: {exc!r}") from exc
    version = _triton_version_tuple(triton)
    if not (SUPPORTED_TRITON[0] <= version < SUPPORTED_TRITON[1]):
        raise TritonRouteRefused(
            f"triton {triton.__version__} is outside the verified window "
            f"{SUPPORTED_TRITON[0]}-{SUPPORTED_TRITON[1]}; the fp32 dot "
            f"lowering is what the numerics claim rests on")

    arch, entry = describe(device)

    if isinstance(model, ChromBPNet):
        submodels = [model.tf_model, model.bias_model]
    elif isinstance(model, BPNet):
        submodels = [model]
    else:
        raise TritonRouteRefused(
            f"{type(model).__name__} is not a BPNet or ChromBPNet")

    for submodel in submodels:
        config = submodel.config
        if config.filters not in SUPPORTED_FILTERS:
            raise TritonRouteRefused(
                f"filters={config.filters} is unsupported; the counts head's "
                f"dense kernel needs C in {SUPPORTED_FILTERS} (it computes "
                f"C // 16 lanes and halves a power-of-two group, and produces "
                f"wrong values silently otherwise)")
        if submodel.first_conv.in_channels != 4 or submodel.first_conv.kernel_size[0] != 21:
            raise TritonRouteRefused("the first convolution must be 4 -> C, width 21")
        for conv in submodel.dilated_convs:
            if conv.kernel_size[0] != 3:
                raise TritonRouteRefused(
                    "every dilated convolution must have width 3; it is a "
                    "constant in the kernels' residual epilogue")
            if conv.in_channels != conv.out_channels:
                raise TritonRouteRefused(
                    "dilated convolutions must preserve width for the residual add")
        if submodel.profile_conv.out_channels != 1:
            raise TritonRouteRefused("the profile head must have one output channel")
        for conv in [submodel.first_conv, *submodel.dilated_convs, submodel.profile_conv]:
            if conv.padding[0] != 0 or conv.stride[0] != 1 or conv.bias is None:
                raise TritonRouteRefused(
                    "convolutions must be unpadded, unstrided and biased")
        if submodel.count_dense.bias is None:
            raise TritonRouteRefused("the counts head must be biased")

        # K1BPNetV3 crops the profile with (Lp - out_len) // 2, which would
        # crop off-centre rather than raise where _center_crop raises.
        profile_length = _profile_length(config)
        margin = profile_length - config.output_length
        if margin < 0 or margin % 2:
            raise TritonRouteRefused(
                f"the profile crop is not symmetric ({profile_length} -> "
                f"{config.output_length}); the kernels would crop off-centre")
    return arch, entry


def _profile_length(config):
    """Length out of `profile_conv`, from the receptive field alone."""
    length = config.input_length - 20                      # first conv, width 21
    for layer in range(1, config.n_dilation_layers + 1):
        length -= 2 * (2 ** layer)                         # width 3, dilation 2**layer
    return length - 74                                     # profile conv, width 75


class _KitView:
    """A `BPNet` in the vocabulary `K1BPNetV3.__init__` reads.

    It holds live references, so there is no second copy of the weights.
    Renaming the real attributes was rejected: it would change every
    `state_dict` key and break `load_model`, the Keras-layout weight import,
    the bias rescaling in `find_chrombpnet_hyperparams`, every existing `.pt`
    and every regression baseline.
    """

    def __init__(self, bpnet):
        self.iconv = bpnet.first_conv
        self.rconvs = list(bpnet.dilated_convs)
        self.fconv = bpnet.profile_conv
        self.linear = bpnet.count_dense


def build(model, device=None, counts="exact"):
    """A callable `(N,4,L) -> (profile, logcounts)` on the Triton kernels.

    Call it only after `model` is on `device`: `K1BPNetV3` snapshots and
    transposes the weights at construction.
    """
    import torch

    from chrombpnet.engines.pytorch.model import ChromBPNet

    arch, entry = require(model, device)
    from chrombpnet.engines.pytorch._vendor.chrombpnet_k1 import kernels

    tile = entry["tile"]
    block_m, block_n, block_k, warps, stages = (int(v) for v in tile.split("x"))
    # conv_dot reads these module globals at call time. Set them here rather
    # than through os.environ, which the kit mutates but this package does not.
    kernels._BM, kernels._BN = block_m, block_n
    kernels._BK, kernels._NW, kernels._NS = block_k, warps, stages

    # The whole record, not its name: exact_counts reads form.get("form")
    # and the group size alongside it.
    counts_form = entry.get("counts_form")
    def _sub(bpnet):
        return kernels.K1BPNetV3(_KitView(bpnet), counts_mode=counts,
                                 head="serial", counts_form=counts_form)

    if isinstance(model, ChromBPNet):
        accessibility = _sub(model.tf_model)
        bias = _sub(model.bias_model)
        accessibility.out_len = int(model.tf_model.config.output_length)
        bias.out_len = int(model.bias_model.config.output_length)

        def forward(x):
            x = _prepare(x)
            profile_acc, counts_acc = accessibility(x)
            profile_bias, counts_bias = bias(x)
            # Same composition as ChromBPNet.forward: add the profiles, and
            # logsumexp the counts in (accessibility, bias) order. The order
            # is load-bearing for agreement with the eager path.
            return (profile_acc + profile_bias,
                    kernels.logsumexp_tf(counts_acc, counts_bias))
    else:
        single = _sub(model)
        single.out_len = int(model.config.output_length)

        def forward(x):
            return single(_prepare(x))

    forward.arch = arch
    forward.tile = tile
    return forward


def _prepare(x):
    """Contiguous fp32 NCL, which the kernels assume and do not check.

    They compute raw memory offsets from the logical shape, so a permuted
    view -- exactly what `PredictionAdapter`'s unoptimized path produces --
    is read as though it were contiguous and yields garbage rather than an
    error. Phase 2 must not depend on phase 3 having laid the input out for
    it, so normalize here. Both calls are no-ops when it is already right.
    """
    _check_bound(x)
    return x.contiguous().float()


def _check_bound(x):
    """The kernels index in int32."""
    elements = int(x.shape[0]) * int(x.shape[1]) * int(x.shape[2])
    if elements >= MAX_ELEMENTS:
        raise TritonRouteRefused(
            f"batch of {int(x.shape[0])} exceeds the kernels' int32 index bound; "
            f"lower --batch-size")
