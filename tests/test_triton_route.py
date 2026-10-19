"""The phase-2 Triton route: vendor integrity, refusals, and GPU agreement.

The golden-artifact harness cannot cover this route at all -- its fixture is
`-fil 4`, which the route refuses by design -- so everything that proves it
lives here. See tests/regression/README.md.
"""

import hashlib
import pathlib
import sys

import pytest
# PyTorch-engine tests: skipped in the TensorFlow and JAX environments.
torch = pytest.importorskip("torch")

from chrombpnet import fused_kernels, triton_forward
from chrombpnet.model import BiasModel, BPNet, ChromBPNet, ModelConfig

VENDOR = pathlib.Path(triton_forward.__file__).parent / "_vendor" / "chrombpnet_k1"

#: Recorded in the vendor README. `kernels.py`'s first 16 hex characters are
#: the `kernels_sha256_16` the kit ships in its JIT identity record, so this
#: also pins that we hold the file its certificates were measured against.
VENDOR_DIGESTS = {
    "kernels.py": "7026b40a233544fbbc49d464731e42e084cd53097ca7f60f606e8cf145dac915",
    "arch_tiles.json": "9a886921533527199b9fcb3c2a46ee19995bd08eb2aa7eb1610bd184a8d7cbfa",
}


def _production(filters=512):
    return BPNet(ModelConfig(2114, 1000, filters=filters, n_dilation_layers=8))


# --------------------------------------------------------------- CPU-only

@pytest.mark.parametrize("name,digest", sorted(VENDOR_DIGESTS.items()))
def test_vendored_files_are_unmodified(name, digest):
    """Vendored byte-identical so it can be diffed against upstream forever."""
    assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest


def test_importing_chrombpnet_does_not_import_triton():
    """`import chrombpnet` must work on a machine without triton, and must
    not pay for it on a CPU-only run."""
    assert "chrombpnet" in sys.modules
    assert "chrombpnet.triton_forward" in sys.modules
    assert "chrombpnet.engines.pytorch._vendor.chrombpnet_k1.kernels" not in sys.modules


def test_the_shim_maps_the_model_onto_the_kernels_vocabulary():
    """The one test that catches a future rename in chrombpnet/model.py."""
    model = _production()
    view = triton_forward._KitView(model)
    assert view.iconv is model.first_conv
    assert view.fconv is model.profile_conv
    assert view.linear is model.count_dense
    assert len(view.rconvs) == 8
    assert [c.dilation[0] for c in view.rconvs] == [2, 4, 8, 16, 32, 64, 128, 256]
    assert view.iconv.kernel_size[0] == 21
    assert view.fconv.kernel_size[0] == 75


def test_profile_length_matches_the_real_architecture():
    """The symmetric-crop check is only as good as this arithmetic."""
    model = _production()
    with torch.no_grad():
        features = model._features(torch.zeros(1, 4, 2114))
        actual = model.profile_conv(features).shape[-1]
    assert triton_forward._profile_length(model.config) == actual


def test_a_cpu_device_is_refused():
    with pytest.raises(triton_forward.TritonRouteRefused, match="CUDA"):
        triton_forward.require(_production(), "cpu")


def test_an_unsupported_width_is_refused():
    """filters outside {128, 512} makes the counts kernel produce zeros or
    drop terms, both silently -- so this must refuse, never degrade."""
    model = BPNet(ModelConfig(2114, 1000, filters=64, n_dilation_layers=8))
    with pytest.raises(triton_forward.TritonRouteRefused, match="filters=64"):
        triton_forward.require(model, "cuda")


def test_a_non_bpnet_is_refused():
    with pytest.raises(triton_forward.TritonRouteRefused, match="not a BPNet"):
        triton_forward.require(torch.nn.Linear(4, 4), "cuda")


def _fake_cuda(monkeypatch, capability, shared=0):
    """`describe` on any host: a compute capability and an opt-in shared
    memory are all it reads from the device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    monkeypatch.setattr(triton_forward, "_shared_memory_optin", lambda index: shared)


@pytest.mark.parametrize("capability,tile", [
    ((9, 0), "64x512x16x16x3"),     # H100/H200
    ((8, 9), "64x256x16x8x3"),      # L40S
    ((8, 0), "64x512x16x16x3"),     # A100
])
def test_a_listed_architecture_uses_the_kits_measured_tile(monkeypatch, capability, tile):
    """The A100 too: the kit's route table defaults it to stock TensorFlow
    only for bitwise equality with stock's output, which an opt-in flag over
    the PyTorch forward does not need."""
    _fake_cuda(monkeypatch, capability)
    arch, entry = triton_forward.describe("cuda")
    assert (entry["tile"], entry["measured"]) == (tile, True)


@pytest.mark.parametrize("shared,tile", [
    (232448, "64x512x16x16x3"),     # room for the large tile (e.g. Blackwell)
    (101376, "64x256x16x8x3"),      # e.g. sm_86
    (0, "64x256x16x8x3"),           # unreadable: the small tile, as the kit does
])
def test_an_unlisted_architecture_gets_the_kits_shared_memory_tile(monkeypatch, shared, tile):
    """The kit's `forward.select_tile` for an architecture its table does
    not list."""
    _fake_cuda(monkeypatch, (12, 0), shared)
    arch, entry = triton_forward.describe("cuda")
    assert (arch, entry["tile"], entry["measured"]) == ("cuda-120", tile, False)


def test_an_unlisted_architecture_is_checked_against_the_eager_forward():
    model = _production(filters=128).eval()
    triton_forward._check_against_eager(model, model, "cpu", "cuda-120", "t")

    def wrong(x):
        profile, counts = model(x)
        return profile + 1, counts

    with pytest.raises(triton_forward.TritonRouteRefused, match="disagree"):
        triton_forward._check_against_eager(wrong, model, "cpu", "cuda-120", "t")

    def broken(x):
        raise RuntimeError("out of resource: shared memory")

    with pytest.raises(triton_forward.TritonRouteRefused, match="do not run"):
        triton_forward._check_against_eager(broken, model, "cpu", "cuda-120", "t")


def test_an_out_of_window_triton_is_refused(monkeypatch):
    monkeypatch.setattr(triton_forward, "SUPPORTED_TRITON", ((99, 0), (99, 1)))
    with pytest.raises(triton_forward.TritonRouteRefused, match="outside the verified"):
        triton_forward.require(_production(), "cuda")


def test_the_module_uses_no_asserts():
    """Validation must survive `python -O`."""
    import re
    source = pathlib.Path(triton_forward.__file__).read_text()
    assert not re.search(r"^\s*assert\b", source, re.M)


# --------------------------------------------------------------- GPU

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def route_and_model():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    model = ChromBPNet(
        ModelConfig(2114, 1000, filters=512, n_dilation_layers=8),
        BiasModel(ModelConfig(2114, 1000, filters=128, n_dilation_layers=4)),
    ).cuda().eval()
    try:
        return triton_forward.build(model, "cuda"), model
    except triton_forward.TritonRouteRefused as exc:
        pytest.skip(f"route refused on this card: {exc}")


@gpu
def test_the_forward_is_batch_invariant(route_and_model):
    """The strongest cheap check available: every output element of these
    kernels is independent of N, so any tiling or masking bug shows up as a
    difference across batch sizes. cuDNN is *not* batch-invariant, so this
    is also a real improvement over the eager path."""
    forward, _ = route_and_model
    x = torch.randint(0, 2, (96, 4, 2114), device="cuda").float()
    with torch.inference_mode():
        whole = forward(x)
        for batch in (1, 7, 32):
            pieces = [forward(x[i:i + batch]) for i in range(0, 96, batch)]
            profile = torch.cat([p for p, _ in pieces])
            counts = torch.cat([c for _, c in pieces])
            assert torch.equal(profile, whole[0]), batch
            assert torch.equal(counts, whole[1]), batch


@gpu
def test_the_forward_is_deterministic(route_and_model):
    forward, _ = route_and_model
    x = torch.randint(0, 2, (16, 4, 2114), device="cuda").float()
    with torch.inference_mode():
        first, second = forward(x), forward(x)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@gpu
def test_a_non_contiguous_input_is_handled(route_and_model):
    """`PredictionAdapter`'s unoptimized path hands over a permuted view. The
    kernels compute raw offsets from the logical shape, so without an
    explicit contiguous() they read garbage *without erroring*."""
    forward, _ = route_and_model
    nlc = torch.randint(0, 2, (8, 2114, 4), device="cuda").float()
    view = nlc.permute(0, 2, 1)
    assert not view.is_contiguous()
    with torch.inference_mode():
        from_view = forward(view)
        from_copy = forward(view.contiguous())
    assert torch.equal(from_view[0], from_copy[0])
    assert torch.equal(from_view[1], from_copy[1])


@gpu
def test_it_agrees_with_the_eager_forward(route_and_model):
    """Tolerances are the measured maxima, rounded up, not guesses: the
    profile head sums the 75-tap kernel in TensorFlow's order rather than
    cuDNN's, which is where the residual difference comes from."""
    forward, model = route_and_model
    x = torch.randint(0, 2, (64, 4, 2114), device="cuda").float()
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False       # compare fp32 against fp32
    try:
        with torch.inference_mode():
            triton_out = forward(x)
            eager_out = model(x)
    finally:
        torch.backends.cudnn.allow_tf32 = previous
    assert torch.allclose(triton_out[0], eager_out[0], atol=5e-5, rtol=0)
    assert torch.allclose(triton_out[1], eager_out[1], atol=5e-6, rtol=0)


@gpu
def test_the_adapter_uses_the_route_only_under_predict_fk(predict_fk):
    from chrombpnet.model import PredictionAdapter
    model = _production().cuda().eval()
    predict_fk(False)
    assert PredictionAdapter(model, device="cuda")._forward is model
    predict_fk(True)
    assert PredictionAdapter(model, device="cuda")._forward is not model
