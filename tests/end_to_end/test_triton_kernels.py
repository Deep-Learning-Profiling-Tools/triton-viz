import pytest
import torch
import triton
import triton.language as tl


COMPACTION_CASES = [
    (8192, 64, 4, 0.5),
    (8192, 64, 4, 1.0),
    (131, 128, 16, 0.6),
    (496, 128, 16, 0.0),
]
RAGGED_METADATA_N_SLICES = [1, 7, 33, 911, 1025]
REMAP_RAGGED_METADATA_N_SLICES = [9, 32, 911, 1025]


TOPK_FORWARD_CASES = [
    (7, 13, 8, True, "float16"),
]
TOPK_BACKWARD_CASES = [
    (7, 13, 8, True, "float16"),
]
SWIGLU_CASES = [(1311, 4352, 1e-2), (1311, 4352, 10)]
MXFP4_TILE_UPCAST_CASES = ["float16", "bfloat16", "float32"]
_mxfp4_tile_upcast = None


@pytest.fixture
def device(pytestconfig):
    requested = pytestconfig.getoption("--triton-kernels-device")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested for triton_kernels tests but unavailable")
    return requested


@pytest.fixture
def sanitizer_only(device):
    config_mod = pytest.importorskip("triton_viz.core.config")
    old_virtual_memory = config_mod.config.virtual_memory
    enabled = not (device == "cuda" and torch.cuda.is_available())
    config_mod.config.virtual_memory = enabled
    try:
        yield enabled
    finally:
        config_mod.config.virtual_memory = old_virtual_memory


def _new_sanitizer():
    pytest.importorskip("triton_viz")
    sanitizer_mod = pytest.importorskip("triton_viz.clients.sanitizer.sanitizer")
    return sanitizer_mod.SymbolicSanitizer(abort_on_error=True)


def _trace_kernel(monkeypatch, module, name, sanitizer):
    triton_viz = pytest.importorskip("triton_viz")
    traced = triton_viz.trace(client=sanitizer)(getattr(module, name))
    monkeypatch.setattr(module, name, traced)


@pytest.mark.parametrize(
    ("n_tokens", "n_cols", "k", "keep_prob"),
    COMPACTION_CASES,
)
def test_triton_viz_sanitizer_masked_compaction(
    monkeypatch, device, sanitizer_only, n_tokens, n_cols, k, keep_prob
):
    compaction_mod = pytest.importorskip("triton_kernels.compaction")

    torch.manual_seed(0)
    yi = (
        torch.rand((n_tokens, n_cols - 1), device=device)
        .argsort(dim=-1)[:, :k]
        .to(torch.int32)
    )
    yv = torch.randn((n_tokens, k), dtype=torch.bfloat16, device=device)

    keep = torch.rand(yi.shape, device=device) < keep_prob
    mask = torch.zeros((n_tokens, n_cols), dtype=torch.int32, device=device)
    if keep.any():
        rows = torch.arange(n_tokens, device=device).unsqueeze(1).expand_as(yi)
        mask[rows[keep], yi[keep]] = 1
    chunks = mask.view(n_tokens, -1, 32)
    weights = 1 << torch.arange(32, dtype=torch.int32, device=device)
    bitmask = (chunks * weights).sum(dim=-1)

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, compaction_mod, "_masked_compaction", sanitizer)
    compaction_mod.compaction(yv, yi, bitmask)
    assert len(sanitizer.records) == 0


@pytest.mark.xfail(
    reason="ragged metadata memset currently reports an upstream padded strided "
    "storage OOB",
    strict=True,
)
@pytest.mark.parametrize("n_slices", RAGGED_METADATA_N_SLICES)
def test_triton_viz_sanitizer_make_ragged_tensor_metadata(
    monkeypatch, device, sanitizer_only, n_slices
):
    ragged_mod = pytest.importorskip("triton_kernels.tensor_details.ragged_tensor")

    torch.manual_seed(0)
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(
        0, max_slice_size, (n_slices,), dtype=torch.int32, device=device
    )
    slice_sizes[torch.randint(0, n_slices, (1,), device=device)] = 0

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_memset", sanitizer)
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_compute", sanitizer)
    ragged_mod.make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    assert len(sanitizer.records) == 0


@pytest.mark.xfail(
    reason="ragged metadata memset currently reports an upstream padded strided "
    "storage OOB",
    strict=True,
)
@pytest.mark.parametrize("n_slices", REMAP_RAGGED_METADATA_N_SLICES)
def test_triton_viz_sanitizer_remap_ragged_tensor_metadata(
    monkeypatch, device, sanitizer_only, n_slices
):
    ragged_mod = pytest.importorskip("triton_kernels.tensor_details.ragged_tensor")

    torch.manual_seed(0)
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(
        0, max_slice_size, (n_slices,), dtype=torch.int32, device=device
    )
    slice_sizes[torch.randint(0, n_slices, (1,), device=device)] = 0
    slice_map = torch.randperm(n_slices, device=device, dtype=torch.int32)
    slice_map[torch.randint(0, n_slices, (min(5, n_slices),), device=device)] = -1

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_memset", sanitizer)
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_compute", sanitizer)
    _trace_kernel(monkeypatch, ragged_mod, "_remap_ragged_tensor_metadata", sanitizer)
    tri_metadata = ragged_mod.make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ragged_mod.remap_ragged_tensor_metadata(tri_metadata, slice_map)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "k", "apply_softmax", "dtype"),
    TOPK_FORWARD_CASES,
)
def test_triton_viz_sanitizer_topk_forward(
    monkeypatch, device, sanitizer_only, n_rows, n_cols, k, apply_softmax, dtype
):
    topk_mod = pytest.importorskip("triton_kernels.topk")

    torch.manual_seed(0)
    x = torch.randn((n_rows, n_cols), dtype=getattr(torch, dtype), device=device)

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, topk_mod, "_topk_forward", sanitizer)
    topk_mod.topk_forward(x, k, apply_softmax=apply_softmax)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "k", "apply_softmax", "dtype"),
    TOPK_BACKWARD_CASES,
)
def test_triton_viz_sanitizer_topk_backward(
    monkeypatch, device, sanitizer_only, n_rows, n_cols, k, apply_softmax, dtype
):
    topk_mod = pytest.importorskip("triton_kernels.topk")

    torch.manual_seed(0)
    x = torch.randn((n_rows, n_cols), device=device, dtype=getattr(torch, dtype))
    y_indx = torch.sort(
        torch.rand(n_rows, n_cols, device=device).argsort(dim=1).int()[:, :k],
        dim=1,
    )[0].to(torch.int16)
    dy_vals = torch.randn((n_rows, k), device=device, dtype=getattr(torch, dtype))
    active = max(1, n_rows // 2)
    active_rows = torch.tensor(active, device=device, dtype=torch.int32)

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, topk_mod, "_topk_backward", sanitizer)
    topk_mod.topk_backward(
        x,
        y_indx,
        dy_vals,
        k=k,
        n_rows=active_rows,
        apply_softmax=apply_softmax,
    )
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(("m", "n", "limit"), SWIGLU_CASES)
def test_triton_viz_sanitizer_swiglu(monkeypatch, device, sanitizer_only, m, n, limit):
    swiglu_mod = pytest.importorskip("triton_kernels.swiglu")

    torch.manual_seed(2)
    x = torch.randn((m, n), device=device, dtype=torch.bfloat16)
    precision_config = swiglu_mod.PrecisionConfig(limit=limit)

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, swiglu_mod, "_swiglu", sanitizer)
    swiglu_mod.swiglu(x, 0.5, precision_config)
    assert len(sanitizer.records) == 0


@triton.jit
def _triton_viz_mxfp4_tile_upcast_kernel(
    out,
    tensor,
    scale,
    stride_out_m,
    stride_out_k,
    stride_tensor_m,
    stride_tensor_k,
    stride_scale_m,
    stride_scale_k,
    BLOCK_M: tl.constexpr,
    PACKED_K: tl.constexpr,
    SCALE_K: tl.constexpr,
    dst_dtype: tl.constexpr,
):
    m_offsets = tl.arange(0, BLOCK_M)[:, None]
    packed_k_offsets = tl.arange(0, PACKED_K)[None, :]
    tensor_tile = tl.load(
        tensor + m_offsets * stride_tensor_m + packed_k_offsets * stride_tensor_k
    )

    scale_k_offsets = tl.arange(0, SCALE_K)[None, :]
    scale_tile = tl.load(
        scale + m_offsets * stride_scale_m + scale_k_offsets * stride_scale_k
    )

    out_tile = _mxfp4_tile_upcast(tensor_tile, scale_tile, dst_dtype)
    k_offsets = tl.arange(0, PACKED_K * 2)[None, :]
    tl.store(out + m_offsets * stride_out_m + k_offsets * stride_out_k, out_tile)


@pytest.mark.parametrize("dst_dtype", MXFP4_TILE_UPCAST_CASES)
def test_triton_viz_sanitizer_mxfp4_tile_upcast(device, sanitizer_only, dst_dtype):
    global _mxfp4_tile_upcast
    mxfp_mod = pytest.importorskip("triton_kernels.numerics_details.mxfp")
    upcast_mod = pytest.importorskip(
        "triton_kernels.numerics_details.mxfp_details._upcast_from_mxfp"
    )
    _mxfp4_tile_upcast = upcast_mod.upcast_mxfp4_tile

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dst_dtype)
    rows, k = 64, 128
    tensor = torch.empty((rows, k // 2), dtype=torch.uint8, device=device)
    scale = torch.empty(
        (rows, k // mxfp_mod.MXFP_BLOCK_SIZE.value),
        dtype=torch.uint8,
        device=device,
    )
    out = torch.empty((rows, k), dtype=torch_dtype, device=device)
    tl_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[torch_dtype]

    kernel = _triton_viz_mxfp4_tile_upcast_kernel
    sanitizer = _new_sanitizer()
    traced = pytest.importorskip("triton_viz").trace(client=sanitizer)(kernel)
    traced[(1,)](
        torch.empty_like(out),
        tensor,
        scale,
        *out.stride(),
        *tensor.stride(),
        *scale.stride(),
        rows,
        tensor.shape[-1],
        scale.shape[-1],
        tl_dtype,
    )
    assert len(sanitizer.records) == 0
