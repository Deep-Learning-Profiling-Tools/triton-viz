import pytest
import torch
import triton
import triton.language as tl


def assert_equal(*args, **kwargs):
    triton_kernels_testing = pytest.importorskip("triton_kernels.testing")
    return triton_kernels_testing.assert_equal(*args, **kwargs)


def assert_close(*args, **kwargs):
    triton_kernels_testing = pytest.importorskip("triton_kernels.testing")
    return triton_kernels_testing.assert_close(*args, **kwargs)


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
REDUCE_CASES = [
    ((3, 15, 25), 0, torch.float16),
    ((3, 15, 25), 1, torch.float16),
    ((3, 15, 25), 2, torch.float16),
    ((4, 4, 4), 0, torch.float32),
    ((4, 4, 4), 1, torch.float32),
    ((4, 4, 4), 2, torch.float32),
    ((15, 345, 789), 2, torch.float16),
]
_mxfp4_tile_upcast = None


def _require_cuda(device):
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("triton-viz sanitizer cases require CUDA")


def _new_sanitizer():
    pytest.importorskip("triton_viz")
    sanitizer_mod = pytest.importorskip("triton_viz.clients.sanitizer.sanitizer")
    return sanitizer_mod.SymbolicSanitizer(abort_on_error=True)


def _trace_kernel(monkeypatch, module, name, sanitizer, *, specialized=False):
    triton_viz = pytest.importorskip("triton_viz")
    traced = triton_viz.trace(client=sanitizer, specialized=specialized)(
        getattr(module, name)
    )
    monkeypatch.setattr(module, name, traced)


def _trace_specialization_get(
    monkeypatch,
    specialization_module,
    sanitizer,
    kernel_names,
    *,
    specialized_global_names=(),
):
    orig_get = specialization_module.get
    traced_modules = set()

    def wrapped_get(**kwargs):
        module = orig_get(**kwargs)
        if id(module) not in traced_modules:
            for name in kernel_names:
                kernel = getattr(module, name)
                # SpecializationModule clones launch kernels into dynamic modules.
                # Patch captured nested helpers in that generated global scope.
                for global_name in specialized_global_names:
                    globals_dict = kernel.fn.__globals__
                    if global_name in globals_dict:
                        triton_viz = pytest.importorskip("triton_viz")
                        traced = triton_viz.trace(
                            client=sanitizer,
                            specialized=True,
                        )(globals_dict[global_name])
                        monkeypatch.setitem(globals_dict, global_name, traced)
                _trace_kernel(monkeypatch, module, name, sanitizer)
            traced_modules.add(id(module))
        return module

    monkeypatch.setattr(specialization_module, "get", wrapped_get)


@pytest.mark.parametrize(
    ("n_tokens", "n_cols", "k", "keep_prob"),
    COMPACTION_CASES,
)
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_masked_compaction(
    monkeypatch, device, n_tokens, n_cols, k, keep_prob
):
    _require_cuda(device)

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

    yv_ref, yi_ref = compaction_mod.compaction_torch(yv, yi, bitmask)
    yv_tri, yi_tri = compaction_mod.compaction(yv, yi, bitmask)

    assert_equal(yi_tri, yi_ref)
    assert_equal(yv_tri, yv_ref)

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
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_make_ragged_tensor_metadata(
    monkeypatch, device, n_slices
):
    _require_cuda(device)

    ragged_mod = pytest.importorskip("triton_kernels.tensor_details.ragged_tensor")

    torch.manual_seed(0)
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(
        0, max_slice_size, (n_slices,), dtype=torch.int32, device=device
    )
    slice_sizes[torch.randint(0, n_slices, (1,), device=device)] = 0

    meta = ragged_mod.make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref = ragged_mod.make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows)

    assert_equal(meta.slice_sizes, ref.slice_sizes)
    assert_equal(meta.slice_offs, ref.slice_offs)
    assert_equal(meta.block_offs_data, ref.block_offs_data)
    assert_equal(meta.block_schedule_data, ref.block_schedule_data)

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
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_remap_ragged_tensor_metadata(
    monkeypatch, device, n_slices
):
    _require_cuda(device)

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

    tri_metadata = ragged_mod.make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref_metadata = ragged_mod.make_ragged_tensor_metadata_torch(
        slice_sizes, n_total_rows
    )
    tri_metadata = ragged_mod.remap_ragged_tensor_metadata(tri_metadata, slice_map)
    ref_metadata = ragged_mod.remap_ragged_tensor_metadata_torch(
        ref_metadata, slice_map
    )

    assert_equal(tri_metadata.slice_sizes, ref_metadata.slice_sizes)
    assert_equal(tri_metadata.slice_offs, ref_metadata.slice_offs)
    assert_equal(tri_metadata.block_offs_data, ref_metadata.block_offs_data)
    assert_equal(tri_metadata.block_schedule_data, ref_metadata.block_schedule_data)

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
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_topk_forward(
    monkeypatch, device, n_rows, n_cols, k, apply_softmax, dtype
):
    _require_cuda(device)

    topk_mod = pytest.importorskip("triton_kernels.topk")

    torch.manual_seed(0)
    x = torch.randn((n_rows, n_cols), dtype=getattr(torch, dtype), device=device)

    sparse_tri = topk_mod.topk(x, k, apply_softmax=apply_softmax)
    sparse_ref = topk_mod.topk_torch(x, k, apply_softmax=apply_softmax)
    assert torch.allclose(
        sparse_tri.vals.float(),
        sparse_ref.vals.float(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert_equal(sparse_tri.indx, sparse_ref.indx)
    assert_equal(sparse_tri.mask.storage.data, sparse_ref.mask.storage.data)
    assert (
        sparse_tri.mask.storage.data.stride() == sparse_ref.mask.storage.data.stride()
    )
    assert sparse_tri.mask.storage.data.shape == sparse_ref.mask.storage.data.shape

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, topk_mod, "_topk_forward", sanitizer)
    topk_mod.topk_forward(x, k, apply_softmax=apply_softmax)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "k", "apply_softmax", "dtype"),
    TOPK_BACKWARD_CASES,
)
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_topk_backward(
    monkeypatch, device, n_rows, n_cols, k, apply_softmax, dtype
):
    _require_cuda(device)

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

    dx = topk_mod.topk_backward(
        x, y_indx, dy_vals, k=k, n_rows=active_rows, apply_softmax=apply_softmax
    )
    torch.cuda.synchronize()
    assert dx.shape == x.shape

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
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_swiglu(monkeypatch, device, m, n, limit):
    _require_cuda(device)

    swiglu_mod = pytest.importorskip("triton_kernels.swiglu")

    torch.manual_seed(2)
    x = torch.randn((m, n), device=device, dtype=torch.bfloat16)
    precision_config = swiglu_mod.PrecisionConfig(limit=limit)

    tri_y = swiglu_mod.swiglu(x, 0.5, precision_config)
    ref_y = swiglu_mod.swiglu_torch(x, 0.5, precision_config)
    assert_close(tri_y, ref_y)

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
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_mxfp4_tile_upcast(device, dst_dtype):
    _require_cuda(device)

    global _mxfp4_tile_upcast
    mxfp_mod = pytest.importorskip("triton_kernels.numerics_details.mxfp")
    upcast_mod = pytest.importorskip(
        "triton_kernels.numerics_details.mxfp_details._upcast_from_mxfp"
    )
    _mxfp4_tile_upcast = upcast_mod.upcast_mxfp4_tile

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dst_dtype)
    rows, k = 64, 128
    block_scales = torch.tensor(
        [0.125, 1.0, 8.0, 64.0], dtype=torch_dtype, device=device
    )
    block_scales = block_scales.repeat_interleave(mxfp_mod.MXFP_BLOCK_SIZE.value)
    x = torch.randn((rows, k), dtype=torch_dtype, device=device) * block_scales
    tensor, scale = mxfp_mod.downcast_to_mxfp(x, torch.uint8, axis=-1)
    ref = mxfp_mod.upcast_from_mxfp(tensor, scale, torch_dtype, axis=-1)
    out = torch.empty_like(ref)
    tl_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[torch_dtype]

    kernel = _triton_viz_mxfp4_tile_upcast_kernel
    kernel[(1,)](
        out,
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
    assert_equal(ref, out)

    sanitizer = _new_sanitizer()
    traced = pytest.importorskip("triton_viz").trace(client=sanitizer)(kernel)
    traced[(1,)](
        torch.empty_like(ref),
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


@pytest.mark.parametrize(("shape", "dim", "dtype"), REDUCE_CASES)
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_reduce(monkeypatch, device, shape, dim, dtype):
    _require_cuda(device)

    reduce_mod = pytest.importorskip("triton_kernels.reduce")

    torch.manual_seed(0)
    x_tri = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    x_ref = x_tri.detach().clone().requires_grad_(True)

    y_tri, _ = reduce_mod.reduce(x_tri, dim=dim)
    y_ref, _ = reduce_mod.reduce_torch(
        x_ref,
        dim=dim,
        x_flex=None,
        y_flex=None,
    )
    assert torch.allclose(y_tri.float(), y_ref.float(), atol=1e-3, rtol=1e-3)

    dy = torch.randn_like(y_tri)
    y_tri.backward(dy)
    y_ref.backward(dy)
    assert torch.allclose(x_tri.grad.float(), x_ref.grad.float(), atol=1e-3, rtol=1e-3)

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, reduce_mod, "_create_row_idxs", sanitizer)
    _trace_specialization_get(
        monkeypatch,
        reduce_mod.forward_specializations,
        sanitizer,
        ("_reduce_forward",),
        specialized_global_names=("_reduce_forward_inner",),
    )
    _trace_specialization_get(
        monkeypatch,
        reduce_mod.backward_specializations,
        sanitizer,
        ("_reduce_backward",),
    )
    x_san = x_tri.detach().clone().requires_grad_(True)
    y_san, _ = reduce_mod.reduce(x_san, dim=dim)
    y_san.backward(torch.randn_like(y_san))
    assert len(sanitizer.records) == 0
