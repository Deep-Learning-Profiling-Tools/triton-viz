import itertools

import pytest
import torch
import triton
import triton.language as tl


def assert_equal(*args, **kwargs):
    triton_kernels_testing = pytest.importorskip("triton_kernels.testing")
    return triton_kernels_testing.assert_equal(*args, **kwargs)


COMPACTION_CASES = [
    (8192, 64, 4, 0.5),
    (8192, 64, 4, 1.0),
    (131, 128, 16, 0.6),
    (496, 128, 16, 0.0),
]

RAGGED_METADATA_N_SLICES = [1, 7, 33, 911, 1025]
REMAP_RAGGED_METADATA_N_SLICES = [9, 32, 911, 1025]
BITMATRIX_METADATA_CASES = [
    (n_rows, n_cols, k)
    for n_rows, n_cols, k in itertools.product([7, 256, 17111], [13, 32, 128, 811], [1, 4, 8])
    if k <= n_cols
]
TOPK_FORWARD_CASES = [
    (n_rows, n_cols, 8, apply_softmax, dtype, use_provided_indx)
    for n_rows, n_cols, apply_softmax, dtype, use_provided_indx in itertools.product(
        [1, 7, 256, 300],
        [13, 32, 128, 200],
        [True, False],
        ["float16", "bfloat16", "float32"],
        [False, True],
    )
]
TOPK_BACKWARD_CASES = [
    (n_rows, n_cols, 8, apply_softmax, dtype, n_rows_arg)
    for n_rows, n_cols, apply_softmax, dtype, n_rows_arg in itertools.product(
        [1, 7, 256, 300],
        [13, 32, 128, 200],
        [True, False],
        ["float16", "bfloat16", "float32"],
        ["int", "tensor"],
    )
]
SWIGLU_CASES = [(1311, 4352, 1e-2), (1311, 4352, 10)]
REDUCE_SHAPES = [
    (311, 384, 384, None),
    (384, 311, 384, None),
    (384, 384, 311, None),
    (512, 512, 512, None),
    (512, 512, 512, "plus_ten"),
    (4, 4, 4, None),
    (3, 15, 25, None),
    (5, 9999, 2345, None),
    (15, 345, 789, None),
]
REDUCE_CASES = [
    ((b, m, n), dim, dtype, mask_mode, scale_mode, postprocess_fn)
    for (b, m, n, postprocess_fn), dtype, mask_mode, dim, scale_mode in itertools.product(
        REDUCE_SHAPES,
        [torch.float16, torch.float32],
        ["none", "full", "broadcast_b", "broadcast_m", "broadcast_n"],
        [0, 1, 2],
        ["none", "full", "broadcast_n"],
    )
]
MATMUL_CASES = [
    (mode, dtype, do_bias, do_gather, do_scatter, split_k, transpose_a, transpose_b, transpose_c)
    for mode, dtype, do_bias, do_gather, do_scatter, split_k, transpose_a, transpose_b, transpose_c in itertools.product(
        ["plain", "batched", "ragged_m"],
        [torch.float16, torch.bfloat16, torch.float32],
        [False, True],
        [False, True],
        [False, True],
        [1, 3],
        [False, True],
        [False, True],
        [False, True],
    )
    if not (mode == "batched" and (do_gather or do_scatter))
    if not (do_gather and do_scatter)
    if not (mode == "ragged_m" and (do_gather or do_scatter))
    if not (dtype == torch.float32 and (split_k != 1 or mode == "batched"))
]


def _require_cuda(device):
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("triton-viz sanitizer cases require CUDA")


def _new_sanitizer():
    pytest.importorskip("triton_viz")
    sanitizer_mod = pytest.importorskip("triton_viz.clients.sanitizer.sanitizer")
    return sanitizer_mod.SymbolicSanitizer(abort_on_error=True)


def _trace_kernel(monkeypatch, module, name, sanitizer):
    triton_viz = pytest.importorskip("triton_viz")
    traced = triton_viz.trace(client=sanitizer)(getattr(module, name))
    monkeypatch.setattr(module, name, traced)


def _trace_specialization_get(monkeypatch, specialization_module, sanitizer, kernel_names):
    orig_get = specialization_module.get
    traced_modules = set()

    def wrapped_get(**kwargs):
        module = orig_get(**kwargs)
        if id(module) not in traced_modules:
            for name in kernel_names:
                _trace_kernel(monkeypatch, module, name, sanitizer)
            traced_modules.add(id(module))
        return module

    monkeypatch.setattr(specialization_module, "get", wrapped_get)


def _init_reduce_mask(mask_mode, b, m, n, device):
    if mask_mode == "none":
        return None
    if mask_mode == "full":
        return (torch.rand((b, m, n), device=device) > 0.3).to(torch.int8)
    if mask_mode == "broadcast_b":
        return (torch.rand((1, m, n), device=device) > 0.3).to(torch.int8).expand(b, m, n)
    if mask_mode == "broadcast_m":
        return (torch.rand((b, 1, n), device=device) > 0.3).to(torch.int8).expand(b, m, n)
    if mask_mode == "broadcast_n":
        return (torch.rand((b, m, 1), device=device) > 0.3).to(torch.int8).expand(b, m, n)
    raise AssertionError(f"unknown mask mode: {mask_mode}")


@pytest.mark.parametrize(
    ("n_tokens", "n_cols", "k", "keep_prob"),
    COMPACTION_CASES,
)
@pytest.mark.parametrize("device", ["cuda"], indirect=True)
def test_triton_viz_sanitizer_masked_compaction(monkeypatch, device, n_tokens, n_cols, k, keep_prob):
    _require_cuda(device)

    compaction_mod = pytest.importorskip("triton_kernels.compaction")

    torch.manual_seed(0)
    yi = torch.rand((n_tokens, n_cols - 1), device=device).argsort(dim=-1)[:, :k].to(torch.int32)
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


@pytest.mark.parametrize("n_slices", RAGGED_METADATA_N_SLICES)
def test_triton_viz_sanitizer_make_ragged_tensor_metadata(monkeypatch, device, n_slices):
    _require_cuda(device)

    from triton_kernels.tensor_details import ragged_tensor as ragged_mod

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_memset", sanitizer)
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_compute", sanitizer)

    torch.manual_seed(0)
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(0, max_slice_size, (n_slices,), dtype=torch.int32, device=device)
    slice_sizes[torch.randint(0, n_slices, (1,), device=device)] = 0

    meta = ragged_mod.make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref = ragged_mod.make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows)

    assert_equal(meta.slice_sizes, ref.slice_sizes)
    assert_equal(meta.slice_offs, ref.slice_offs)
    assert_equal(meta.block_offs_data, ref.block_offs_data)
    assert_equal(meta.block_schedule_data, ref.block_schedule_data)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize("n_slices", REMAP_RAGGED_METADATA_N_SLICES)
def test_triton_viz_sanitizer_remap_ragged_tensor_metadata(monkeypatch, device, n_slices):
    _require_cuda(device)

    from triton_kernels.tensor_details import ragged_tensor as ragged_mod

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_memset", sanitizer)
    _trace_kernel(monkeypatch, ragged_mod, "_ragged_tensor_metadata_compute", sanitizer)
    _trace_kernel(monkeypatch, ragged_mod, "_remap_ragged_tensor_metadata", sanitizer)

    torch.manual_seed(0)
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(0, max_slice_size, (n_slices,), dtype=torch.int32, device=device)
    slice_sizes[torch.randint(0, n_slices, (1,), device=device)] = 0
    slice_map = torch.randperm(n_slices, device=device, dtype=torch.int32)
    slice_map[torch.randint(0, n_slices, (min(5, n_slices),), device=device)] = -1

    tri_metadata = ragged_mod.make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref_metadata = ragged_mod.make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows)
    tri_metadata = ragged_mod.remap_ragged_tensor_metadata(tri_metadata, slice_map)
    ref_metadata = ragged_mod.remap_ragged_tensor_metadata_torch(ref_metadata, slice_map)

    assert_equal(tri_metadata.slice_sizes, ref_metadata.slice_sizes)
    assert_equal(tri_metadata.slice_offs, ref_metadata.slice_offs)
    assert_equal(tri_metadata.block_offs_data, ref_metadata.block_offs_data)
    assert_equal(tri_metadata.block_schedule_data, ref_metadata.block_schedule_data)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "k"),
    BITMATRIX_METADATA_CASES,
)
def test_triton_viz_sanitizer_bitmatrix_metadata(monkeypatch, device, n_rows, n_cols, k):
    _require_cuda(device)

    from triton_kernels.tensor import wrap_torch_tensor
    from triton_kernels.tensor_details import bitmatrix as bitmatrix_mod
    from triton_kernels.tensor_details.bitmatrix_details import sum_bitmatrix_rows as sum_mod
    from triton_kernels.tensor_details.dtype import BIT

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, sum_mod, "_sum_bitmatrix_rows", sanitizer)
    _trace_kernel(monkeypatch, bitmatrix_mod, "_bitmatrix_metadata_compute_stage1", sanitizer)
    _trace_kernel(monkeypatch, bitmatrix_mod, "_bitmatrix_metadata_compute_stage2", sanitizer)

    torch.manual_seed(0)
    if k > n_cols:
        pytest.skip("k must be <= n_cols")
    indx = torch.rand(n_rows, n_cols, device=device).argsort(dim=1).int()[:, :k]
    indx = torch.sort(indx, dim=1)[0]
    rows = torch.arange(n_rows, device=device).unsqueeze(1).expand_as(indx)
    bitmask_data = torch.zeros((n_rows, (n_cols + 31) // 32), dtype=torch.int32, device=device)
    bitmask_data.index_put_((rows, indx // 32), 1 << (indx % 32), accumulate=True)
    bitmask = wrap_torch_tensor(bitmask_data.view(torch.uint32), dtype=BIT, shape=(n_rows, n_cols))

    metadata_tri = bitmatrix_mod.make_bitmatrix_metadata(indx, bitmask)
    metadata_ref = bitmatrix_mod.make_bitmatrix_metadata_torch(indx, bitmask)

    assert_equal(metadata_tri.col_sum, metadata_ref.col_sum)
    assert_equal(metadata_tri.row_sorted_indx, metadata_ref.row_sorted_indx)
    assert_equal(metadata_tri.col_sorted_indx, metadata_ref.col_sorted_indx)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "k", "apply_softmax", "dtype", "use_provided_indx"),
    TOPK_FORWARD_CASES,
)
def test_triton_viz_sanitizer_topk_forward(
    monkeypatch, device, n_rows, n_cols, k, apply_softmax, dtype, use_provided_indx
):
    _require_cuda(device)

    import triton_kernels.topk as topk_mod

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, topk_mod, "_topk_forward", sanitizer)

    torch.manual_seed(0)
    x = torch.randn((n_rows, n_cols), dtype=getattr(torch, dtype), device=device)
    y_indx = None
    if use_provided_indx:
        y_indx = torch.sort(torch.rand(n_rows, n_cols, device=device).argsort(dim=1).int()[:, :k], dim=1)[0]

    sparse_tri = topk_mod.topk(x, k, apply_softmax=apply_softmax, y_indx=y_indx)
    sparse_ref = topk_mod.topk_torch(x, k, apply_softmax=apply_softmax)
    if use_provided_indx:
        selected = torch.gather(x, 1, y_indx.long())
        if apply_softmax:
            selected = torch.softmax(selected.float(), dim=1).to(x.dtype)
        assert torch.allclose(sparse_tri.vals.float(), selected.float(), atol=1e-3, rtol=1e-3)
        assert_equal(sparse_tri.indx, y_indx)
    else:
        assert torch.allclose(sparse_tri.vals.float(), sparse_ref.vals.float(), atol=1e-3, rtol=1e-3)
        assert_equal(sparse_tri.indx, sparse_ref.indx)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "k", "apply_softmax", "dtype", "n_rows_arg"),
    TOPK_BACKWARD_CASES,
)
def test_triton_viz_sanitizer_topk_backward(
    monkeypatch, device, n_rows, n_cols, k, apply_softmax, dtype, n_rows_arg
):
    _require_cuda(device)

    import triton_kernels.topk as topk_mod

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, topk_mod, "_topk_backward", sanitizer)

    torch.manual_seed(0)
    x = torch.randn((n_rows, n_cols), device=device, dtype=getattr(torch, dtype))
    y_indx = torch.sort(torch.rand(n_rows, n_cols, device=device).argsort(dim=1).int()[:, :k], dim=1)[0].to(
        torch.int16
    )
    dy_vals = torch.randn((n_rows, k), device=device, dtype=getattr(torch, dtype))
    active = max(1, n_rows // 2)
    active_rows = active if n_rows_arg == "int" else torch.tensor(active, device=device, dtype=torch.int32)

    dx = topk_mod.topk_backward(x, y_indx, dy_vals, k=k, n_rows=active_rows, apply_softmax=apply_softmax)
    torch.cuda.synchronize()

    assert dx.shape == x.shape
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(("m", "n", "limit"), SWIGLU_CASES)
def test_triton_viz_sanitizer_swiglu(monkeypatch, device, m, n, limit):
    _require_cuda(device)

    import triton_kernels.swiglu as swiglu_mod
    from triton_kernels.swiglu import PrecisionConfig, swiglu_torch

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, swiglu_mod, "_swiglu", sanitizer)

    torch.manual_seed(2)
    x = torch.randn((m, n), device=device, dtype=torch.bfloat16)
    precision_config = PrecisionConfig(limit=limit)

    tri_y = swiglu_mod.swiglu(x, 0.5, precision_config)
    ref_y = swiglu_torch(x, 0.5, precision_config)

    assert torch.allclose(tri_y.float(), ref_y.float(), atol=1e-3, rtol=1e-3)
    assert len(sanitizer.records) == 0


@triton.jit
def _plus_a_reduce(x, a):
    y = x + a
    return tl.sum(y.reshape([x.shape[0], x.shape[1] // 2, 2]), axis=2)


@pytest.mark.parametrize(
    ("shape", "dim", "dtype", "mask_mode", "scale_mode", "postprocess"),
    REDUCE_CASES,
)
def test_triton_viz_sanitizer_reduce(monkeypatch, device, shape, dim, dtype, mask_mode, scale_mode, postprocess):
    _require_cuda(device)

    import triton_kernels.reduce as reduce_mod

    sanitizer = _new_sanitizer()
    _trace_kernel(monkeypatch, reduce_mod, "_create_row_idxs", sanitizer)
    _trace_specialization_get(monkeypatch, reduce_mod.forward_specializations, sanitizer, ("_reduce_forward",))
    _trace_specialization_get(monkeypatch, reduce_mod.backward_specializations, sanitizer, ("_reduce_backward",))

    torch.manual_seed(0)
    b, m, n = shape
    x_tri = torch.randn(shape, device=device, dtype=dtype, requires_grad=not postprocess)
    x_ref = x_tri.detach().clone().requires_grad_(not postprocess)
    mask = _init_reduce_mask(mask_mode, b, m, n, device)
    scale = None
    if scale_mode == "full":
        scale = torch.rand(shape, device=device, dtype=torch.float32)
    elif scale_mode == "broadcast_n":
        scale = torch.rand((b, m, 1), device=device, dtype=torch.float32).expand(shape)
    elif scale_mode != "none":
        raise AssertionError(f"unknown scale mode: {scale_mode}")

    if postprocess == "plus_ten":
        postprocess_tri = reduce_mod.PostprocessFn(
            specs=reduce_mod.FnSpecs("plus_a", _plus_a_reduce, ("a",), reduction_n=2),
            fn_args=(10,),
        )
        postprocess_ref = lambda x: (x + 10).reshape([x.shape[0], x.shape[1] // 2, 2]).sum(dim=2)
    else:
        postprocess_tri = postprocess_ref = None

    y_tri, _ = reduce_mod.reduce(x_tri, dim=dim, mask=mask, scale=scale, postprocess_fn1=postprocess_tri)
    y_ref, _ = reduce_mod.reduce_torch(x_ref, dim=dim, mask=mask, scale=scale, postprocess_fn1=postprocess_ref)
    assert torch.allclose(y_tri.float(), y_ref.float(), atol=1e-3, rtol=1e-3)

    if not postprocess:
        dy = torch.randn_like(y_tri)
        y_tri.backward(dy)
        y_ref.backward(dy)
        assert torch.allclose(x_tri.grad.float(), x_ref.grad.float(), atol=1e-3, rtol=1e-3)
    assert len(sanitizer.records) == 0


@pytest.mark.parametrize(
    ("mode", "dtype", "do_bias", "do_gather", "do_scatter", "split_k", "transpose_a", "transpose_b", "transpose_c"),
    MATMUL_CASES,
)
def test_triton_viz_sanitizer_matmul(
    monkeypatch,
    device,
    mode,
    dtype,
    do_bias,
    do_gather,
    do_scatter,
    split_k,
    transpose_a,
    transpose_b,
    transpose_c,
):
    _require_cuda(device)

    import triton_kernels.matmul as matmul_mod
    from triton_kernels.tensor import make_ragged_tensor_metadata

    sanitizer = _new_sanitizer()
    _trace_specialization_get(monkeypatch, matmul_mod.specializations, sanitizer, ("_matmul", "_p_matmul"))

    torch.manual_seed(0)
    m, n, k = 8, 12, 16

    def make_matrix(shape, transposed):
        if not transposed:
            return torch.randn(shape, device=device, dtype=dtype)
        return torch.randn(shape[:-2] + (shape[-1], shape[-2]), device=device, dtype=dtype).transpose(-1, -2)

    if mode == "batched":
        a = make_matrix((2, m, k), transpose_a)
        b = make_matrix((2, k, n), transpose_b)
    else:
        a = make_matrix((m, k), transpose_a)
        b = make_matrix((k, n), transpose_b)
    bias = torch.randn((n,), device=device, dtype=torch.float32) if do_bias else None
    gather_indx = torch.tensor([7, 0, 2, 4, 6, 1, 3, 5], device=device, dtype=torch.int32) if do_gather else None
    scatter_indx = torch.tensor([7, 0, 2, 4, 6, 1, 3, 5], device=device, dtype=torch.int32) if do_scatter else None
    a_ragged_metadata = None
    if mode == "ragged_m":
        slice_sizes = torch.tensor([3, 0, 5], device=device, dtype=torch.int32)
        a_ragged_metadata = make_ragged_tensor_metadata(slice_sizes, int(slice_sizes.sum().item()))
    c_rows = scatter_indx.numel() if scatter_indx is not None else m
    c_shape = (2, c_rows, n) if mode == "batched" else (c_rows, n)
    if transpose_c:
        c = torch.empty(c_shape[:-2] + (n, c_rows), device=device, dtype=dtype).transpose(-1, -2)
    else:
        c = torch.empty(c_shape, device=device, dtype=dtype)

    with matmul_mod.scoped_opt_flags_constraints({"split_k": split_k}):
        tri = matmul_mod.matmul(
            a,
            b,
            bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            c=c,
        )
    ref = matmul_mod.matmul_torch(
        a,
        b,
        bias,
        a_ragged_metadata=a_ragged_metadata,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
    )

    assert torch.allclose(tri.float(), ref.float(), atol=5e-2, rtol=5e-2)
    assert len(sanitizer.records) == 0
