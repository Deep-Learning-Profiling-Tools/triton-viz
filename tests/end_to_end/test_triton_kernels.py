import pytest
import torch


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


TOPK_FORWARD_CASES = [
    (7, 13, 8, True, "float16"),
]
TOPK_BACKWARD_CASES = [
    (7, 13, 8, True, "float16"),
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
