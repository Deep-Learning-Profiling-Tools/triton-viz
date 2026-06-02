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
