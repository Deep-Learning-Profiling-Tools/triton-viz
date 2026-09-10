import pytest
import torch
import triton
import triton.language as tl

from triton_viz.performance.triton_observe import observe
from triton_viz.performance.gpu_controls import cases, prepare, check_output
from triton_viz.performance.gpu import expand


@triton.jit
def _add(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offsets, offsets < n, other=0)
    tl.store(y + offsets, value + 1, offsets < n)


def test_observe_masked_launch_without_compilation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Observe must not compile the target")

    monkeypatch.setattr(triton.compiler, "compile", forbidden)
    monkeypatch.setattr(triton.runtime.jit.JITFunction, "warmup", forbidden)
    x = torch.arange(13, dtype=torch.float32)
    y = torch.empty_like(x)
    trace = observe(_add, (2,), x, y, 13, 8)
    torch.testing.assert_close(y, x + 1)
    assert trace["program_count"] == 2
    memory = [e for e in trace["events"] if e["op"] in {"load", "store"}]
    assert sum(e["bytes"] for e in memory) == 13 * 4 * 2
    assert any(e["masked"] for e in memory)
    assert any(e.get("primitive") == "add" for e in trace["events"])


@pytest.mark.parametrize(
    "case", cases("control")[:9] + cases("holdout")[:5], ids=lambda c: c["id"]
)
def test_pilot_source_coverage(case):
    import triton_viz

    kernel, grid, args, output = prepare(case, "cpu")
    source = observe(kernel, grid, *args)
    check_output(case, output)
    work = expand(source, sm_count=48)
    assert not work["ood_reasons"]
    assert work["features"]["global_sectors"] > 0
    memory = [
        e
        for e in source["events"]
        if e["op"] in {"load", "raw_load", "store", "raw_store"}
    ]
    expected_bytes = case["n"] * 8
    if case["kind"] == "reduction" and case["mode"] < 2:
        expected_bytes = case["n"] * 4 + case["n"] // case["block"] * 4
    assert sum(e["bytes"] for e in memory) == expected_bytes
    if case["kind"] == "dot":
        assert work["features"]["tensor_flops"] == (
            2 * case["n"] * case["block"] * case["repeat"]
        )
    if case["kind"] == "reduction":
        reductions = work["rules"]["gpu.reduction.tree"]
        per_program = 2 if case["mode"] == 2 else 1
        assert reductions == case["n"] // case["block"] * per_program
    triton_viz.clear()
