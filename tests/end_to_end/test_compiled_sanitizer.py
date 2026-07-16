"""End-to-end tests for Sanitizer(compile=True).

The analyzer-level path (parse + Z3) runs without a GPU; the trace-level
tests need a CUDA driver for the warmup compilation and are skipped without
one.
"""

from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.compiled.client import CompiledSanitizer


def _runtime_int(value: int):
    """An arg_cvt that is NOT identical to the int arg, so arg_callback treats
    it as a runtime scalar (substituted) rather than a constexpr."""
    return object()


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="warmup compilation needs a CUDA driver"
)


class _FakeKernel:
    """Stand-in for a triton CompiledKernel: only .asm is read."""

    def __init__(self, asm):
        self.asm = asm


def test_factory_dispatch():
    eager = Sanitizer()
    compiled = Sanitizer(compile=True)
    assert type(eager).__name__ == "SymbolicSanitizer"
    assert isinstance(compiled, CompiledSanitizer)
    # compile kwarg must not leak into __init__ and must honor abort flag.
    assert Sanitizer(compile=True, abort_on_error=False).abort_on_error is False


def test_compile_false_dispatches_to_eager():
    """Sanitizer(compile=False) must dispatch to the eager SymbolicSanitizer.
    Regression: the factory returns a Sanitizer SUBCLASS, so Python re-invokes
    its __init__ with the ORIGINAL kwargs (including compile=False). If that
    __init__ does not tolerate the stray `compile`, this legitimate-looking
    spelling raises TypeError instead of constructing an eager sanitizer."""
    det = Sanitizer(compile=False, abort_on_error=False)
    assert type(det).__name__ == "SymbolicSanitizer"
    assert det.abort_on_error is False
    # The default spelling (no compile kwarg) must keep working too.
    assert type(Sanitizer(abort_on_error=True)).__name__ == "SymbolicSanitizer"


def test_stale_ttir_does_not_leak_across_launches():
    """A captured TTIR is per-launch input. If a later launch's warmup yields
    no TTIR, finalize must report unsupported — never re-analyze the previous
    kernel's graph against the current launch's metadata."""
    add_ttir = (
        Path(__file__).resolve().parents[1] / "golden" / "ttgir" / "add_sm80.ttir"
    ).read_text()

    det = CompiledSanitizer(abort_on_error=False)

    # Launch 1: warmup captures a real TTIR, finalize analyzes it.
    det.pre_warmup_callback(None)
    det.post_warmup_callback(None, _FakeKernel({"ttir": add_ttir}))
    det.grid_callback((1, 1, 1))
    det.arg_callback("x_ptr", torch.empty(8), None)
    det.arg_callback("y_ptr", torch.empty(8), None)
    det.arg_callback("out_ptr", torch.empty(8), None)
    det.arg_callback("n_elements", 8, _runtime_int(8))
    det.finalize()
    assert det.last_status == "ok"

    # Launch 2: warmup produces NO TTIR. The stale add graph must not be used.
    det.pre_warmup_callback(None)
    det.post_warmup_callback(None, _FakeKernel({}))  # no "ttir" key
    det.grid_callback((4, 1, 1))
    det.finalize()
    assert det.last_status == "unsupported"
    assert det.records == []


def test_unsupported_is_reported_only_no_auto_eager_fallback():
    """v1 contract: an unsupported construct yields last_status='unsupported'
    with empty records — it does NOT silently fall back to interpreted
    (eager) checking. This pins the documented semantics so the PR text, the
    API, and the implementation can't drift."""
    # A gather kernel's TTIR has a data-dependent address -> unsupported.
    gather_ttir = (
        Path(__file__).resolve().parents[1] / "golden" / "ttgir" / "gather_sm80.ttir"
    ).read_text()

    det = CompiledSanitizer(abort_on_error=False)
    det.pre_warmup_callback(None)
    det.post_warmup_callback(None, _FakeKernel({"ttir": gather_ttir}))
    det.grid_callback((1, 1, 1))
    det.finalize()
    assert det.last_status == "unsupported"
    assert det.records == []  # reported, not interpreted


def test_disable_flag_overrides_compile_mode():
    """ENABLE_SANITIZER=0 must disable compiled mode too: Sanitizer(compile=
    True) collapses to NullSanitizer (so trace() leaves the kernel
    untraced), instead of warming up and aborting on OOB."""
    from triton_viz.core.config import config as cfg

    saved = cfg.enable_sanitizer
    try:
        cfg.enable_sanitizer = False
        off = Sanitizer(compile=True, abort_on_error=False)
        assert type(off).__name__ == "NullSanitizer"
        cfg.enable_sanitizer = True
        on = Sanitizer(compile=True, abort_on_error=False)
        assert isinstance(on, CompiledSanitizer)
    finally:
        cfg.enable_sanitizer = saved


@requires_cuda
def test_correct_kernel_is_proven_in_bounds():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 4096
    x, y, out = torch.randn(n), torch.randn(n), torch.empty(n)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    assert det.last_status == "ok"
    assert det.records == []


@requires_cuda
def test_unmasked_tail_oob_is_reported():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)  # missing mask -> OOB on a ragged tail
        tl.store(out_ptr + offs, x)

    n = 3000  # not a multiple of BLOCK -> grid=3 covers [0,3072)
    x, out = torch.randn(n), torch.empty(n)
    add_nomask[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    assert det.last_status == "ok"
    assert len(det.records) >= 1
    kinds = {r.op_type.__name__ for r in det.records}
    assert "Load" in kinds
    # Report carries the offending source line and the kernel function name.
    tb = det.records[0].user_code_tracebacks[0]
    assert tb.func_name == "add_nomask"
    assert "tl.load" in tb.line_of_code


@requires_cuda
def test_abort_on_error_raises_system_exit():
    det = Sanitizer(compile=True, abort_on_error=True)

    @triton_viz.trace(det)
    @triton.jit
    def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, x)

    n = 3000
    x, out = torch.randn(n), torch.empty(n)
    with pytest.raises(SystemExit):
        add_nomask[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)


@requires_cuda
def test_indirect_gather_is_unsupported_not_silent():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def gather(idx_ptr, src_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(idx_ptr + offs, mask=mask)
        vals = tl.load(src_ptr + idx, mask=mask)  # data-dependent address
        tl.store(out_ptr + offs, vals, mask=mask)

    n = 1024
    idx = torch.zeros(n, dtype=torch.int32)
    src, out = torch.randn(n), torch.empty(n)
    gather[(triton.cdiv(n, 256),)](idx, src, out, n, BLOCK=256)
    assert det.last_status == "unsupported"
    assert det.unsupported_reason is not None
    assert "data-dependent" in det.unsupported_reason
    assert det.records == []


@requires_cuda
def test_nested_loops_are_unsupported_not_silently_proven():
    """Nested loops carry independent induction variables the single-loop
    model can't represent. Regression: the nested-loop guard keyed on a flag
    only set at loop close, so an inner loop slipped through and the outer
    induction var leaked unbounded — a silent false "ok" proof for a real
    OOB."""
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def nested(in_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
        for i in range(0, M):
            for j in range(0, N):
                offs = (i * N + j) * BLOCK + tl.arange(0, BLOCK)
                x = tl.load(in_ptr + offs)  # no mask -> would be OOB
                tl.store(out_ptr + offs, x)

    inp, out = torch.randn(20), torch.empty(20)
    nested[(1,)](inp, out, 8, 2, BLOCK=4)  # real footprint needs numel 64
    assert det.last_status == "unsupported"
    assert "loop" in (det.unsupported_reason or "")
    assert det.records == []


@requires_cuda
def test_store_loop_without_accumulator_is_analyzed():
    """A pure side-effect loop (no yielded accumulator, so the TTIR scf.for
    has no '-> (types)') must still be recognized as a loop and checked, not
    skipped — its induction variable drives the address."""
    correct = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(correct)
    @triton.jit
    def store_loop(out_ptr, iters, BLOCK: tl.constexpr):
        for i in range(0, iters):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.full((BLOCK,), 1.0, tl.float32))

    out = torch.zeros(16)
    store_loop[(1,)](out, 4, BLOCK=4)  # 4*4 == 16, exactly fits
    assert correct.last_status == "ok"
    assert correct.records == []

    buggy = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(buggy)
    @triton.jit
    def store_loop2(out_ptr, iters, BLOCK: tl.constexpr):
        for i in range(0, iters):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.full((BLOCK,), 1.0, tl.float32))

    out2 = torch.zeros(16)
    store_loop2[(1,)](out2, 6, BLOCK=4)  # 6*4 == 24 > 16
    assert buggy.last_status == "ok"
    assert len(buggy.records) >= 1


@requires_cuda
def test_second_launch_recomputes_per_launch_metadata():
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, x)

    # First launch: exact multiple -> in bounds.
    n1 = 4096
    x1, out1 = torch.randn(n1), torch.empty(n1)
    add_nomask[(triton.cdiv(n1, 1024),)](x1, out1, n1, BLOCK=1024)
    assert det.last_status == "ok"
    assert det.records == []

    # Second launch: ragged tail -> OOB. Per-launch metadata must be fresh.
    n2 = 3000
    x2, out2 = torch.randn(n2), torch.empty(n2)
    add_nomask[(triton.cdiv(n2, 1024),)](x2, out2, n2, BLOCK=1024)
    assert det.last_status == "ok"
    assert len(det.records) >= 1


@requires_cuda
def test_modeled_branch_condition_proves_no_false_witness():
    """`if t > 0: load(p + offs - B)` never reads offset -1: the t == 0
    iteration takes the other branch. The condition (Param - LoopVar > 0) is
    modelable, so the branch load carries it as a path constraint and the
    launch gets a PRECISE proof — no abstention, and certainly no false
    witness at the unreachable t == 0 state. This is the TritonBench
    diag_ssm_triton backward-kernel shape (pre-S2 this abstained as
    'branch-guarded unsupported')."""
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def guarded_scan(x_ptr, out_ptr, n_steps, n_cols, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        acc = tl.zeros((BLOCK,), tl.float32)
        for i in range(n_steps):
            t = n_steps - 1 - i
            if t > 0:
                prev = tl.load(x_ptr + t * n_cols + offs - n_cols, mask=mask, other=0)
            else:
                prev = tl.zeros((BLOCK,), tl.float32)
            acc += prev
        tl.store(out_ptr + offs, acc, mask=mask)

    n_steps, n_cols = 5, 8
    x = torch.randn(n_steps * n_cols, device="cuda")
    out = torch.empty(n_cols, device="cuda")
    guarded_scan[(1,)](x, out, n_steps, n_cols, BLOCK=8)
    assert det.last_status == "ok"
    assert det.records == []  # proved: the guarded load never reaches -1


@requires_cuda
def test_data_dependent_branch_still_abstains():
    """A branch condition derived from loaded DATA cannot be modeled. A
    potential OOB behind it must abstain (unsupported) — never a witness
    from a possibly-untaken branch, never a silent 'ok'."""
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def flag_gated(flag_ptr, x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        flag = tl.load(flag_ptr)
        acc = tl.zeros((BLOCK,), tl.float32)
        if flag > 0:
            # offs - n_cols is negative for every active lane: definite OOB
            # if the branch runs — but whether it runs depends on data.
            acc = tl.load(x_ptr + offs - n_cols, mask=mask, other=0)
        tl.store(out_ptr + offs, acc, mask=mask)

    n_cols = 8
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    x = torch.randn(n_cols, device="cuda")
    out = torch.empty(n_cols, device="cuda")
    flag_gated[(1,)](flag, x, out, n_cols, BLOCK=8)
    assert det.last_status == "unsupported"
    assert "branch-guarded" in (det.unsupported_reason or "")
    assert det.records == []  # no false witness from the untaken branch


@requires_cuda
def test_unguarded_oob_reported_despite_branch_in_kernel():
    """A kernel may mix a branch (whose accesses can only be proven, not
    witnessed) with plain accesses. A real OOB on a plain access is a
    reachable violation and must still be reported — the mere presence of an
    scf.if must not blind the analyzer (chunk_gla_fwd's OOB h-load sits
    before an scf.if in the same loop body)."""
    det = Sanitizer(compile=True, abort_on_error=False)

    @triton_viz.trace(det)
    @triton.jit
    def mixed(x_ptr, out_ptr, n, flag, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)  # unguarded; OOB when numel < BLOCK
        if flag > 0:
            x += tl.load(x_ptr + offs, mask=offs < n, other=0)  # guarded, safe
        tl.store(out_ptr + offs, x, mask=offs < n)

    n = 8
    x = torch.randn(n, device="cuda")
    out = torch.empty(n, device="cuda")
    mixed[(1,)](x, out, n, 1, BLOCK=16)  # BLOCK 16 > numel 8
    assert det.last_status == "ok", det.unsupported_reason
    assert len(det.records) >= 1
    assert det.records[0].op_type.__name__ == "Load"


@requires_cuda
def test_grouped_swizzle_matmul_min_rem_modeled():
    """The tutorial-03 grouped swizzle (pid // and % over launch quantities,
    min() for the last partial group) lowers to arith.divsi/remsi/minsi —
    all launch-affine. The analyzer must see through it: TritonBench's
    matmul_triton2 (this swizzle with the `% M`/`% N` row clamps removed)
    used to abstain as "data-dependent" instead of catching a real OOB."""

    @triton.jit
    def swizzle_matmul(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):  # fmt: skip
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # no `% M` clamp
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # no `% N` clamp
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

    def launch(det, m, n, k):
        traced = triton_viz.trace(det)(swizzle_matmul)
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        c = torch.empty(m, n, device="cuda")
        grid = (triton.cdiv(m, 32) * triton.cdiv(n, 32),)
        traced[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, GROUP_M=8,
        )  # fmt: skip

    # M = N = K = 64: every block index stays inside both operands — a proof,
    # despite the divsi/remsi/minsi swizzle in every address.
    clean = Sanitizer(compile=True, abort_on_error=False)
    launch(clean, 64, 64, 64)
    assert clean.last_status == "ok", clean.unsupported_reason
    assert clean.records == []

    # M = N = K = 16 < BLOCK: the K-only masks leave rows 16..31 of A (and
    # cols 16..31 of B) unguarded — the real OOB compute-sanitizer flags on
    # TritonBench's matmul_triton2.
    buggy = Sanitizer(compile=True, abort_on_error=False)
    launch(buggy, 16, 16, 16)
    assert buggy.last_status == "ok", buggy.unsupported_reason
    assert len(buggy.records) >= 1
    kinds = {r.op_type.__name__ for r in buggy.records}
    assert "Load" in kinds
