import pytest
import torch
import numpy as np

import triton
import triton.language as tl

import triton_viz
from triton_viz.core.data import Load, RawLoad
from triton_viz.clients.symbolic_engine import SymbolicExpr, Z3Expr, RangeWrapper
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicSanitizer,
    _range_to_iterator_constraint,
)
from triton_viz.core.callbacks import ForLoopCallbacks
from triton_viz.core.config import config
from z3.z3 import BoolRef


@pytest.fixture
def _isolate_virtual_memory():
    """Save and restore config.virtual_memory around a test."""
    saved = config.virtual_memory
    yield
    config.virtual_memory = saved


# ======== Helpers ===========


class LoadIndexChecker(SymbolicSanitizer):
    """
    Record all offsets, then union into a set.
    """

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._offset_lists: list[list[int]] = list()

    @property
    def observed_offsets(self) -> list[list[int]]:
        return self._offset_lists

    def register_op_callback(self, op_type):
        op_callbacks = super().register_op_callback(op_type)
        if op_type not in (Load, RawLoad) or op_callbacks.op_overrider is None:
            return op_callbacks

        orig_overrider = op_callbacks.op_overrider

        def _sum_offsets_from_addptr(expr):
            offsets = []

            cur = expr
            while cur.op == "addptr":
                off = cur.offset
                if (
                    off.op == "const"
                ):  # If any offset is not constant, we cannot sum it.
                    offsets.append(off.to_py())
                cur = cur.ptr

            if len(offsets) == 0:
                return None
            return np.sum(offsets, axis=0)

        def _new_load_overrider(ptr, *args, **kwargs):
            # exec original overrider
            load_expr = orig_overrider(ptr, *args, **kwargs)
            p = load_expr.ptr
            offs = _sum_offsets_from_addptr(p)
            if offs is not None:
                self._offset_lists.append(offs.tolist())
            return load_expr

        # Return OpCallbacks with the new overrider, preserving other callbacks
        from triton_viz.core.callbacks import OpCallbacks

        return OpCallbacks(
            before_callback=op_callbacks.before_callback,
            after_callback=op_callbacks.after_callback,
            op_overrider=_new_load_overrider,
        )


load_index_checker: LoadIndexChecker = LoadIndexChecker(abort_on_error=False)


class LoopBoundsChecker(SymbolicSanitizer):
    """
    Record concretized loop bounds.
    """

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._bounds: list[tuple[int, int, int]] = list()

    @property
    def observed_bounds(self) -> list[tuple[int, int, int]]:
        return self._bounds

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        callbacks = super().register_for_loop_callback()
        orig_before = callbacks.before_loop_callback

        def _before_loop(lineno, iterable):
            if orig_before is not None:
                orig_before(lineno, iterable)
            if isinstance(iterable, RangeWrapper):
                self._bounds.append((iterable.start, iterable.stop, iterable.step))

        return ForLoopCallbacks(
            range_wrapper_factory=callbacks.range_wrapper_factory,
            range_type_callback=callbacks.range_type_callback,
            before_loop_callback=_before_loop,
            loop_iter_overrider=callbacks.loop_iter_overrider,
            loop_iter_listener=callbacks.loop_iter_listener,
            after_loop_callback=callbacks.after_loop_callback,
        )


loop_bounds_checker: LoopBoundsChecker = LoopBoundsChecker(abort_on_error=True)


class LoopDeferredCheckRecorder(SymbolicSanitizer):
    """
    Record when deferred checks are executed after loop exit.
    """

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.after_loop_pending: list[int] = []
        self.check_inside_loop: list[tuple[Z3Expr, BoolRef | None]] = []
        self.iterator_constraints: list[BoolRef] = []

    def _check_range_satisfiable(
        self,
        access_addr: Z3Expr,
        expr_constraints: BoolRef | None,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        self.check_inside_loop.append((access_addr, expr_constraints))
        super()._check_range_satisfiable(
            access_addr, expr_constraints, symbolic_expr, source_location
        )

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        callbacks = super().register_for_loop_callback()
        orig_after = callbacks.after_loop_callback

        def _after_loop(lineno: int) -> None:
            pending = 0
            if self.loop_stack and self.loop_stack[-1].lineno == lineno:
                ctx = self.loop_stack[-1]
                pending = len(ctx.pending_checks)
                self.iterator_constraints.append(
                    _range_to_iterator_constraint(
                        ctx.idx_z3, start=ctx.start, stop=ctx.stop, step=ctx.step
                    )
                )
            if orig_after is not None:
                orig_after(lineno)
            self.after_loop_pending.append(pending)

        return ForLoopCallbacks(
            range_wrapper_factory=callbacks.range_wrapper_factory,
            range_type_callback=callbacks.range_type_callback,
            before_loop_callback=callbacks.before_loop_callback,
            loop_iter_overrider=callbacks.loop_iter_overrider,
            loop_iter_listener=callbacks.loop_iter_listener,
            after_loop_callback=_after_loop,
        )


loop_deferred_check_recorder: LoopDeferredCheckRecorder = LoopDeferredCheckRecorder(
    abort_on_error=False
)


# ======== Kernels ===========


@triton_viz.trace(client=load_index_checker)
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    indices = tl.load(idx_ptr + offsets)
    out_val = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offsets, out_val)


@triton_viz.trace(client=loop_bounds_checker)
@triton.jit
def loop_bounds_kernel(start_ptr, stop_ptr, out_ptr):
    start = tl.load(start_ptr)
    stop = tl.load(stop_ptr)
    for _ in range(start, stop):
        pass
    tl.store(out_ptr, start)


@triton_viz.trace(client=loop_bounds_checker)
@triton.jit
def loop_bounds_pid_kernel(out_ptr):
    pid = tl.program_id(0)
    start = pid
    stop = pid + 2
    for _ in range(start, stop):
        pass
    tl.store(out_ptr + pid, start)


@triton_viz.trace(client=loop_deferred_check_recorder)
@triton.jit
def loop_deferred_check_kernel(out_ptr):
    for i in range(0, 4):
        idx = i + 1
        tl.store(out_ptr + idx, idx)


@triton_viz.trace(client=load_index_checker)
@triton.jit
def loop_deferred_check_simplify_kernel(out_ptr):
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0) + 1
    for i in range(0, num_blocks):
        idx = pid + 1
        tl.store(out_ptr + idx, idx)


# ======== Indirect Load/Store Tests ===========


def test_indirect_load():
    load_index_checker.observed_offsets.clear()

    idx = torch.arange(128, dtype=torch.int32)
    src = torch.rand(128)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
    indirect_load_kernel[grid](idx, src, dst, BLOCK_SIZE=32)

    expected_offsets = idx.cpu().numpy().tolist()  # Ground truth
    observed_offsets = [
        x for sublist in load_index_checker.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


# ======== Loop Tests ===========


def test_loop_bounds_from_load():
    loop_bounds_checker.observed_bounds.clear()
    start = torch.tensor([2], dtype=torch.int32)
    stop = torch.tensor([6], dtype=torch.int32)
    out = torch.empty((1,), dtype=torch.int32)

    loop_bounds_kernel[(1,)](start, stop, out)

    assert loop_bounds_checker.observed_bounds == [(2, 6, 1)]


def test_loop_bounds_from_pid():
    loop_bounds_checker.observed_bounds.clear()
    out = torch.empty((2,), dtype=torch.int32)

    loop_bounds_pid_kernel[(2,)](out)

    assert loop_bounds_checker.observed_bounds == [(0, 2, 1), (1, 3, 1)]


def test_loop_deferred_checks_after_context():
    loop_deferred_check_recorder.after_loop_pending.clear()
    loop_deferred_check_recorder.check_inside_loop.clear()
    loop_deferred_check_recorder.iterator_constraints.clear()
    loop_deferred_check_recorder.records.clear()

    out = torch.empty((2,), dtype=torch.int32)

    loop_deferred_check_kernel[(1,)](out)

    assert loop_deferred_check_recorder.after_loop_pending == [1]
    addr_expr, _ = loop_deferred_check_recorder.check_inside_loop[0]
    assert "4*loop_i_2" in str(addr_expr)
    iterator_constraints_str = " ".join(
        str(c) for c in loop_deferred_check_recorder.iterator_constraints
    )
    assert "loop_i_2 >= 0" in iterator_constraints_str
    assert "loop_i_2 < 4" in iterator_constraints_str
    assert loop_deferred_check_recorder.records


def test_loop_deferred_checks_simplify():
    load_index_checker.observed_offsets.clear()

    out = torch.empty((3,), dtype=torch.int32)

    loop_deferred_check_simplify_kernel[(2,)](out)

    assert load_index_checker.observed_offsets == []


# Dedicated sanitizer for nested loop regression test
nested_loop_checker = SymbolicSanitizer()


@triton_viz.trace(client=nested_loop_checker)
@triton.jit
def nested_loop_outer_dep_kernel(out_ptr):
    for i in range(0, 3):
        for j in range(0, 4):
            idx = i * 4 + j
            tl.store(out_ptr + idx, idx)


def test_nested_loop_no_false_positive():
    nested_loop_checker.records.clear()

    # Buffer size = 12, exactly fits i*4+j where i in [0,3), j in [0,4)
    out = torch.empty((12,), dtype=torch.int32)
    nested_loop_outer_dep_kernel[(1,)](out)

    assert (
        len(nested_loop_checker.records) == 0
    ), f"False positive OOB detected. Records: {nested_loop_checker.records}"


# ======== Line Number Reporting Tests ===========


# Create a dedicated sanitizer for line number tests
# abort_on_error=False is on purpose: so OOB violations are recorded
line_number_checker: SymbolicSanitizer = SymbolicSanitizer(abort_on_error=False)


@triton_viz.trace(client=line_number_checker)
@triton.jit
def oob_in_loop_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel where OOB occurs inside a loop at the tl.load line."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for _ in range(2):
        # This line should be reported in the traceback when OOB occurs
        val = tl.load(ptr + offsets + 1000)  # OOB access due to +1000 offset
        tl.store(ptr + offsets, val)


def test_loop_oob_reports_correct_line_number():
    """
    Test that sanitizer reports the correct line number for OOB errors in loops.

    Previously, the sanitizer would report the function definition line instead
    of the actual tl.load/tl.store line that caused the error. This was because
    traceback was captured at loop exit time rather than when the memory
    operation was executed.
    """
    line_number_checker.records.clear()

    data = torch.zeros((16,), dtype=torch.float32)

    oob_in_loop_kernel[(1,)](data, N=16, BLOCK_SIZE=16)

    # Verify that OOB was detected
    assert len(line_number_checker.records) > 0, "Expected OOB to be detected"

    record = line_number_checker.records[0]
    assert len(record.user_code_tracebacks) > 0, "Expected traceback info"

    tb_info = record.user_code_tracebacks[0]

    # The error should point to the tl.load line, not the function definition
    assert "tl.load" in tb_info.line_of_code, (
        f"Expected traceback to point to tl.load line, "
        f"but got: {tb_info.line_of_code!r}"
    )

    # Verify the line contains the OOB offset
    assert (
        "+1000" in tb_info.line_of_code or "1000" in tb_info.line_of_code
    ), f"Expected line to contain the OOB offset, but got: {tb_info.line_of_code!r}"

    # Verify function name
    assert (
        tb_info.func_name == "oob_in_loop_kernel"
    ), f"Expected func_name to be 'oob_in_loop_kernel', but got: {tb_info.func_name!r}"


# ---------------------------------------------------------------------------
# gemm_oob call-stack extraction test
# ---------------------------------------------------------------------------
# Runs the actual examples/sanitizer/gemm_oob.py and verifies that the
# sanitizer output (Code Context, Call Stack) matches docs/index.html.


def test_gemm_oob_call_stack():
    """
    Run examples/sanitizer/gemm_oob.py and verify the reported call-stack
    matches what is documented in docs/index.html:

      ━━━ Code Context ━━━
      File: triton-viz/examples/sanitizer/gemm_oob.py
      Function: gemm_kernel
      Line 25:
        22 │     for k_block in range(K // TILE_SIZE):
        ...
      → 25 │         A_tile = tl.load(A + A_off + 1) # Out-Of-Bounds Access HERE!
      ━━━ Call Stack ━━━
      #1 gemm_kernel at gemm_oob.py:25
         └─ A_tile = tl.load(A + A_off + 1) # Out-Of-Bounds Access HERE!
    """
    import pathlib
    import subprocess
    import sys

    example_file = (
        pathlib.Path(__file__).resolve().parents[2]
        / "examples"
        / "sanitizer"
        / "gemm_oob.py"
    )
    assert example_file.exists(), f"Example file not found: {example_file}"

    import re

    result = subprocess.run(
        [sys.executable, str(example_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Strip ANSI escape codes so plain-text assertions work
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)

    # ---- Code Context checks (matching docs/index.html) ----
    assert "Code Context" in output, "Missing 'Code Context' section in output"
    assert "gemm_oob.py" in output, "Missing filename 'gemm_oob.py' in output"
    assert (
        "Function: gemm_kernel" in output
    ), "Missing 'Function: gemm_kernel' in output"
    assert "Line 25:" in output, "Missing 'Line 25:' in output"
    assert (
        "tl.load(A + A_off + 1)" in output
    ), "Missing 'tl.load(A + A_off + 1)' in Code Context"

    # ---- Call Stack checks ----
    assert (
        "#1 gemm_kernel at gemm_oob.py:25" in output
    ), "Missing expected call stack entry '#1 gemm_kernel at gemm_oob.py:25'"


# ======== Block Tensor (Block Pointer) Tests ===========


# abort_on_error=False is on purpose: so OOB violations are recorded
block_sanitizer = SymbolicSanitizer(abort_on_error=False)


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_load_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    tl.load(block_ptr, boundary_check=(0,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_load_oob_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # No boundary_check -> OOB is undefined behavior
    tl.load(block_ptr)


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_store_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    val = tl.full((BLOCK_SIZE,), value=1.0, dtype=tl.float32)
    tl.store(block_ptr, val, boundary_check=(0,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_store_oob_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    val = tl.full((BLOCK_SIZE,), value=1.0, dtype=tl.float32)
    # No boundary_check -> OOB is undefined behavior
    tl.store(block_ptr, val)


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_load_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.load(block_ptr, boundary_check=(0, 1))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_load_oob_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # No boundary_check -> OOB is undefined behavior
    tl.load(block_ptr)


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_boundary_check_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Block extends past tensor end but boundary_check masks it."""
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # boundary_check=(0,) means hardware masks OOB positions
    tl.load(block_ptr, boundary_check=(0,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_loop_advance_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    for _ in range(N // BLOCK_SIZE):
        tl.load(block_ptr, boundary_check=(0,))
        block_ptr = tl.advance(block_ptr, (BLOCK_SIZE,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_loop_advance_oob_kernel(
    ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # One extra iteration -> OOB, NO boundary_check
    for _ in range(N // BLOCK_SIZE + 1):
        tl.load(block_ptr)
        block_ptr = tl.advance(block_ptr, (BLOCK_SIZE,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_loop_advance_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D block pointer advanced along K (N) dimension in a loop."""
    pid_m = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    for _ in range(N // BLOCK_N):
        tl.load(block_ptr, boundary_check=(0, 1))
        block_ptr = tl.advance(block_ptr, (0, BLOCK_N))


def test_block_tensor_1d_load_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (N // BLOCK_SIZE,)
    block_tensor_1d_load_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_1d_store_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.zeros(N, dtype=torch.float32)
    grid = (N // BLOCK_SIZE,)
    block_tensor_1d_store_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_2d_load_non_oob():
    block_sanitizer.records.clear()
    M, N = 4, 4
    BLOCK_M, BLOCK_N = 2, 2
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    grid = (M // BLOCK_M, N // BLOCK_N)
    block_tensor_2d_load_kernel[grid](data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_boundary_check_masks_oob():
    block_sanitizer.records.clear()
    # N=48 is not divisible by BLOCK_SIZE=32, so last block extends past end.
    # boundary_check=(0,) means hardware masks the OOB positions.
    N = 48
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (2,)  # 2 blocks: [0,32) and [32,64) -- second extends past N=48
    block_tensor_boundary_check_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records with boundary_check, got {len(block_sanitizer.records)}"


def test_block_tensor_1d_load_oob():
    block_sanitizer.records.clear()
    N = 48
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (2,)  # Second block [32,64) goes past N=48, no boundary_check
    block_tensor_1d_load_oob_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_1d_store_oob():
    block_sanitizer.records.clear()
    N = 48
    BLOCK_SIZE = 32
    data = torch.zeros(N, dtype=torch.float32)
    grid = (2,)  # Second block [32,64) goes past N=48, no boundary_check
    block_tensor_1d_store_oob_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_2d_load_oob():
    block_sanitizer.records.clear()
    M, N = 4, 4
    BLOCK_M, BLOCK_N = 2, 2
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    # 3x2 grid but matrix is only 4x4 with BLOCK_M=2 -> pid_m=2 goes OOB
    grid = (3, 2)  # pid_m in [0,3), but M//BLOCK_M=2, so pid_m=2 is OOB
    block_tensor_2d_load_oob_kernel[grid](
        data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_loop_advance_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    block_tensor_loop_advance_kernel[(1,)](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_loop_advance_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    block_tensor_loop_advance_oob_kernel[(1,)](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_2d_loop_advance_non_oob():
    block_sanitizer.records.clear()
    M, N = 4, 8
    BLOCK_M, BLOCK_N = 4, 4
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    # 1 program, loops N//BLOCK_N=2 times
    block_tensor_2d_loop_advance_kernel[(1,)](
        data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_cli_code_context_points_to_kernel():
    """
    Run a minimal OOB kernel via ``triton-sanitizer`` CLI and verify the
    Code Context section points to the actual kernel line, not the CLI
    entry-point wrapper.
    """
    import os
    import re
    import subprocess
    import sys
    import tempfile
    import textwrap

    kernel_src = textwrap.dedent(
        """\
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def oob_kernel(ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            tl.load(ptr + offs)  # OOB: BLOCK > tensor size, no mask

        oob_kernel[(1,)](torch.zeros(4), BLOCK=16)
    """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(kernel_src)
        tmp_path = tmp.name

    try:
        sanitizer = os.path.join(os.path.dirname(sys.executable), "triton-sanitizer")
        result = subprocess.run(
            [sanitizer, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)

        # Code Context must reference the kernel, not the CLI wrapper
        assert "Code Context" in output, "Missing 'Code Context' section"
        assert (
            "Function: oob_kernel" in output
        ), f"Code Context should point to 'oob_kernel', got:\n{output}"
        assert (
            "tl.load(ptr + offs)" in output
        ), f"Code Context should show the tl.load line, got:\n{output}"
    finally:
        os.unlink(tmp_path)


# ======== Reduce with return_indices Tests ===========

reduce_indices_sanitizer = SymbolicSanitizer(abort_on_error=False)


@triton_viz.trace(client=reduce_indices_sanitizer)
@triton.jit
def max_return_indices_kernel(inp_ptr, out_val_ptr, out_idx_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    inp = tl.load(inp_ptr + offs)
    max_val, max_idx = tl.max(inp, axis=0, return_indices=True)
    tl.store(out_val_ptr, max_val)
    tl.store(out_idx_ptr, max_idx)


@triton_viz.trace(client=reduce_indices_sanitizer)
@triton.jit
def min_return_indices_kernel(inp_ptr, out_val_ptr, out_idx_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    inp = tl.load(inp_ptr + offs)
    min_val, min_idx = tl.min(inp, axis=0, return_indices=True)
    tl.store(out_val_ptr, min_val)
    tl.store(out_idx_ptr, min_idx)


def test_tl_max_return_indices():
    """tl.max with return_indices=True should stay on the symbolic path."""
    reduce_indices_sanitizer.records.clear()

    N = 32
    inp = torch.arange(N, dtype=torch.float32)
    out_val = torch.empty(1, dtype=torch.float32)
    out_idx = torch.empty(1, dtype=torch.int32)

    max_return_indices_kernel[(1,)](inp, out_val, out_idx, N=N)

    assert (
        len(reduce_indices_sanitizer.records) == 0
    ), f"Sanitizer reported {len(reduce_indices_sanitizer.records)} error(s)"


def test_tl_min_return_indices():
    """tl.min with return_indices=True should stay on the symbolic path."""
    reduce_indices_sanitizer.records.clear()

    N = 32
    inp = torch.arange(N, dtype=torch.float32)
    out_val = torch.empty(1, dtype=torch.float32)
    out_idx = torch.empty(1, dtype=torch.int32)

    min_return_indices_kernel[(1,)](inp, out_val, out_idx, N=N)

    assert (
        len(reduce_indices_sanitizer.records) == 0
    ), f"Sanitizer reported {len(reduce_indices_sanitizer.records)} error(s)"


# ======== Reduce + Broadcast Tests ===========

reduce_broadcast_sanitizer = SymbolicSanitizer()


@triton_viz.trace(client=reduce_broadcast_sanitizer)
@triton.jit
def reduce_broadcast_kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0) * 1 + tl.arange(0, 1)[:, None]
    cols = tl.arange(0, N)[None, :]
    x = tl.load(in_ptr + row * N + cols)
    s = tl.sum(x, axis=1)[:, None]  # reduce axis=1, re-expand
    result = x - s  # broadcast back to [1, N]
    tl.store(out_ptr + row * N + cols, result)


def test_reduce_broadcast():
    """
    Verify the symbolic engine handles reduce + broadcast without raising
    ValueError: Cannot broadcast, rank mismatch.

    Pattern: 2D load -> tl.sum(axis=1) -> [:, None] -> arithmetic with 2D tensor.
    """
    reduce_broadcast_sanitizer.records.clear()

    M, N = 1, 16
    inp = torch.randn(M, N, dtype=torch.float32)
    out = torch.empty_like(inp)

    reduce_broadcast_kernel[(M,)](inp, out, M=M, N=N)

    # No OOB expected — the kernel accesses exactly M*N elements
    assert (
        len(reduce_broadcast_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(reduce_broadcast_sanitizer.records)}"


# ======== Fake Tensor (Virtual Memory) OOB Tests ===========

fake_tensor_sanitizer = SymbolicSanitizer(abort_on_error=True)


@triton_viz.trace(client=fake_tensor_sanitizer)
@triton.jit
def fake_tensor_oob_kernel(x_ptr, out_ptr, N: tl.constexpr):
    # Intentionally read out-of-bounds: offset N is beyond the valid range [0, N)
    val = tl.load(x_ptr + N)
    tl.store(out_ptr, val)


def test_oob_with_fake_tensor(_isolate_virtual_memory):
    fake_tensor_sanitizer.records.clear()

    config.virtual_memory = True
    x = torch.randn(8)
    out = torch.empty(1)
    with pytest.raises(SystemExit):
        fake_tensor_oob_kernel[(1,)](x, out, N=8)


@triton_viz.trace(client=SymbolicSanitizer())
@triton.jit
def block_ptr_sum_kernel(
    s_ptr,
    z_ptr,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    o_i = tl.arange(0, BT)
    m = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BS], dtype=tl.float32)
    p_s = tl.make_block_ptr(s_ptr, (T, S), (S, 1), (0, 0), (BT, BS), (1, 0))
    p_z = tl.make_block_ptr(z_ptr, (T, S), (S, 1), (0, 0), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_c = b_z[None, :] + tl.dot(m, b_s, allow_tf32=False)
    tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))
    b_z += tl.sum(b_s, 0)


def test_reduce_symbolic_core_dtype():
    """tl.sum on cast block_ptr data must preserve block_type dtype."""
    s = torch.randn(16, 16, device="cpu")
    z = torch.empty_like(s)
    block_ptr_sum_kernel[(1,)](s, z, T=16, S=16, BT=16, BS=16)


@triton_viz.trace(client=SymbolicSanitizer())
@triton.jit
def softmax_kernel(output_ptr, input_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(input_ptr + row * N + offs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, 0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, 0)
    tl.store(output_ptr + row * N + offs, exp_x / sum_exp, mask=mask)


def test_reduce_symbolic_nonetype():
    """Softmax kernel: tl.exp -> tl.max / tl.sum must propagate dtype correctly."""
    x = torch.randn(4, 64, device="cpu")
    out = torch.empty_like(x)
    softmax_kernel[(4,)](out, x, 64, BLOCK=64)


@triton_viz.trace(client=SymbolicSanitizer())
@triton.jit
def exp_expand_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    w = tl.exp(x)
    m = w[None, :] * tl.load(x_ptr + offs)[:, None]
    tl.store(out_ptr + offs, tl.sum(m, axis=1))


def test_expand_dims_scalar_attr():
    """tl.exp followed by expand_dims must propagate dtype correctly."""
    x = torch.randn(8, device="cpu")
    out = torch.empty(8, device="cpu")
    exp_expand_kernel[(1,)](x, out, N=8)


# ======== Non-contiguous Expanded Tensor Regression Test ===========


@triton_viz.trace(client=SymbolicSanitizer())
@triton.jit
def read_expanded_kernel(inp, out, stride_row, stride_col, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    ptrs = inp + row * stride_row + offs * stride_col
    x = tl.load(ptrs, mask=mask, other=0)
    tl.store(out + row * N + offs, x, mask=mask)


def test_non_contiguous_expanded_tensor():
    """expand()-ed tensors (stride-0, non-contiguous) must not crash the sanitizer."""
    M, N = 4, 8
    row = torch.arange(N, device="cpu", dtype=torch.float32)
    x = row.unsqueeze(0).expand(M, N)  # shape (4,8), strides (0, 1)
    assert not x.is_contiguous()
    out = torch.empty(M, N, device="cpu")
    read_expanded_kernel[(M,)](x, out, x.stride(0), x.stride(1), M, N, BLOCK_N=8)


# ======== TensorWrapper Regression Test ===========


@triton_viz.trace(client=SymbolicSanitizer())
@triton.jit
def copy_kernel(src, dst, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)


# ======== Reduce on Dot Result Tests ===========


reduce_dot_sanitizer = SymbolicSanitizer()


@triton_viz.trace(client=reduce_dot_sanitizer)
@triton.jit
def dot_row_max_kernel(
    Q,
    K,
    Out,
    stride_qm,
    stride_kn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :])
    k = tl.load(K + offs_d[:, None] + offs_n[None, :] * stride_kn)

    qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
    row_max = tl.max(qk, axis=1)  # reduce axis=1 -> [BLOCK_M]

    tl.store(Out + offs_m, row_max)


def test_reduce_on_dot_result():
    """tl.max on the result of tl.dot must not crash the symbolic engine.

    Regression: ReduceSymbolicExpr previously raised
    'expects block input with non-empty shape' when reducing a dot product.
    """
    reduce_dot_sanitizer.records.clear()

    M, N, D = 16, 16, 16
    q = torch.randn(M, D, dtype=torch.float16)
    k = torch.randn(D, N, dtype=torch.float16)
    out = torch.empty(M, dtype=torch.float32)

    dot_row_max_kernel[(1,)](
        q,
        k,
        out,
        q.stride(0),
        k.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        D=D,
    )

    assert (
        len(reduce_dot_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(reduce_dot_sanitizer.records)}"


@triton_viz.trace(client=reduce_dot_sanitizer)
@triton.jit
def batched_dot_row_max_kernel(
    A,
    B,
    Out,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    B_DIM: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    offs_b = tl.arange(0, B_DIM)
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a = tl.load(
        A
        + offs_b[:, None, None] * stride_ab
        + offs_m[None, :, None] * stride_am
        + offs_k[None, None, :] * stride_ak
    )
    b = tl.load(
        B
        + offs_b[:, None, None] * stride_bb
        + offs_k[None, :, None] * stride_bk
        + offs_n[None, None, :] * stride_bn
    )

    c = tl.dot(a, b)  # [B_DIM, M, N]
    row_max = tl.max(c, axis=2)  # reduce axis=2 -> [B_DIM, M]

    tl.store(Out + offs_b[:, None] * stride_ob + offs_m[None, :] * stride_om, row_max)


def test_reduce_on_batched_dot_result():
    """tl.max on the result of a 3D batched tl.dot must not crash the symbolic engine."""
    reduce_dot_sanitizer.records.clear()

    B, M, N, K = 2, 16, 16, 16
    a = torch.randn(B, M, K, dtype=torch.float16)
    b = torch.randn(B, K, N, dtype=torch.float16)
    out = torch.empty(B, M, dtype=torch.float32)

    batched_dot_row_max_kernel[(1,)](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        B_DIM=B,
        M=M,
        N=N,
        K=K,
    )

    assert (
        len(reduce_dot_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(reduce_dot_sanitizer.records)}"


def test_reinterpret_tensor_wrapper():
    """triton.reinterpret() produces a TensorWrapper; sanitizer must handle it."""
    N = 64
    x = torch.ones(N, dtype=torch.float16, device="cpu")
    y = torch.empty(N, dtype=torch.float16, device="cpu")
    copy_kernel[(1,)](triton.reinterpret(x, tl.float16), y, N, BLOCK=64)


# ======== Constexpr Positional Argument Tests ===========

constexpr_str_annotation_sanitizer = SymbolicSanitizer(abort_on_error=False)


@triton_viz.trace(client=constexpr_str_annotation_sanitizer)
@triton.jit
def constexpr_str_annotation_kernel(X, Out, N, BLOCK_SIZE: "tl.constexpr"):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Out + offs, x + 1.0, mask=mask)


def test_constexpr_string_annotation_positional_arg():
    """String-annotated tl.constexpr passed positionally must not raise ValueError.

    On Python 3.14+ the AST rewriter double-quotes string annotations,
    so the rewritten function's __annotations__ contains "'tl.constexpr'"
    instead of "tl.constexpr", making GridExecutor.constexprs empty.
    The fix derives constexpr names from the original JITFunction.params.
    """
    constexpr_str_annotation_sanitizer.records.clear()

    N = 16
    BLOCK_SIZE = 4
    x = torch.arange(N, dtype=torch.float32)
    out = torch.empty_like(x)

    # BLOCK_SIZE passed as positional arg with string annotation
    constexpr_str_annotation_kernel[(N // BLOCK_SIZE,)](x, out, N, BLOCK_SIZE)

    assert (
        len(constexpr_str_annotation_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(constexpr_str_annotation_sanitizer.records)}"
