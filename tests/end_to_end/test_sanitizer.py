import torch
import numpy as np
from typing import Optional

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
from z3.z3 import BoolRef


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
        self.check_inside_loop: list[tuple[Z3Expr, Optional[BoolRef]]] = []
        self.iterator_constraints: list[BoolRef] = []

    def _check_range_satisfiable(
        self,
        access_addr: Z3Expr,
        expr_constraints: Optional[BoolRef],
        symbolic_expr: SymbolicExpr,
        source_location: Optional[tuple[str, int, str]] = None,
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
            if self._loop_stack and self._loop_stack[-1].lineno == lineno:
                ctx = self._loop_stack[-1]
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
nested_loop_checker = SymbolicSanitizer(abort_on_error=False)


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
    assert "+1000" in tb_info.line_of_code or "1000" in tb_info.line_of_code, (
        f"Expected line to contain the OOB offset, "
        f"but got: {tb_info.line_of_code!r}"
    )

    # Verify function name
    assert tb_info.func_name == "oob_in_loop_kernel", (
        f"Expected func_name to be 'oob_in_loop_kernel', "
        f"but got: {tb_info.func_name!r}"
    )


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
