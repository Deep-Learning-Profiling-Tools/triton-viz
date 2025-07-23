import pytest
import torch
import numpy as np

import triton
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

import triton_viz
from triton_viz import config as cfg
from triton_viz.core.data import AddPtr, Load, RawLoad
from triton_viz.core.client import Client
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    SanitizerBruteForce,
    NullSanitizer,
    SanitizerSymbolicExecution,
)

cfg.sanitizer_backend = "symexec"


# ======== Init ===========
def test_init_brute_force():
    cfg.sanitizer_backend = "brute_force"
    s1 = Sanitizer(abort_on_error=True)
    assert isinstance(s1, SanitizerBruteForce) and s1.abort_on_error is True


def test_init_null_sanitizer():
    cfg.sanitizer_backend = "off"
    s2 = Sanitizer(abort_on_error=True)
    assert isinstance(s2, NullSanitizer)


def test_init_symbolic_execution():
    cfg.sanitizer_backend = "symexec"
    s3 = Sanitizer(abort_on_error=True)
    assert isinstance(s3, SanitizerSymbolicExecution) and s3.abort_on_error is True


def test_init_default_sanitizer():
    s = Sanitizer()
    assert isinstance(s, Client)


# ======== AddPtr =========


def test_addptr_expr_eval():
    base = SymbolicExpr.from_value(1000)  # Synthetic base address
    base.dtype_tt = tl.pointer_type(
        tl.int32
    )  # Simulate a pointer type, int32 = 4 bytes
    offset = SymbolicExpr.from_value(3)
    expr = SymbolicExpr("addptr", base, offset)
    assert expr.eval()[0] == 1000 + 3 * 4


def test_addptr_overrider():
    # Run through sanitizer's overrider
    ptr_dtype = tl.pointer_type(tl.int32)
    ptr_th = TensorHandle(np.array([1000]), ptr_dtype)
    offset_th = TensorHandle(np.array([3]), tl.int32)
    sanitizer = SanitizerSymbolicExecution(abort_on_error=False)
    op_callbacks = sanitizer.register_op_callback(AddPtr)
    assert op_callbacks.op_overrider is not None
    expr = op_callbacks.op_overrider(ptr_th, offset_th)  # offset = 3
    assert expr.op == "addptr"
    assert expr.eval()[0] == 1000 + 3 * 4  # element_bytewidth = 4


# ======== Null Sanitizer =========


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def null_sanitizer_kernel(idx_ptr):
    a = tl.load(idx_ptr)
    tl.store(idx_ptr, a + 5)


def test_null_sanitizer():
    cfg.sanitizer_backend = "off"
    idx = torch.arange(128, dtype=torch.int32)
    null_sanitizer_kernel[(1,)](idx)


# ======== Indirect Load/Store =========
def test_const_dtype_inference():
    x = SymbolicExpr.from_value((1, 2, 3))
    y = SymbolicExpr.from_value((1, 2, 3))
    z = SymbolicExpr("add", x, y)

    assert x.dtype_tt == tl.int32
    assert y.dtype_tt == tl.int32
    assert z.dtype_tt == tl.int32

    expected_tree = (
        "\n"
        "add [dtype=int32]\n"
        "├── lhs: const=(1, 2, 3) [dtype=int32]\n"
        "└── rhs: const=(1, 2, 3) [dtype=int32]"
    )
    assert z.to_tree_str() == expected_tree


def _sum_offsets_from_addptr(expr):
    """
    Traverse an addptr SymbolicExpr and sum all constant offsets.
    If any offset is not constant, return None.
    """
    offsets = []
    non_const_offset = None

    cur = expr
    while cur.op == "addptr":
        off = cur.offset
        if off.op != "const":  # If any offset is not constant, we cannot sum it.
            non_const_offset = off
            break
        offsets.append(np.asarray(off.to_py(), dtype=np.int64))
        cur = cur.ptr

    if non_const_offset:
        raise ValueError(
            f"Some non-constant offsets found ({non_const_offset}) in the addptr chain."
        )
    return np.sum(offsets, axis=0)


class LoadIndexChecker(SanitizerSymbolicExecution):
    """
    Record all offsets, then union into a set.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._offset_lists: list[list[int]] = list()

    @property
    def observed_offsets(self):
        return self._offset_lists

    def register_op_callback(self, op_type):
        op_callbacks = super().register_op_callback(op_type)
        if op_type not in (Load, RawLoad) or op_callbacks.op_overrider is None:
            return op_callbacks

        orig_overrider = op_callbacks.op_overrider

        def new_load_overrider(ptr, *a, **k):
            # exec original overrider
            load_expr = orig_overrider(ptr, *a, **k)

            # Important: We only record pointers accessing fp32!
            # This is because fp32 is usually the outermost dtype of a "load of a load" chain.
            # This is the case in all unittests.
            p = load_expr.ptr
            if (
                hasattr(p, "dtype_tt")
                and isinstance(p.dtype_tt, tl.pointer_type)
                and p.dtype_tt.element_ty is tl.float32
            ):  # filtering fp32 pointers
                offs = _sum_offsets_from_addptr(p)
                if offs is not None:
                    self._offset_lists.append(offs.tolist())
            return load_expr

        # Return OpCallbacks with the new overrider, preserving other callbacks
        from triton_viz.core.callbacks import OpCallbacks

        return OpCallbacks(
            before_callback=op_callbacks.before_callback,
            after_callback=op_callbacks.after_callback,
            op_overrider=new_load_overrider,
        )


@triton_viz.trace(clients=(san1 := LoadIndexChecker(abort_on_error=True)))
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    indices = tl.load(idx_ptr + offsets)
    out_val = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offsets, out_val)


def test_indirect_load():
    cfg.sanitizer_backend = "symexec"
    idx = torch.arange(128, dtype=torch.int32)
    src = torch.rand(128)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
    indirect_load_kernel[grid](idx, src, dst, BLOCK_SIZE=32)

    expected_offsets = idx.cpu().numpy().tolist()  # Ground truth
    observed_offsets = [
        x for sublist in san1.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


@triton_viz.trace(clients=(san2 := LoadIndexChecker(abort_on_error=True)))
@triton.jit
def triple_indirect_load_kernel(
    idx1_ptr,  # int32*
    idx2_ptr,  # int32*
    src_ptr,  # fp32*
    dst_ptr,  # fp32*
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    idx1_val = tl.load(idx1_ptr + offsets)
    idx2_val = tl.load(idx2_ptr + idx1_val)
    out_val = tl.load(src_ptr + idx2_val)

    tl.store(dst_ptr + offsets, out_val)


def test_triple_indirect_load():
    cfg.sanitizer_backend = "symexec"
    N = 128
    device = "cpu"

    src = torch.rand(N, device=device, dtype=torch.float32)
    idx2 = torch.randint(0, N, (N,), device=device, dtype=torch.int32)
    idx1 = torch.randint(0, N, (N,), device=device, dtype=torch.int32)

    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    triple_indirect_load_kernel[grid](
        idx1,
        idx2,
        src,
        dst,
        BLOCK_SIZE=32,
    )

    expected_offsets = idx2[idx1].cpu().numpy().tolist()  # Ground Truth
    observed_offsets = [
        x for sublist in san2.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


@triton_viz.trace(clients=(san3 := LoadIndexChecker(abort_on_error=True)))
@triton.jit
def dual_offset_load_kernel(
    idx_a_ptr,  # int32*
    idx_b_ptr,  # int32*
    src_ptr,  # fp32*
    dst_ptr,  # fp32*
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    a = tl.load(idx_a_ptr + offsets)
    b = tl.load(idx_b_ptr + offsets)
    out_val = tl.load(src_ptr + a + b)

    tl.store(dst_ptr + offsets, out_val)


def test_dual_offset_load():
    cfg.sanitizer_backend = "symexec"
    N = 128
    device = "cpu"

    src = torch.rand(N, device=device, dtype=torch.float32)
    # Generate indices so that a + b is always in-range (0 ≤ a + b < N)
    idx_a = torch.randint(0, N // 2, (N,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, N // 2, (N,), device=device, dtype=torch.int32)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    dual_offset_load_kernel[grid](
        idx_a,
        idx_b,
        src,
        dst,
        BLOCK_SIZE=32,
    )

    expected_offsets = (idx_a + idx_b).cpu().numpy().tolist()  # Ground Truth
    observed_offsets = [
        x for sublist in san3.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


# ======== Sanitizer Backend Tests =========
def test_switch_backend():
    """Switch back and forth at runtime."""
    original = cfg.sanitizer_backend

    cfg.sanitizer_backend = "symexec"
    assert cfg.sanitizer_backend == "symexec"

    cfg.sanitizer_backend = "off"
    assert cfg.sanitizer_backend == "off"

    cfg.sanitizer_backend = original


def test_invalid_backend_raises():
    """Setting an unknown backend should raise ValueError."""
    with pytest.raises(ValueError):
        cfg.sanitizer_backend = "does_not_exist"
