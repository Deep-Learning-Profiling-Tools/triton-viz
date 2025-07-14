"""Tests for the Sanitizer client functionality."""

import tempfile

import torch
import triton
import triton.language as tl
from triton_viz import config as cfg
from triton_viz.sanitizer import SanitizerBruteForceMasking, SanitizerSymbolicExecution
from triton_viz.sanitizer.symbolic_execution.symbolic_expr import SymbolicExpr, AtomType
from triton_viz.sanitizer.symbolic_execution.addptr_overrider import (
    AddPtrSymbolicExecOverrider,
)
from triton_viz.trace import trace
from triton_viz.clients import Sanitizer


# =============================================================================
# Test autotune functionality with sanitizer
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
    ],
    key=["x_size"],
)
@triton.jit
def autotune_add_kernel(x_ptr, output_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_size
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    output = x + 5
    tl.store(output_ptr + offsets, output, mask=mask)


def test_autotune_add_inrange():
    cfg.sanitizer_backend = "symexec"
    device = torch.cuda.current_device()
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=device)
    output = torch.zeros_like(x)
    grid = lambda meta: (triton.cdiv(x.size(0), meta["BLOCK_SIZE"]),)
    with tempfile.TemporaryDirectory() as tmpdir:
        with trace(autotune_add_kernel, log_dir=tmpdir):
            autotune_add_kernel[grid](x, output, x.size(0))


def test_autotune_add_out_of_bound():
    cfg.sanitizer_backend = "symexec"
    device = torch.cuda.current_device()
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=device)
    output = torch.zeros_like(x)
    grid = lambda meta: (triton.cdiv(x.size(0), meta["BLOCK_SIZE"]),)
    with tempfile.TemporaryDirectory() as tmpdir:
        with trace(autotune_add_kernel, log_dir=tmpdir):
            autotune_add_kernel[grid](x, output, x.size(0) - 1)


# =============================================================================
# Test null sanitizer functionality
# =============================================================================


@triton.jit
def null_sanitizer_kernel(x_ptr, output_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_size
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    output = x + 5
    tl.store(output_ptr + offsets, output, mask=mask)


def test_null_sanitizer():
    cfg.sanitizer_backend = "off"
    device = torch.cuda.current_device()
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=device)
    output = torch.zeros_like(x)
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(x.size(0), BLOCK_SIZE),)
    with tempfile.TemporaryDirectory() as tmpdir:
        with trace(null_sanitizer_kernel, log_dir=tmpdir):
            null_sanitizer_kernel[grid](x, output, x.size(0), BLOCK_SIZE=BLOCK_SIZE)
    assert torch.all(output == x + 5)


# =============================================================================
# Test sanitizer with nested function calls and traceback printing
# =============================================================================


@triton.jit
def kernel_B(x_ptr, output_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_size
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    output = x + 5
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def kernel_A(x_ptr, output_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    kernel_B(x_ptr, output_ptr, x_size, BLOCK_SIZE)


def test_print_nested_functions():
    cfg.sanitizer_backend = "symexec"
    device = torch.cuda.current_device()
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=device)
    output = torch.zeros_like(x)
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(x.size(0), BLOCK_SIZE),)
    with tempfile.TemporaryDirectory() as tmpdir:
        with trace(kernel_A, log_dir=tmpdir):
            kernel_A[grid](x, output, x.size(0) - 1, BLOCK_SIZE=BLOCK_SIZE)


# =============================================================================
# Test constant data type inference for symbolic expressions
# =============================================================================


def test_const_dtype_inference():
    atom1 = SymbolicExpr(atom_type=AtomType.CONST_INT, value=1)
    atom2 = SymbolicExpr(atom_type=AtomType.CONST_INT, value=2)
    atom3 = SymbolicExpr(atom_type=AtomType.CONST_INT, value=3)

    add1 = atom1 + atom2
    add2 = add1 + atom3

    print("add1:", add1)
    print("add2:", add2)

    assert add1.value() == 3
    assert add2.value() == 6

    def build_tree_repr(expr, depth=0):
        if expr.is_leaf():
            return " " * depth + str(expr.value) + ", " + str(expr.atom_type)
        else:
            result = " " * depth + str(expr.op_type)
            for operand in expr.operands:
                result += "\n" + build_tree_repr(operand, depth + 2)
            return result

    print("Tree representation of add1:")
    print(build_tree_repr(add1))
    print("Tree representation of add2:")
    print(build_tree_repr(add2))


# =============================================================================
# Test indirect load functionality
# =============================================================================


# Helper class for capturing sanitizer behavior
class CaptureSanitizer(SanitizerSymbolicExecution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_offsets = {}

    def check_one_pointer(self, ptr, *args):
        offset_sum = _sum_offsets_from_addptr(ptr)
        self.observed_offsets[ptr.handle] = offset_sum
        return super().check_one_pointer(ptr, *args)


def _sum_offsets_from_addptr(addptr):
    op_type = addptr.op_type
    assert str(op_type) == "OpType.ADDPTR"

    offset_val = 0

    def visit(node):
        nonlocal offset_val
        if node.is_leaf():
            if node.atom_type == AtomType.CONST_INT:
                offset_val += node.value
            return

        if str(node.op_type) == "OpType.ADD":
            for operand in node.operands:
                visit(operand)
        elif str(node.op_type) == "OpType.MUL":
            operand0, operand1 = node.operands
            if operand0.atom_type == AtomType.CONST_INT:
                const_val = operand0.value
                visit(operand1)
                offset_val *= const_val
            elif operand1.atom_type == AtomType.CONST_INT:
                const_val = operand1.value
                visit(operand0)
                offset_val *= const_val

    offset_expr = addptr.operands[1]
    visit(offset_expr)
    return offset_val


@triton.jit
def indirect_load_kernel(
    idx1_ptr,
    idx2_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.where(offsets < n_elements, offsets, 0)

    idx2 = tl.load(idx2_ptr + offsets)
    idx1 = tl.load(idx1_ptr + idx2)

    tl.store(output_ptr + offsets, idx1)


def test_indirect_load():
    cfg.sanitizer_backend = "symexec"
    cfg.sanitizer_impl = CaptureSanitizer

    device = torch.cuda.current_device()
    n_elements = 512

    idx1 = torch.arange(n_elements, dtype=torch.int32, device=device)
    idx2 = torch.randint(0, n_elements, (n_elements,), dtype=torch.int32, device=device)
    output = torch.zeros_like(idx1)

    BLOCK_SIZE = 64
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    with tempfile.TemporaryDirectory() as tmpdir:
        with trace(indirect_load_kernel, log_dir=tmpdir):
            indirect_load_kernel[grid](
                idx1, idx2, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )

    expected = idx1[idx2]
    assert torch.all(output == expected)

    cfg.sanitizer_impl = SanitizerSymbolicExecution


@triton.jit
def triple_indirect_load_kernel(
    idx1_ptr,
    idx2_ptr,
    offsets_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.where(offsets < n_elements, offsets, 0)

    loaded_offsets = tl.load(offsets_ptr + offsets)
    idx2 = tl.load(idx2_ptr + loaded_offsets)
    idx1 = tl.load(idx1_ptr + idx2)

    tl.store(output_ptr + offsets, idx1)


def test_triple_indirect_load():
    cfg.sanitizer_backend = "symexec"
    cfg.sanitizer_impl = CaptureSanitizer

    device = torch.cuda.current_device()
    n_elements = 512

    idx1 = torch.arange(n_elements, dtype=torch.int32, device=device)
    idx2 = torch.randint(0, n_elements, (n_elements,), dtype=torch.int32, device=device)
    offsets = torch.randint(
        0, n_elements, (n_elements,), dtype=torch.int32, device=device
    )
    output = torch.zeros_like(idx1)

    BLOCK_SIZE = 64
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    with tempfile.TemporaryDirectory() as tmpdir:
        with trace(triple_indirect_load_kernel, log_dir=tmpdir):
            triple_indirect_load_kernel[grid](
                idx1, idx2, offsets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )

    expected = idx1[idx2[offsets]]
    assert torch.all(output == expected)

    cfg.sanitizer_impl = SanitizerSymbolicExecution


@triton.jit
def dual_offset_load_kernel(
    ptr,
    offset_a_ptr,
    offset_b_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.where(offsets < n_elements, offsets, 0)

    offset_a = tl.load(offset_a_ptr + offsets)
    offset_b = tl.load(offset_b_ptr + offsets)

    val = tl.load(ptr + offset_a + offset_b)

    tl.store(output_ptr + offsets, val)


def test_dual_offset_load():
    cfg.sanitizer_backend = "symexec"
    cfg.sanitizer_impl = CaptureSanitizer

    device = torch.cuda.current_device()
    n_elements = 512

    data = torch.arange(n_elements * 2, dtype=torch.int32, device=device)
    offset_a = torch.randint(
        0, n_elements, (n_elements,), dtype=torch.int32, device=device
    )
    offset_b = torch.randint(
        0, n_elements, (n_elements,), dtype=torch.int32, device=device
    )
    output = torch.zeros(n_elements, dtype=torch.int32, device=device)

    BLOCK_SIZE = 64
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    with tempfile.TemporaryDirectory() as tmpdir:
        with trace(dual_offset_load_kernel, log_dir=tmpdir):
            dual_offset_load_kernel[grid](
                data, offset_a, offset_b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )

    expected = data[offset_a + offset_b]
    assert torch.all(output == expected)

    cfg.sanitizer_impl = SanitizerSymbolicExecution


# =============================================================================
# Test address pointer arithmetic in symbolic execution
# =============================================================================


def test_addptr_expr_eval():
    base_ptr = SymbolicExpr(atom_type=AtomType.ARG, arg_id=0, shape=[1000])
    idx = SymbolicExpr(atom_type=AtomType.RANGE, size=32, start=0)
    offset = SymbolicExpr(atom_type=AtomType.CONST_INT, value=10)

    ptr = AddPtrSymbolicExecOverrider._addptr(
        ptr=base_ptr, offset=idx + offset, element_size=4
    )

    store_handles = ptr.get_handles()
    assert len(store_handles) == 1
    handle = store_handles[0]

    stride = handle.stride
    start = handle.offset_expr
    size = handle.size

    # Check stride calculation (idx coefficient)
    stride_val = stride.value()
    assert stride_val == 4  # element_size

    # Check start calculation (constant term)
    start_val = start.value()
    assert start_val == 40  # offset * element_size = 10 * 4

    # Check size (from idx range)
    assert size == 32


def test_addptr_overrider():
    base_ptr = SymbolicExpr(atom_type=AtomType.ARG, arg_id=0, shape=[1000])
    range_expr = SymbolicExpr(atom_type=AtomType.RANGE, size=64, start=0)
    const_expr = SymbolicExpr(atom_type=AtomType.CONST_INT, value=5)

    ptr = AddPtrSymbolicExecOverrider._addptr(
        ptr=base_ptr, offset=range_expr + const_expr, element_size=8
    )

    store_handles = ptr.get_handles()
    assert len(store_handles) == 1
    handle = store_handles[0]

    # Verify ptr calculation
    assert handle.stride.value() == 8  # element_size for range coefficient
    assert handle.offset_expr.value() == 40  # const * element_size = 5 * 8
    assert handle.size == 64  # from range size


# =============================================================================
# Test sanitizer initialization with different backends
# =============================================================================


def test_brute_force():
    cfg.sanitizer_backend = "brute_force"
    sanitizer = cfg.create_sanitizer(None, None)
    assert isinstance(sanitizer, SanitizerBruteForceMasking)


def test_null_sanitizer_init():
    cfg.sanitizer_backend = "off"
    sanitizer = cfg.create_sanitizer(None, None)
    assert sanitizer is None


def test_symbolic_execution():
    cfg.sanitizer_backend = "symexec"
    sanitizer = cfg.create_sanitizer(None, None)
    assert isinstance(sanitizer, SanitizerSymbolicExecution)


def test_default_sanitizer():
    # Test default sanitizer creation without explicitly setting backend
    original_backend = cfg.sanitizer_backend
    cfg.sanitizer_backend = None  # Reset to default
    sanitizer = cfg.create_sanitizer(None, None)
    # Should default to symbolic execution
    assert isinstance(sanitizer, SanitizerSymbolicExecution)
    cfg.sanitizer_backend = original_backend  # Restore original


# =============================================================================
# Test adding sanitizer client via decorator
# =============================================================================


def test_sanitizer_decorator():
    """Test that sanitizer can be added via @trace decorator."""
    import triton_viz

    @triton_viz.trace("sanitizer")
    @triton.jit
    def simple_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    # Verify the kernel is wrapped as a Trace object
    from triton_viz.core.trace import Trace

    assert isinstance(simple_kernel, Trace)

    # Verify sanitizer client was added
    clients = simple_kernel.client_manager.clients
    assert "sanitizer" in clients


def test_sanitizer_instance_decorator():
    """Test that sanitizer instance can be added via @trace decorator."""
    import triton_viz

    @triton_viz.trace(Sanitizer(abort_on_error=True))
    @triton.jit
    def simple_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))

    # Verify the kernel is wrapped as a Trace object
    from triton_viz.core.trace import Trace

    assert isinstance(simple_kernel, Trace)

    # Verify sanitizer client was added
    clients = simple_kernel.client_manager.clients
    assert "sanitizer" in clients
