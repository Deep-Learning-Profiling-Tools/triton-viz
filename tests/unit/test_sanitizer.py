"""Unit tests for sanitizer functionality."""

import pytest
import numpy as np

import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from triton_viz.core.config import config as cfg
from triton_viz.core.data import AddPtr, Trans
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    SanitizerSymbolicExecution,
)

from .conftest import (
    _build_layernorm_like_index_expr,
)


# ======== Init ===========
def test_init_symbolic_execution():
    s3 = Sanitizer(abort_on_error=True)
    assert isinstance(s3, SanitizerSymbolicExecution) and s3.abort_on_error is True


def test_init_default_sanitizer():
    from triton_viz.core.client import Client

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
    offset_th = TensorHandle(np.array([3]).astype(np.int32), tl.int32)
    sanitizer = SanitizerSymbolicExecution(abort_on_error=False)
    op_callbacks = sanitizer.register_op_callback(AddPtr)
    assert op_callbacks.op_overrider is not None
    expr = op_callbacks.op_overrider(ptr_th, offset_th)  # offset = 3
    assert expr.op == "addptr"
    assert expr.eval()[0] == 1000 + 3 * 4  # element_bytewidth = 4


# ======== Trans =========


def test_trans_overrider_passthrough_and_dtype():
    base = SymbolicExpr.from_value((1, 2, 3))
    base.dtype_tt = tl.float32  # ensure dtype is preserved through trans

    sanitizer = SanitizerSymbolicExecution(abort_on_error=False)
    op_callbacks = sanitizer.register_op_callback(Trans)
    assert op_callbacks.op_overrider is not None

    perm = (1, 0)
    trans_expr = op_callbacks.op_overrider(base, perm)

    assert isinstance(trans_expr, SymbolicExpr)
    assert trans_expr.op == "trans"
    assert trans_expr.arg is base
    assert trans_expr.permutation.op == "const"
    assert trans_expr.permutation.to_py() == perm
    assert trans_expr.dtype_tt == base.dtype_tt

    def _z3_to_int_list(values):
        vals = values if isinstance(values, list) else [values]
        return [v.as_long() if hasattr(v, "as_long") else int(str(v)) for v in vals]

    trans_vals, trans_constraints = trans_expr.eval()
    base_vals, base_constraints = base.eval()

    assert _z3_to_int_list(trans_vals) == _z3_to_int_list(base_vals)
    assert trans_constraints == base_constraints


# ======== Miscellaneous =========


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


# ======== Sanitizer Backend Tests =========


def test_switch_backend():
    """Switch back and forth at runtime."""
    original = cfg.disable_sanitizer

    cfg.disable_sanitizer = False  # Enable sanitizer
    assert cfg.disable_sanitizer is False

    cfg.disable_sanitizer = True  # Disable sanitizer
    assert cfg.disable_sanitizer is True

    cfg.disable_sanitizer = original


def test_invalid_backend_raises():
    """Setting a non-boolean value should raise TypeError."""
    with pytest.raises(TypeError):
        cfg.disable_sanitizer = "not_a_bool"


# ======== Reduce Operations (max, min) =========


def test_reduce_max_expr_eval():
    """Test that max reduce operation evaluates correctly."""
    # Test with array of values - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr("const", np.array([1, 5, 3, 2]), tl.int32)
    max_expr = SymbolicExpr("max", input_arr, None, False)
    result, _ = max_expr.eval()
    assert result == 5


def test_reduce_min_expr_eval():
    """Test that min reduce operation evaluates correctly."""
    # Test with array of values - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr("const", np.array([1, 5, 3, 2]), tl.int32)
    min_expr = SymbolicExpr("min", input_arr, None, False)
    result, _ = min_expr.eval()
    assert result == 1


def test_reduce_max_single_element():
    """Test that max reduce operation works with single element."""
    # Test with single element - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr("const", np.array([42]), tl.int32)
    max_expr = SymbolicExpr("max", input_arr, None, False)
    result, _ = max_expr.eval()
    assert result == 42


def test_reduce_min_single_element():
    """Test that min reduce operation works with single element."""
    # Test with single element - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr("const", np.array([42]), tl.int32)
    min_expr = SymbolicExpr("min", input_arr, None, False)
    result, _ = min_expr.eval()
    assert result == 42


def test_reduce_max_empty_array():
    """Test that max reduce operation raises ValueError for empty array."""
    input_arr = SymbolicExpr("const", np.array([]), tl.int32)
    max_expr = SymbolicExpr("max", input_arr, None, False)
    with pytest.raises(ValueError, match="Cannot compute max of empty array"):
        max_expr.eval()


def test_reduce_min_empty_array():
    """Test that min reduce operation raises ValueError for empty array."""
    input_arr = SymbolicExpr("const", np.array([]), tl.int32)
    min_expr = SymbolicExpr("min", input_arr, None, False)
    with pytest.raises(ValueError, match="Cannot compute min of empty array"):
        min_expr.eval()


# ---- Z3 Expression Comparison Tests ----


def test_layernorm_like_memory_access_pattern():
    """
    Test that simulates the layernorm kernel scenario with multiple loads/stores
    sharing the same index expressions. Verifies Z3 expression consistency
    with cache ON/OFF.
    """
    original_cache_setting = cfg.enable_symbol_cache

    def test_with_cache_mode(enable_cache: bool):
        """Run test with specified cache mode and return results."""
        cfg.enable_symbol_cache = enable_cache
        SymbolicExpr.ARANGE_COUNTER = 0

        index_expr = _build_layernorm_like_index_expr()

        # Simulate multiple uses of the same index (like multiple tl.load/tl.store)
        results = []
        for _ in range(3):
            z3_expr, constraints = index_expr._to_z3()
            results.append((str(z3_expr), sorted([str(c) for c in constraints])))

        return results

    try:
        # Test with cache ON
        results_cache_on = test_with_cache_mode(True)

        # Test with cache OFF
        results_cache_off = test_with_cache_mode(False)

        # Compare all access results
        for i, (r_on, r_off) in enumerate(zip(results_cache_on, results_cache_off)):
            expr_on, constraints_on = r_on
            expr_off, constraints_off = r_off

            assert (
                expr_on == expr_off
            ), f"Access #{i+1} expression differs: cache ON={expr_on}, cache OFF={expr_off}"
            assert (
                constraints_on == constraints_off
            ), f"Access #{i+1} constraints differ: cache ON has {len(constraints_on)}, cache OFF has {len(constraints_off)}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting


def test_z3_expression_capture_comparison():
    """
    Test that Z3 expressions captured with Symbol Cache ON vs OFF produce identical
    final results for the same symbolic expression trees.

    Note: Cache ON will have fewer _to_z3 calls (due to caching), but the final
    expressions should be identical. This test compares the unique expressions.
    """
    original_cache_setting = cfg.enable_symbol_cache

    def capture_final_z3_expressions(enable_cache: bool):
        """
        Run a series of symbolic operations and capture the FINAL Z3 expressions
        (not intermediate calls, as caching affects call count).
        """
        cfg.enable_symbol_cache = enable_cache
        SymbolicExpr.ARANGE_COUNTER = 0

        # Build a complex expression tree similar to real kernel patterns
        grid = (8, 1, 1)

        # Expression 1: arange + pid * block_size
        arange1 = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 16, tl.int32),
        )
        pid1 = SymbolicExpr(
            "pid",
            SymbolicExpr("const", grid, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
        )
        block_size1 = SymbolicExpr("const", 16, tl.int32)
        offset1 = SymbolicExpr("add", arange1, SymbolicExpr("mul", pid1, block_size1))

        # Expression 2: nested operations (abs, sum)
        arange2 = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 8, tl.int32),
        )
        abs_expr = SymbolicExpr("abs", arange2)
        sum_expr = SymbolicExpr("sum", abs_expr, 0, False)

        # Expression 3: binary operations chain
        const_a = SymbolicExpr("const", 2, tl.int32)
        const_b = SymbolicExpr("const", 3, tl.int32)
        mul_expr = SymbolicExpr("mul", offset1, const_a)
        add_expr = SymbolicExpr("add", mul_expr, const_b)

        # Get final Z3 expressions
        results = {}

        z3_offset1, constraints_offset1 = offset1._to_z3()
        results["offset1"] = {
            "expr": str(z3_offset1),
            "constraints": sorted([str(c) for c in constraints_offset1]),
        }

        z3_sum, constraints_sum = sum_expr._to_z3()
        results["sum"] = {
            "expr": str(z3_sum),
            "constraints": sorted([str(c) for c in constraints_sum]),
        }

        z3_add, constraints_add = add_expr._to_z3()
        results["add"] = {
            "expr": str(z3_add),
            "constraints": sorted([str(c) for c in constraints_add]),
        }

        return results

    try:
        # Capture with cache ON
        results_cache_on = capture_final_z3_expressions(True)

        # Capture with cache OFF
        results_cache_off = capture_final_z3_expressions(False)

        # Compare final expressions
        for key in results_cache_on:
            expr_on = results_cache_on[key]
            expr_off = results_cache_off[key]

            assert (
                expr_on["expr"] == expr_off["expr"]
            ), f"Expression '{key}' differs: cache ON={expr_on['expr']}, cache OFF={expr_off['expr']}"

            assert (
                expr_on["constraints"] == expr_off["constraints"]
            ), f"Constraints for '{key}' differ: cache ON={expr_on['constraints']}, cache OFF={expr_off['constraints']}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting
