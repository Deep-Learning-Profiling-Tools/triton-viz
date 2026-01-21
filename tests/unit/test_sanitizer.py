"""Unit tests for sanitizer functionality."""

import pytest
import numpy as np

import triton
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from triton_viz import trace
from triton_viz.core.config import config as cfg
from triton_viz.core.data import AddPtr, Trans
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    NullSanitizer,
    SanitizerSymbolicExecution,
)

from .conftest import (
    _build_cache_test_expression,
    _build_shared_index_expr,
    _build_layernorm_like_index_expr,
)


# ======== Init ===========
def test_init_null_sanitizer():
    try:
        cfg.disable_sanitizer = True
        s2 = Sanitizer(abort_on_error=True)
        assert isinstance(s2, NullSanitizer)
    finally:
        cfg.disable_sanitizer = False


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


# ======== Cache Ablation Tests =========

# ---- Symbol Cache Tests ----


def test_symbol_cache_consistency():
    """
    Test that Symbol Cache produces the same Z3 expressions when enabled/disabled.
    """
    original_cache_setting = cfg.enable_symbol_cache
    grid = (8, 1, 1)

    try:
        # Test with cache ENABLED
        cfg.enable_symbol_cache = True
        SymbolicExpr.ARANGE_COUNTER = 0
        expr_cache_on = _build_cache_test_expression(grid)
        z3_expr_on, constraints_on = expr_cache_on._to_z3()

        # Second call should use cache
        z3_expr_on_2, constraints_on_2 = expr_cache_on._to_z3()

        # Test with cache DISABLED
        cfg.enable_symbol_cache = False
        SymbolicExpr.ARANGE_COUNTER = 0
        expr_cache_off = _build_cache_test_expression(grid)
        z3_expr_off, constraints_off = expr_cache_off._to_z3()

        # Second call should rebuild since cache is off
        z3_expr_off_2, constraints_off_2 = expr_cache_off._to_z3()

        # Compare expressions (as strings since they're Z3 objects)
        expr_on_str = str(z3_expr_on)
        expr_off_str = str(z3_expr_off)
        constraints_on_str = sorted([str(c) for c in constraints_on])
        constraints_off_str = sorted([str(c) for c in constraints_off])

        # Verify cache-on repeated calls return same result
        assert str(z3_expr_on) == str(
            z3_expr_on_2
        ), "Cache ON: repeated calls should return same expression"
        assert sorted([str(c) for c in constraints_on]) == sorted(
            [str(c) for c in constraints_on_2]
        ), "Cache ON: repeated calls should return same constraints"

        # Verify cache-on and cache-off produce equivalent results
        assert (
            expr_on_str == expr_off_str
        ), f"Z3 expressions differ: cache ON={expr_on_str}, cache OFF={expr_off_str}"
        assert (
            constraints_on_str == constraints_off_str
        ), f"Constraints differ: cache ON={constraints_on_str}, cache OFF={constraints_off_str}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting


def test_unary_op_cache_side_effect():
    """
    Test specifically for side effects in unary ops (e.g., abs, fabs).
    Ensures that unary operations don't cause inconsistencies with cache on/off.
    """
    original_cache_setting = cfg.enable_symbol_cache

    try:
        # Test with cache ON
        cfg.enable_symbol_cache = True
        SymbolicExpr.ARANGE_COUNTER = 0

        arange_expr_on = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 10, tl.int32),
        )
        abs_expr_on = SymbolicExpr("abs", arange_expr_on)
        z3_on, constraints_on = abs_expr_on._to_z3()

        # Test with cache OFF
        cfg.enable_symbol_cache = False
        SymbolicExpr.ARANGE_COUNTER = 0

        arange_expr_off = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 10, tl.int32),
        )
        abs_expr_off = SymbolicExpr("abs", arange_expr_off)
        z3_off, constraints_off = abs_expr_off._to_z3()

        # Compare results
        assert str(z3_on) == str(
            z3_off
        ), f"Unary op Z3 expressions differ: cache ON={z3_on}, cache OFF={z3_off}"
        assert (
            sorted([str(c) for c in constraints_on])
            == sorted([str(c) for c in constraints_off])
        ), f"Unary op constraints differ: cache ON={constraints_on}, cache OFF={constraints_off}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting


def test_nested_expression_reuse():
    """
    Test the case where a sub-expression is reused in multiple places.
    This is common in kernels where the same index calculation is used
    for multiple loads/stores (like in layernorm).
    """
    original_cache_setting = cfg.enable_symbol_cache
    grid = (8, 1, 1)

    try:
        # Test with cache ENABLED - Reuse shared index expression
        cfg.enable_symbol_cache = True
        SymbolicExpr.ARANGE_COUNTER = 0

        shared_index1 = _build_shared_index_expr(grid)

        # First "load" - convert index to Z3
        z3_1, constraints_1 = shared_index1._to_z3()

        # Second "load" - should use cached result
        z3_2, constraints_2 = shared_index1._to_z3()

        # Test with cache DISABLED - Reuse shared index expression
        cfg.enable_symbol_cache = False
        SymbolicExpr.ARANGE_COUNTER = 0

        shared_index2 = _build_shared_index_expr(grid)

        # First "load"
        z3_3, constraints_3 = shared_index2._to_z3()

        # Second "load" - should rebuild since cache is off
        z3_4, constraints_4 = shared_index2._to_z3()

        # First use should be identical between cache on/off
        assert str(z3_1) == str(
            z3_3
        ), f"First use differs: cache ON={z3_1}, cache OFF={z3_3}"
        assert (
            sorted([str(c) for c in constraints_1])
            == sorted([str(c) for c in constraints_3])
        ), f"First use constraints differ: cache ON={constraints_1}, cache OFF={constraints_3}"

        # Second use should also be identical
        assert str(z3_2) == str(
            z3_4
        ), f"Second use differs: cache ON={z3_2}, cache OFF={z3_4}"
        assert (
            sorted([str(c) for c in constraints_2])
            == sorted([str(c) for c in constraints_4])
        ), f"Second use constraints differ: cache ON={constraints_2}, cache OFF={constraints_4}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting


def test_sum_operation_side_effect():
    """
    Test for side effects in operations that assign to self._constraints.
    Specifically tests the "sum" operation which does:
        arr, self._constraints = self.input._to_z3()
    """
    original_cache_setting = cfg.enable_symbol_cache

    try:
        # Test with cache ON
        cfg.enable_symbol_cache = True
        SymbolicExpr.ARANGE_COUNTER = 0

        # Build: sum(arange(0, 10) * 2, axis=0)
        arange_expr1 = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 10, tl.int32),
        )
        const_2_1 = SymbolicExpr("const", 2, tl.int32)
        mul_expr1 = SymbolicExpr("mul", arange_expr1, const_2_1)
        sum_expr1 = SymbolicExpr("sum", mul_expr1, 0, False)

        z3_on, constraints_on = sum_expr1._to_z3()

        # Test with cache OFF
        cfg.enable_symbol_cache = False
        SymbolicExpr.ARANGE_COUNTER = 0

        arange_expr2 = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 10, tl.int32),
        )
        const_2_2 = SymbolicExpr("const", 2, tl.int32)
        mul_expr2 = SymbolicExpr("mul", arange_expr2, const_2_2)
        sum_expr2 = SymbolicExpr("sum", mul_expr2, 0, False)

        z3_off, constraints_off = sum_expr2._to_z3()

        # Compare results
        assert str(z3_on) == str(
            z3_off
        ), f"Sum operation Z3 differs: cache ON={z3_on}, cache OFF={z3_off}"
        assert (
            sorted([str(c) for c in constraints_on])
            == sorted([str(c) for c in constraints_off])
        ), f"Sum operation constraints differ: cache ON={constraints_on}, cache OFF={constraints_off}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting


def test_node_state_after_single_call():
    """
    Test that node._z3 and node._constraints are consistent with cache ON/OFF
    after a single _to_z3() call.
    """
    original_cache_setting = cfg.enable_symbol_cache

    def build_expr():
        """Build expression: abs(arange(0, 10))"""
        SymbolicExpr.ARANGE_COUNTER = 0
        arange = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 10, tl.int32),
        )
        abs_expr = SymbolicExpr("abs", arange)
        return abs_expr, arange

    try:
        # Test with cache ON
        cfg.enable_symbol_cache = True
        abs_on, arange_on = build_expr()
        z3_on, constraints_on = abs_on._to_z3()

        # Test with cache OFF
        cfg.enable_symbol_cache = False
        abs_off, arange_off = build_expr()
        z3_off, constraints_off = abs_off._to_z3()

        # Check arange node state consistency
        assert str(arange_on._z3) == str(
            arange_off._z3
        ), f"arange._z3 differs: ON={arange_on._z3}, OFF={arange_off._z3}"
        assert (
            sorted([str(c) for c in arange_on._constraints])
            == sorted([str(c) for c in arange_off._constraints])
        ), f"arange._constraints differs: ON={arange_on._constraints}, OFF={arange_off._constraints}"

        # Check abs node state consistency
        assert str(abs_on._z3) == str(
            abs_off._z3
        ), f"abs._z3 differs: ON={abs_on._z3}, OFF={abs_off._z3}"
        assert (
            sorted([str(c) for c in abs_on._constraints])
            == sorted([str(c) for c in abs_off._constraints])
        ), f"abs._constraints differs: ON={abs_on._constraints}, OFF={abs_off._constraints}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting


def test_node_state_after_multiple_calls():
    """
    Test that node state remains consistent after multiple _to_z3() calls
    on the same expression, with cache ON vs OFF.
    """
    original_cache_setting = cfg.enable_symbol_cache

    def build_and_call_multiple(enable_cache):
        """Build arange expression and call _to_z3 three times."""
        SymbolicExpr.ARANGE_COUNTER = 0
        cfg.enable_symbol_cache = enable_cache

        arange = SymbolicExpr(
            "arange",
            SymbolicExpr("const", tl.int32, tl.int32),
            SymbolicExpr("const", 0, tl.int32),
            SymbolicExpr("const", 10, tl.int32),
        )

        results = []
        for _ in range(3):
            z3_expr, constraints = arange._to_z3()
            results.append((str(z3_expr), sorted([str(c) for c in constraints])))

        final_state = {
            "_z3": str(arange._z3),
            "_constraints": sorted([str(c) for c in arange._constraints]),
        }

        return results, final_state

    try:
        # Test with cache ON
        results_on, state_on = build_and_call_multiple(True)

        # Test with cache OFF
        results_off, state_off = build_and_call_multiple(False)

        # Compare all returned results from each call
        for i, (result_on, result_off) in enumerate(zip(results_on, results_off), 1):
            assert (
                result_on == result_off
            ), f"Call #{i} results differ: ON={result_on}, OFF={result_off}"

        # Compare final node states
        assert (
            state_on == state_off
        ), f"Final node state differs: ON={state_on}, OFF={state_off}"

    finally:
        cfg.enable_symbol_cache = original_cache_setting


# ---- Kernel Cache Tests ----


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _cache_test_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_kernel_cache_enables_dummy_benchmarker():
    """Test that dummy_benchmarker is installed when kernel cache is enabled."""
    original_setting = cfg.enable_kernel_cache

    try:
        cfg.enable_kernel_cache = True
        traced = trace(clients=Sanitizer())(_cache_test_add_kernel)

        # Check if dummy benchmarker was installed
        if hasattr(traced, "runner") and hasattr(traced.runner, "_do_bench"):
            bench_fn = traced.runner._do_bench
            assert (
                bench_fn is not None and bench_fn.__name__ == "dummy_benchmarker"
            ), f"Expected dummy_benchmarker when cache enabled, got: {bench_fn}"
    finally:
        cfg.enable_kernel_cache = original_setting


def test_kernel_cache_disabled_no_dummy_benchmarker():
    """Test that dummy_benchmarker is NOT installed when kernel cache is disabled."""
    original_setting = cfg.enable_kernel_cache

    try:
        cfg.enable_kernel_cache = False

        # Create a new kernel to avoid caching issues
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
            ],
            key=["n"],
        )
        @triton.jit
        def cache_disabled_add_kernel(
            x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        traced = trace(clients=Sanitizer())(cache_disabled_add_kernel)

        # Check that dummy benchmarker was NOT installed
        if hasattr(traced, "runner") and hasattr(traced.runner, "_do_bench"):
            bench_fn = traced.runner._do_bench
            assert (
                bench_fn is None or bench_fn.__name__ != "dummy_benchmarker"
            ), f"dummy_benchmarker should not be installed when cache disabled, but found: {bench_fn}"
    finally:
        cfg.enable_kernel_cache = original_setting


def test_kernel_cache_autotune_with_dummy_benchmarker():
    """
    Test that kernel cache enabled installs dummy_benchmarker on autotuned kernels.
    """
    original_setting = cfg.enable_kernel_cache

    try:
        cfg.enable_kernel_cache = True

        # Create a fresh autotuned kernel inside the test to avoid state corruption
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
                triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
                triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def autotune_add_kernel_cache_on(
            x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        traced_kernel = trace(clients=Sanitizer())(autotune_add_kernel_cache_on)

        # Verify dummy benchmarker is installed
        if hasattr(traced_kernel, "runner") and hasattr(
            traced_kernel.runner, "_do_bench"
        ):
            bench_fn = traced_kernel.runner._do_bench
            assert (
                bench_fn is not None and bench_fn.__name__ == "dummy_benchmarker"
            ), f"Expected dummy_benchmarker when cache enabled, got: {bench_fn}"

    finally:
        cfg.enable_kernel_cache = original_setting


def test_kernel_cache_autotune_real_benchmarker():
    """
    Test that kernel cache disabled does NOT install dummy_benchmarker on autotuned kernels.
    """
    original_setting = cfg.enable_kernel_cache

    try:
        cfg.enable_kernel_cache = False

        # Create a fresh autotuned kernel to avoid cached config
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
                triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def fresh_add_kernel(
            x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        traced_kernel = trace(clients=Sanitizer())(fresh_add_kernel)

        # Verify dummy_benchmarker is NOT installed
        if hasattr(traced_kernel, "runner") and hasattr(
            traced_kernel.runner, "_do_bench"
        ):
            bench_fn = traced_kernel.runner._do_bench
            assert (
                bench_fn is None or bench_fn.__name__ != "dummy_benchmarker"
            ), "dummy_benchmarker should not be installed when cache disabled"

    finally:
        cfg.enable_kernel_cache = original_setting


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
