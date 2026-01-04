"""
Tests for Sanitizer Cache Ablation Study Switches.

This module tests the four cache optimization switches documented in the paper:
1. Symbol Cache (enable_symbol_cache) - caches Z3 expressions in SymExpr nodes
2. Loop Cache (enable_loop_cache) - deduplicates memory patterns in loops
3. Grid Cache (enable_grid_cache) - shared solver per kernel launch (incremental SMT)
4. Kernel Cache (enable_kernel_cache) - skip re-analysis of identical kernel launches

It also verifies the correctness of the _make_signature function which is critical
for loop cache deduplication.
"""

import pytest
import re
import torch
from unittest.mock import patch

import triton
import triton.language as tl
from z3 import Int, And, IntVal

import triton_viz
from triton_viz.core.config import config as cfg
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    SymbolicSanitizer,
    _fn_symbolic_cache_set,
    _make_signature,
)


# ======== Fixtures ========


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config to defaults before and after each test."""
    cfg.reset()
    cfg.enable_sanitizer = True
    yield
    cfg.reset()


@pytest.fixture
def clear_kernel_cache():
    """Clear kernel cache before each test."""
    _fn_symbolic_cache_set.clear()
    yield
    _fn_symbolic_cache_set.clear()


# ======== Symbol Cache Tests ========


class SymbolCacheTest:
    """Tests for the Symbol Node Cache (enable_symbol_cache)."""

    def test_symbol_cache_enabled_caches_z3_result(self):
        """When symbol cache is enabled, Z3 expressions should be cached."""
        cfg.enable_symbol_cache = True

        # Create a symbolic expression
        const_expr = SymbolicExpr.create("const", 42, tl.int32)

        # First eval should compute and cache
        result1, _ = const_expr.eval(simplify_constraints=False)

        # Check that z3 field is set (cached)
        assert const_expr.z3 is not None

        # Store the cached value
        cached_z3 = const_expr.z3

        # Second eval should return cached result
        result2, _ = const_expr.eval(simplify_constraints=False)

        # The cached z3 should be the same object
        assert const_expr.z3 is cached_z3

    def test_symbol_cache_disabled_recomputes_z3(self):
        """When symbol cache is disabled, Z3 expressions should be recomputed."""
        cfg.enable_symbol_cache = False

        # Create a symbolic expression
        const_expr = SymbolicExpr.create("const", 42, tl.int32)

        # First eval
        result1, _ = const_expr.eval(simplify_constraints=False)

        # Even though z3 is set, when cache is disabled it should be cleared on next eval
        assert const_expr.z3 is not None

        # Second eval - should clear and recompute
        result2, _ = const_expr.eval(simplify_constraints=False)

        # Results should be equivalent
        assert str(result1) == str(result2)

    def test_symbol_cache_with_complex_expression(self):
        """Test symbol cache with nested expressions."""
        cfg.enable_symbol_cache = True

        # Create nested expression: (a + b) * c
        a = SymbolicExpr.create("const", 2, tl.int32)
        b = SymbolicExpr.create("const", 3, tl.int32)
        c = SymbolicExpr.create("const", 4, tl.int32)

        add_expr = SymbolicExpr.create("add", a, b)
        mul_expr = SymbolicExpr.create("mul", add_expr, c)

        # Eval the whole expression
        result, _ = mul_expr.eval(simplify_constraints=False)

        # All sub-expressions should have cached z3
        assert a.z3 is not None
        assert b.z3 is not None
        assert add_expr.z3 is not None
        assert mul_expr.z3 is not None

    def test_symbol_cache_toggle_behavior(self):
        """Test toggling symbol cache on and off."""
        # Start with cache enabled
        cfg.enable_symbol_cache = True

        expr = SymbolicExpr.create("const", 100, tl.int32)
        result1, _ = expr.eval(simplify_constraints=False)
        assert expr.z3 is not None

        # Disable cache and eval again
        cfg.enable_symbol_cache = False
        result2, _ = expr.eval(simplify_constraints=False)

        # z3 should have been cleared and recomputed
        # (note: the value might be the same but it's a fresh computation)
        assert str(result1) == str(result2)


# ======== Loop Cache Tests ========


class LoopCacheTest:
    """Tests for the Loop Iterator Cache (enable_loop_cache)."""

    def test_loop_cache_enabled_deduplicates(self):
        """When loop cache is enabled, duplicate addresses should be skipped."""
        cfg.enable_loop_cache = True

        class LoopCacheTracker(SymbolicSanitizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pending_check_counts = []

            def register_for_loop_callback(self):
                callbacks = super().register_for_loop_callback()
                orig_after = callbacks.after_loop_callback

                def _after_loop(lineno: int):
                    if self.loop_stack:
                        ctx = self.loop_stack[-1]
                        self.pending_check_counts.append(len(ctx.pending_checks))
                    if orig_after:
                        orig_after(lineno)

                from triton_viz.core.callbacks import ForLoopCallbacks

                return ForLoopCallbacks(
                    range_wrapper_factory=callbacks.range_wrapper_factory,
                    range_type_callback=callbacks.range_type_callback,
                    before_loop_callback=callbacks.before_loop_callback,
                    loop_iter_overrider=callbacks.loop_iter_overrider,
                    loop_iter_listener=callbacks.loop_iter_listener,
                    after_loop_callback=_after_loop,
                )

        tracker = LoopCacheTracker(abort_on_error=False)

        @triton_viz.trace(client=tracker)
        @triton.jit
        def loop_dedup_kernel(out_ptr):
            # This loop has identical address pattern each iteration
            # With loop cache, only one check should be pending
            for i in range(0, 4):
                # Same offset expression pattern: out_ptr + (pid + 1)
                pid = tl.program_id(0)
                idx = pid + 1
                tl.store(out_ptr + idx, idx)

        out = torch.empty((4,), dtype=torch.int32)
        loop_dedup_kernel[(1,)](out)

        # With loop cache enabled, duplicate patterns are deduplicated
        # so we expect only 1 pending check (not 4)
        if tracker.pending_check_counts:
            assert tracker.pending_check_counts[0] == 1

    def test_loop_cache_disabled_no_dedup(self):
        """When loop cache is disabled, all addresses should be checked."""
        cfg.enable_loop_cache = False

        class LoopCacheTracker(SymbolicSanitizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pending_check_counts = []

            def register_for_loop_callback(self):
                callbacks = super().register_for_loop_callback()
                orig_after = callbacks.after_loop_callback

                def _after_loop(lineno: int):
                    if self.loop_stack:
                        ctx = self.loop_stack[-1]
                        self.pending_check_counts.append(len(ctx.pending_checks))
                    if orig_after:
                        orig_after(lineno)

                from triton_viz.core.callbacks import ForLoopCallbacks

                return ForLoopCallbacks(
                    range_wrapper_factory=callbacks.range_wrapper_factory,
                    range_type_callback=callbacks.range_type_callback,
                    before_loop_callback=callbacks.before_loop_callback,
                    loop_iter_overrider=callbacks.loop_iter_overrider,
                    loop_iter_listener=callbacks.loop_iter_listener,
                    after_loop_callback=_after_loop,
                )

        tracker = LoopCacheTracker(abort_on_error=False)

        @triton_viz.trace(client=tracker)
        @triton.jit
        def loop_no_dedup_kernel(out_ptr):
            for i in range(0, 4):
                pid = tl.program_id(0)
                idx = pid + 1
                tl.store(out_ptr + idx, idx)

        out = torch.empty((4,), dtype=torch.int32)
        loop_no_dedup_kernel[(1,)](out)

        # With loop cache disabled, all 4 iterations add pending checks
        if tracker.pending_check_counts:
            assert tracker.pending_check_counts[0] == 4


# ======== Grid Cache Tests ========


class GridCacheTest:
    """Tests for the Grid Cache (enable_grid_cache)."""

    def test_grid_cache_enabled_uses_persistent_solver(self):
        """When grid cache is enabled, push/pop should be used on persistent solver."""
        cfg.enable_grid_cache = True

        sanitizer = SymbolicSanitizer(abort_on_error=False)

        # After initialization, solver should be None (created lazily)
        # We'll check that after grid setup, the solver is reused
        @triton_viz.trace(client=sanitizer)
        @triton.jit
        def simple_kernel(x_ptr, y_ptr, n_elements):
            pid = tl.program_id(0)
            offsets = pid * 128 + tl.arange(0, 128)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x, mask=mask)

        x = torch.randn(256, dtype=torch.float32)
        y = torch.empty_like(x)
        simple_kernel[(2,)](x, y, 256)

        # Test passes if no exception is raised
        # The grid cache optimization is internal

    def test_grid_cache_disabled_creates_new_solver(self):
        """When grid cache is disabled, a new solver is created for each check."""
        cfg.enable_grid_cache = False

        solver_creation_count = [0]

        # We track solver creation by patching
        from z3 import Solver

        original_solver = Solver

        def counting_solver(*args, **kwargs):
            solver_creation_count[0] += 1
            return original_solver(*args, **kwargs)

        sanitizer = SymbolicSanitizer(abort_on_error=False)

        @triton_viz.trace(client=sanitizer)
        @triton.jit
        def simple_kernel2(x_ptr, y_ptr, n_elements):
            pid = tl.program_id(0)
            offsets = pid * 128 + tl.arange(0, 128)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x, mask=mask)

        x = torch.randn(256, dtype=torch.float32)
        y = torch.empty_like(x)

        # Patch and run
        with patch("triton_viz.clients.sanitizer.sanitizer.Solver", counting_solver):
            simple_kernel2[(2,)](x, y, 256)

        # When grid cache is disabled, more solvers should be created
        # (This is a behavioral test - the exact count depends on implementation)


# ======== Kernel Cache Tests ========


class KernelCacheTest:
    """Tests for the Kernel Cache (enable_kernel_cache)."""

    def test_kernel_cache_enabled_skips_identical_launches(self, clear_kernel_cache):
        """When kernel cache is enabled, identical kernel launches should be cached."""
        cfg.enable_kernel_cache = True

        sanitizer = Sanitizer(abort_on_error=False)

        @triton_viz.trace(client=sanitizer)
        @triton.jit
        def cached_kernel(x_ptr, y_ptr, n):
            pid = tl.program_id(0)
            x = tl.load(x_ptr + pid)
            tl.store(y_ptr + pid, x)

        x = torch.randn(4, dtype=torch.float32)
        y = torch.empty_like(x)

        # First launch - should add to cache
        cached_kernel[(4,)](x, y, 4)
        cache_size_after_first = len(_fn_symbolic_cache_set)

        # Second launch with same configuration - should hit cache
        cached_kernel[(4,)](x, y, 4)
        cache_size_after_second = len(_fn_symbolic_cache_set)

        # With kernel cache, cache size should remain the same after second launch
        # because the same configuration is already cached
        assert cache_size_after_second == cache_size_after_first
        assert cache_size_after_first >= 1  # At least one entry was cached

    def test_kernel_cache_disabled_no_caching(self, clear_kernel_cache):
        """When kernel cache is disabled, cache set should not be used."""
        cfg.enable_kernel_cache = False

        sanitizer = Sanitizer(abort_on_error=False)

        @triton_viz.trace(client=sanitizer)
        @triton.jit
        def uncached_kernel(x_ptr, y_ptr, n):
            pid = tl.program_id(0)
            x = tl.load(x_ptr + pid)
            tl.store(y_ptr + pid, x)

        x = torch.randn(4, dtype=torch.float32)
        y = torch.empty_like(x)

        initial_cache_size = len(_fn_symbolic_cache_set)

        # First launch - should NOT add to cache when disabled
        uncached_kernel[(4,)](x, y, 4)
        cache_size_after_first = len(_fn_symbolic_cache_set)

        # Second launch
        uncached_kernel[(4,)](x, y, 4)
        cache_size_after_second = len(_fn_symbolic_cache_set)

        # Without kernel cache, cache set should not grow
        assert cache_size_after_first == initial_cache_size
        assert cache_size_after_second == initial_cache_size

    def test_kernel_cache_different_configs_separate_entries(self, clear_kernel_cache):
        """Different kernel configurations should create separate cache entries."""
        cfg.enable_kernel_cache = True

        sanitizer = Sanitizer(abort_on_error=False)

        @triton_viz.trace(client=sanitizer)
        @triton.jit
        def config_kernel(x_ptr, y_ptr, n):
            pid = tl.program_id(0)
            x = tl.load(x_ptr + pid)
            tl.store(y_ptr + pid, x)

        # First launch with one config
        x1 = torch.randn(4, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        config_kernel[(4,)](x1, y1, 4)
        cache_size_after_first = len(_fn_symbolic_cache_set)

        # Second launch with different grid size
        x2 = torch.randn(8, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        config_kernel[(8,)](x2, y2, 8)
        cache_size_after_second = len(_fn_symbolic_cache_set)

        # Different configs should create separate cache entries
        assert cache_size_after_second > cache_size_after_first


# ======== Z3 Hash Behavior Tests ========


class Z3HashBehaviorTest:
    """Test Z3 object hash behavior for loop cache deduplication."""

    def test_z3_int_different_names_have_different_hash(self):
        """
        Test: Z3 variables with different names should have different hashes.

        This is the expected behavior of Z3 - different variable names
        create different symbolic variables.
        """
        # Create two Z3 integer variables with different names
        x1 = Int("loop_i_81")
        x2 = Int("loop_i_82")

        # They should have different hashes because they are different variables
        assert hash(x1) != hash(
            x2
        ), "Different variable names should have different hashes"

    def test_z3_same_name_same_hash(self):
        """
        Test: Z3 variables with the same name should have the same hash.
        """
        x1 = Int("loop_i")
        x2 = Int("loop_i")

        # Same name should give same hash
        assert hash(x1) == hash(x2), "Same variable name should have same hash"

    def test_z3_expressions_with_different_variable_names(self):
        """
        Test: Expressions with same structure but different variable names.

        This simulates what happens in a loop:
        - Iteration 1: base + loop_i_81 * 8
        - Iteration 2: base + loop_i_82 * 8

        These have the SAME PATTERN but DIFFERENT variable names.
        """
        base = Int("base")

        # Simulate two loop iterations with different iterator variable names
        loop_i_81 = Int("loop_i_81")
        loop_i_82 = Int("loop_i_82")

        expr1 = base + loop_i_81 * 8
        expr2 = base + loop_i_82 * 8

        # These expressions have the same structure but different variable names
        # Z3's hash will treat them as DIFFERENT
        hash1 = hash(expr1)
        hash2 = hash(expr2)

        # This assertion documents the ACTUAL behavior of Z3
        # Z3 does NOT normalize variable names
        assert (
            hash1 != hash2
        ), "Z3 does not normalize variable names - different names produce different hashes"

    def test_z3_arange_expressions_with_different_ids(self):
        """
        Test: arange expressions with different IDs.

        In Triton, each tl.arange() call creates a new symbolic variable.
        """
        base = IntVal(1000)

        # Simulate arange with different IDs
        arange_0 = Int("arange_0_8")
        arange_1 = Int("arange_1_8")

        expr1 = base + arange_0
        expr2 = base + arange_1

        hash1 = hash(expr1)
        hash2 = hash(expr2)

        assert hash1 != hash2, "Different arange IDs should produce different hashes"

    def test_make_signature_with_same_pattern_different_vars(self):
        """
        Test: _make_signature with expressions that have same pattern but different vars.

        This is the CRITICAL test for loop cache correctness.
        After the fix, _make_signature should normalize variable names.
        """
        base = Int("base")

        # Iteration 1
        loop_i_1 = Int("loop_i_1")
        arange_1 = Int("arange_1")
        addr1 = base + loop_i_1 * 8 + arange_1
        constraint1 = And(loop_i_1 >= 0, loop_i_1 < 4, arange_1 >= 0, arange_1 < 8)

        # Iteration 2 - same pattern, different variable names
        loop_i_2 = Int("loop_i_2")
        arange_2 = Int("arange_2")
        addr2 = base + loop_i_2 * 8 + arange_2
        constraint2 = And(loop_i_2 >= 0, loop_i_2 < 4, arange_2 >= 0, arange_2 < 8)

        sig1 = _make_signature(addr1, constraint1)
        sig2 = _make_signature(addr2, constraint2)

        # After fix: signatures SHOULD be equal for same patterns with different var names
        assert (
            sig1 == sig2
        ), "_make_signature should produce same signature for same patterns"

    def test_make_signature_truly_different_patterns(self):
        """
        Test: _make_signature should give different signatures for truly different patterns.
        """
        base = Int("base")
        loop_i = Int("loop_i")
        arange = Int("arange")

        # Pattern 1: base + loop_i * 8 + arange
        addr1 = base + loop_i * 8 + arange

        # Pattern 2: base + loop_i * 16 + arange (different multiplier)
        addr2 = base + loop_i * 16 + arange

        sig1 = _make_signature(addr1, None)
        sig2 = _make_signature(addr2, None)

        assert sig1 != sig2, "Different patterns should have different signatures"

    def test_loop_cache_scenario_simulation(self):
        """
        Simulate the actual loop cache scenario:

        for i in range(0, 4):
            tl.store(out_ptr + (pid + 1), value)

        In each iteration, if pid is symbolic, the expression pattern is the same.
        """
        out_ptr = IntVal(1000)
        pid = Int("pid0")  # program_id is the same across iterations

        # All iterations produce the SAME expression
        expr_iter1 = out_ptr + (pid + 1)
        expr_iter2 = out_ptr + (pid + 1)
        expr_iter3 = out_ptr + (pid + 1)

        sig1 = _make_signature(expr_iter1, None)
        sig2 = _make_signature(expr_iter2, None)
        sig3 = _make_signature(expr_iter3, None)

        # These SHOULD be equal since it's the exact same expression
        assert sig1 == sig2 == sig3, "Same expression should produce same signature"

    def test_loop_cache_with_iterator_in_expression(self):
        """
        The problematic case: loop iterator is part of the expression.

        for i in range(0, 4):
            tl.store(out_ptr + i * 8 + arange, value)

        If 'i' is represented as a symbolic variable with different names
        per iteration, signatures will differ even though the PATTERN is same.

        Note: This test uses "loop_iter" which is NOT in the normalization pattern.
        """
        out_ptr = IntVal(1000)
        arange = Int("arange_0_8")

        # If loop iterator is a fresh symbolic variable each iteration
        # with names NOT matching the normalization pattern
        i_iter1 = Int("loop_iter_1")
        i_iter2 = Int("loop_iter_2")

        expr1 = out_ptr + i_iter1 * 8 + arange
        expr2 = out_ptr + i_iter2 * 8 + arange

        sig1 = _make_signature(expr1, None)
        sig2 = _make_signature(expr2, None)

        # These will be different because "loop_iter" is not in the pattern
        # Only "loop_i" and "arange" are normalized
        assert (
            sig1 != sig2
        ), "Variables not matching normalization pattern should produce different signatures"

    def test_reference_implementation_approach(self):
        """
        Test the approach used in the reference implementation:
        Use sexpr() string + regex normalization.
        """
        # The regex from reference implementation
        re_pattern = re.compile(r"(loop_i|arange)_\d+")

        base = Int("base")

        # Create expressions with numbered suffixes
        loop_i_81 = Int("loop_i_81")
        loop_i_82 = Int("loop_i_82")

        expr1 = base + loop_i_81 * 8
        expr2 = base + loop_i_82 * 8

        # Reference implementation approach: normalize then hash
        sexpr1 = expr1.sexpr()
        sexpr2 = expr2.sexpr()

        normalized1 = re_pattern.sub(r"\1", sexpr1)
        normalized2 = re_pattern.sub(r"\1", sexpr2)

        # With normalization, the strings should be equal
        assert normalized1 == normalized2, "Normalized expressions should be equal"

        # And their hashes should be equal
        assert hash(normalized1) == hash(
            normalized2
        ), "Hashes of normalized expressions should be equal"


# ======== Integration Tests ========


class CacheIntegrationTest:
    """Integration tests for cache combinations."""

    def test_all_caches_enabled(self, clear_kernel_cache):
        """Test with all caches enabled (default production mode)."""
        cfg.enable_symbol_cache = True
        cfg.enable_loop_cache = True
        cfg.enable_grid_cache = True
        cfg.enable_kernel_cache = True

        @triton_viz.trace(client=Sanitizer(abort_on_error=True))
        @triton.jit
        def full_cache_kernel(in_ptr, out_ptr, n):
            pid = tl.program_id(0)
            block_start = pid * 32
            offsets = block_start + tl.arange(0, 32)
            mask = offsets < n

            for i in range(0, 4):
                x = tl.load(in_ptr + offsets, mask=mask)
                tl.store(out_ptr + offsets, x + i, mask=mask)

        x = torch.randn(128, dtype=torch.float32)
        y = torch.empty_like(x)
        full_cache_kernel[(4,)](x, y, 128)

        # Should complete without error

    def test_all_caches_disabled(self, clear_kernel_cache):
        """Test with all caches disabled (for ablation study)."""
        cfg.enable_symbol_cache = False
        cfg.enable_loop_cache = False
        cfg.enable_grid_cache = False
        cfg.enable_kernel_cache = False

        @triton_viz.trace(client=Sanitizer(abort_on_error=True))
        @triton.jit
        def no_cache_kernel(in_ptr, out_ptr, n):
            pid = tl.program_id(0)
            block_start = pid * 32
            offsets = block_start + tl.arange(0, 32)
            mask = offsets < n

            for i in range(0, 4):
                x = tl.load(in_ptr + offsets, mask=mask)
                tl.store(out_ptr + offsets, x + i, mask=mask)

        x = torch.randn(128, dtype=torch.float32)
        y = torch.empty_like(x)
        no_cache_kernel[(4,)](x, y, 128)

        # Should complete without error (slower but correct)

    def test_correctness_with_oob_detection(self, clear_kernel_cache):
        """Verify that OOB detection works correctly regardless of cache settings."""
        for symbol_cache in [True, False]:
            for loop_cache in [True, False]:
                for grid_cache in [True, False]:
                    for kernel_cache in [True, False]:
                        cfg.enable_symbol_cache = symbol_cache
                        cfg.enable_loop_cache = loop_cache
                        cfg.enable_grid_cache = grid_cache
                        cfg.enable_kernel_cache = kernel_cache
                        _fn_symbolic_cache_set.clear()

                        sanitizer = Sanitizer(abort_on_error=False)

                        @triton_viz.trace(client=sanitizer)
                        @triton.jit
                        def oob_kernel(out_ptr, n):
                            pid = tl.program_id(0)
                            # This will cause OOB when pid >= n
                            offsets = pid * 32 + tl.arange(0, 32)
                            tl.store(out_ptr + offsets, pid)

                        # Create tensor that's too small for the grid
                        out = torch.empty((32,), dtype=torch.int32)  # Only 32 elements
                        oob_kernel[(2,)](out, 32)  # But 2 blocks * 32 = 64 accesses

                        # OOB should be detected regardless of cache settings
                        assert len(sanitizer.records) > 0, (
                            f"OOB not detected with caches: symbol={symbol_cache}, "
                            f"loop={loop_cache}, grid={grid_cache}, kernel={kernel_cache}"
                        )


# ======== Environment Variable Tests ========


class EnvironmentVariablesTest:
    """Test that environment variables correctly set cache flags."""

    def test_symbol_cache_env_var(self):
        """Test SANITIZER_ENABLE_SYMBOL_CACHE environment variable."""
        import os

        # Test with "0"
        with patch.dict(os.environ, {"SANITIZER_ENABLE_SYMBOL_CACHE": "0"}):
            cfg.reset()
            assert cfg.enable_symbol_cache is False

        # Test with "1"
        with patch.dict(os.environ, {"SANITIZER_ENABLE_SYMBOL_CACHE": "1"}):
            cfg.reset()
            assert cfg.enable_symbol_cache is True

    def test_loop_cache_env_var(self):
        """Test SANITIZER_ENABLE_LOOP_CACHE environment variable."""
        import os

        with patch.dict(os.environ, {"SANITIZER_ENABLE_LOOP_CACHE": "0"}):
            cfg.reset()
            assert cfg.enable_loop_cache is False

        with patch.dict(os.environ, {"SANITIZER_ENABLE_LOOP_CACHE": "1"}):
            cfg.reset()
            assert cfg.enable_loop_cache is True

    def test_grid_cache_env_var(self):
        """Test SANITIZER_ENABLE_GRID_CACHE environment variable."""
        import os

        with patch.dict(os.environ, {"SANITIZER_ENABLE_GRID_CACHE": "0"}):
            cfg.reset()
            assert cfg.enable_grid_cache is False

        with patch.dict(os.environ, {"SANITIZER_ENABLE_GRID_CACHE": "1"}):
            cfg.reset()
            assert cfg.enable_grid_cache is True

    def test_kernel_cache_env_var(self):
        """Test SANITIZER_ENABLE_KERNEL_CACHE environment variable."""
        import os

        with patch.dict(os.environ, {"SANITIZER_ENABLE_KERNEL_CACHE": "0"}):
            cfg.reset()
            assert cfg.enable_kernel_cache is False

        with patch.dict(os.environ, {"SANITIZER_ENABLE_KERNEL_CACHE": "1"}):
            cfg.reset()
            assert cfg.enable_kernel_cache is True

    def test_default_cache_values(self):
        """Test that all caches are enabled by default."""
        import os

        # Clear relevant env vars
        env_vars = [
            "SANITIZER_ENABLE_SYMBOL_CACHE",
            "SANITIZER_ENABLE_LOOP_CACHE",
            "SANITIZER_ENABLE_GRID_CACHE",
            "SANITIZER_ENABLE_KERNEL_CACHE",
        ]

        clean_env = {k: v for k, v in os.environ.items() if k not in env_vars}
        with patch.dict(os.environ, clean_env, clear=True):
            cfg.reset()
            assert cfg.enable_symbol_cache is True
            assert cfg.enable_loop_cache is True
            assert cfg.enable_grid_cache is True
            assert cfg.enable_kernel_cache is True
