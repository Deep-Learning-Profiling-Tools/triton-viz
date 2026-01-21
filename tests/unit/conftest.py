"""Shared fixtures and helper functions for unit tests."""

import triton.language as tl
from triton_viz.clients.sanitizer.sanitizer import SymbolicExpr


def _build_cache_test_expression(grid):
    """
    Helper to build a common test expression: arange(0, 10) + pid_0 * 16
    This simulates a typical memory access pattern.
    """
    arange_expr = SymbolicExpr(
        "arange",
        SymbolicExpr("const", tl.int32, tl.int32),
        SymbolicExpr("const", 0, tl.int32),
        SymbolicExpr("const", 10, tl.int32),
    )
    pid_expr = SymbolicExpr(
        "pid",
        SymbolicExpr("const", grid, tl.int32),
        SymbolicExpr("const", 0, tl.int32),
    )
    block_size = SymbolicExpr("const", 16, tl.int32)
    pid_times_block = SymbolicExpr("mul", pid_expr, block_size)
    final_expr = SymbolicExpr("add", arange_expr, pid_times_block)
    return final_expr


def _build_shared_index_expr(grid):
    """Build a shared index expression: pid_0 * 16 + arange(0, 16)"""
    pid_expr = SymbolicExpr(
        "pid",
        SymbolicExpr("const", grid, tl.int32),
        SymbolicExpr("const", 0, tl.int32),
    )
    block_size = SymbolicExpr("const", 16, tl.int32)
    pid_times_block = SymbolicExpr("mul", pid_expr, block_size)

    arange_expr = SymbolicExpr(
        "arange",
        SymbolicExpr("const", tl.int32, tl.int32),
        SymbolicExpr("const", 0, tl.int32),
        SymbolicExpr("const", 16, tl.int32),
    )

    index_expr = SymbolicExpr("add", pid_times_block, arange_expr)
    return index_expr


def _build_layernorm_like_index_expr():
    """
    Simulate layernorm-like memory access pattern:
    Index: pid_0 * BLOCK_SIZE + arange(0, BLOCK_SIZE)
    """
    grid = (128, 1, 1)
    BLOCK_SIZE = 64

    pid_0 = SymbolicExpr(
        "pid",
        SymbolicExpr("const", grid, tl.int32),
        SymbolicExpr("const", 0, tl.int32),
    )

    block_size_const = SymbolicExpr("const", BLOCK_SIZE, tl.int32)
    offset = SymbolicExpr("mul", pid_0, block_size_const)

    arange = SymbolicExpr(
        "arange",
        SymbolicExpr("const", tl.int32, tl.int32),
        SymbolicExpr("const", 0, tl.int32),
        SymbolicExpr("const", BLOCK_SIZE, tl.int32),
    )

    # Shared index: pid_0 * BLOCK_SIZE + arange
    shared_index = SymbolicExpr("add", offset, arange)
    return shared_index
