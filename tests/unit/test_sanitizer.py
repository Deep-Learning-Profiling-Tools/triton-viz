import pytest
import torch
import numpy as np
import triton.language as tl
from typing import cast
from triton.runtime.interpreter import TensorHandle, _implicit_cvt
from z3 import Solver, sat, unsat

from triton_viz.core.config import config as cfg
from triton_viz.clients import Sanitizer
from triton_viz.clients.symbolic_engine import SymbolicExpr
from triton_viz.clients.sanitizer.report import _classify_layout_and_segments
from triton_viz.clients.sanitizer.sanitizer import (
    NullSanitizer,
    SymbolicSanitizer,
    _fn_symbolic_cache_set,
)

_EXHAUSTIVE_TENSOR_BYTES = 128 * 1024**3


# ======== Init Tests ===========


@pytest.fixture
def _isolate_sanitizer_cfg():
    """Save and restore cfg.enable_sanitizer around every test."""
    saved = cfg.enable_sanitizer
    saved_exhaustive_mode = cfg.sanitizer_exhaustive_mode
    yield
    cfg.enable_sanitizer = saved
    cfg.sanitizer_exhaustive_mode = saved_exhaustive_mode
    SymbolicExpr.clear_exhaustive_scalar_inputs()


def test_sanitizer_init(_isolate_sanitizer_cfg):
    cfg.enable_sanitizer = True
    assert isinstance(Sanitizer(), SymbolicSanitizer)

    cfg.enable_sanitizer = False
    assert isinstance(Sanitizer(), NullSanitizer)


def test_exhaustive_mode_config_and_constructor(_isolate_sanitizer_cfg):
    cfg.sanitizer_exhaustive_mode = True
    assert SymbolicSanitizer(abort_on_error=False).exhaustive_mode is True

    assert (
        SymbolicSanitizer(abort_on_error=False, exhaustive_mode=False).exhaustive_mode
        is False
    )

    cfg.sanitizer_exhaustive_mode = False
    assert (
        SymbolicSanitizer(abort_on_error=False, exhaustive_mode=True).exhaustive_mode
        is True
    )


def test_exhaustive_mode_widens_tensor_address_range(_isolate_sanitizer_cfg):
    tensor = torch.empty(4, dtype=torch.int32)

    normal = SymbolicSanitizer(abort_on_error=False, exhaustive_mode=False)
    normal_start, normal_end, _ = normal._tensor_physical_addresses("x", tensor)[0]
    assert normal_end - normal_start + 1 == (
        (tensor.numel() - 1) * tensor.element_size() + 1
    )

    exhaustive = SymbolicSanitizer(abort_on_error=False, exhaustive_mode=True)
    start, end, resolved = exhaustive._tensor_physical_addresses("x", tensor)[0]
    assert start == tensor.data_ptr()
    assert end - start + 1 == _EXHAUSTIVE_TENSOR_BYTES
    assert resolved is tensor


def _assert_scalar_range(arg, arg_cvt, lower: int, upper: int) -> None:
    sanitizer = SymbolicSanitizer(abort_on_error=False, exhaustive_mode=True)
    sanitizer.arg_callback("n", arg, arg_cvt)

    expr = SymbolicExpr.from_value(arg_cvt)
    z3_expr, constraints = expr.eval(simplify_constraints=False)

    solver = Solver()
    solver.add(constraints)
    solver.push()
    solver.add(z3_expr == lower)
    assert solver.check() == sat
    solver.pop()

    solver.push()
    solver.add(z3_expr == upper)
    assert solver.check() == unsat
    solver.pop()

    SymbolicExpr.clear_exhaustive_scalar_inputs()


def test_exhaustive_mode_signed_scalar_uses_dtype_range(_isolate_sanitizer_cfg):
    _assert_scalar_range(3, _implicit_cvt(3), -(2**31), 2**31)


def test_exhaustive_mode_unsigned_scalar_uses_dtype_range(_isolate_sanitizer_cfg):
    arg_cvt = tl.tensor(
        TensorHandle(np.array([1], dtype=np.uint32), tl.uint32), tl.uint32
    )
    _assert_scalar_range(1, arg_cvt, 0, 2**32)


# ======== Launch-lifecycle regression tests ===========
#
# These guard the two spots that break most easily when SymbolicClient's
# launch-lifecycle machinery is reshuffled:
#   1. the cross-launch fn cache (sanitizer's pre_run_callback short-circuit)
#   2. the post_run_callback guard that must NOT wipe launch state mid-
#      full-grid enumeration.


@pytest.fixture
def _isolate_fn_symbolic_cache():
    """Reset the cross-launch sanitizer fn-cache around each test."""
    saved = set(_fn_symbolic_cache_set)
    _fn_symbolic_cache_set.clear()
    yield _fn_symbolic_cache_set
    _fn_symbolic_cache_set.clear()
    _fn_symbolic_cache_set.update(saved)


def _populate_for_launch(sanitizer: SymbolicSanitizer, grid: tuple[int, ...]) -> None:
    """Prime a sanitizer as if ``arg_callback`` + ``grid_callback`` had fired.

    Uses a fake cache_args payload instead of real tensors so the test stays
    focused on pre_run_callback's cache-hit logic.
    """
    sanitizer.cache_args = [("fake_shape",)]
    sanitizer.cache_grid = grid


def test_pre_run_callback_cache_hit_on_replay(_isolate_fn_symbolic_cache):
    """Identical (fn, grid, args) launched twice: 1st runs, 2nd is cached-skip."""

    def kernel():
        pass

    first = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(first, (2, 1, 1))
    assert first.pre_run_callback(kernel) is True
    # pre_run_callback clears the cache after hashing, matching production flow.
    assert first.cache_grid is None and first.cache_args == []
    assert len(_isolate_fn_symbolic_cache) == 1

    second = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(second, (2, 1, 1))
    assert second.pre_run_callback(kernel) is False
    assert len(_isolate_fn_symbolic_cache) == 1


def test_pre_run_callback_cache_miss_on_grid_change(_isolate_fn_symbolic_cache):
    """Changing the grid must NOT trigger a cache hit — it's part of the key."""

    def kernel():
        pass

    a = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(a, (2, 1, 1))
    assert a.pre_run_callback(kernel) is True

    b = SymbolicSanitizer(abort_on_error=False)
    _populate_for_launch(b, (4, 1, 1))
    assert b.pre_run_callback(kernel) is True
    assert len(_isolate_fn_symbolic_cache) == 2


def test_post_run_callback_preserves_state_mid_full_grid():
    """When need_full_grid=True and we're NOT on the last block, launch state
    (tensors / tensor_addrs / tensor_names / cache_*) must survive
    post_run_callback so later blocks still see registered tensors."""
    sanitizer = SymbolicSanitizer(abort_on_error=False)

    # Simulate state populated during arg_callback + grid_callback.
    sentinel_tensor = object()
    sanitizer.tensors.append(cast(object, sentinel_tensor))  # type: ignore[arg-type]
    sanitizer.tensor_addrs.append((0, 0, cast(object, sentinel_tensor)))  # type: ignore[arg-type]
    sanitizer.tensor_names[id(sentinel_tensor)] = {"X"}
    sanitizer.cache_args = [("fake",)]
    sanitizer.cache_grid = (4, 1, 1)

    sanitizer.last_grid = (3, 0, 0)  # last block is (3,0,0) in a 4-block grid
    sanitizer.grid_idx = (1, 0, 0)  # we're mid-enumeration
    sanitizer.need_full_grid = True

    ret = sanitizer.post_run_callback(lambda: None)

    assert ret is True
    assert sanitizer.tensors == [sentinel_tensor]
    assert sanitizer.tensor_addrs == [(0, 0, sentinel_tensor)]
    assert sanitizer.tensor_names == {id(sentinel_tensor): {"X"}}
    assert sanitizer.cache_args == [("fake",)]
    assert sanitizer.cache_grid == (4, 1, 1)


def test_post_run_callback_clears_state_on_last_block():
    """On the final block we DO want launch state cleared."""
    sanitizer = SymbolicSanitizer(abort_on_error=False)

    sentinel_tensor = object()
    sanitizer.tensors.append(cast(object, sentinel_tensor))  # type: ignore[arg-type]
    sanitizer.tensor_addrs.append((0, 0, cast(object, sentinel_tensor)))  # type: ignore[arg-type]
    sanitizer.tensor_names[id(sentinel_tensor)] = {"X"}
    sanitizer.cache_args = [("fake",)]
    sanitizer.cache_grid = (4, 1, 1)

    sanitizer.last_grid = (3, 0, 0)
    sanitizer.grid_idx = (3, 0, 0)  # final block
    sanitizer.need_full_grid = True

    sanitizer.post_run_callback(lambda: None)

    assert sanitizer.tensors == []
    assert sanitizer.tensor_addrs == []
    assert sanitizer.tensor_names == {}
    assert sanitizer.cache_args == []
    assert sanitizer.cache_grid is None


# ======== Report Layout Classification Tests ===========


def _normalize(layout_segments, tensor):
    """Translate ``(layout, segments)`` to ``(layout, relative_segments)``.

    Segment endpoints are made relative to ``tensor.data_ptr()`` so the asserts
    do not depend on PyTorch's allocator returning specific addresses.
    """
    layout, segments = layout_segments
    base = tensor.data_ptr() if tensor.numel() else 0
    return layout, [(s - base, e - base) for s, e in segments]


def test_classify_contiguous_1d():
    t = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    assert _normalize(_classify_layout_and_segments(t), t) == (
        "contiguous",
        [(0, 15)],
    )


def test_classify_dim0_contiguous_transpose():
    # ``t.T`` alone is storage-contiguous (just F-order), so it falls into the
    # single-segment path. Slice it to force the dim-0-contiguous bucket:
    # ``t.T[1:]`` has strides (1, 8) and a non-zero storage_offset.
    t = torch.arange(32, dtype=torch.int32).reshape(4, 8).T[1:]  # shape (7, 4)
    layout, rel = _normalize(_classify_layout_and_segments(t), t)
    assert layout == "dim-0-contiguous"
    # 4 segments (outer dim 1 size = 4), each covering 7 int32 elements = 28 bytes,
    # separated by stride[1]=8 elements = 32 bytes.
    assert len(rel) == 4
    assert rel[0] == (0, 27)
    assert rel[1] == (32, 59)


def test_classify_dim1_contiguous_slice():
    # ``t[:, 2:6]``: storage-discontiguous but inner dim 1 still has stride 1.
    base_t = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    t = base_t[:, 2:6]
    layout, rel = _normalize(_classify_layout_and_segments(t), t)
    assert layout == "dim-1-contiguous"
    # 4 row segments, each 4 int32 wide = 16 bytes; rows separated by 32 bytes.
    assert len(rel) == 4
    assert rel[0] == (0, 15)
    assert rel[1] == (32, 47)


def test_classify_per_element_strided_view():
    base_t = torch.tensor([10, 99, 11, 99, 12], dtype=torch.int32)
    view = base_t[0::2]
    assert _normalize(_classify_layout_and_segments(view), view) == (
        "per-element",
        [(0, 3), (8, 11), (16, 19)],
    )


def test_classify_per_element_nonzero_storage_offset():
    base_t = torch.tensor([99, 10, 99, 11, 99, 12], dtype=torch.int32)
    view = base_t[1::2]
    assert view.storage_offset() == 1
    assert _normalize(_classify_layout_and_segments(view), view) == (
        "per-element",
        [(0, 3), (8, 11), (16, 19)],
    )


def test_classify_empty_tensor():
    t = torch.empty((0,), dtype=torch.int32)
    assert _classify_layout_and_segments(t) == ("empty", [])
