import pytest
from typing import cast

from triton_viz.core.config import config as cfg
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    NullSanitizer,
    SymbolicSanitizer,
    _fn_symbolic_cache_set,
)


# ======== Init Tests ===========


@pytest.fixture
def _isolate_sanitizer_cfg():
    """Save and restore cfg.enable_sanitizer around every test."""
    saved = cfg.enable_sanitizer
    yield
    cfg.enable_sanitizer = saved


def test_sanitizer_init(_isolate_sanitizer_cfg):
    cfg.enable_sanitizer = True
    assert isinstance(Sanitizer(), SymbolicSanitizer)

    cfg.enable_sanitizer = False
    assert isinstance(Sanitizer(), NullSanitizer)


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
