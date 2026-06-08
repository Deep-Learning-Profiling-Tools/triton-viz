import itertools
import sys
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, overload

import numpy as np
import numpy.typing as npt
import torch

from ..utils.traceback_utils import (  # noqa: F401
    extract_complete_statement_from_line,
)


P = ParamSpec("P")
R = TypeVar("R")
_TIMER_STATE = threading.local()


def _get_timer_stack() -> list[dict[str, float]]:
    stack = getattr(_TIMER_STATE, "stack", None)
    if stack is None:
        stack = []
        _TIMER_STATE.stack = stack
    return stack


@overload
def frame_timer(fn: Callable[P, R], *, label: str = "") -> Callable[P, R]:
    ...


@overload
def frame_timer(*, label: str = "") -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...


def frame_timer(
    fn: Callable[P, R] | None = None, *, label: str = ""
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            stack = _get_timer_stack()
            frame = {"child_time": 0.0}
            stack.append(frame)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                stack.pop()
                if stack:
                    stack[-1]["child_time"] += elapsed
                exclusive_ms = (elapsed - frame["child_time"]) * 1000.0
                if label == "z3":
                    self = args[0] if args else None
                    op_name = getattr(self, "op", type(self).__name__)
                    class_name = type(self).__name__
                    name = f"{label} {class_name}({op_name})"
                elif label:
                    name = f"{label} {func.__name__}"
                else:
                    name = func.__name__
                print(
                    f"{name} {exclusive_ms:.3f}ms",
                    file=sys.stderr,
                    flush=True,
                )

        return wrapper

    if fn is None:
        return decorator
    return decorator(fn)


def check_out_of_bounds_access(
    ptrs: npt.NDArray, masks: npt.NDArray[np.bool_], tensor: torch.Tensor
):
    offsets = ptrs - tensor.data_ptr()
    max_valid_offset = np.prod(tensor.shape) * tensor.element_size()
    valid_access_masks = (offsets >= 0) & (offsets < max_valid_offset)
    invalid_access_masks = (~valid_access_masks) & masks
    corrected_offsets = np.where(valid_access_masks, offsets, 0)
    return {
        "tensor": tensor,
        "offsets": offsets,
        "masks": masks,
        "valid_access_masks": valid_access_masks & masks,
        "invalid_access_masks": invalid_access_masks,
        "corrected_offsets": corrected_offsets,
    }


def check_storage_contiguous(tensor: torch.Tensor):
    # Note that this is different from if a tensor is accessed contiguously, so we cannot use tensor.is_contiguous()
    # 1. Sort strides from smallest to largest
    # 2. If the tensor is contiguous, the stride product should be the same of the shape product of all previous dimensions
    from triton.runtime.jit import TensorWrapper

    if isinstance(tensor, TensorWrapper):
        tensor = tensor.base
    assert (
        type(tensor) == torch.Tensor
    ), f"Only torch.Tensor is supported, but found {type(tensor)}"
    shape_prod = 1
    indices = sorted(range(len(tensor.stride())), key=tensor.stride().__getitem__)
    for i, index in enumerate(indices):
        stride = tensor.stride(index)
        shape = tensor.shape[index]
        if i == 0 and stride != 1:
            return False
        if i != 0 and stride != shape_prod:
            return False
        shape_prod *= shape
    return True


def check_inner_stride_equal_to_one(tensor: torch.Tensor):
    non_zero = [s for s in tensor.stride() if s != 0]
    return len(non_zero) > 0 and sorted(non_zero)[0] == 1


def get_physical_addr_from_tensor_slice(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """One contiguous byte segment per outer-index tuple, covering the run of
    ``inner_dim_size`` elements along the dim whose stride is 1.

    Segments are byte-inclusive ``(start, end)`` ranges, matching the
    convention used by :func:`get_physical_addr_per_element`. ``data_ptr()``
    already points at the view's element-0 address (it folds in
    ``storage_offset``), so element offsets are computed relative to it.
    """
    non_zero = [s for s in tensor.stride() if s != 0]
    if len(non_zero) == 0 or sorted(non_zero)[0] != 1:
        raise ValueError("inner dim must be contiguous!")
    dims = tensor.dim()
    # Inner dim is the non-zero dim with smallest stride
    inner_dim = min(
        (d for d in range(dims) if tensor.stride(d) != 0),
        key=lambda d: tensor.stride(d),
    )
    # Stride-0 dims access the same memory for all indices, so collapse to size 1
    outer_dims = [d for d in range(dims) if d != inner_dim]
    itemsize = tensor.element_size()
    base = tensor.data_ptr()
    inner_dim_size = int(tensor.size(inner_dim))

    segments = []
    for idxs in itertools.product(
        *(range(1 if tensor.stride(d) == 0 else tensor.size(d)) for d in outer_dims)
    ):
        # tensor.data_ptr() already accounts for storage_offset, so offsets here
        # are measured relative to data_ptr().
        offset = sum(idx * int(tensor.stride(d)) for idx, d in zip(idxs, outer_dims))
        start = base + offset * itemsize
        end = start + inner_dim_size * itemsize - 1
        segments.append((start, end))
    return segments


def get_physical_addr_per_element(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """One (start, end) byte segment per logical element of ``tensor``.

    Each segment covers exactly ``element_size()`` bytes, so the resulting
    list precisely describes the view footprint — bytes between logical
    elements are *not* covered. Stride-0 dims collapse to size 1 since they
    alias the same memory for every index along that axis.

    Use this when neither the storage-contiguous nor the inner-stride-equal-
    to-one fast paths apply (e.g. a 1-D view with stride > 1).
    """
    itemsize = tensor.element_size()
    # ``tensor.data_ptr()`` already points at the view's element-0 address
    # (i.e. it folds in ``storage_offset``), so element indices are measured
    # *relative to* it and must not add storage_offset back in.
    base = tensor.data_ptr()
    dims = tensor.dim()
    if dims == 0:
        return [(base, base + itemsize - 1)]

    strides = [int(tensor.stride(d)) for d in range(dims)]
    sizes = [1 if strides[d] == 0 else int(tensor.size(d)) for d in range(dims)]

    # TODO(perf): the consumer OR's one Z3 clause per segment, so cost
    # scales with numel. For stride>1 views the legal set can be expressed
    # as a stride equation instead -- roughly
    #   0 <= (addr - base) < numel * stride * itemsize
    #   (addr - base) % (stride * itemsize) < itemsize
    # generalising to N-D by combining per-axis constraints. That keeps
    # the Z3 expression O(dims) instead of O(numel).
    segments = []
    for idxs in itertools.product(*(range(s) for s in sizes)):
        offset = sum(i * st for i, st in zip(idxs, strides))
        start = base + offset * itemsize
        segments.append((start, start + itemsize - 1))
    return segments
