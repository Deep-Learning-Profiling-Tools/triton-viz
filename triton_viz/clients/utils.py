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
    return sorted(tensor.stride())[0] == 1


def get_physical_addr_from_tensor_slice(tensor: torch.Tensor) -> list[tuple[int, int]]:
    if sorted(tensor.stride())[0] != 1:
        raise ValueError("inner dim must be contiguous!")
    dims = tensor.dim()
    inner_dim = min(range(dims), key=lambda d: tensor.stride(d))
    outer_dims = [d for d in range(dims) if d != inner_dim]

    segments = []
    for idxs in itertools.product(*(range(tensor.size(d)) for d in outer_dims)):
        offset = int(tensor.storage_offset()) + sum(
            idx * int(tensor.stride(d)) for idx, d in zip(idxs, outer_dims)
        )
        inner_dim_size = int(tensor.size(inner_dim))
        segments.append(
            (
                tensor.data_ptr() + offset * tensor.element_size(),
                tensor.data_ptr()
                + (offset + inner_dim_size - 1) * tensor.element_size(),
            )
        )
    return segments
