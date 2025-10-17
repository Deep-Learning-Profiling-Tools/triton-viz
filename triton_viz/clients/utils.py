import numpy as np
import numpy.typing as npt
import torch
import itertools


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
