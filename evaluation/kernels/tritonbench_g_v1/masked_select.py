import logging

import torch
import triton
import triton.language as tl
from typing import Tuple


Shape = Tuple[int]

def broadcastable(s1: Shape, s2: Shape) -> bool:
    r1 = len(s1)
    if r1 == 0:
        return True
    r2 = len(s2)
    if r2 == 0:
        return True

    s1, s2 = (s1, s2) if r1 >= r2 else (s2, s1)
    r1, r2 = (r1, r2) if r1 >= r2 else (r2, r1)

    d = r1 - r2
    for i in range(r2):
        if s1[d + i] == 1 or s2[i] == 1 or s1[d + i] == s2[i]:
            continue
        return False
    return True


def cfggen():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for w in [4, 8, 16, 32]
        for bs in [256, 512, 1024, 2048, 4096]
    ]
    return configs



@triton.autotune(configs=cfggen(), key=["n_elements"])
@triton.jit
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0.0) - 1

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))


def masked_select(inp, mask):
    logging.debug("GEMS MASKED SELECT")

    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    inp, mask = torch.broadcast_tensors(inp, mask)

    inp = inp.contiguous()
    mask = mask.contiguous()

    mask_flattened = mask.ravel()

    prefix_sum = mask_flattened.cumsum(axis=0)
    out = torch.empty(prefix_sum[-1].item(), dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(inp.device):
        masked_select_kernel[grid](inp, mask_flattened, prefix_sum, out, n_elements)
    return out




##################################################################################################################################################


def test_masked_select():
    # Initialize a dictionary to store results
    results = {}

    # Test case 9: Random mask for 2D tensor, float32
    x_random = torch.rand((4, 4), device='cuda', dtype=torch.float32)
    mask_random = torch.randint(0, 2, (4, 4), dtype=torch.bool, device='cuda')
    result_random = masked_select(x_random, mask_random)
    results['test_case_0'] = result_random

    # Test case 3: 3D tensor, float64, mask with all True
    x_3d = torch.rand((2, 3, 4), dtype=torch.float64, device='cuda')
    mask_3d = torch.ones((2, 3, 4), dtype=torch.bool, device='cuda')
    result_3d = masked_select(x_3d, mask_3d)
    results['test_case_1'] = result_3d

    # Test case 4: 4D tensor, int64, mask with all False
    x_4d = torch.randint(0, 100, (2, 2, 2, 2), dtype=torch.int64, device='cuda')
    mask_4d = torch.zeros((2, 2, 2, 2), dtype=torch.bool, device='cuda')
    result_4d = masked_select(x_4d, mask_4d)
    results['test_case_2'] = result_4d


    # Test case 13: Large tensor, float32, random mask
    x_large = torch.rand((512, 1024), device='cuda', dtype=torch.float32)
    mask_large = torch.randint(0, 2, (512, 1024), dtype=torch.bool, device='cuda')
    result_large = masked_select(x_large, mask_large)
    results['test_case_3'] = result_large

    return results

result_gold = test_masked_select()
