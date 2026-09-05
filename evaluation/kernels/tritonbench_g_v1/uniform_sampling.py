import logging

import torch
import triton
import triton.language as tl
import functools
import operator
from typing import Tuple

Shape = Tuple[int]


try:
    uint_to_uniform_float = tl.uint_to_uniform_float
except AttributeError:
    # Copied from triton.language package for compatibility
    @triton.jit
    def uint_to_uniform_float(x):
        """
        Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
        """
        # TODO: fix frontend issues and cleanup
        # conditions can be simplified
        # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
        if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
            # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
            x = x.to(tl.int32, bitcast=True)
            scale = 4.6566127342e-10
        else:
            tl.static_assert(
                tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)
            )
            x = x.to(tl.int64, bitcast=True)
            scale = 1.0842020432385337e-19
        x = tl.where(x < 0, -x - 1, x)
        return x * scale


def philox_cuda_seed_offset(increment, device=None):
    device = device or torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state_copy = gen.get_state()
    c0, c1 = state_copy.view(torch.int64)
    seed, offset = int(c0), int(c1)
    increment = (increment + 3) // 4 * 4
    c1 += increment
    # get_state returns a new tensor, so it needs set_state to update the actual generator state.
    gen.set_state(state_copy)
    return seed, offset


def heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "BLOCK": heur_block,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def uniform_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    from_,
    to,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0) * (to - from_) + from_
    r1 = uint_to_uniform_float(r1) * (to - from_) + from_
    r2 = uint_to_uniform_float(r2) * (to - from_) + from_
    r3 = uint_to_uniform_float(r3) * (to - from_) + from_
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")



def volume(shape: Shape) -> int:
    return functools.reduce(operator.mul, shape, 1)


UNROLL = 4


def uniform_(self, from_=0.0, to=1.0, *, generator=None):
    logging.debug("GEMS UNIFORM")
    N = volume(self.shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_cuda_seed_offset(increment)
    with torch.cuda.device(self.device):
        uniform_kernel[grid_fn](self, N, philox_seed, philox_offset, from_, to)
    return self




##################################################################################################################################################


def test_uniform_():
    results = {}

    # Test case 1: 1D tensor, float32, default from=0, to=1
    x_1d = torch.empty((10,), device='cuda', dtype=torch.float32)
    uniform_(x_1d)
    results['test_case_1'] = x_1d

    # Test case 2: 2D tensor, float32, from=2, to=5
    x_2d = torch.empty((4, 4), device='cuda', dtype=torch.float32)
    uniform_(x_2d, from_=2.0, to=5.0)
    results['test_case_2'] = x_2d

    # Test case 3: 3D tensor, float64, from=-1, to=1
    x_3d = torch.empty((2, 3, 4), device='cuda', dtype=torch.float64)
    uniform_(x_3d, from_=-1.0, to=1.0)
    results['test_case_3'] = x_3d

    # Test case 4: Empty tensor, float32, from=0, to=1
    x_empty = torch.empty((0,), device='cuda', dtype=torch.float32)
    uniform_(x_empty)
    results['test_case_4'] = x_empty

    # Test case 5: Single-element tensor, float32, from=5, to=10
    x_single = torch.empty((1,), device='cuda', dtype=torch.float32)
    uniform_(x_single, from_=5.0, to=10.0)
    results['test_case_5'] = x_single

    # Test case 6: Large tensor, float32, from=-10, to=10
    x_large = torch.empty((1024, 1024), device='cuda', dtype=torch.float32)
    uniform_(x_large, from_=-10.0, to=10.0)
    results['test_case_6'] = x_large

    # Test case 7: 1D tensor, float16, default from=0, to=1
    x_float16 = torch.empty((10,), device='cuda', dtype=torch.float16)
    uniform_(x_float16)
    results['test_case_7'] = x_float16

    # Test case 8: Tensor with specific shapes (non-square), float32, from=-5, to=5
    x_shape1 = torch.empty((3, 7), device='cuda', dtype=torch.float32)
    x_shape2 = torch.empty((5, 3, 2), device='cuda', dtype=torch.float32)
    uniform_(x_shape1, from_=-5.0, to=5.0)
    uniform_(x_shape2, from_=-5.0, to=5.0)
    results['test_case_8_1'] = x_shape1
    results['test_case_8_2'] = x_shape2

    # Test case 9: Tensor with 4D shape, float32, from=0.5, to=2.5
    x_4d = torch.empty((2, 2, 3, 4), device='cuda', dtype=torch.float32)
    uniform_(x_4d, from_=0.5, to=2.5)
    results['test_case_9'] = x_4d

    # Test case 10: Tensor with nan values, float32, checking if the range is filled correctly
    x_nan = torch.full((10,), float('nan'), device='cuda', dtype=torch.float32)
    uniform_(x_nan, from_=1.0, to=2.0)
    results['test_case_10'] = x_nan

    # Test case 11: Tensor with negative shape values (expecting an exception)
    try:
        x_negative = torch.empty((-1,), device='cuda', dtype=torch.float32)
        uniform_(x_negative)
    except Exception as e:
        results['test_case_11'] = str(e)

    # Test case 12: Tensor with mixed positive and negative shape values (expecting an exception)
    try:
        x_invalid = torch.empty((3, -4), device='cuda', dtype=torch.float32)
        uniform_(x_invalid)
    except Exception as e:
        results['test_case_12'] = str(e)

    # Test case 13: Very large tensor, checking if it handles allocation and computation
    x_very_large = torch.empty((4096, 4096), device='cuda', dtype=torch.float32)
    uniform_(x_very_large, from_=-10.0, to=10.0)
    results['test_case_13'] = x_very_large

    # Test case 14: Tensor with extreme value ranges
    x_extreme = torch.empty((10,), device='cuda', dtype=torch.float32)
    uniform_(x_extreme, from_=-1e5, to=1e5)
    results['test_case_14'] = x_extreme

    # Test case 15: Edge case where from_ == to
    x_equal = torch.empty((10,), device='cuda', dtype=torch.float32)
    uniform_(x_equal, from_=5.0, to=5.0)
    results['test_case_15'] = x_equal

    # Test case 16: Tensor with NaN, Inf values for from_ and to (expecting exception)
    try:
        x_nan_inf = torch.empty((10,), device='cuda', dtype=torch.float32)
        uniform_(x_nan_inf, from_=float('nan'), to=float('inf'))
    except Exception as e:
        results['test_case_16'] = str(e)

    return results

result_gold = test_uniform_()
