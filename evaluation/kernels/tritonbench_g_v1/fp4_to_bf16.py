import torch
import triton
import triton.language as tl

@triton.jit
def _fp4_packed_to_bf16(
    x_packed,
    sign_mask_f4,
    mantissa_mask_f4,
    mbits_f4_e2m1,
    ebits_f4_e2m1,
    f4_e2m1_exp_bias,
    mbits_f32,
    ebits_f32,
    f32_exp_bias,
    zero_bits_f32,
    zero_point_five_bits_f32,
):
    """
    Input: a tensor of packed fp4 values
    Output: a tensor of bfloat16 values
    """

    # low-bits: original location 0:3
    # high-bits: original location 4:7
    x_low_bits = x_packed >> 4
    x_high_bits = x_packed & 0xF
    x = tl.interleave(x_low_bits, x_high_bits)

    # cast logic below
    # output = x_unpacked.to(tl.float32)

    # save the sign
    sign_f4 = x & sign_mask_f4

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_f4

    # Special case zero
    zero_mask = x_pos == 0

    # There is only one denormal value in fp4: s001, which is 0.5 in f32
    # Special case it.
    # TODO(later): will it be faster to repeat this for all 8 positive
    # values instead of the bit manipulations?
    denormal_mask = x_pos == 1

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_f4 = x_pos >> mbits_f4_e2m1
    exp_biased_f32 = exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias
    exp_biased_f32 = exp_biased_f32.to(tl.int32) << mbits_f32

    # shift the mantissa to bits 10:32 of the result
    mantissa_f4 = x_pos & mantissa_mask_f4
    mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
    output = mantissa_f32

    # combine the pieces
    result = exp_biased_f32 | mantissa_f32
    # result[zero_mask] = ZERO_BITS_F32
    result = tl.where(zero_mask, zero_bits_f32, result)
    # result[denormal_mask] = ZERO_POINT_FIVE_BITS_F32
    result = tl.where(denormal_mask, zero_point_five_bits_f32, result)

    # add sign back
    sign_f32 = sign_f4.to(tl.int32) << (
        mbits_f32 - mbits_f4_e2m1 + ebits_f32 - ebits_f4_e2m1
    )
    result = result | sign_f32

    # The bit shifting above is for float32, so for now we
    # bitcast to float32 and then regular cast to bfloat16
    # TODO(later): it should be pretty easy to cast directly to bf16, just
    # need to adjust the mbits/ebits/special values. Perf impact is likely
    # to be small as we would not be chaning memory access patterns.
    output = result.to(tl.float32, bitcast=True)
    output = output.to(tl.bfloat16)
    return output

@triton.jit
def triton_f4_to_bf16_kernel(
    x_ptr,
    output_ptr,
    n_elements_in,
    sign_mask_f4: tl.constexpr,
    mantissa_mask_f4: tl.constexpr,
    mbits_f4_e2m1: tl.constexpr,
    ebits_f4_e2m1: tl.constexpr,
    f4_e2m1_exp_bias: tl.constexpr,
    mbits_f32: tl.constexpr,
    ebits_f32: tl.constexpr,
    f32_exp_bias: tl.constexpr,
    zero_bits_f32: tl.constexpr,
    zero_point_five_bits_f32: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_elements_out = n_elements_in * 2
    BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2

    block_start_in = pid * BLOCK_SIZE_IN
    offsets_in = block_start_in + tl.arange(0, BLOCK_SIZE_IN)

    mask_in = offsets_in < n_elements_in

    # packed uint8
    x_packed = tl.load(x_ptr + offsets_in, mask=mask_in)
    output = _fp4_packed_to_bf16(
        x_packed,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
    )

    # set up output offsets
    block_start_out = pid * BLOCK_SIZE_OUT
    offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT)
    mask_out = offsets_out < n_elements_out

    tl.store(output_ptr + offsets_out, output, mask=mask_out)

EBITS_F32, MBITS_F32 = 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2

SIGN_MASK_F4 = 0x8  # 1000
MANTISSA_MASK_F4 = 0x1  # 0001

ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000
F4_E2M1_EXP_BIAS = 1
F32_EXP_BIAS = 127

def triton_f4_to_bf16(x: torch.Tensor):
    """
    Input: a tensor of packed fp4 values
    Output: a tensor of bfloat16 values

    Note: this function is only used in testing, so we can test
      the numerical correctness of the cast without the scaling.
    """
    new_shape = (*x.shape[:-1], x.shape[-1] * 2)
    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)
    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda
    n_elements_in = x.numel()
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(n_elements_in, meta["BLOCK_SIZE_IN"]),
    )  # noqa: E731,E501
    triton_f4_to_bf16_kernel[grid](
        x,
        output,
        n_elements_in,
        sign_mask_f4=SIGN_MASK_F4,
        mantissa_mask_f4=MANTISSA_MASK_F4,
        mbits_f4_e2m1=MBITS_F4_E2M1,
        ebits_f4_e2m1=EBITS_F4_E2M1,
        f4_e2m1_exp_bias=F4_E2M1_EXP_BIAS,
        mbits_f32=MBITS_F32,
        ebits_f32=EBITS_F32,
        f32_exp_bias=F32_EXP_BIAS,
        zero_bits_f32=ZERO_BITS_F32,
        zero_point_five_bits_f32=ZERO_POINT_FIVE_BITS_F32,
        BLOCK_SIZE_IN=512,
    )
    return output




##################################################################################################################################################


import torch

def test_triton_f4_to_bf16():
    results = {}
    
    # Test case 1
    n_elements_in = 1024
    x = torch.randint(0, 256, (n_elements_in,), dtype=torch.uint8, device='cuda')
    output = triton_f4_to_bf16(x)
    results["test_case_1"] = output

    # Test case 2
    n_elements_in = 2048
    x = torch.randint(0, 256, (n_elements_in,), dtype=torch.uint8, device='cuda')
    output = triton_f4_to_bf16(x)
    results["test_case_2"] = output

    # Test case 3
    n_elements_in = 512
    x = torch.randint(0, 256, (n_elements_in,), dtype=torch.uint8, device='cuda')
    output = triton_f4_to_bf16(x)
    results["test_case_3"] = output

    # Test case 4
    n_elements_in = 256
    x = torch.randint(0, 256, (n_elements_in,), dtype=torch.uint8, device='cuda')
    output = triton_f4_to_bf16(x)
    results["test_case_4"] = output

    return results

result_gold = test_triton_f4_to_bf16()
