
import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 1024,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0,
    in_out_ptr1,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tmp4
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = rnumel  # 4096.0
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 - tmp6
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tmp13
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = rnumel  # 4096.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp18, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp6
        tmp22 = tmp21 * tmp18
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr0 + (r1 + (rnumel * x0)), tmp29, rmask)

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    S, D = primals_3.shape
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf1 = buf0
        del buf0  # reuse
        buf2 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf3 = reinterpret_tensor(buf2, (S, 1), (1, 1), 0)
        del buf2  # reuse
        buf4 = empty_strided_cuda((S, D), (D, 1), torch.bfloat16)
        stream0 = get_raw_stream(0)
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_no_welford[grid](
            buf1, buf3, primals_3, primals_1, primals_2, buf4, S, D
        )
    return (
        buf4,
        primals_3,
        buf1,
        buf3,
    )




##################################################################################################################################################


import torch

def test_fused_native_layer_norm_no_welford():
    # Define the input shapes
    S = 128  # Number of sequences
    D = 4096  # Dimension of each sequence

    # Create input tensors with appropriate shapes and data types
    primals_1 = torch.randn(D, dtype=torch.bfloat16, device='cuda')  # Weight tensor
    primals_2 = torch.randn(D, dtype=torch.bfloat16, device='cuda')  # Bias tensor
    primals_3 = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')  # Input tensor

    # Test the fused_native_layer_norm_no_welford function
    test_case_1 = fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3)

    # Additional test cases to cover all branches
    # Test case 2: Different input size
    S2 = 256
    primals_3_case2 = torch.randn(S2, D, dtype=torch.bfloat16, device='cuda')
    test_case_2 = fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3_case2)

    # Test case 3: Different dimension size
    D2 = 2048
    primals_1_case3 = torch.randn(D2, dtype=torch.bfloat16, device='cuda')
    primals_2_case3 = torch.randn(D2, dtype=torch.bfloat16, device='cuda')
    primals_3_case3 = torch.randn(S, D2, dtype=torch.bfloat16, device='cuda')
    test_case_3 = fused_native_layer_norm_no_welford(primals_1_case3, primals_2_case3, primals_3_case3)

    # Test case 4: Edge case with minimal size
    S4 = 1
    D4 = 1
    primals_1_case4 = torch.randn(D4, dtype=torch.bfloat16, device='cuda')
    primals_2_case4 = torch.randn(D4, dtype=torch.bfloat16, device='cuda')
    primals_3_case4 = torch.randn(S4, D4, dtype=torch.bfloat16, device='cuda')
    test_case_4 = fused_native_layer_norm_no_welford(primals_1_case4, primals_2_case4, primals_3_case4)

    return {
        "test_case_1": test_case_1,
        "test_case_2": test_case_2,
        "test_case_3": test_case_3,
        "test_case_4": test_case_4,
    }

result_gold = test_fused_native_layer_norm_no_welford()
