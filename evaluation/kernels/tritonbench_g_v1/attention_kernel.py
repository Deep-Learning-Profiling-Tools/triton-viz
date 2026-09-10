
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale,
    Out,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vk, stride_vn,
    stride_oh, stride_om, stride_on,
    stride_b0h, stride_b0m,
    Z,
    H,
    N_CTX,
    P_SEQ,
    OUT_DTYPE: tl.constexpr,
    BIAS_LAST_SIZE: tl.constexpr,
    B0_NUMEL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)  # , boundary_check=(1, 0), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_hz * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :])
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        # , boundary_check=(0, 1), padding_option="zero")
        k = tl.load(K_block_ptr)
        # , boundary_check=(1, 0), padding_option="zero")
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k) #, out_dtype=OUT_DTYPE)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias
        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N)
        qk += ((b0 + b1) * 1.44269504)

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE))


def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, o,
                                                 BLOCK_M,
                                                 BLOCK_N,
                                                 num_warps,
                                                 num_stages):
    _, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
    assert q.size() == k.size()
    assert q.size() == v.size()
    assert q.size(-2) == rel_h_w.size(-2)
    assert (q.dtype == torch.bfloat16 or q.dtype == torch.float16)
    assert k.dtype == q.dtype
    assert v.dtype == k.dtype
    assert o.dtype == v.dtype
    assert rel_h_w.dtype == q.dtype
    assert rel_h_w.size(-1) == 128
    # assert rel_h_w.size(-1) == 2 * BLOCK_N

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    # print("q.shape[0] * q.shape[1]: ", q.shape[0] * q.shape[1])
    P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]
    assert P_SEQ == 0
    assert rel_h_w.is_contiguous(), str(rel_h_w.stride())
    OUT_DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    _fwd_kernel_aligned[grid](
        q, k, v,
        rel_h_w,
        sm_scale,
        o,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        rel_h_w.stride(1), rel_h_w.stride(2),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        P_SEQ,
        OUT_DTYPE=OUT_DTYPE,
        BIAS_LAST_SIZE=(rel_h_w.size(-1) // 2),
        B0_NUMEL=rel_h_w.size(-1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=num_stages)




##################################################################################################################################################


import torch

def test_attention_rel_h_rel_w_kernel_aligned_device():
    # Define the input parameters
    BATCH_SIZE = 2
    HEADS = 4
    N_CTX = 128
    BLOCK_M = 64
    BLOCK_N = 64
    D_MODEL = 128
    SM_SCALE = 0.1

    # Create input tensors with appropriate shapes and data types
    q = torch.randn((BATCH_SIZE, HEADS, N_CTX, D_MODEL), dtype=torch.float16, device='cuda')
    k = torch.randn((BATCH_SIZE, HEADS, N_CTX, D_MODEL), dtype=torch.float16, device='cuda')
    v = torch.randn((BATCH_SIZE, HEADS, N_CTX, D_MODEL), dtype=torch.float16, device='cuda')
    rel_h_w = torch.randn((BATCH_SIZE, HEADS, N_CTX, 128), dtype=torch.float16, device='cuda')
    o = torch.empty((BATCH_SIZE, HEADS, N_CTX, D_MODEL), dtype=torch.float16, device='cuda')

    # Create a dictionary to store the results of different test cases
    test_case_results = {}

    # Test case 1: Default case with P_SEQ = 0
    P_SEQ = 0
    _attention_rel_h_rel_w_kernel_aligned_device(
        q, k, v, rel_h_w, SM_SCALE, o,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    test_case_results['test_case_1'] = o.clone()

    # Test case 2: Change P_SEQ to a non-zero value
    P_SEQ = 10  # Arbitrary non-zero value
    _attention_rel_h_rel_w_kernel_aligned_device(
        q, k, v, rel_h_w, SM_SCALE, o,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    test_case_results['test_case_2'] = o.clone()

    # Test case 3: Change number of warps
    num_warps = 8  # Arbitrary non-zero value
    _attention_rel_h_rel_w_kernel_aligned_device(
        q, k, v, rel_h_w, SM_SCALE, o,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=2
    )
    test_case_results['test_case_3'] = o.clone()

    # Test case 4: Change number of stages
    num_stages = 4  # Arbitrary non-zero value
    _attention_rel_h_rel_w_kernel_aligned_device(
        q, k, v, rel_h_w, SM_SCALE, o,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=num_stages
    )
    test_case_results['test_case_4'] = o.clone()

    return test_case_results


# Execute the test function
result_gold = test_attention_rel_h_rel_w_kernel_aligned_device()
