
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC", "BK"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BT"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))

    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC", "BK"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A,
    A2,
    T: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_c * BC >= T:
        return
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_bh + i_k*n_bh) * T * BC, (T, BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BV", "BT"],
)
@triton.jit
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT):
    B, H, T, K = q.shape
    BC = 16
    NC = triton.cdiv(BT, BC)
    NT = triton.cdiv(T, BT)

    BK = min(64, triton.next_power_of_2(K))
    A = q.new_empty(B, H, T, BT, dtype=torch.float32)
    grid = (NT, NC * NC, B * H)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q, k, g, A,
        k.stride(1), k.stride(2),
        scale,
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
    )
    grid = (NT, NC, B * H)
    if K <= 256:
        BK = triton.next_power_of_2(K)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
            q, k, g, A,
            k.stride(1), k.stride(2),
            scale,
            T=T, K=K, BT=BT, BC=BC, BK=BK
        )
    else:
        BK = 128
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, H, BT, BC, dtype=torch.float32)
        grid = (NK, NT * NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](
            q, k, g, A_intra,
            k.stride(1), k.stride(2),
            scale,
            T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
        )
        grid = (NT, NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](
            A_intra, A,
            T=T, BT=BT, BC=BC, NK=NK
        )
    return A

def chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, BT, scale):
    B, H, T, K, V = *q.shape, v.shape[-1]
    BK = min(32, triton.next_power_of_2(K))
    BV = min(32, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)

    grid = (NV, NT, B * H)
    o = torch.empty_like(v)
    chunk_gla_fwd_kernel_o[grid](
        q, v, g_cumsum, h, o, A,
        q.stride(1), q.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
    )
    return o




##################################################################################################################################################


def test_chunk_gla_fwd():
    # 测试正常的输入规模
    B = 2  # batch size
    H = 2  # number of heads
    T = 128  # sequence length
    K = 256  # key length
    V = 64  # value length
    BT = 16  # block size for T
    BC = 16  # block size for C (head dimension)
    BK = 64  # block size for K (key length)
    scale = 1.0  # scaling factor

    q = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    g = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    h = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
    
    A = chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT)
    o = chunk_fwd_o_gated_gk_fn(q, v, g.cumsum(dim=-1), A, h, BT, scale)

    result = {}
    result['test_case_1'] = o.shape

    # 测试 K > 256 的情况
    B = 2
    H = 2
    T = 128
    K = 512  # 设置 K > 256
    V = 64
    BT = 16
    BC = 16
    BK = 128
    scale = 1.0

    q = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')
    g = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    h = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')

    A = chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT)
    o = chunk_fwd_o_gated_gk_fn(q, v, g.cumsum(dim=-1), A, h, BT, scale)

    result['test_case_3'] = o.shape

    return result

result_gold = test_chunk_gla_fwd()
