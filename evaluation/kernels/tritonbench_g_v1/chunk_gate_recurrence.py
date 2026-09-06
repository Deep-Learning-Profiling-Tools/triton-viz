import torch
import triton
import triton.language as tl

torch.backends.cudnn.allow_tf32 = True

@triton.jit
def _fwd_recurrence(
    S, d, 
    O,
    NUM_HEAD, NUM_BLOCK, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    last_kv: tl.tensor  # 不再使用 Optional
):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    if last_kv is not None:
        last_kv = last_kv + offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
        acc = tl.load(last_kv).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V
    d = d + offset_bh * NUM_BLOCK
    for i in range(NUM_BLOCK-1):
        d_i = tl.load(d)
        S_i = tl.load(S) 
        acc = acc * d_i + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        d += 1
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V
     

## NUM_SPLIT_K/V. K/V dimension split into NUM_SPLIT_K/V parts with equal size BLOCK_MODEL
@triton.jit
def _bwd_recurrence(
    S, d, 
    DI, DG, DL, DS, 
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    
 ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    # offset_h = offset_bh % NUM_HEAD
    NUM_K = D_MODEL_K // BLOCK_MODEL_K
    NUM_V = D_MODEL_V // BLOCK_MODEL_V
    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    DI = DI + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk  
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    DG = DG + offset_bh * NUM_BLOCK * NUM_K * NUM_V + offset_d * NUM_V + offset_s + (NUM_BLOCK - 2) * NUM_K * NUM_V

    d = d + offset_bh * NUM_BLOCK + (NUM_BLOCK - 1)

    Dacc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32) 

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        d_i = tl.load(d)
        Dacc = Dacc * d_i + DS_i
        DG_i = tl.sum(Dacc * S_i.to(tl.float32))

        tl.store(DG, DG_i.to(DG.dtype.element_ty))
        tl.store(DI, Dacc.to(DI.dtype.element_ty))    

        S -= D_MODEL_K * D_MODEL_V
        DI -= D_MODEL_K * D_MODEL_V 
        DS -= D_MODEL_K * D_MODEL_V
        DG -= NUM_K * NUM_V
        d -= 1
    
    DL = DL + offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K  +  tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]
    DS_i = tl.load(DS)
    d_i = tl.load(d)
    Dacc = Dacc * d_i + DS_i
    tl.store(DL, Dacc.to(DL.dtype.element_ty))  

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        cross_decay = cross_decay.contiguous()
        kv = kv.contiguous()

        B, H, N, D_k, D_v = kv.shape 
        output = torch.empty_like(kv)        
        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 16
    
        assert D_k % BLOCK_MODEL_K == 0
        assert D_v % BLOCK_MODEL_V == 0

        grid = (B*H, D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)
        ctx.grid = grid
        ctx.have_last_kv = last_kv is not None
        ctx.BLOCK_MODEL_K = BLOCK_MODEL_K
        ctx.BLOCK_MODEL_V = BLOCK_MODEL_V

        _fwd_recurrence[grid](
            kv,
            cross_decay,
            output,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N, NUM_HEAD=H,
            BLOCK_MODEL_K=BLOCK_MODEL_K,
            BLOCK_MODEL_V=BLOCK_MODEL_V,
            last_kv=last_kv
        )

        ctx.save_for_backward(output, cross_decay)        
        return output

    @staticmethod
    def backward(ctx, DO):
        DO = DO.contiguous()

        output, cross_decay = ctx.saved_tensors 

        B, H, N, D_k, D_v = output.shape 
        
        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 16

        grid = (B*H, D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)

        DI = torch.empty_like(DO)
        DG = torch.empty(B*H, N, D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V, device=cross_decay.device, dtype=cross_decay.dtype)
        DL = torch.empty(B, H, D_k, D_v, device=output.device, dtype=output.dtype)
        _bwd_recurrence[grid](
            output, cross_decay,
            DI, DG, DL, DO, 
            NUM_HEAD=H, NUM_BLOCK = N, 
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL_K=BLOCK_MODEL_K,
            BLOCK_MODEL_V=BLOCK_MODEL_V,
        )

        DI[:, :, -1] = 0
        DG[:, -1] = 0
        DG = DG.view(B, H, N, -1).sum(dim=-1)
        return DI, DG, DL if ctx.have_last_kv else None

chunk_gate_recurrent = ChunkGateRecurrent.apply




##################################################################################################################################################


import torch

def test_chunk_gate_recurrent():
    # 定义测试参数
    B = 2        # Batch size
    H = 4        # Number of heads
    N = 64       # Number of blocks (sequence length)
    D_k = 64     # Key dimension
    D_v = 64     # Value dimension

    # 创建测试输入张量
    kv = torch.randn(B, H, N, D_k, D_v, device='cuda', dtype=torch.float32, requires_grad=True)
    cross_decay = torch.randn(B, H, N, device='cuda', dtype=torch.float32, requires_grad=True)

    # 可选的 last_kv
    last_kv = torch.randn(B, H, D_k, D_v, device='cuda', dtype=torch.float32, requires_grad=True)

    # 前向传播
    output1 = chunk_gate_recurrent(kv, cross_decay, last_kv)
    output2 = chunk_gate_recurrent(kv, cross_decay, None)

    # 测试反向传播
    # 对输出求和，保证所有元素都对梯度有贡献
    loss1 = output1.sum()
    loss1.backward()

    # 检查梯度是否计算成功
    result = {
        "test_case_1": (kv.grad is not None, cross_decay.grad is not None, last_kv.grad is not None),
        "test_case_2": (kv.grad is not None, cross_decay.grad is not None)
    }

    return result

result_gold = test_chunk_gate_recurrent()
