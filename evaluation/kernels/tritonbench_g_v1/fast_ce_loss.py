
import triton
import triton.language as tl
import torch
# from .utils import triton_tanh
from triton.language.extra import libdevice

triton_tanh = libdevice.tanh
next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE : int = 65536

def calculate_settings(n : int) -> (int, int):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
            if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
        tl.store(logsumexp_ptr, logsumexp)


@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE
    
    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial
    
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE
    
    if DO_SOFTCAPPING:
        y = y * (1.0 - partial*partial)
    
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)

MAX_FUSED_SIZE = 65536

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0, logit_scaling=0):
        n_rows, vocab_size = logits.shape

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        DO_SOFTCAPPING = (logit_softcapping != 0)
        DO_LOGIT_SCALING = (logit_scaling != 0)

        if n_chunks == 1:
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=DO_LOGIT_SCALING,
                LOGIT_SCALE=logit_scaling,
                num_warps=num_warps,
            )
        else:
            logsumexp = torch.empty((n_rows, n_chunks,), dtype=torch.float32, device="cuda:0")

            _chunked_cross_entropy_forward[(n_rows, n_chunks,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=DO_LOGIT_SCALING,
                LOGIT_SCALE=logit_scaling,
                num_warps=32,
            )
            logsumexp = torch.logsumexp(logsumexp, dim=1)
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)
        
        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        ctx.DO_LOGIT_SCALING = DO_LOGIT_SCALING
        ctx.logit_scaling = logit_scaling
        return losses
    
    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        _cross_entropy_backward[(n_rows, n_blocks,)](
            logits, logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=ctx.DO_SOFTCAPPING,
            SOFTCAP=ctx.logit_softcapping,
            DO_LOGIT_SCALING=ctx.DO_LOGIT_SCALING,
            LOGIT_SCALE=ctx.logit_scaling,
            num_warps=8,
        )
        return logits, None, None, None,
    
def fast_cross_entropy_loss(
    logits,
    labels,
    logit_softcapping=0,
    logit_scaling=0,
    n_items=None,
):
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch*seq_len, d),
        labels.view(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items




##################################################################################################################################################


import torch

def test_fast_cross_entropy_loss_with_backward():
    # Test case 1: Basic test without softcapping or logit scaling
    logits = torch.randn(2, 3, 5, device='cuda:0', requires_grad=True)  # Batch size 2, sequence length 3, vocab size 5
    labels = torch.tensor([[1, 2, 3], [0, 1, 4]], device='cuda:0')  # Corresponding labels
    loss = fast_cross_entropy_loss(logits, labels)

    # Perform backward pass
    loss.backward()

    # Reset gradients
    logits.grad.zero_()

    # Test case 2: With logit softcapping
    logit_softcapping = 0.5
    loss = fast_cross_entropy_loss(logits, labels, logit_softcapping=logit_softcapping)

    # Perform backward pass
    loss.backward()

    # Reset gradients
    logits.grad.zero_()

    # Test case 3: With logit scaling
    logit_scaling = 1.5
    loss = fast_cross_entropy_loss(logits, labels, logit_scaling=logit_scaling)

    # Perform backward pass
    loss.backward()

    # Reset gradients
    logits.grad.zero_()

    # Test case 4: With both softcapping and logit scaling
    loss = fast_cross_entropy_loss(logits, labels, logit_softcapping=logit_softcapping, logit_scaling=logit_scaling)

    # Perform backward pass
    loss.backward()

    # Reset gradients
    logits.grad.zero_()

    # Test case 5: Handling ignore index (-100)
    labels_with_ignore = torch.tensor([[1, -100, 3], [0, 1, -100]], device='cuda:0')
    loss = fast_cross_entropy_loss(logits, labels_with_ignore)

    # Perform backward pass
    loss.backward()

    return {
        "test_case_1": loss.item(),
        "test_case_2": loss.item(),
        "test_case_3": loss.item(),
        "test_case_4": loss.item(),
        "test_case_5": loss.item()
    }

result_gold = test_fast_cross_entropy_loss_with_backward()
