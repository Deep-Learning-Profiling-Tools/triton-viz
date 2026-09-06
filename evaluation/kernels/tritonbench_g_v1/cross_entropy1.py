
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0,
    }
)
@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,  # data ptrs
    lse_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    n_rows,
    logits_row_stride,  # strides
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    # if SPLIT (e.g. tensor parallel), don't include the LSE in the loss since it's not the final LSE
    SPLIT: tl.constexpr,
):
    # Triton kernel implementation for the forward pass of cross-entropy with label smoothing.
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols, (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx)
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            loss += lse_square_scale * lse * lse
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)

@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0,
    }
)
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    # Triton kernel implementation for the backward pass of cross-entropy with label smoothing.
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignored_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, dloss * probs, mask=col_offsets < n_cols)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        smoothing,
        lse_square_scale=0.0,
        ignored_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        # CrossEntropyLoss forward function leveraging the Triton kernel.
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        class_start_idx = rank * n_cols

        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        MAX_BLOCK_SIZE = 64 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else (16 if BLOCK_SIZE < 128 * 1024 else 32))
        split = world_size > 1 or n_cols > MAX_BLOCK_SIZE
        n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
        losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        with torch.cuda.device(logits.device.index):
            cross_entropy_fwd_kernel[(n_rows, n_splits)](
                losses,  # data ptrs
                lse,
                logits,
                labels,
                smoothing,
                lse_square_scale,
                ignored_index,
                total_classes,
                class_start_idx,
                n_cols,  # shapes
                n_rows,
                logits.stride(0),  # strides
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
                SPLIT=split,
            )

        if split:
            if world_size > 1:
                lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
                torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
                handle_losses = torch.distributed.all_reduce(
                    losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
                )
                lse = torch.logsumexp(lse_allgather, dim=0)
                handle_losses.wait()
            else:
                lse = torch.logsumexp(lse, dim=0)
                losses = losses.sum(dim=0)
            losses += lse
            if lse_square_scale != 0.0:
                losses += lse_square_scale * lse.square()
            losses.masked_fill_(labels == ignored_index, 0.0)

        ctx.save_for_backward(logits, lse, labels)
        ctx.smoothing = smoothing
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        logits, lse, labels = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        grid = lambda META: (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))  # noqa
        with torch.cuda.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](
                dlogits,  # data ptrs
                grad_losses,
                logits,
                lse,
                labels,
                ctx.smoothing,
                ctx.lse_square_scale,
                ctx.ignored_index,
                ctx.total_classes,
                ctx.class_start_idx,
                n_cols,  # shapes
                logits.stride(0),  # strides
                dlogits.stride(0),
                grad_losses.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
            )
        return dlogits, None, None, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    ignored_index=-100,
    inplace_backward: bool = False,
    process_group=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CrossEntropyLoss wrapper function for calling the custom autograd Function.
    """
    return CrossEntropyLoss.apply(
        logits,
        labels,
        label_smoothing,
        lse_square_scale,
        ignored_index,
        inplace_backward,
        process_group,
    )




##################################################################################################################################################


import torch

def test_cross_entropy_loss():
    results = {}
    # Test case 1: Basic test without label smoothing
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], device='cuda')
    labels = torch.tensor([0, 1], device='cuda')
    loss, _ = cross_entropy_loss(logits, labels)
    results['test_case_1'] = loss

    # Test case 2: Test with label smoothing
    label_smoothing = 0.1
    loss, _ = cross_entropy_loss(logits, labels, label_smoothing=label_smoothing)
    results['test_case_2'] = loss

    # Test case 3: Test with ignored index
    ignored_index = 1
    labels_with_ignored = torch.tensor([0, ignored_index], device='cuda')
    loss, _ = cross_entropy_loss(logits, labels_with_ignored, ignored_index=ignored_index)
    results['test_case_3'] = loss

    # Test case 4: Test with tensor parallelism (simulated)
    # Assuming a process group is set up for distributed training
    # For simplicity, we simulate this by using a single process
    process_group = None  # Replace with actual process group in distributed setting
    loss, _ = cross_entropy_loss(logits, labels, process_group=process_group)
    results['test_case_4'] = loss

    return results

result_gold = test_cross_entropy_loss()
