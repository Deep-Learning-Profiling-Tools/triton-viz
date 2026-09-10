from typing import Literal
import torch
import triton
import triton.language as tl


MAX_FUSED_SIZE = 65536 // 4  # 65536 // 4 or 8 works the best
REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,  # [B, S], prediction ptr, the kernel expects the prediction in log-space
    y_stride,  # int, prediction stride
    gt_ptr,  # [B, S], ground truth ptr
    gt_stride,  # int, ground truth stride
    loss_ptr,  # [B] or [B, S] if reduction == _REDUCTION_MODE_NONE, output ptr
    loss_stride,  # int, output stride
    n_cols,  # int, number of columns in the input tensor
    eps,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == 0:  # _REDUCTION_MODE_NONE
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction != 0:
        tl.store(loss_ptr, loss_sum)


@triton.jit
def _kldiv_kernel_backward(
    target_ptr,
    target_stride,
    new_grads_ptr,
    new_grads_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)

    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)

        tl.store(new_grads_ptr + offsets, res, mask=mask)


def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):  # [BT, V]
    BT, V = y_pred.shape

    BLOCK_SIZE = min(16384, triton.next_power_of_2(V))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32

    grid = (BT,)
    reduction = {"none": 0, "sum": 1, "mean": 2, "batchmean": 3}[reduction]

    out_size = (BT, V) if reduction == 0 else (BT,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch.float32)

    _kldiv_kernel_forward[grid](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        output_tensor,
        output_tensor.stride(0),
        V,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
        reduction=reduction,
    )

    if reduction == 3:  # _REDUCTION_MODE_BATCHMEAN
        return output_tensor.sum() / BT
    elif reduction == 1:  # _REDUCTION_MODE_SUM
        return output_tensor.sum(dim=0)
    elif reduction == 2:  # _REDUCTION_MODE_MEAN
        return output_tensor.sum() / (BT * V)
    else:
        return output_tensor


def kldiv_backward_triton(target, grad_output, new_grads, log_target):
    BT, V = target.shape

    BLOCK_SIZE = min(16384, triton.next_power_of_2(V))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32

    grid = (BT,)

    _kldiv_kernel_backward[grid](
        target,
        target.stride(0),
        new_grads,
        new_grads.stride(0),
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
    )

    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return new_grads

    return new_grads * grad_output




##################################################################################################################################################


import torch

# Test cases for kldiv_forward_triton
def test_kldiv():
    # Define input tensors
    y_pred = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], device='cuda', dtype=torch.float32).log()
    y_true = torch.tensor([[0.1, 0.4, 0.5], [0.2, 0.5, 0.3]], device='cuda', dtype=torch.float32)
    eps = 1e-6

    # Test different reduction modes
    results = {}
    for i, reduction in enumerate(["none", "sum", "mean", "batchmean"]):
        output = kldiv_forward_triton(y_pred, y_true, log_target=False, reduction=reduction, eps=eps)
        results[f"test_case_{i+1}"] = output

    # Test with log_target=True
    y_true_log = y_true.log()
    output_log_target = kldiv_forward_triton(y_pred, y_true_log, log_target=True, reduction="sum", eps=eps)
    results["test_case_5"] = output_log_target

    # Define input tensors
    target = torch.tensor([[0.1, 0.4, 0.5], [0.2, 0.5, 0.3]], device='cuda', dtype=torch.float32)
    grad_output = torch.tensor(1.0, device='cuda', dtype=torch.float32)
    new_grads = torch.zeros_like(target)

    # Test with log_target=False
    backward_output = kldiv_backward_triton(target, grad_output, new_grads, log_target=False)
    results["test_case_6"] = backward_output

    # Test with log_target=True
    target_log = target.log()
    backward_output_log_target = kldiv_backward_triton(target_log, grad_output, new_grads, log_target=True)
    results["test_case_7"] = backward_output_log_target

    return results

# Run tests
result_gold = test_kldiv()
