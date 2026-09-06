import torch
import triton
import triton.language as tl
import numpy as np

def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


MAX_FUSED_SIZE = 65536 // 4

_REDUCTION_MODE_NONE = tl.constexpr(0)
_REDUCTION_MODE_SUM = tl.constexpr(1)
_REDUCTION_MODE_MEAN = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,  # [B, S], prediction ptr, the kernel expects the prediction in log-space
    y_stride,  # int, prediction stride
    gt_ptr,  # [B, S], ground truth ptr
    gt_stride,  # int, ground truth stride
    loss_ptr,  # [B] or [B, S] if reduction == _REDUCTION_MODE_NONE, output ptr
    loss_stride,  # int, output stride
    n_cols,  # int, number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(y_true) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.sum(loss, axis=0)
            tl.store(loss_ptr, loss)
            loss_ptr += 1

@triton.jit
def _kldiv_kernel_backward(
    input_ptr,
    input_stride,
    target_ptr,
    target_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)

    input_ptr += pid * input_stride
    target_ptr += pid * target_stride

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

        tl.store(input_ptr + offsets, res, mask=mask)

def kldiv_forward_triton(y_pred, y_true, log_target, reduction):  # [B, S]  # [B, S]
    B, S = y_pred.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(S))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (B,)
    reduction = _str_to_reduction_mode[reduction]

    out_size = (B, S) if reduction == _REDUCTION_MODE_NONE.value else (B,)
    output_tensor = torch.zeros(
        out_size,
        dtype=torch.float32,
        device="cuda",
    )

    _kldiv_kernel_forward[grid](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        output_tensor,
        output_tensor.stride(0),
        S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
        reduction=reduction,
    )

    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / B
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(axis=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.mean(axis=0)
    else:
        return output_tensor

def kldiv_backward_triton(input, target, grad_output, log_target):
    B, S = input.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(S))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (B,)

    _kldiv_kernel_backward[grid](
        input,
        input.stride(0),
        target,
        target.stride(0),
        S,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
    )

    if torch.equal(
        grad_output,
        torch.tensor(
                1.0,
            dtype=grad_output.dtype,
            device="cuda",
        ),
    ):
        return input

    return input * grad_output




##################################################################################################################################################


def test_kldiv_triton():
    # Initialize random inputs
    B, S = 4, 8
    y_pred_np = np.random.rand(B, S).astype(np.float32)
    y_true_np = np.random.rand(B, S).astype(np.float32)

    # Parameters
    log_target = False
    reduction_modes = ['none', 'sum', 'mean', 'batchmean']

    # Ensure input tensors have requires_grad=True
    y_pred_torch = torch.tensor(y_pred_np, requires_grad=True, device="cuda")
    y_true_torch = torch.tensor(y_true_np, device="cuda")

    results = {}

    for i, reduction in enumerate(reduction_modes):
        # Triton forward
        triton_loss = kldiv_forward_triton(y_pred_torch, y_true_torch, log_target, reduction)

        # Reset PyTorch gradient
        y_pred_torch.grad = None

        # Triton backward
        grad_output_triton = torch.ones_like(triton_loss)  # Ensure shape consistency
        triton_grad = kldiv_backward_triton(y_pred_torch, y_true_torch, grad_output_triton, log_target)

        # Store results
        results[f'test_case_{i+1}'] = triton_grad.detach().cpu().numpy()

    return results

result_gold = test_kldiv_triton()
