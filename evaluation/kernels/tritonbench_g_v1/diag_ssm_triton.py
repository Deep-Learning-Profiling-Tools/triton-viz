import torch
import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(s_ptr, x_ptr, lambda_ptr, y_ptr, length,
                            batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    前向传播核函数（实数版本）

    参数:
        s_ptr: [batch_size, dim]
        x_ptr: [length, batch_size, dim]
        lambda_ptr: [dim]
        y_ptr: [length, batch_size, dim]
    """
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim
    s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)
    for t in range(length):
        offsets = t * batch_size * dim + col_offsets
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        s = s * Lambda + x
        tl.store(y_ptr + offsets, s, mask=mask)

@triton.jit
def diag_ssm_backward_kernel(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    反向传播核函数（实数版本）

    参数:
        s_ptr: [batch_size, dim]
        lambda_ptr: [dim]
        y_ptr: [length, batch_size, dim]
        grad_s_ptr: [batch_size, dim]
        grad_x_ptr: [length, batch_size, dim]
        grad_lambda_ptr: [batch_size, dim]
        grad_y_ptr: [length, batch_size, dim]
    """

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)

    # 初始化梯度为零
    grad_s = tl.zeros_like(Lambda)
    grad_Lambda = tl.zeros_like(Lambda)

    for i in range(length):
        # Triton 不支持 range(length - 1, -1, -1)
        t = length - 1 - i
        offsets = t * batch_size * dim + col_offsets

        grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        if t > 0:
            s = tl.load(
                y_ptr + offsets - batch_size * dim, mask=mask, other=0)
        else:
            s = tl.load(s_ptr + col_offsets, mask=mask, other=0)

        grad_s = grad_y + grad_s
        grad_x = grad_s
        grad_Lambda += grad_s * s
        grad_s = grad_s * Lambda

        tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

    tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)

@triton.jit
def diag_ssm_forward_kernel_complex(s_ptr, x_ptr, y_ptr, lambda_ptr,
                                    length, batch_size, dim,
                                    BLOCK_SIZE: tl.constexpr):
    """
    前向传播核函数（复数版本）

    参数:
        s_ptr: [batch_size, dim, 2]
        x_ptr: [length, batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
    """
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # 加载's'和'Lambda'的实部和虚部
    s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
    s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    for t in range(length):
        offsets = (t * batch_size * dim + col_offsets) * 2
        # 加载'x'的实部和虚部
        x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)

        # 复数的乘法和加法
        new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
        new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag

        # 存储更新后的实部和虚部
        tl.store(y_ptr + offsets, new_s_real, mask=mask)
        tl.store(y_ptr + offsets + 1, new_s_imag, mask=mask)

        # 更新's'以进行下一次迭代
        s_real, s_imag = new_s_real, new_s_imag

@triton.jit
def diag_ssm_backward_kernel_complex(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    反向传播核函数（复数版本）

    参数:
        s_ptr: [batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
        grad_s_ptr: [batch_size, dim, 2]
        grad_x_ptr: [length, batch_size, dim, 2]
        grad_lambda_ptr: [batch_size, dim, 2]
        grad_y_ptr: [length, batch_size, dim, 2]
    """

    # 复数自导数计算 \partial f / \partial z^*
    # 因此在计算过程中需要取共轭
    # 参考：https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
    # 所以在加载/存储梯度的虚部时，需要取反

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # 加载'Lambda'的实部和虚部
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    # 初始化梯度为零
    grad_s_real = tl.zeros_like(lambda_real)
    grad_s_imag = tl.zeros_like(lambda_imag)
    grad_lambda_real = tl.zeros_like(lambda_real)
    grad_lambda_imag = tl.zeros_like(lambda_imag)

    for i in range(length):
        # Triton 不支持 range(length - 1, -1, -1)
        t = length - 1 - i
        offsets = (t * batch_size * dim + col_offsets) * 2

        grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        grad_y_imag = -tl.load(
            grad_y_ptr + offsets + 1, mask=mask, other=0)
        if t > 0:
            s_real = tl.load(
                y_ptr + offsets - 2 * batch_size * dim, mask=mask, other=0)
            s_imag = tl.load(
                y_ptr + offsets - 2 * batch_size * dim + 1,
                mask=mask,
                other=0)
        else:
            s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
            s_imag = tl.load(
                s_ptr + 2 * col_offsets + 1, mask=mask, other=0)

        grad_s_real = grad_y_real + grad_s_real
        grad_s_imag = grad_y_imag + grad_s_imag
        grad_x_real = grad_s_real
        grad_x_imag = grad_s_imag
        grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
        grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
        grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
        grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real

        tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
        tl.store(grad_x_ptr + offsets + 1, -grad_x_imag, mask=mask)

    # 存储最终的梯度
    tl.store(grad_s_ptr + col_offsets * 2, grad_s_real, mask=mask)
    tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag, mask=mask)
    tl.store(
        grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
    tl.store(
        grad_lambda_ptr + col_offsets * 2 + 1,
        -grad_lambda_imag,
        mask=mask)

class _ssm_forward(torch.autograd.Function):
    # TODO 使用 @triton.autotune 选择最佳的 BLOCK_SIZE
    # 对于3090，BLOCK_SIZE = 128似乎效果良好
    BLOCK_SIZE = 128

    @staticmethod
    def forward(ctx, s, x, Lambda):
        assert s.is_contiguous() and x.is_contiguous() and Lambda.is_contiguous()
        length, batch_size, dim = x.shape
        n = batch_size * dim
        y = torch.zeros_like(x)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )

        if Lambda.dtype == torch.complex64:
            # 确保s和x是复数张量
            if not torch.is_complex(s):
                raise ValueError("当Lambda为复数时，s必须是复数张量")
            if not torch.is_complex(x):
                raise ValueError("当Lambda为复数时，x必须是复数张量")
            diag_ssm_forward_kernel_complex[grid](
                torch.view_as_real(s), torch.view_as_real(x),
                torch.view_as_real(y), torch.view_as_real(Lambda), length,
                batch_size, dim, _ssm_forward.BLOCK_SIZE)
        elif Lambda.dtype.is_floating_point:
            diag_ssm_forward_kernel[grid](s, x, Lambda, y, length,
                                          batch_size, dim,
                                          _ssm_forward.BLOCK_SIZE)
        else:
            raise ValueError("不支持的 dtype: %s" % Lambda.dtype)
        ctx.save_for_backward(s, y, Lambda)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        s, y, Lambda = ctx.saved_tensors
        length, batch_size, dim = y.shape
        grad_y = grad_y.contiguous()
        n = batch_size * dim
        grad_s = torch.empty_like(s)
        grad_x = torch.empty_like(grad_y)
        # grad_lambda 存储每个批次中 Lambda 的梯度
        # 我们将在内核完成后进行求和
        grad_lambda = torch.empty_like(s)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
        if Lambda.dtype == torch.complex64:
            diag_ssm_backward_kernel_complex[grid](
                torch.view_as_real(s), torch.view_as_real(Lambda),
                torch.view_as_real(y), torch.view_as_real(grad_s),
                torch.view_as_real(grad_x),
                torch.view_as_real(grad_lambda),
                torch.view_as_real(grad_y), length, batch_size, dim,
                _ssm_forward.BLOCK_SIZE)
        else:
            diag_ssm_backward_kernel[grid](
                s, Lambda, y, grad_s, grad_x, grad_lambda, grad_y, length,
                batch_size, dim, _ssm_forward.BLOCK_SIZE)
        return grad_s, grad_x, grad_lambda.sum(dim=0)

diag_ssm_forward_triton = _ssm_forward.apply

##################################################################################################################################################

def test_diag_ssm_triton():
    # 测试参数
    batch_size, dim, length = 2, 3, 5  # 定义测试张量的维度
    BLOCK_SIZE = 128  # Triton核的块大小

    # 初始化输入张量，确保 requires_grad=True
    # 实数张量
    s_real = torch.randn((batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True)
    x_real = torch.randn((length, batch_size, dim), dtype=torch.float32, device="cuda", requires_grad=True)
    Lambda_real = torch.rand((dim,), dtype=torch.float32, device="cuda", requires_grad=True)
    
    # 复数张量
    s_complex = torch.randn((batch_size, dim), dtype=torch.complex64, device="cuda", requires_grad=True)
    x_complex = torch.randn((length, batch_size, dim), dtype=torch.complex64, device="cuda", requires_grad=True)
    Lambda_complex = torch.rand((dim,), dtype=torch.complex64, device="cuda", requires_grad=True)

    # Triton前向传播，对于实数Lambda
    y_triton_real = diag_ssm_forward_triton(s_real, x_real, Lambda_real)
    # Triton前向传播，对于复数Lambda
    y_triton_complex = diag_ssm_forward_triton(s_complex, x_complex, Lambda_complex)

    # Triton反向传播，对于实数Lambda
    grad_output_real = torch.ones_like(y_triton_real, device="cuda")
    y_triton_real.backward(grad_output_real)
    # Triton反向传播，对于复数Lambda
    grad_output_complex = torch.ones_like(y_triton_complex, device="cuda")
    y_triton_complex.backward(grad_output_complex)

    results = {
        "test_case_1": {
            "y_triton_real": y_triton_real,
            "grad_s_real": s_real.grad.clone(),
            "grad_x_real": x_real.grad.clone(),
            "grad_Lambda_real": Lambda_real.grad.clone(),
        },
        "test_case_2": {
            "y_triton_complex": y_triton_complex,
            "grad_s_complex": s_complex.grad.clone(),
            "grad_x_complex": x_complex.grad.clone(),
            "grad_Lambda_complex": Lambda_complex.grad.clone(),
        }
    }
    
    return results

if __name__ == "__main__":
    result_gold = test_diag_ssm_triton()
    # 输出结果
    for test_case, outputs in result_gold.items():
        print(f"{test_case}:")
        for name, tensor in outputs.items():
            print(f"  {name}: {tensor}")
