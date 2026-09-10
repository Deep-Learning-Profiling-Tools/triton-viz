import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_forward_kernel(
    input_pointer, weight_pointer, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width,
    out_feat_dim, out_height, out_width,
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
    weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
    kernel_height: tl.constexpr, kernel_width: tl.constexpr,
    stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr,
    groups: tl.constexpr, fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    ):
    """
    2D-convolves over the input using weights.

    Args:
        input_pointer: Pointer to the input to convolve over.
            The input must be of shape [batch_dim, in_feat_dim, in_height, in_width].
        weight_pointer: Pointer to the weights input is convolved over by.
            The weights must be of shape [out_feat_dim, in_feat_dim, kernel_height, kernel_width].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, out_feat_dim, out_height, out_width].
        batch_dim: Batch dimension of the input and output.
        in_feat_dim: Dimensionality of the input features.
        in_height: Input height.
        in_width: Input width.
        out_feat_dim: Dimensionality of the output features.
        out_height: Output height.
        out_width: Output width.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_in_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_height_stride: Stride necessary to jump one element along the
            input's height dimension.
        input_width_stride: Stride necessary to jump one element along the
            input's width dimension.
        weight_out_feat_stride: Stride necessary to jump one element along the
            weights' output feature dimension.
        weight_in_feat_stride: Stride necessary to jump one element along the
            weights' input feature dimension.
        weight_height_stride: Stride necessary to jump one element along the
            weights' height dimension.
        weight_width_stride: Stride necessary to jump one element along the
            weights' width dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output's batch dimension.
        output_out_feat_stride: Stride necessary to jump one element along the
            output's feature dimension.
        output_height_stride: Stride necessary to jump one element along the
            output's height dimension.
        output_width_stride: Stride necessary to jump one element along the
            output's width dimension.
        kernel_height: Kernel height.
        kernel_width: Kernel width.
        stride_height: Stride of kernel across the height dimension.
        stride_width: Stride of kernel across the width dimension.
        padding_height: Padding applied to the input across the height dimension.
        padding_width: Padding applied to the input across the width dimension.
        groups: Number of groups for the convolution.
        fp16: Flag for loading the input and weights in FP16.
        tf32: Flag for performing matrix products in TF32.
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH: Block size across the batch, height, and
            width dimensions.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
    """
    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)

    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    batch_height_width_offset = (batch_height_width_pid * BLOCK_SIZE_BATCH_HEIGHT_WIDTH +
                                 tl.arange(0, BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
    batch_height_offset = batch_height_width_offset // out_width
    batch_offset = batch_height_offset // out_height

    output_feat_offset = (out_feat_pid * BLOCK_SIZE_OUT_FEAT +
                          tl.arange(0, BLOCK_SIZE_OUT_FEAT))
    output_height_offset = batch_height_offset % out_height
    output_width_offset = batch_height_width_offset % out_width

    input_pointer += (input_batch_stride * batch_offset +
                      input_in_feat_stride * group_pid * in_group_dim)[:, None]
    weight_pointer += (weight_out_feat_stride * output_feat_offset +
                       weight_out_feat_stride * group_pid * out_group_dim)[None, :]

    accum = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT),
                     dtype=tl.float32)

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_IN_FEAT):
                input_feat_offset = c + tl.arange(0, BLOCK_SIZE_IN_FEAT)
                input_height_offset = (h - padding_height +
                                       stride_height * output_height_offset)
                input_width_offset = (w - padding_width +
                                      stride_width * output_width_offset)

                curr_input_pointer = (input_pointer +
                                     (input_in_feat_stride * input_feat_offset)[None, :] +
                                     (input_height_stride * input_height_offset)[:, None] +
                                     (input_width_stride * input_width_offset)[:, None])
                curr_weight_pointer = (weight_pointer +
                                      (weight_in_feat_stride * input_feat_offset)[:, None] +
                                      (weight_height_stride * h) +
                                      (weight_width_stride * w))

                input_mask = ((batch_offset < batch_dim)[:, None] &
                              (input_feat_offset < in_group_dim)[None, :] &
                              (0 <= input_height_offset)[:, None] &
                              (input_height_offset < in_height)[:, None] &
                              (0 <= input_width_offset)[:, None] &
                              (input_width_offset < in_width)[:, None])
                weight_mask = ((input_feat_offset < in_group_dim)[:, None] &
                               (output_feat_offset < out_group_dim)[None, :])

                input_block = tl.load(curr_input_pointer, mask=input_mask)
                weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

                if fp16:
                    input_block = input_block.to(tl.float16)
                    weight_block = weight_block.to(tl.float16)

                accum += tl.dot(input_block, weight_block, allow_tf32=tf32)

    output_pointer += ((output_batch_stride * batch_offset)[:, None] +
                       (output_out_feat_stride * (group_pid * out_group_dim + output_feat_offset))[None, :] +
                       (output_height_stride * output_height_offset)[:, None] +
                       (output_width_stride * output_width_offset)[:, None])
    output_mask = ((batch_offset < batch_dim)[:, None] &
                   (output_feat_offset < out_group_dim)[None, :] &
                   (output_height_offset < out_height)[:, None] &
                   (output_width_offset < out_width)[:, None])

    tl.store(output_pointer, accum, mask=output_mask)


def conv2d_forward(input_tensor, weight_tensor, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, groups, fp16=False, tf32=False):
    # Extract dimensions
    batch_dim, in_feat_dim, in_height, in_width = input_tensor.shape
    out_feat_dim, _, _, _ = weight_tensor.shape
    out_height = (in_height + 2 * padding_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * padding_width - kernel_width) // stride_width + 1

    # Calculate strides
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride = input_tensor.stride()
    weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride = weight_tensor.stride()
    output_tensor = torch.empty((batch_dim, out_feat_dim, out_height, out_width), device=input_tensor.device, dtype=torch.float32)
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride = output_tensor.stride()

    # Define block sizes
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH = 128
    BLOCK_SIZE_IN_FEAT = 32
    BLOCK_SIZE_OUT_FEAT = 32

    # Launch the Triton kernel
    grid = (triton.cdiv(batch_dim * out_height * out_width, BLOCK_SIZE_BATCH_HEIGHT_WIDTH),
            triton.cdiv(out_feat_dim, BLOCK_SIZE_OUT_FEAT),
            groups)
    
    conv2d_forward_kernel[grid](
        input_tensor, weight_tensor, output_tensor,
        batch_dim, in_feat_dim, in_height, in_width,
        out_feat_dim, out_height, out_width,
        input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
        weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride,
        output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width,
        groups, fp16, tf32,
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT
    )
    
    return output_tensor




##################################################################################################################################################


# Test cases
def test_conv2d_forward():
    results = {}

    # Test case 1: Basic test
    input_tensor = torch.randn(1, 3, 32, 32, device='cuda', dtype=torch.float32)
    weight_tensor = torch.randn(16, 3, 3, 3, device='cuda', dtype=torch.float32)
    output_tensor = conv2d_forward(input_tensor, weight_tensor, 3, 3, 1, 1, 0, 0, 1)
    results["test_case_1"] = output_tensor

    # Test case 2: With padding and stride
    input_tensor = torch.randn(1, 3, 32, 32, device='cuda', dtype=torch.float32)
    weight_tensor = torch.randn(16, 3, 3, 3, device='cuda', dtype=torch.float32)
    output_tensor = conv2d_forward(input_tensor, weight_tensor, 3, 3, 2, 2, 1, 1, 1)
    results["test_case_2"] = output_tensor

    # Test case 3: With groups
    input_tensor = torch.randn(1, 6, 32, 32, device='cuda', dtype=torch.float32)
    weight_tensor = torch.randn(16, 3, 3, 3, device='cuda', dtype=torch.float32)
    output_tensor = conv2d_forward(input_tensor, weight_tensor, 3, 3, 1, 1, 0, 0, 2)
    results["test_case_3"] = output_tensor

    # Test case 4: Different kernel size
    input_tensor = torch.randn(1, 3, 32, 32, device='cuda', dtype=torch.float32)
    weight_tensor = torch.randn(16, 3, 5, 5, device='cuda', dtype=torch.float32)
    output_tensor = conv2d_forward(input_tensor, weight_tensor, 5, 5, 1, 1, 0, 0, 1)
    results["test_case_4"] = output_tensor

    return results

# Execute test
result_gold = test_conv2d_forward()
