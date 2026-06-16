import torch
import triton
import triton.language.extra.libdevice as libdevice
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import (
    TensorDescriptor,
    TensorDescriptorIm2Col,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    clc,
    tensor_memory_descriptor,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.blackwell.float2 import (
    Float2Tensor,
    fma as float2_fma,
    full_like as float2_full_like,
    pack as float2_pack,
    pack2,
    unpack as float2_unpack,
    unpack2,
)
from triton.experimental.gluon.language.nvidia.blackwell import (
    float2 as blackwell_float2,
)
from triton.experimental.gluon.language.nvidia.blackwell import tma as blackwell_tma
from triton.experimental.gluon.language.amd import cdna3 as amd_cdna3
from triton.experimental.gluon.language.amd import cdna4 as amd_cdna4
from triton.experimental.gluon.language.amd import rdna4 as amd_rdna4
from triton.experimental.gluon.language.amd import (
    warp_pipeline_stage as amd_warp_pipeline_stage,
)
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from triton.experimental.gluon.language.amd.gfx1250 import (
    async_copy as amd_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (
    buffer_load as amd_buffer_load,
)
from triton.experimental.gluon.language.amd.gfx1250 import (
    buffer_store as amd_buffer_store,
)
from triton.experimental.gluon.language.amd.gfx1250 import cluster as amd_cluster
from triton.experimental.gluon.language.amd.gfx1250 import tdm as amd_tdm
from triton.experimental.gluon.language.amd.gfx1250 import wmma as amd_wmma
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)

import triton_viz
from triton_viz.core.data import Load
from triton_viz.clients.sanitizer.sanitizer import SymbolicSanitizer


@gluon.jit
def _intro_memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


@gluon.jit
def _layout_memcpy_1d_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    XBLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    pid = gl.program_id(0)
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel
    value = gl.load(in_ptr + offsets, mask=mask, other=0.0)
    gl.store(out_ptr + offsets, value, mask=mask)


@gluon.jit
def _layout_memcpy_2d_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_in,
    ystride_in,
    xstride_out,
    ystride_out,
    layout: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)
    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    indices_x = start_x + gl.arange(
        0,
        XBLOCK,
        layout=gl.SliceLayout(dim=1, parent=layout),
    )
    indices_y = start_y + gl.arange(
        0,
        YBLOCK,
        layout=gl.SliceLayout(dim=0, parent=layout),
    )

    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    gl.store(out_ptr + out_offsets, value, mask=mask)


@gluon.jit
def _mask_and_offsets(
    start_x,
    start_y,
    xnumel,
    ynumel,
    xstride,
    ystride,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    indices_x = start_x + gl.arange(
        0,
        XBLOCK,
        layout=gl.SliceLayout(dim=1, parent=layout),
    )
    indices_y = start_y + gl.arange(
        0,
        YBLOCK,
        layout=gl.SliceLayout(dim=0, parent=layout),
    )
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)
    offsets = xstride * indices_x[:, None] + ystride * indices_y[None, :]
    return mask, offsets


@gluon.jit
def _layout_memcpy_2d_inout_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    ynumel,
    xstride_in,
    ystride_in,
    xstride_out,
    ystride_out,
    layout_in: gl.constexpr,
    layout_out: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)
    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    mask_in, in_offsets = _mask_and_offsets(
        start_x,
        start_y,
        xnumel,
        ynumel,
        xstride_in,
        ystride_in,
        XBLOCK,
        YBLOCK,
        layout_in,
    )
    mask_out, out_offsets = _mask_and_offsets(
        start_x,
        start_y,
        xnumel,
        ynumel,
        xstride_out,
        ystride_out,
        XBLOCK,
        YBLOCK,
        layout_out,
    )
    value = gl.load(in_ptr + in_offsets, mask=mask_in, other=0.0)
    gl.store(out_ptr + out_offsets, gl.convert_layout(value, layout_out), mask=mask_out)


@gluon.jit
def _elementwise_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_c,
    ystride_c,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask, other=0.0)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask, other=0.0)
        gl.store(c_ptrs + ystride_c * yoffs[None, :], a_val + b_val, mask=mask)


@gluon.jit
def _cpasync_memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    cp.async_load(smem, in_ptr + offsets, mask=mask)
    cp.commit_group()
    cp.wait_group(0)
    gl.store(out_ptr + offsets, smem.load(layout), mask=mask)


@gluon.jit
def _cpasync_elementwise_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_c,
    ystride_c,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    smem_layout: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    yoffs = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
    a_smem = gl.allocate_shared_memory(gl.float32, [XBLOCK, YBLOCK], smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float32, [XBLOCK, YBLOCK], smem_layout)

    cp.async_load(
        a_smem,
        a_ptr + xstride_a * xoffs[:, None] + ystride_a * yoffs[None, :],
        mask=mask,
    )
    cp.async_load(
        b_smem,
        b_ptr + xstride_b * xoffs[:, None] + ystride_b * yoffs[None, :],
        mask=mask,
    )
    cp.commit_group()
    cp.wait_group(0)
    c_val = a_smem.load(layout) + b_smem.load(layout)
    gl.store(
        c_ptr + xstride_c * xoffs[:, None] + ystride_c * yoffs[None, :],
        c_val,
        mask=mask,
    )


@gluon.jit
def _issue_cpasync_loads(
    copy_idx,
    a_smem,
    b_smem,
    a_ptrs,
    ystride_a,
    b_ptrs,
    xmask,
    ynumel,
    y_idx,
    ystride_b,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
):
    yoffs = copy_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    cp.async_load(
        a_smem.index(copy_idx % num_buffers),
        a_ptrs + ystride_a * yoffs[None, :],
        mask=mask,
    )
    cp.async_load(
        b_smem.index(copy_idx % num_buffers),
        b_ptrs + ystride_b * yoffs[None, :],
        mask=mask,
    )
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def _perform_cpasync_add(
    read_idx,
    a_smem,
    b_smem,
    c_ptrs,
    ynumel,
    ystride_c,
    y_idx,
    xmask,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
    layout: gl.constexpr,
):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    yoffs = read_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    gl.store(c_ptrs + ystride_c * yoffs[None, :], a_val + b_val, mask=mask)
    return read_idx + 1


@gluon.jit
def _cpasync_elementwise_add_pipelined_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_c,
    ystride_c,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    smem_layout: gl.constexpr,
    num_buffers: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    y_idx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    xmask = (xoffs < xnumel)[:, None]
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]
    a_smem = gl.allocate_shared_memory(
        gl.float32,
        [num_buffers, XBLOCK, YBLOCK],
        smem_layout,
    )
    b_smem = gl.allocate_shared_memory(
        gl.float32,
        [num_buffers, XBLOCK, YBLOCK],
        smem_layout,
    )
    copy_idx = 0
    read_idx = 0
    for _ in gl.static_range(num_buffers - 1):
        copy_idx = _issue_cpasync_loads(
            copy_idx,
            a_smem,
            b_smem,
            a_ptrs,
            ystride_a,
            b_ptrs,
            xmask,
            ynumel,
            y_idx,
            ystride_b,
            YBLOCK,
            num_buffers,
        )
    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_idx = _issue_cpasync_loads(
            copy_idx,
            a_smem,
            b_smem,
            a_ptrs,
            ystride_a,
            b_ptrs,
            xmask,
            ynumel,
            y_idx,
            ystride_b,
            YBLOCK,
            num_buffers,
        )
        cp.wait_group(num_buffers - 1)
        read_idx = _perform_cpasync_add(
            read_idx,
            a_smem,
            b_smem,
            c_ptrs,
            ynumel,
            ystride_c,
            y_idx,
            xmask,
            YBLOCK,
            num_buffers,
            layout,
        )
    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = _perform_cpasync_add(
            read_idx,
            a_smem,
            b_smem,
            c_ptrs,
            ynumel,
            ystride_c,
            y_idx,
            xmask,
            YBLOCK,
            num_buffers,
            layout,
        )


@gluon.jit
def _tma_memcpy_1d_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], in_desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], in_desc.layout)
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid * XBLOCK], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    tma.async_copy_shared_to_global(out_desc, [pid * XBLOCK], smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _issue_tma_loads(
    copy_index,
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    bars,
    xoff,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
):
    yoff = copy_index * YBLOCK
    bar = bars.index(copy_index % num_buffers)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_load(a_desc, [xoff, yoff], bar, a_smem.index(copy_index % num_buffers))
    tma.async_load(b_desc, [xoff, yoff], bar, b_smem.index(copy_index % num_buffers))
    return copy_index + 1


@gluon.jit
def _perform_tma_add(
    read_index,
    bars,
    a_smem,
    b_smem,
    c_smem,
    c_desc,
    xoff,
    layout: gl.constexpr,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
):
    mbarrier.wait(bars.index(read_index % num_buffers), read_index // num_buffers & 1)
    c_smem.store(
        a_smem.index(read_index % num_buffers).load(layout)
        + b_smem.index(read_index % num_buffers).load(layout)
    )
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [xoff, read_index * YBLOCK], c_smem)
    return read_index + 1


@gluon.jit
def _tma_elementwise_add_kernel(
    a_desc,
    b_desc,
    c_desc,
    xnumel,
    ynumel,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
    num_buffers: gl.constexpr,
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoff = pid * XBLOCK
    dtype: gl.constexpr = a_desc.type.block_type.element_ty
    a_smem = gl.allocate_shared_memory(
        dtype,
        [num_buffers, XBLOCK, YBLOCK],
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        dtype,
        [num_buffers, XBLOCK, YBLOCK],
        b_desc.layout,
    )
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], c_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], a_desc.layout)
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    copy_index = 0
    read_index = 0
    for _ in gl.static_range(num_buffers - 1):
        copy_index = _issue_tma_loads(
            copy_index,
            a_desc,
            b_desc,
            a_smem,
            b_smem,
            bars,
            xoff,
            YBLOCK,
            num_buffers,
        )
    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_index = _issue_tma_loads(
            copy_index,
            a_desc,
            b_desc,
            a_smem,
            b_smem,
            bars,
            xoff,
            YBLOCK,
            num_buffers,
        )
        read_index = _perform_tma_add(
            read_index,
            bars,
            a_smem,
            b_smem,
            c_smem,
            c_desc,
            xoff,
            layout,
            YBLOCK,
            num_buffers,
        )
    for _ in gl.static_range(num_buffers - 1):
        read_index = _perform_tma_add(
            read_index,
            bars,
            a_smem,
            b_smem,
            c_smem,
            c_desc,
            xoff,
            layout,
            YBLOCK,
            num_buffers,
        )
    tma.store_wait(pendings=0)


@gluon.jit
def _tma_message_passing_kernel(
    message_desc, ready, output, MESSAGE_SIZE: gl.constexpr
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, MESSAGE_SIZE, layout)
    smem = gl.allocate_shared_memory(
        message_desc.dtype,
        message_desc.block_shape,
        message_desc.layout,
    )

    if pid == 0:
        smem.store(offsets + 1000)
        fence_async_shared()
        tma.async_copy_shared_to_global(message_desc, [0], smem)
        tma.store_wait(pendings=0, read_only=False)
        gl.atomic_xchg(ready, 1, sem="release", scope="gpu")
    else:
        ready_value = 0
        while ready_value != 1:
            ready_value = gl.atomic_add(ready, 0, sem="acquire", scope="gpu")
        bar = gl.allocate_shared_memory(gl.int64, [1], message_desc.layout)
        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, message_desc.block_type.nbytes)
        tma.async_load(message_desc, [0], bar, smem)
        mbarrier.wait(bar, phase=0, deps=[smem])
        mbarrier.invalidate(bar)
        gl.store(output + offsets, smem.load(layout))


@gluon.jit
def _tma_multicast_copy_kernel(in_desc, out_desc):
    gl.static_assert(gl.num_ctas() == 2)
    smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.nbytes_per_cta)
    tma.async_load(in_desc, [0, 0], bar, smem, multicast=True)
    mbarrier.wait(bar, phase=0, deps=[smem])
    tma.async_copy_shared_to_global(out_desc, [0, 0], smem)


@gluon.jit
def _tma_alias_shared_reshape_kernel(in_desc, out_desc):
    smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(in_desc, [0, 0], bar, smem)
    mbarrier.wait(bar, phase=0, deps=[smem])
    reshaped = smem.reshape((2, 16)).reshape((4, 8)).permute((1, 0))
    tma.async_copy_shared_to_global(out_desc, [0, 0], reshaped)


@gluon.jit
def _tma_im2col_kernel(
    in_desc,
    out_desc,
    coord_n,
    coord_h,
    coord_w,
    coord_c,
    offset_h: gl.constexpr,
    offset_w: gl.constexpr,
):
    smem = gl.allocate_shared_memory(in_desc.dtype, in_desc.block_shape, in_desc.layout)
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load_im2col(
        in_desc,
        [coord_n, coord_h, coord_w, coord_c],
        [offset_h, offset_w],
        bar,
        smem,
    )
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    tma.async_copy_shared_to_global(out_desc, [0, 0], smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _conv2d_im2col_wgmma_kernel(
    in_desc,
    weight_desc,
    out_desc,
    R: gl.constexpr,
    S: gl.constexpr,
    Ci: gl.constexpr,
    out_h: gl.constexpr,
    out_w: gl.constexpr,
    pad_h: gl.constexpr,
    pad_w: gl.constexpr,
    stride_h: gl.constexpr,
    stride_w: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    dtype: gl.constexpr = in_desc.dtype
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    offs_m = pid_m * BLOCK_M
    batch_id = offs_m // (out_h * out_w)
    m_residual = offs_m % (out_h * out_w)
    out_y = m_residual // out_w
    out_x = m_residual % out_w

    a_smem = gl.allocate_shared_memory(dtype, in_desc.block_shape, in_desc.layout)
    b_smem = gl.allocate_shared_memory(
        dtype, weight_desc.block_shape, weight_desc.layout
    )
    mma_layout: gl.constexpr = _wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    ci_num_blocks: gl.constexpr = gl.cdiv(Ci, BLOCK_K)
    total_k_iters: gl.constexpr = R * S * ci_num_blocks
    for k_iter in range(total_k_iters):
        ci_block: gl.constexpr = k_iter % ci_num_blocks
        rs_idx: gl.constexpr = k_iter // ci_num_blocks
        r: gl.constexpr = rs_idx // S
        s: gl.constexpr = rs_idx % S
        mbarrier.expect(bar, in_desc.block_type.nbytes + weight_desc.block_type.nbytes)
        tma.async_load_im2col(
            in_desc,
            [
                batch_id,
                out_y * stride_h - pad_h,
                out_x * stride_w - pad_w,
                ci_block * BLOCK_K,
            ],
            [r, s],
            bar,
            a_smem,
        )
        k_offset: gl.constexpr = r * S * Ci + s * Ci + ci_block * BLOCK_K
        tma.async_load(weight_desc, [pid_n * BLOCK_N, k_offset], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1
        acc = warpgroup_mma(a_smem, b_smem.permute((1, 0)), acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)
    out_smem = gl.allocate_shared_memory(
        out_desc.dtype, out_desc.block_shape, out_desc.layout
    )
    out_smem.store(acc.to(out_desc.dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(out_desc, [offs_m, pid_n * BLOCK_N], out_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _tma_oob_kernel(
    x,
    m: gl.constexpr,
    n: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    layout: gl.constexpr,
):
    desc = tma.make_tensor_descriptor(
        x,
        [m, n],
        [n, 1],
        [block_m, block_n],
        layout,
    )
    smem = gl.allocate_shared_memory(gl.float32, [block_m, block_n], layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], layout)
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.nbytes_per_cta)
    tma.async_load(desc, [m, 0], bar, smem)


@gluon.aggregate
class _Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return _Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return _Counter(index, phase, self.num_barriers)


@gluon.aggregate
class _PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = gl.minimum(pid_start + pid_per_kernel, num_pid)
        return _PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


def _GroupedPersistentTileScheduler(GROUP_SIZE_M):
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    @gluon.aggregate
    class _GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return _GroupedPersistentTileSchedulerImpl(
                start_pid,
                num_pid_m,
                num_pid_in_group,
                num_pid,
            )

        @gluon.jit
        def get_num_tiles(self):
            remaining_tiles = self.num_pid - self.start_pid
            return gl.cdiv(remaining_tiles, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = gl.minimum(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    return _GroupedPersistentTileSchedulerImpl


@gluon.jit
def _layout_anchor_and_barrier_kernel(in_ptr, out_ptr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, N, layout)
    values = gl.load(in_ptr + offsets)
    values = gl.set_auto_layout(values, layout)
    gl.barrier()
    gl.store(out_ptr + offsets, values + 1.0)


@gluon.jit
def _float2_helpers_kernel(
    f0_ptr,
    f1_ptr,
    out_sum0,
    out_sum1,
    out_fma0,
    out_fma1,
    out_full0,
    out_full1,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout)
    f0 = gl.load(f0_ptr + offsets)
    f1 = gl.load(f1_ptr + offsets)
    lhs = pack2(f0, f1)
    rhs = pack2(f1, f0)
    sum0, sum1 = unpack2(lhs + rhs)
    fma0, fma1 = unpack2(float2_fma(lhs, rhs, lhs))
    two = gl.full((BLOCK,), 2.0, gl.float32, layout)
    full0, full1 = unpack2(lhs * float2_full_like(lhs, two))
    gl.store(out_sum0 + offsets, sum0)
    gl.store(out_sum1 + offsets, sum1)
    gl.store(out_fma0 + offsets, fma0)
    gl.store(out_fma1 + offsets, fma1)
    gl.store(out_full0 + offsets, full0)
    gl.store(out_full1 + offsets, full1)


@gluon.jit
def _float2_pack_reduce_kernel(values_ptr, roundtrip_ptr, even_sum_ptr, odd_sum_ptr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, 4, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, 8, gl.SliceLayout(0, layout))
    values = gl.load(values_ptr + offs_m[:, None] * 8 + offs_n[None, :])
    packed = blackwell_float2.pack(values, axis=1)
    unpacked = blackwell_float2.unpack(packed, axis=1)
    reduced = packed.sum(axis=1)
    even_sum, odd_sum = blackwell_float2.unpack2(reduced)
    gl.store(roundtrip_ptr + offs_m[:, None] * 8 + offs_n[None, :], unpacked)
    row_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    row_offsets = gl.arange(0, 4, row_layout)
    gl.store(even_sum_ptr + row_offsets, gl.convert_layout(even_sum, row_layout))
    gl.store(odd_sum_ptr + row_offsets, gl.convert_layout(odd_sum, row_layout))


@gluon.jit
def _float2_constructor_convert_kernel(values_ptr, out_ptr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, 2, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, 4, gl.SliceLayout(0, layout))
    values = gl.load(values_ptr + offs_m[:, None] * 4 + offs_n[None, :])
    packed = float2_pack(values, axis=1)
    converted = gl.convert_layout(
        packed.value,
        packed.value.type.layout,
        assert_trivial=True,
    )
    rebuilt = Float2Tensor(converted)
    scale = gl.full(
        rebuilt.value.shape,
        2.0,
        gl.float32,
        layout=rebuilt.value.type.layout,
    )
    scaled = float2_fma(rebuilt, float2_full_like(rebuilt, scale), rebuilt)
    unpacked = float2_unpack(scaled, axis=1)
    gl.store(out_ptr + offs_m[:, None] * 4 + offs_n[None, :], unpacked)


@gluon.constexpr_function
def _wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    n = min(BLOCK_N, 64)
    while BLOCK_N % n != 0:
        n -= 8
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[num_warps, 1],
        instr_shape=[m, n, 256 // dtype.primitive_bitwidth],
    )


@gluon.aggregate
class _PersistentWGMMA:
    acc: gl.tensor
    use_acc: gl.tensor

    @gluon.jit
    def initialize(
        dtype: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        layout: gl.constexpr = _wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=layout)
        return _PersistentWGMMA(acc, gl.to_tensor(False))

    @gluon.jit
    def issue_async_mma(self, a, b):
        acc = warpgroup_mma(
            a,
            b,
            self.acc,
            is_async=True,
            use_acc=self.use_acc,
        )
        return _PersistentWGMMA(acc, gl.to_tensor(True))

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc,))
        return _PersistentWGMMA(acc, self.use_acc)

    @gluon.jit
    def take_result(self, splitn: gl.constexpr = False):
        del splitn
        return self.acc, _PersistentWGMMA(self.acc, gl.to_tensor(False))


@gluon.jit
def _small_wgmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    d_desc,
    LHS_IN_REG: gl.constexpr,
    num_warps: gl.constexpr,
):
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_type.shape,
        b_desc.layout,
    )
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype,
        c_desc.block_type.shape,
        c_desc.layout,
    )
    mbarrier.expect(
        bar,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [0, 0], bar, a_smem)
    tma.async_load(b_desc, [0, 0], bar, b_smem)
    tma.async_load(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    c_layout: gl.constexpr = _wgmma_layout(
        a_desc.dtype,
        d_desc.block_type.shape[0],
        d_desc.block_type.shape[1],
        num_warps,
    )
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_desc.dtype.primitive_bitwidth,
    )
    gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
    gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))
    a = a_smem.load(a_reg_layout) if LHS_IN_REG else a_smem
    c = c_smem.load(c_layout)
    d = warpgroup_mma(a, b_smem, c, is_async=True, use_acc=True)
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))

    d_smem = gl.allocate_shared_memory(
        d_desc.dtype,
        d_desc.block_type.shape,
        d_desc.layout,
    )
    d_smem.store(d)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _blocked_wgmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    TRANSPOSE_B: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    K = a_desc.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    mma_layout: gl.constexpr = _wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k], bar, a_smem)
        if TRANSPOSE_B:
            tma.async_load(b_desc, [off_n, k], bar, b_smem)
            b = b_smem.permute((1, 0))
        else:
            tma.async_load(b_desc, [k, off_n], bar, b_smem)
            b = b_smem
        mbarrier.wait(bar, phase=phase)
        phase ^= 1
        acc = warpgroup_mma(a_smem, b, acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _pipelined_wgmma_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    K = a_desc.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    a_smem = gl.allocate_shared_memory(
        dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )
    index = 0
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    mma_layout: gl.constexpr = _wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = warpgroup_mma_init(
        gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        a = a_smem.index(index)
        b = b_smem.index(index)
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k], bar, a)
        tma.async_load(b_desc, [k, off_n], bar, b)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        acc = warpgroup_mma(a, b, acc, is_async=True)
        index ^= 1

    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.invalidate(bar)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _persistent_wgmma_matmul_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    SchedulerImpl: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    K = a_desc.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mma = _PersistentWGMMA.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    phase = 0
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        a_smem = gl.allocate_shared_memory(
            dtype, a_desc.block_type.shape, a_desc.layout
        )
        b_smem = gl.allocate_shared_memory(
            dtype, b_desc.block_type.shape, b_desc.layout
        )
        for k in range(0, K, BLOCK_K):
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_load(a_desc, [off_m, k], bar, a_smem)
            tma.async_load(b_desc, [k, off_n], bar, b_smem)
            mbarrier.wait(bar, phase=phase)
            phase ^= 1
            mma = mma.wait_num_outstanding(0)
            mma = mma.issue_async_mma(a_smem, b_smem)

        mma = mma.wait_num_outstanding(0)
        c_smem = gl.allocate_shared_memory(
            dtype, c_desc.block_type.shape, c_desc.layout
        )
        c, mma = mma.take_result()
        c_smem.store(c.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)
    mbarrier.invalidate(bar)


@gluon.jit
def _issue_loads_stealb(
    producer,
    a_desc,
    b_desc,
    off_m,
    off_n,
    k,
    bars,
    a_bufs,
    b_bufs,
    STEALB: gl.constexpr,
    num_buffers: gl.constexpr,
    pred=True,
):
    index = producer % num_buffers
    b_index = producer % (num_buffers + STEALB)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(
        bar,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes,
        pred=pred,
    )
    tma.async_load(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_load(b_desc, [k, off_n], bar, b_bufs.index(b_index), pred)
    return producer


@gluon.jit
def _issue_mma_stealb(
    consumer,
    mma,
    bars,
    a_bufs,
    b_bufs,
    STEALB: gl.constexpr,
    num_buffers: gl.constexpr,
):
    index = consumer % num_buffers
    b_index = consumer % (num_buffers + STEALB)
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(b_index))
    return consumer, mma


@gluon.jit
def _persistent_wgmma_pipelined_kernel(
    a_desc,
    b_desc,
    c_desc,
    c_half_desc,
    M,
    N,
    SchedulerImpl: gl.constexpr,
    num_buffers: gl.constexpr,
    STEALB: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    K = a_desc.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    use_split_n_load: gl.constexpr = STEALB and BLOCK_M != BLOCK_K
    gl.static_assert(num_buffers >= 3, "expected at least 3 buffers")
    gl.static_assert(
        not use_split_n_load or BLOCK_M == BLOCK_K * 2,
        "split-N epilogue expects a B tile to hold one C half",
    )

    a_bufs = gl.allocate_shared_memory(
        dtype,
        [num_buffers] + a_desc.block_type.shape,
        a_desc.layout,
    )
    b_bufs = gl.allocate_shared_memory(
        dtype,
        [num_buffers + STEALB] + b_desc.block_type.shape,
        b_desc.layout,
    )
    if not STEALB:
        c_smem = gl.allocate_shared_memory(
            dtype, c_desc.block_type.shape, c_desc.layout
        )
    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    producer = 0
    consumer = 0
    mma = _PersistentWGMMA.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = _issue_loads_stealb(
            producer,
            a_desc,
            b_desc,
            off_m,
            off_n,
            ki,
            bars,
            a_bufs,
            b_bufs,
            STEALB,
            num_buffers,
        )
    k = BLOCK_K * (num_buffers - 2)
    producer = _issue_loads_stealb(
        producer,
        a_desc,
        b_desc,
        off_m,
        off_n,
        k,
        bars,
        a_bufs,
        b_bufs,
        STEALB,
        num_buffers,
    )

    for _ in range(num_tiles):
        consumer, mma = _issue_mma_stealb(
            consumer,
            mma,
            bars,
            a_bufs,
            b_bufs,
            STEALB,
            num_buffers,
        )
        if STEALB:
            tma.store_wait(pendings=0)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = _issue_loads_stealb(
                producer,
                a_desc,
                b_desc,
                off_m,
                off_n,
                k,
                bars,
                a_bufs,
                b_bufs,
                STEALB,
                num_buffers,
            )
            consumer, mma = _issue_mma_stealb(
                consumer,
                mma,
                bars,
                a_bufs,
                b_bufs,
                STEALB,
                num_buffers,
            )

        epilogue_off_m = off_m
        epilogue_off_n = off_n
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = _issue_loads_stealb(
                producer,
                a_desc,
                b_desc,
                off_m,
                off_n,
                ki,
                bars,
                a_bufs,
                b_bufs,
                STEALB,
                num_buffers,
                pred,
            )
            consumer, mma = _issue_mma_stealb(
                consumer,
                mma,
                bars,
                a_bufs,
                b_bufs,
                STEALB,
                num_buffers,
            )
        k = BLOCK_K * (num_buffers - 2)
        producer = _issue_loads_stealb(
            producer,
            a_desc,
            b_desc,
            off_m,
            off_n,
            k,
            bars,
            a_bufs,
            b_bufs,
            STEALB,
            num_buffers,
            pred,
        )

        mma = mma.wait_num_outstanding(0)
        c, mma = mma.take_result(splitn=use_split_n_load)
        c = c.to(dtype)
        if not STEALB:
            c_buf = c_smem
            tma.store_wait(pendings=0)
            c_buf.store(c)
            fence_async_shared()
            tma.async_copy_shared_to_global(
                c_desc,
                [epilogue_off_m, epilogue_off_n],
                c_buf,
            )
        elif use_split_n_load:
            c0, c1 = c.reshape((BLOCK_M, 2, BLOCK_N // 2)).permute(0, 2, 1).split()
            c0_buf = b_bufs.index(producer % (num_buffers + STEALB))._reinterpret(
                shape=c_half_desc.block_type.shape,
                layout=c_half_desc.layout,
            )
            c1_buf = b_bufs.index((producer + 1) % (num_buffers + STEALB))._reinterpret(
                shape=c_half_desc.block_type.shape,
                layout=c_half_desc.layout,
            )
            c0_buf.store(c0)
            c1_buf.store(c1)
            fence_async_shared()
            tma.async_copy_shared_to_global(
                c_half_desc,
                [epilogue_off_m, epilogue_off_n],
                c0_buf,
            )
            tma.async_copy_shared_to_global(
                c_half_desc,
                [epilogue_off_m, epilogue_off_n + BLOCK_N // 2],
                c1_buf,
            )
        else:
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))
            c_buf.store(c)
            fence_async_shared()
            tma.async_copy_shared_to_global(
                c_desc,
                [epilogue_off_m, epilogue_off_n],
                c_buf,
            )
    tma.store_wait(pendings=0)


@gluon.jit
def _tmem_roundtrip_kernel(
    in_ptr,
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    num_warps: gl.constexpr,
):
    global_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, num_warps],
        [1, 0],
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = gl.load(in_ptr + offs)
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
    )
    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )
    value = gl.convert_layout(value, tmem.get_reg_layout())
    tmem.store(value)
    out = gl.convert_layout(tmem.load(), global_layout)
    gl.store(out_ptr + offs, out)


@gluon.jit
def _tcgen05_small_mma_kernel(
    a_desc,
    b_desc,
    c_desc,
    d_desc,
    tmem_block: gl.constexpr,
    LHS_IN_TMEM: gl.constexpr,
    USE_COMMIT: gl.constexpr,
):
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_type.shape,
        b_desc.layout,
    )
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype,
        c_desc.block_type.shape,
        c_desc.layout,
    )
    mbarrier.expect(
        bar,
        a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes,
    )
    tma.async_load(a_desc, [0, 0], bar, a_smem)
    tma.async_load(b_desc, [0, 0], bar, b_smem)
    tma.async_load(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)

    M: gl.constexpr = d_desc.block_type.shape[0]
    N: gl.constexpr = d_desc.block_type.shape[1]
    K: gl.constexpr = a_desc.block_type.shape[1]
    acc_tmem_layout: gl.constexpr = TensorMemoryLayout(
        tmem_block,
        col_stride=32 // d_desc.dtype.primitive_bitwidth,
    )
    acc_tmem = allocate_tensor_memory(d_desc.dtype, [M, N], acc_tmem_layout)
    acc = c_smem.load(acc_tmem.get_reg_layout())
    acc_tmem.store(acc)

    if LHS_IN_TMEM:
        lhs_tmem_layout: gl.constexpr = TensorMemoryLayout(tmem_block, col_stride=1)
        lhs_tmem = allocate_tensor_memory(a_desc.dtype, [M, K], lhs_tmem_layout)
        lhs = a_smem.load(lhs_tmem.get_reg_layout())
        lhs_tmem.store(lhs)
        a = lhs_tmem
    else:
        a = a_smem

    if USE_COMMIT:
        tcgen05_mma(a, b_smem, acc_tmem)
        tcgen05_commit(bar)
    else:
        tcgen05_mma(a, b_smem, acc_tmem, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    d_smem = gl.allocate_shared_memory(
        d_desc.dtype,
        d_desc.block_type.shape,
        d_desc.layout,
    )
    d_smem.store(acc_tmem.load())
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _tcgen05_scaled_mma_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    VEC: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, layout))
    offs_k = gl.arange(0, K, gl.SliceLayout(0, layout))
    offs_scale = gl.arange(0, K // VEC, gl.SliceLayout(0, layout))

    a = gl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = gl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    a_scales = gl.load(a_scale_ptr + offs_m[:, None] * (K // VEC) + offs_scale[None, :])
    b_scales = gl.load(b_scale_ptr + offs_n[:, None] * (K // VEC) + offs_scale[None, :])

    smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [M, K], a_ptr.dtype.element_ty
    )
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [K, N],
        b_ptr.dtype.element_ty,
    )
    a_smem = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [M, K], smem_layout)
    b_smem = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [K, N], b_smem_layout)
    a_smem.store(a)
    b_smem.store(b)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(
        a_scale_ptr.dtype.element_ty,
        [M, K // VEC],
        scale_layout,
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale_ptr.dtype.element_ty,
        [N, K // VEC],
        scale_layout,
    )
    a_scale_tmem.store(a_scales)
    b_scale_tmem.store(b_scales)

    acc_layout: gl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [M, N], acc_layout)
    tcgen05_mma_scaled(
        a_smem,
        b_smem,
        acc,
        a_scale_tmem,
        b_scale_tmem,
        "e4m3",
        "e4m3",
        use_acc=False,
    )
    result = gl.convert_layout(acc.load(), layout)
    gl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], result)


@gluon.jit
def _tcgen05_descriptor_scaled_mma_kernel(
    a_desc,
    b_desc,
    c_desc,
    a_scale_ptr,
    b_scale_ptr,
    VEC: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    K = a_desc.shape[1]
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_type.shape,
        b_desc.layout,
    )
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(
        a_scale_ptr.dtype.element_ty,
        [BLOCK_M, BLOCK_K // VEC],
        scale_layout,
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_K // VEC],
        scale_layout,
    )
    acc_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], acc_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    use_acc = False
    phase = 0
    scale_k: gl.constexpr = BLOCK_K // VEC
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    scale_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    scale_n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    scale_cols = gl.arange(0, scale_k, gl.SliceLayout(0, layout))
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k], bar, a_smem)
        tma.async_load(b_desc, [off_n, k], bar, b_smem)
        mbarrier.wait(bar, phase)

        a_scale_base = (off_m + scale_m[:, None]) * (K // VEC) + k // VEC
        b_scale_base = (off_n + scale_n[:, None]) * (K // VEC) + k // VEC
        a_scales = gl.load(a_scale_ptr + a_scale_base + scale_cols[None, :])
        b_scales = gl.load(b_scale_ptr + b_scale_base + scale_cols[None, :])
        a_scale_tmem.store(gl.convert_layout(a_scales, a_scale_tmem.get_reg_layout()))
        b_scale_tmem.store(gl.convert_layout(b_scales, b_scale_tmem.get_reg_layout()))

        tcgen05_mma_scaled(
            a_smem,
            b_smem.permute((1, 0)),
            acc,
            a_scale_tmem,
            b_scale_tmem,
            "e4m3",
            "e4m3",
            use_acc=use_acc,
        )
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    c_smem.store(acc.load().to(c_desc.dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(0)


@gluon.jit
def _tcgen05_descriptor_scaled_tma_pipeline_kernel(
    a_desc,
    b_desc,
    c_desc,
    a_scale_desc,
    b_scale_desc,
    VEC: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    SCALE_K: gl.constexpr = BLOCK_K // VEC
    K = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        b_desc.block_type.shape,
        b_desc.layout,
    )
    a_scale_smem = gl.allocate_shared_memory(
        a_scale_desc.dtype,
        a_scale_desc.block_type.shape,
        a_scale_desc.layout,
    )
    b_scale_smem = gl.allocate_shared_memory(
        b_scale_desc.dtype,
        b_scale_desc.block_type.shape,
        b_scale_desc.layout,
    )
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(
        a_scale_desc.dtype,
        [BLOCK_M, SCALE_K],
        scale_layout,
    )
    b_scale_tmem = allocate_tensor_memory(
        b_scale_desc.dtype,
        [BLOCK_N, SCALE_K],
        scale_layout,
    )
    acc_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], acc_layout)
    load_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(load_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    phase = 0
    use_acc = False
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(
            load_bar,
            a_desc.block_type.nbytes
            + b_desc.block_type.nbytes
            + a_scale_desc.block_type.nbytes
            + b_scale_desc.block_type.nbytes,
        )
        tma.async_load(a_desc, [0, k], load_bar, a_smem)
        tma.async_load(b_desc, [0, k], load_bar, b_smem)
        tma.async_load(a_scale_desc, [0, k // VEC], load_bar, a_scale_smem)
        tma.async_load(b_scale_desc, [0, k // VEC], load_bar, b_scale_smem)
        mbarrier.wait(load_bar, phase)
        tcgen05_copy(a_scale_smem, a_scale_tmem)
        tcgen05_copy(b_scale_smem, b_scale_tmem)
        tcgen05_mma_scaled(
            a_smem,
            b_smem.permute((1, 0)),
            acc,
            a_scale_tmem,
            b_scale_tmem,
            "e4m3",
            "e4m3",
            use_acc=use_acc,
        )
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase)
        use_acc = True
        phase ^= 1

    mbarrier.invalidate(load_bar)
    mbarrier.invalidate(mma_bar)
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    c_smem.store(acc.load().to(c_desc.dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [0, 0], c_smem)
    tma.store_wait(0)


@gluon.jit
def _two_cta_tcgen05_kernel(a_desc, b_desc, c_desc):
    gl.static_assert(gl.num_ctas() == 2)
    cluster_m: gl.constexpr = a_desc.block_shape[0]
    tile_n: gl.constexpr = b_desc.block_shape[1]
    cta_m: gl.constexpr = cluster_m // 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout

    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_load(a_desc, [0, 0], tma_bar, smem_a)
    tma.async_load(b_desc, [0, 0], tma_bar, smem_b)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)

    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(cta_m, tile_n),
        col_stride=1,
        cga_layout=cga_layout,
        two_ctas=True,
    )
    acc = allocate_tensor_memory(gl.float32, [cluster_m, tile_n], acc_layout)
    tcgen05_mma(smem_a, smem_b, acc, use_acc=False, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_bar)

    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)
    c_smem.store(acc.load().to(c_desc.dtype))
    tma.async_copy_shared_to_global(c_desc, [0, 0], c_smem)


@gluon.jit
def _tma_tcgen05_multicast_kernel(
    a_desc,
    b_desc,
    out_desc,
    NUM_K_TILES: gl.constexpr,
    acc_tmem_layout: gl.constexpr,
):
    block_m: gl.constexpr = a_desc.block_shape[0]
    block_k: gl.constexpr = a_desc.block_shape[1]
    block_n: gl.constexpr = b_desc.block_shape[1]

    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    acc_tmem = allocate_tensor_memory(gl.float32, [block_m, block_n], acc_tmem_layout)
    tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(
        mma_bar,
        count=tcgen05_mma_barrier_count(
            [smem_a, smem_b],
            multicast=True,
            two_ctas=acc_tmem.type.layout.two_ctas,
        ),
    )

    phase_tma = 0
    phase_mma = 0
    for k in range(NUM_K_TILES):
        mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [0, k * block_k], tma_bar, smem_a, multicast=True)
        tma.async_load(b_desc, [k * block_k, 0], tma_bar, smem_b, multicast=True)
        mbarrier.wait(tma_bar, phase=phase_tma, deps=[smem_a, smem_b])
        phase_tma ^= 1
        tcgen05_mma(
            smem_a,
            smem_b,
            acc_tmem,
            use_acc=(k != 0),
            multicast=True,
            mbarriers=[mma_bar],
        )
        mbarrier.wait(mma_bar, phase=phase_mma, deps=[smem_a, smem_b])
        phase_mma ^= 1

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)
    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        out_desc.block_shape,
        out_desc.layout,
    )
    out_smem.store(acc_tmem.load().to(out_desc.dtype))
    tma.async_copy_shared_to_global(out_desc, [0, 0], out_smem)


@gluon.jit
def _tcgen05_copy_kernel(
    in_ptr,
    in_stride0,
    in_stride1,
    out_ptr,
    out_stride0,
    out_stride1,
    M: gl.constexpr,
    N: gl.constexpr,
    smem_layout: gl.constexpr,
    tmem_layout: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, layout))
    value = gl.load(
        in_ptr + offs_m[:, None] * in_stride0 + offs_n[None, :] * in_stride1
    )
    smem = gl.allocate_shared_memory(value.dtype, (M, N), smem_layout)
    tmem = allocate_tensor_memory(value.dtype, (M, N), tmem_layout)
    smem.store(value)
    fence_async_shared()
    tcgen05_copy(smem, tmem)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_commit(bar)
    mbarrier.wait(bar, 0)
    output = gl.convert_layout(tmem.load(), layout)
    gl.store(
        out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :] * out_stride1, output
    )


@gluon.aggregate
class _AccumPartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    d_ptr: gl.tensor
    d_stride_m: gl.tensor
    d_stride_n: gl.tensor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    c_buf: gl.shared_memory_descriptor
    c_empty_bar: gl.shared_memory_descriptor
    c_ready_bar: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SchedulerImpl: gl.constexpr


@gluon.jit
def _matmul_accumulate_load_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 1
    state = _Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(
        p.c_desc.shape[0],
        p.c_desc.shape[1],
        BLOCK_M,
        BLOCK_N,
    )
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        mbarrier.wait(p.c_empty_bar, c_phase)
        mbarrier.expect(p.c_ready_bar, p.c_desc.block_type.nbytes)
        tma.async_load(p.c_desc, [off_m, off_n], p.c_ready_bar, p.c_buf)
        c_phase ^= 1
        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            mbarrier.expect(
                bar,
                p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes,
            )
            tma.async_load(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_load(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()


@gluon.jit
def _matmul_accumulate_mma_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    c_phase = 0
    load_state = _Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = _Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(
        p.c_desc.shape[0],
        p.c_desc.shape[1],
        BLOCK_M,
        BLOCK_N,
    )
    for _ in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.c_ready_bar, c_phase)
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        tcgen05_copy(p.c_buf, acc_buf)
        tcgen05_commit(p.c_empty_bar)
        c_phase ^= 1
        for _ in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
            tcgen05_mma(
                p.a_bufs.index(load_state.index),
                p.b_bufs.index(load_state.index),
                acc_buf,
                use_acc=True,
            )
            tcgen05_commit(p.load_empty_bars.index(load_state.index))
            load_state = load_state.next()
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def _matmul_accumulate_epilogue_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.c_desc.block_type.shape[1]

    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))

    acc_state = _Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = p.SchedulerImpl.initialize(
        p.c_desc.shape[0],
        p.c_desc.shape[1],
        BLOCK_M,
        BLOCK_N,
    )
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc = p.acc_bufs.index(acc_state.index).load()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
        acc = gl.convert_layout(acc, layout)
        gl.store(
            p.d_ptr
            + (off_m + offs_m)[:, None] * p.d_stride_m
            + (off_n + offs_n)[None, :] * p.d_stride_n,
            acc,
        )


@gluon.jit
def _tcgen05_matmul_accumulate_kernel(
    a_desc,
    b_desc,
    c_desc,
    d_ptr,
    d_stride_m,
    d_stride_n,
    SchedulerImpl: gl.constexpr,
    num_buffers: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(
        dtype,
        [num_buffers] + a_desc.block_type.shape,
        a_desc.layout,
    )
    b_bufs = gl.allocate_shared_memory(
        dtype,
        [num_buffers] + b_desc.block_type.shape,
        b_desc.layout,
    )
    load_empty_bars = gl.allocate_shared_memory(
        gl.int64,
        [num_buffers, 1],
        mbarrier.MBarrierLayout(),
    )
    load_ready_bars = gl.allocate_shared_memory(
        gl.int64,
        [num_buffers, 1],
        mbarrier.MBarrierLayout(),
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    c_buf = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    c_empty_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    c_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(c_empty_bar, count=1)
    mbarrier.init(c_ready_bar, count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_bufs = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    acc_ready_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = _AccumPartitionArgs(
        a_desc,
        b_desc,
        c_desc,
        d_ptr,
        d_stride_m,
        d_stride_n,
        a_bufs,
        b_bufs,
        load_empty_bars,
        load_ready_bars,
        c_buf,
        c_empty_bar,
        c_ready_bar,
        acc_bufs,
        acc_empty_bars,
        acc_ready_bars,
        SchedulerImpl,
    )
    gl.warp_specialize(
        [
            (_matmul_accumulate_epilogue_partition, (p,)),
            (_matmul_accumulate_mma_partition, (p,)),
            (_matmul_accumulate_load_partition, (p,)),
        ],
        [1, 1],
        [24, 24],
    )


@gluon.jit
def _tensor_memory_load_reduction_kernel(
    in_ptr,
    values_out,
    max_out,
    min_out,
    M: gl.constexpr,
    N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, layout))
    data = gl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :])
    tmem_layout: gl.constexpr = TensorMemoryLayout((64, 64), col_stride=1)
    tmem = allocate_tensor_memory(data.dtype, [M, N], tmem_layout)
    tmem.store(data)
    sliced = tmem.slice(8, 16)
    values, maxima = sliced.load_max()
    _, minima = sliced.load_min(abs=True)
    out_n = gl.arange(0, 16, gl.SliceLayout(0, layout))
    gl.store(values_out + offs_m[:, None] * 16 + out_n[None, :], values)
    row_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    row_offsets = gl.arange(0, M, row_layout)
    gl.store(max_out + row_offsets, gl.convert_layout(maxima, row_layout))
    gl.store(min_out + row_offsets, gl.convert_layout(minima, row_layout))


@gluon.jit
def _tma_gather_kernel(
    out_ptr,
    out_stride_x,
    out_stride_y,
    tensor_desc,
    x_offsets_ptr,
    y_offset,
    BLOCK_X: gl.constexpr,
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout(
        [1],
        [32],
        [gl.num_warps()],
        [0],
    )
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))
    offsets_layout: gl.constexpr = gl.SliceLayout(
        0,
        gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)
    smem_dest = gl.allocate_shared_memory(
        tensor_desc.dtype,
        [BLOCK_X, BLOCK_Y],
        tensor_desc.layout,
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, BLOCK_X * tensor_desc.block_type.nbytes)
    blackwell_tma.async_gather(
        tensor_desc,
        x_offsets,
        y_offset,
        barrier=bar,
        result=smem_dest,
    )
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    out = smem_dest.load(coalesced_2d_layout)
    indices_x = (
        gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None]
        * out_stride_x
    )
    indices_y = (
        gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :]
        * out_stride_y
    )
    gl.store(out_ptr + indices_x + indices_y, out)


@gluon.jit
def _tma_scatter_kernel(
    tensor_desc,
    x_offsets_ptr,
    y_offset,
    src_ptr,
    src_stride_x,
    src_stride_y,
    BLOCK_X: gl.constexpr,
):
    BLOCK_Y: gl.constexpr = tensor_desc.block_type.shape[1]
    coalesced_2d_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    indices_x = (
        gl.arange(0, BLOCK_X, gl.SliceLayout(1, coalesced_2d_layout))[:, None]
        * src_stride_x
    )
    indices_y = (
        gl.arange(0, BLOCK_Y, gl.SliceLayout(0, coalesced_2d_layout))[None, :]
        * src_stride_y
    )
    src = gl.load(src_ptr + indices_x + indices_y)
    coalesced_1d_layout: gl.constexpr = gl.BlockedLayout(
        [1],
        [32],
        [gl.num_warps()],
        [0],
    )
    x_offsets = gl.load(x_offsets_ptr + gl.arange(0, BLOCK_X, coalesced_1d_layout))
    offsets_layout: gl.constexpr = gl.SliceLayout(
        0,
        gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    x_offsets = gl.convert_layout(x_offsets, offsets_layout)
    smem_src = gl.allocate_shared_memory(
        tensor_desc.dtype,
        [BLOCK_X, BLOCK_Y],
        tensor_desc.layout,
    )
    smem_src.store(src)
    fence_async_shared()
    blackwell_tma.async_scatter(tensor_desc, x_offsets, y_offset, smem_src)
    blackwell_tma.store_wait(0)


@gluon.jit
def _tma_gather_scatter_matmul_kernel(
    x_desc,
    w_desc,
    out_desc,
    gather_indices,
    scatter_indices,
    BLOCK_M: gl.constexpr,
):
    BLOCK_K: gl.constexpr = w_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = w_desc.block_type.shape[1]
    offsets_layout: gl.constexpr = gl.SliceLayout(
        0,
        gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    offsets = gl.arange(0, BLOCK_M, offsets_layout)
    gather_offsets = gl.load(gather_indices + offsets)
    scatter_offsets = gl.load(scatter_indices + offsets)

    x_smem = gl.allocate_shared_memory(
        x_desc.dtype,
        [BLOCK_M, BLOCK_K],
        x_desc.layout,
    )
    w_smem = gl.allocate_shared_memory(
        w_desc.dtype,
        w_desc.block_type.shape,
        w_desc.layout,
    )
    load_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(load_bar, count=1)
    mbarrier.expect(
        load_bar,
        BLOCK_M * x_desc.block_type.nbytes + w_desc.block_type.nbytes,
    )
    blackwell_tma.async_gather(
        x_desc,
        gather_offsets,
        0,
        barrier=load_bar,
        result=x_smem,
    )
    tma.async_load(w_desc, [0, 0], load_bar, w_smem)
    mbarrier.wait(load_bar, phase=0)
    mbarrier.invalidate(load_bar)

    acc_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], acc_layout)
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    tcgen05_mma(x_smem, w_smem, acc, use_acc=False, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0)
    mbarrier.invalidate(mma_bar)

    out_smem = gl.allocate_shared_memory(
        out_desc.dtype,
        [BLOCK_M, BLOCK_N],
        out_desc.layout,
    )
    out_smem.store(acc.load().to(out_desc.dtype))
    fence_async_shared()
    blackwell_tma.async_scatter(out_desc, scatter_offsets, 0, out_smem)
    blackwell_tma.store_wait(0)


@gluon.jit
def _tma_async_atomic_float_kernel(
    add_desc,
    min_desc,
    max_desc,
    src_ptr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    src = gl.load(src_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    smem = gl.allocate_shared_memory(src.dtype, [BLOCK_M, BLOCK_N], add_desc.layout)
    smem.store(src)
    fence_async_shared()
    tma.async_atomic_add(add_desc, [1, 2], smem)
    tma.async_atomic_min(min_desc, [1, 2], smem)
    tma.async_atomic_max(max_desc, [1, 2], smem)
    tma.store_wait(0)


@gluon.jit
def _tma_async_atomic_bitwise_kernel(
    and_desc,
    or_desc,
    xor_desc,
    src_ptr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    src = gl.load(src_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    smem = gl.allocate_shared_memory(src.dtype, [BLOCK_M, BLOCK_N], and_desc.layout)
    smem.store(src)
    fence_async_shared()
    blackwell_tma.async_atomic_and(and_desc, [1, 1], smem)
    blackwell_tma.async_atomic_or(or_desc, [1, 1], smem)
    blackwell_tma.async_atomic_xor(xor_desc, [1, 1], smem)
    blackwell_tma.store_wait(0)


def test_gluon_simulator_runs_intro_scalar_range_memcpy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.empty_like(inp)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_intro_memcpy_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert kernel.client_manager.launch.grid == (1, 1, 1)


def test_gluon_simulator_runs_layout_1d_memcpy_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    layout = gl.BlockedLayout([1], [32], [1], [0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(_layout_memcpy_1d_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, layout, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_layout_2d_memcpy_on_cpu():
    inp = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    out = torch.full_like(inp, -1)
    layout = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(_layout_memcpy_2d_kernel)

    kernel[(1,)](
        inp,
        out,
        *inp.shape,
        *inp.stride(),
        *out.stride(),
        layout,
        8,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_layout_2d_inout_memcpy_on_cpu():
    inp = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    out = torch.full_like(inp, -1)
    layout_in = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    layout_out = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _layout_memcpy_2d_inout_kernel
    )

    kernel[(1, 1)](
        inp,
        out,
        *inp.shape,
        *inp.stride(),
        *out.stride(),
        layout_in,
        layout_out,
        8,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_elementwise_add_on_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    b = 10 + a
    c = torch.empty_like(a)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_elementwise_add_kernel)

    kernel[(1,)](
        a,
        b,
        c,
        *a.shape,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        8,
        8,
        num_warps=4,
    )

    torch.testing.assert_close(c, a + b, atol=0, rtol=0)


def test_gluon_simulator_runs_cpasync_memcpy_1d_synchronously_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_cpasync_memcpy_1d_kernel)

    kernel[(1,)](inp, out, inp.numel(), 64, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_cpasync_elementwise_add_synchronously_on_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    b = 10 + a
    c = torch.empty_like(a)
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _cpasync_elementwise_add_kernel
    )

    kernel[(1,)](
        a,
        b,
        c,
        *a.shape,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        8,
        8,
        smem_layout,
        num_warps=4,
    )

    torch.testing.assert_close(c, a + b, atol=0, rtol=0)


def test_gluon_simulator_runs_pipelined_cpasync_add_on_cpu():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    b = 10 + a
    c = torch.empty_like(a)
    smem_layout = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[2, 1, 0],
    )
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _cpasync_elementwise_add_pipelined_kernel
    )

    kernel[(1,)](
        a,
        b,
        c,
        *a.shape,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        8,
        4,
        smem_layout,
        2,
        num_warps=4,
    )

    torch.testing.assert_close(c, a + b, atol=0, rtol=0)


def test_gluon_simulator_runs_tma_memcpy_1d_synchronously_on_cpu():
    inp = torch.arange(40, dtype=torch.float32)
    out = torch.full_like(inp, -1)
    layout = gl.NVMMASharedLayout.get_default_for([64], gl.float32)
    in_desc = TensorDescriptor.from_tensor(inp, [64], layout)
    out_desc = TensorDescriptor.from_tensor(out, [64], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_memcpy_1d_kernel)

    kernel[(1,)](in_desc, out_desc, 64, num_warps=1)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_pipelined_tma_add_on_cpu():
    a = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b = 10 + a
    c = torch.empty_like(a)
    layout = gl.NVMMASharedLayout.get_default_for([8, 4], gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, [8, 4], layout)
    b_desc = TensorDescriptor.from_tensor(b, [8, 4], layout)
    c_desc = TensorDescriptor.from_tensor(c, [8, 4], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_elementwise_add_kernel)

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        *a.shape,
        8,
        4,
        2,
        num_warps=4,
    )

    torch.testing.assert_close(c, a + b, atol=0, rtol=0)


def test_gluon_simulator_runs_blackwell_tma_gather_on_cpu():
    block_x = 8
    block_y = 8
    y_offset = -2
    inp = torch.arange(6 * 12, dtype=torch.float32).reshape(6, 12)
    x_offsets = torch.tensor([-1, 0, 4, 2, 6, 5, 1, 3], dtype=torch.int32)
    out = torch.full((block_x, block_y), -1.0)
    layout = gl.NVMMASharedLayout.get_default_for([block_x, block_y], gl.float32)
    desc = TensorDescriptor.from_tensor(inp, [1, block_y], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_gather_kernel)

    kernel[(1,)](
        out,
        *out.stride(),
        desc,
        x_offsets,
        y_offset,
        block_x,
        num_warps=4,
    )

    expected = torch.zeros_like(out)
    for out_row, src_row in enumerate(x_offsets.tolist()):
        for out_col in range(block_y):
            src_col = y_offset + out_col
            if 0 <= src_row < inp.shape[0] and 0 <= src_col < inp.shape[1]:
                expected[out_row, out_col] = inp[src_row, src_col]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_blackwell_tma_scatter_on_cpu():
    block_x = 8
    block_y = 8
    y_offset = 6
    inp = torch.full((6, 12), -1.0)
    x_offsets = torch.tensor([0, 5, 6, 3, 2, 8, 1, 4], dtype=torch.int32)
    src = torch.arange(block_x * block_y, dtype=torch.float32).reshape(
        block_x,
        block_y,
    )
    layout = gl.NVMMASharedLayout.get_default_for([block_x, block_y], gl.float32)
    desc = TensorDescriptor.from_tensor(inp, [1, block_y], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_scatter_kernel)

    kernel[(1,)](
        desc,
        x_offsets,
        y_offset,
        src,
        *src.stride(),
        block_x,
        num_warps=4,
    )

    expected = torch.full_like(inp, -1.0)
    for src_row, dst_row in enumerate(x_offsets.tolist()):
        for src_col in range(block_y):
            dst_col = y_offset + src_col
            if 0 <= dst_row < inp.shape[0] and 0 <= dst_col < inp.shape[1]:
                expected[dst_row, dst_col] = src[src_row, src_col]
    torch.testing.assert_close(inp, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_fused_tma_gather_scatter_matmul_on_cpu():
    torch.manual_seed(19)
    block_m = 8
    block_k = 16
    block_n = 16
    x = torch.randn((12, block_k), dtype=torch.float16) / 4
    w = torch.randn((block_k, block_n), dtype=torch.float16) / 4
    out = torch.full((12, block_n), -7.0, dtype=torch.float16)
    gather = torch.tensor([9, 1, 7, 0, 4, 11, 2, 6], dtype=torch.int32)
    scatter = torch.tensor([3, 10, 0, 8, 5, 1, 11, 4], dtype=torch.int32)
    x_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    w_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    out_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    x_desc = TensorDescriptor.from_tensor(x, [1, block_k], x_layout)
    w_desc = TensorDescriptor.from_tensor(w, [block_k, block_n], w_layout)
    out_desc = TensorDescriptor.from_tensor(out, [1, block_n], out_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tma_gather_scatter_matmul_kernel
    )

    kernel[(1,)](
        x_desc,
        w_desc,
        out_desc,
        gather,
        scatter,
        block_m,
        num_warps=4,
    )

    expected = torch.full_like(out, -7.0)
    expected[scatter.long()] = (x[gather.long()].float() @ w.float()).half()
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_gluon_simulator_runs_tma_async_atomic_float_ops_on_cpu():
    block_m = 2
    block_n = 4
    src = torch.tensor(
        [[3.0, -2.0, 0.5, 8.0], [1.5, 7.0, -4.0, 0.25]],
        dtype=torch.float32,
    )
    add_dst = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    min_dst = add_dst + 10.0
    max_dst = add_dst - 10.0
    layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float32)
    add_desc = TensorDescriptor.from_tensor(add_dst, [block_m, block_n], layout)
    min_desc = TensorDescriptor.from_tensor(min_dst, [block_m, block_n], layout)
    max_desc = TensorDescriptor.from_tensor(max_dst, [block_m, block_n], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tma_async_atomic_float_kernel
    )

    expected_add = add_dst.clone()
    expected_min = min_dst.clone()
    expected_max = max_dst.clone()
    expected_add[1:3, 2:6] += src
    expected_min[1:3, 2:6] = torch.minimum(expected_min[1:3, 2:6], src)
    expected_max[1:3, 2:6] = torch.maximum(expected_max[1:3, 2:6], src)

    kernel[(1,)](add_desc, min_desc, max_desc, src, block_m, block_n, num_warps=4)

    torch.testing.assert_close(add_dst, expected_add, atol=0, rtol=0)
    torch.testing.assert_close(min_dst, expected_min, atol=0, rtol=0)
    torch.testing.assert_close(max_dst, expected_max, atol=0, rtol=0)


def test_gluon_simulator_runs_tma_async_atomic_bitwise_ops_on_cpu():
    block_m = 2
    block_n = 4
    src = torch.tensor(
        [[0x0F, 0x33, 0x55, 0xAA], [0xF0, 0xCC, 0x5A, 0xA5]],
        dtype=torch.int32,
    )
    base = torch.arange(40, dtype=torch.int32).reshape(5, 8) + 0x80
    and_dst = base.clone()
    or_dst = base.clone()
    xor_dst = base.clone()
    layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.int32)
    and_desc = TensorDescriptor.from_tensor(and_dst, [block_m, block_n], layout)
    or_desc = TensorDescriptor.from_tensor(or_dst, [block_m, block_n], layout)
    xor_desc = TensorDescriptor.from_tensor(xor_dst, [block_m, block_n], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tma_async_atomic_bitwise_kernel
    )

    expected_and = and_dst.clone()
    expected_or = or_dst.clone()
    expected_xor = xor_dst.clone()
    expected_and[1:3, 1:5] = torch.bitwise_and(expected_and[1:3, 1:5], src)
    expected_or[1:3, 1:5] = torch.bitwise_or(expected_or[1:3, 1:5], src)
    expected_xor[1:3, 1:5] = torch.bitwise_xor(expected_xor[1:3, 1:5], src)

    kernel[(1,)](and_desc, or_desc, xor_desc, src, block_m, block_n, num_warps=4)

    torch.testing.assert_close(and_dst, expected_and, atol=0, rtol=0)
    torch.testing.assert_close(or_dst, expected_or, atol=0, rtol=0)
    torch.testing.assert_close(xor_dst, expected_xor, atol=0, rtol=0)


def test_gluon_simulator_runs_tma_message_passing_synchronously_on_cpu():
    message_size = 16
    message = torch.full((message_size,), -1, dtype=torch.int32)
    ready = torch.zeros(1, dtype=torch.int32)
    output = torch.full((message_size,), -1, dtype=torch.int32)
    layout = gl.NVMMASharedLayout.get_default_for([message_size], gl.int32)
    message_desc = TensorDescriptor.from_tensor(message, [message_size], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_message_passing_kernel)

    kernel[(2,)](message_desc, ready, output, message_size, num_warps=1)

    expected = torch.arange(message_size, dtype=torch.int32) + 1000
    torch.testing.assert_close(message, expected, atol=0, rtol=0)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert ready.item() == 1


def test_gluon_simulator_runs_tma_multicast_copy_on_cpu():
    inp = torch.arange(16 * 16, dtype=torch.float16).reshape(16, 16)
    out = torch.empty_like(inp)
    layout = gl.NVMMASharedLayout.get_default_for(
        list(inp.shape),
        gl.float16,
        cga_layout=[[0, 0]],
    )
    in_desc = TensorDescriptor.from_tensor(inp, list(inp.shape), layout)
    out_desc = TensorDescriptor.from_tensor(out, list(out.shape), layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_multicast_copy_kernel)

    kernel[(1,)](in_desc, out_desc, num_warps=4, num_ctas=2)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_tma_alias_with_shared_reshape_on_cpu():
    inp = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    out = torch.empty((8, 4), dtype=torch.float32)
    in_layout = gl.NVMMASharedLayout.get_default_for([4, 8], gl.float32)
    out_layout = gl.NVMMASharedLayout.get_default_for([8, 4], gl.float32)
    in_desc = TensorDescriptor.from_tensor(inp, [4, 8], in_layout)
    out_desc = TensorDescriptor.from_tensor(out, [8, 4], out_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tma_alias_shared_reshape_kernel
    )

    kernel[(1,)](in_desc, out_desc, num_warps=4)

    torch.testing.assert_close(out, inp.T.contiguous(), atol=0, rtol=0)


def _run_im2col_case(
    inp,
    pixel_box_lower_corner,
    pixel_box_upper_corner,
    coord,
    offsets,
):
    out = torch.zeros((16, 32), dtype=torch.float32)
    layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=32,
        rank=2,
    )
    in_desc = TensorDescriptorIm2Col.from_tensor(
        inp,
        [16, 32],
        layout,
        padding="zero",
        element_strides=[1, 1, 1, 1],
        pixel_box_lower_corner=pixel_box_lower_corner,
        pixel_box_upper_corner=pixel_box_upper_corner,
    )
    out_desc = TensorDescriptor.from_tensor(out, [16, 32], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_im2col_kernel)

    kernel[(1,)](in_desc, out_desc, *coord, *offsets, num_warps=1)

    return out


def test_gluon_simulator_runs_tma_im2col_simple_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    out = _run_im2col_case(inp, [0, 0], [0, 0], [0, 0, 0, 0], [0, 0])

    torch.testing.assert_close(out, inp.reshape(16, 32), atol=0, rtol=0)


def test_gluon_simulator_runs_tma_im2col_padded_and_offset_on_cpu():
    inp = torch.arange(1, 17, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(1, 4, 4, 32)

    padded = _run_im2col_case(inp, [-1, -1], [-1, -1], [0, -1, -1, 0], [0, 0])
    expected_padded = torch.tensor(
        [0, 0, 0, 0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11],
        dtype=torch.float32,
    )
    torch.testing.assert_close(padded[:, 0], expected_padded, atol=0, rtol=0)

    shifted = _run_im2col_case(inp, [-1, -1], [-1, -1], [0, -1, -1, 0], [1, 1])
    torch.testing.assert_close(shifted, inp.reshape(16, 32), atol=0, rtol=0)


def test_gluon_simulator_runs_tma_im2col_multi_batch_on_cpu():
    inp = torch.arange(1, 33, dtype=torch.float32).unsqueeze(1).repeat(1, 32)
    inp = inp.reshape(2, 4, 4, 32)

    out = _run_im2col_case(inp, [0, 0], [0, 0], [0, 1, 3, 0], [0, 0])
    expected = torch.tensor(
        [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out[:, 0], expected, atol=0, rtol=0)

    padded = _run_im2col_case(inp, [-1, -1], [-1, -1], [0, 1, 2, 0], [0, 0])
    expected_padded = torch.tensor(
        [7, 0, 9, 10, 11, 0, 0, 0, 0, 0, 17, 18, 19, 0, 21, 22],
        dtype=torch.float32,
    )
    torch.testing.assert_close(padded[:, 0], expected_padded, atol=0, rtol=0)


def test_gluon_simulator_runs_conv2d_tma_im2col_wgmma_on_cpu():
    torch.manual_seed(14)
    input_nhwc = torch.randn((1, 4, 4, 16), dtype=torch.float16) / 4
    weight = torch.randn((16, 3, 3, 16), dtype=torch.float16) / 4
    stride = 1
    padding = 1
    block_m, block_n, block_k = 16, 16, 16
    n, h, w, ci = input_nhwc.shape
    co, r, s, ci_w = weight.shape
    assert ci == ci_w
    out_h = (h + 2 * padding - r) // stride + 1
    out_w = (w + 2 * padding - s) // stride + 1
    output = torch.empty((n * out_h * out_w, co), dtype=torch.float16)

    upper_h = (out_h - 1) * stride + 1 - h - padding
    upper_w = (out_w - 1) * stride + 1 - w - padding
    input_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    in_desc = TensorDescriptorIm2Col.from_tensor(
        input_nhwc,
        [block_m, block_k],
        input_layout,
        padding="zero",
        element_strides=[1, stride, stride, 1],
        pixel_box_lower_corner=[-padding, -padding],
        pixel_box_upper_corner=[upper_h, upper_w],
    )
    weight_2d = weight.reshape(co, r * s * ci)
    weight_layout = gl.NVMMASharedLayout.get_default_for([block_n, block_k], gl.float16)
    weight_desc = TensorDescriptor.from_tensor(
        weight_2d, [block_n, block_k], weight_layout
    )
    out_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    out_desc = TensorDescriptor.from_tensor(output, [block_m, block_n], out_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_conv2d_im2col_wgmma_kernel)

    kernel[(1, 1)](
        in_desc,
        weight_desc,
        out_desc,
        r,
        s,
        ci,
        out_h,
        out_w,
        padding,
        padding,
        stride,
        stride,
        block_m,
        block_n,
        block_k,
        num_warps=4,
    )

    expected = torch.nn.functional.conv2d(
        input_nhwc.permute(0, 3, 1, 2).float(),
        weight.permute(0, 3, 1, 2).float(),
        padding=padding,
        stride=stride,
    ).permute(0, 2, 3, 1)
    actual = output.reshape(n, out_h, out_w, co).float()
    torch.testing.assert_close(actual, expected.half().float(), atol=1e-2, rtol=1e-2)


def test_gluon_simulator_runs_small_wgmma_on_cpu():
    torch.manual_seed(0)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 32, dtype=torch.float16) / 4
    c = torch.randn(64, 32, dtype=torch.float32) / 4
    d = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, a.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, c.shape, c_layout)
    d_desc = TensorDescriptor.from_tensor(d, d.shape, c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_small_wgmma_kernel)

    kernel[(1,)](a_desc, b_desc, c_desc, d_desc, False, num_warps=4)

    torch.testing.assert_close(d, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


def test_gluon_simulator_runs_small_wgmma_with_lhs_registers_on_cpu():
    torch.manual_seed(1)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 32, dtype=torch.float16) / 4
    c = torch.randn(64, 32, dtype=torch.float32) / 4
    d = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, a.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, c.shape, c_layout)
    d_desc = TensorDescriptor.from_tensor(d, d.shape, c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_small_wgmma_kernel)

    kernel[(1,)](a_desc, b_desc, c_desc, d_desc, True, num_warps=4)

    torch.testing.assert_close(d, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


def test_gluon_simulator_runs_blocked_wgmma_matmul_on_cpu():
    torch.manual_seed(2)
    a = torch.randn(96, 64, dtype=torch.float16) / 4
    b = torch.randn(64, 64, dtype=torch.float16) / 4
    c = torch.empty((96, 64), dtype=torch.float16)
    block_m, block_n, block_k = 64, 32, 32
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_blocked_wgmma_kernel)

    kernel[(triton.cdiv(c.shape[0], block_m), triton.cdiv(c.shape[1], block_n))](
        a_desc,
        b_desc,
        c_desc,
        False,
        num_warps=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_blocked_wgmma_with_transposed_b_on_cpu():
    torch.manual_seed(3)
    a = torch.randn(96, 64, dtype=torch.float16) / 4
    b = torch.randn(64, 64, dtype=torch.float16) / 4
    c = torch.empty((96, 64), dtype=torch.float16)
    block_m, block_n, block_k = 64, 32, 32
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_n, block_k], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_n, block_k], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_blocked_wgmma_kernel)

    kernel[(triton.cdiv(c.shape[0], block_m), triton.cdiv(c.shape[1], block_n))](
        a_desc,
        b_desc,
        c_desc,
        True,
        num_warps=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.T.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_pipelined_wgmma_matmul_on_cpu():
    torch.manual_seed(4)
    a = torch.randn(96, 64, dtype=torch.float16) / 4
    b = torch.randn(64, 64, dtype=torch.float16) / 4
    c = torch.empty((96, 64), dtype=torch.float16)
    block_m, block_n, block_k = 64, 32, 32
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_pipelined_wgmma_kernel)

    kernel[(triton.cdiv(c.shape[0], block_m), triton.cdiv(c.shape[1], block_n))](
        a_desc,
        b_desc,
        c_desc,
        num_warps=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_persistent_wgmma_matmul_on_cpu():
    torch.manual_seed(14)
    a = torch.randn(32, 16, dtype=torch.float16) / 4
    b = torch.randn(16, 32, dtype=torch.float16) / 4
    c = torch.empty((32, 32), dtype=torch.float16)
    block_m, block_n, block_k = 16, 16, 16
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _persistent_wgmma_matmul_kernel
    )

    kernel[(2,)](
        a_desc,
        b_desc,
        c_desc,
        *c.shape,
        _PersistentTileScheduler,
        num_warps=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_grouped_persistent_wgmma_matmul_on_cpu():
    torch.manual_seed(15)
    a = torch.randn(48, 16, dtype=torch.float16) / 4
    b = torch.randn(16, 32, dtype=torch.float16) / 4
    c = torch.empty((48, 32), dtype=torch.float16)
    block_m, block_n, block_k = 16, 16, 16
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _persistent_wgmma_matmul_kernel
    )

    kernel[(2,)](
        a_desc,
        b_desc,
        c_desc,
        *c.shape,
        _GroupedPersistentTileScheduler(2),
        num_warps=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_pipelined_persistent_wgmma_matmul_on_cpu():
    torch.manual_seed(20)
    a = torch.randn(32, 64, dtype=torch.float16) / 4
    b = torch.randn(64, 32, dtype=torch.float16) / 4
    c = torch.empty((32, 32), dtype=torch.float16)
    block_m, block_n, block_k = 16, 16, 16
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _persistent_wgmma_pipelined_kernel
    )

    kernel[(2,)](
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        *c.shape,
        _PersistentTileScheduler,
        3,
        1,
        num_warps=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_splitn_pipelined_persistent_wgmma_matmul_on_cpu():
    torch.manual_seed(21)
    a = torch.randn(32, 64, dtype=torch.float16) / 4
    b = torch.randn(64, 32, dtype=torch.float16) / 4
    c = torch.empty((32, 32), dtype=torch.float16)
    block_m, block_n, block_k = 32, 32, 16
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float16)
    c_half_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_n // 2],
        gl.float16,
    )
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    c_half_desc = TensorDescriptor.from_tensor(
        c,
        [block_m, block_n // 2],
        c_half_layout,
    )
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _persistent_wgmma_pipelined_kernel
    )

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        c_half_desc,
        *c.shape,
        _PersistentTileScheduler,
        3,
        2,
        num_warps=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_tensor_memory_roundtrip_on_cpu():
    inp = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    out = torch.empty_like(inp)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tmem_roundtrip_kernel)

    kernel[(1,)](inp, out, *inp.shape, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_tcgen05_small_mma_on_cpu():
    torch.manual_seed(5)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 64, dtype=torch.float16) / 4
    c = torch.randn(64, 64, dtype=torch.float32) / 4
    d = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, a.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, c.shape, c_layout)
    d_desc = TensorDescriptor.from_tensor(d, d.shape, c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tcgen05_small_mma_kernel)

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        (64, 64),
        False,
        True,
        num_warps=4,
    )

    torch.testing.assert_close(d, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


def test_gluon_simulator_runs_tcgen05_small_mma_with_lhs_tmem_on_cpu():
    torch.manual_seed(6)
    a = torch.randn(64, 32, dtype=torch.float16) / 4
    b = torch.randn(32, 64, dtype=torch.float16) / 4
    c = torch.randn(64, 64, dtype=torch.float32) / 4
    d = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for(a.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(b.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(c.shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, a.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(b, b.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(c, c.shape, c_layout)
    d_desc = TensorDescriptor.from_tensor(d, d.shape, c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tcgen05_small_mma_kernel)

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        (64, 64),
        True,
        False,
        num_warps=4,
    )

    torch.testing.assert_close(d, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


def test_gluon_simulator_runs_tcgen05_scaled_mma_on_cpu():
    a = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -0.5]],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [
            [1.0, -1.0, 0.5, 2.0],
            [2.0, 0.0, -0.5, 1.0],
            [0.5, 1.5, 1.0, -1.0],
            [1.0, -2.0, 2.0, 0.5],
        ],
        dtype=torch.float32,
    )
    a_scale = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    b_scale = torch.tensor(
        [[127, 128], [128, 126], [127, 127], [126, 128]],
        dtype=torch.uint8,
    )
    out = torch.empty((2, 4), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tcgen05_scaled_mma_kernel)

    kernel[(1,)](
        a,
        b,
        a_scale,
        b_scale,
        out,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        2,
        num_warps=4,
    )

    a_scale_float = torch.pow(
        2.0,
        a_scale.float() - 127.0,
    ).repeat_interleave(2, dim=1)
    b_scale_float = torch.pow(
        2.0,
        b_scale.float() - 127.0,
    ).repeat_interleave(2, dim=1)
    expected = (a * a_scale_float) @ (b * b_scale_float.T)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_gluon_simulator_runs_descriptor_tcgen05_scaled_mma_on_cpu():
    a = torch.tensor(
        [[1.0, -2.0, 0.5, 4.0], [0.25, 3.0, -1.5, 2.0]],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [
            [1.0, 0.5, -1.0, 2.0],
            [-0.5, 1.5, 2.0, -1.0],
            [2.0, -1.0, 0.25, 0.5],
            [0.75, 2.0, -0.5, -1.5],
        ],
        dtype=torch.float32,
    )
    a_scale = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    b_scale = torch.tensor(
        [[128, 127], [127, 126], [126, 128], [128, 128]],
        dtype=torch.uint8,
    )
    out = torch.empty((2, 4), dtype=torch.float32)
    a_layout = gl.NVMMASharedLayout.get_default_for(list(a.shape), gl.float32)
    b_layout = gl.NVMMASharedLayout.get_default_for(list(b.shape), gl.float32)
    c_layout = gl.NVMMASharedLayout.get_default_for(list(out.shape), gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape), a_layout)
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape), b_layout)
    c_desc = TensorDescriptor.from_tensor(out, list(out.shape), c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tcgen05_descriptor_scaled_mma_kernel
    )

    kernel[(1, 1)](a_desc, b_desc, c_desc, a_scale, b_scale, 2, num_warps=4)

    a_scale_float = torch.pow(2.0, a_scale.float() - 127.0).repeat_interleave(
        2,
        dim=1,
    )
    b_scale_float = torch.pow(2.0, b_scale.float() - 127.0).repeat_interleave(
        2,
        dim=1,
    )
    expected = (a * a_scale_float) @ (b * b_scale_float).T
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_gluon_simulator_runs_tma_loaded_scale_tcgen05_pipeline_on_cpu():
    a = torch.tensor(
        [
            [1.0, -2.0, 0.5, 4.0],
            [0.25, 3.0, -1.5, 2.0],
        ],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [
            [1.0, 0.5, -1.0, 2.0],
            [-0.5, 1.5, 2.0, -1.0],
            [2.0, -1.0, 0.25, 0.5],
            [0.75, 2.0, -0.5, -1.5],
        ],
        dtype=torch.float32,
    )
    a_scale_values = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    b_scale_values = torch.tensor(
        [[128, 127], [127, 126], [126, 128], [128, 128]],
        dtype=torch.uint8,
    )
    a_scale = torch.zeros((2, 16), dtype=torch.uint8)
    b_scale = torch.zeros((4, 16), dtype=torch.uint8)
    a_scale[:, :2] = a_scale_values
    b_scale[:, :2] = b_scale_values
    out = torch.empty((2, 4), dtype=torch.float32)
    block_m, block_n, block_k, vec = 2, 4, 2, 2
    a_layout = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float32)
    b_layout = gl.NVMMASharedLayout.get_default_for([block_n, block_k], gl.float32)
    c_layout = gl.NVMMASharedLayout.get_default_for(list(out.shape), gl.float32)
    scale_a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_k // vec],
        gl.uint8,
    )
    scale_b_layout = gl.NVMMASharedLayout.get_default_for(
        [block_n, block_k // vec],
        gl.uint8,
    )
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_n, block_k], b_layout)
    c_desc = TensorDescriptor.from_tensor(out, list(out.shape), c_layout)
    a_scale_desc = TensorDescriptor.from_tensor(
        a_scale,
        [block_m, block_k // vec],
        scale_a_layout,
    )
    b_scale_desc = TensorDescriptor.from_tensor(
        b_scale,
        [block_n, block_k // vec],
        scale_b_layout,
    )
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tcgen05_descriptor_scaled_tma_pipeline_kernel
    )

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        a_scale_desc,
        b_scale_desc,
        vec,
        num_warps=4,
    )

    a_scale_float = torch.pow(2.0, a_scale_values.float() - 127.0).repeat_interleave(
        vec,
        dim=1,
    )
    b_scale_float = torch.pow(2.0, b_scale_values.float() - 127.0).repeat_interleave(
        vec,
        dim=1,
    )
    expected = (a * a_scale_float) @ (b * b_scale_float).T
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_gluon_simulator_runs_two_cta_tcgen05_on_cpu():
    torch.manual_seed(12)
    a = torch.randn(32, 16, dtype=torch.float16) / 4
    b = torch.randn(16, 32, dtype=torch.float16) / 4
    c = torch.empty((32, 32), dtype=torch.float16)
    a_layout = gl.NVMMASharedLayout.get_default_for(
        list(a.shape),
        gl.float16,
        cga_layout=[(1, 0)],
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        list(b.shape),
        gl.float16,
        cga_layout=[(0, 1)],
    )
    c_layout = gl.NVMMASharedLayout.get_default_for(
        list(c.shape),
        gl.float16,
        cga_layout=[(1, 0)],
    )
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape), a_layout)
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape), b_layout)
    c_desc = TensorDescriptor.from_tensor(c, list(c.shape), c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_two_cta_tcgen05_kernel)

    kernel[(1,)](a_desc, b_desc, c_desc, num_warps=4, num_ctas=2)

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_tma_tcgen05_multicast_on_cpu():
    torch.manual_seed(21)
    block_m = 32
    block_n = 32
    block_k = 16
    num_k_tiles = 2
    a = torch.randn((block_m, block_k * num_k_tiles), dtype=torch.float16) / 4
    b = torch.randn((block_k * num_k_tiles, block_n), dtype=torch.float16) / 4
    c = torch.empty((block_m, block_n), dtype=torch.float16)
    cga_layout_a = ((1, 0), (2, 0))
    cga_layout_b = ((0, 1), (0, 0))
    cga_layout_c = ((1, 0), (2, 0))
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_k],
        gl.float16,
        cga_layout=cga_layout_a,
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [block_k, block_n],
        gl.float16,
        cga_layout=cga_layout_b,
    )
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [block_m, block_n],
        gl.float16,
        cga_layout=cga_layout_c,
    )
    acc_layout = TensorMemoryLayout(
        block=(16, block_n),
        col_stride=1,
        cga_layout=cga_layout_c,
        two_ctas=True,
    )
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [block_m, block_n], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tma_tcgen05_multicast_kernel)

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        num_k_tiles,
        acc_layout,
        num_warps=4,
        num_ctas=4,
    )

    torch.testing.assert_close(
        c.float(),
        (a.float() @ b.float()).half().float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_gluon_simulator_runs_tcgen05_copy_on_cpu():
    inp = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    out = torch.empty_like(inp)
    smem_layout = gl.NVMMASharedLayout.get_default_for([64, 64], gl.float32)
    tmem_layout = TensorMemoryLayout((64, 64), col_stride=1)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tcgen05_copy_kernel)

    kernel[(1,)](
        inp,
        *inp.stride(),
        out,
        *out.stride(),
        *inp.shape,
        smem_layout,
        tmem_layout,
        num_warps=4,
    )

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


def test_gluon_simulator_runs_tcgen05_accumulate_pipeline_on_cpu():
    torch.manual_seed(18)
    a = torch.randn(64, 64, dtype=torch.float16) / 4
    b = torch.randn(64, 64, dtype=torch.float16) / 4
    c = torch.randn(64, 64, dtype=torch.float32) / 4
    d = torch.empty_like(c)
    a_layout = gl.NVMMASharedLayout.get_default_for([64, 32], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([32, 64], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([64, 64], gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, [64, 32], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [32, 64], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [64, 64], c_layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tcgen05_matmul_accumulate_kernel
    )

    kernel[(1,)](
        a_desc,
        b_desc,
        c_desc,
        d,
        *d.stride(),
        _GroupedPersistentTileScheduler(1),
        2,
        num_warps=4,
    )

    torch.testing.assert_close(d, a.float() @ b.float() + c, atol=1e-2, rtol=1e-2)


def test_gluon_simulator_runs_tensor_memory_load_reductions_on_cpu():
    inp = (torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) - 2048) / 64
    values_out = torch.empty((64, 16), dtype=torch.float32)
    max_out = torch.empty((64,), dtype=torch.float32)
    min_abs_out = torch.empty((64,), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _tensor_memory_load_reduction_kernel
    )

    kernel[(1,)](
        inp,
        values_out,
        max_out,
        min_abs_out,
        *inp.shape,
        num_warps=4,
    )

    sliced = inp[:, 8:24]
    torch.testing.assert_close(values_out, sliced, atol=0, rtol=0)
    torch.testing.assert_close(max_out, sliced.max(dim=1).values, atol=0, rtol=0)
    torch.testing.assert_close(min_abs_out, sliced.abs().min(dim=1).values)


def test_gluon_simulator_sanitizer_reports_tma_descriptor_oob_on_cpu():
    block_m = 16
    block_n = 16
    x = torch.zeros((block_m, block_n), dtype=torch.float32)
    layout = gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float32)
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(sanitizer, frontend="gluon")(_tma_oob_kernel)

    kernel[(1,)](x, block_m, block_n, block_m, block_n, layout, num_warps=4)

    assert len(sanitizer.records) == 1
    assert sanitizer.records[0].op_type is Load
    assert sanitizer.records[0].tensor.data_ptr() == x.data_ptr()
    assert tuple(sanitizer.records[0].tensor.shape) == tuple(x.shape)
    assert sanitizer.records[0].violation_address == (
        x.data_ptr() + block_m * block_n * x.element_size()
    )
    assert "descriptor_access" in str(sanitizer.records[0].symbolic_expr)


def test_gluon_simulator_runs_layout_anchor_and_barrier_on_cpu():
    values = torch.arange(8, dtype=torch.float32)
    out = torch.empty_like(values)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _layout_anchor_and_barrier_kernel
    )

    kernel[(1,)](values, out, values.numel(), num_warps=1)

    torch.testing.assert_close(out, values + 1.0, atol=0, rtol=0)


def test_gluon_simulator_runs_float2_helpers_on_cpu():
    f0 = torch.tensor([1.0, -2.0, 0.25, 8.0], dtype=torch.float32)
    f1 = torch.tensor([3.0, 4.5, -0.5, 0.125], dtype=torch.float32)
    out_sum0 = torch.empty_like(f0)
    out_sum1 = torch.empty_like(f0)
    out_fma0 = torch.empty_like(f0)
    out_fma1 = torch.empty_like(f0)
    out_full0 = torch.empty_like(f0)
    out_full1 = torch.empty_like(f0)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_float2_helpers_kernel)

    kernel[(1,)](
        f0,
        f1,
        out_sum0,
        out_sum1,
        out_fma0,
        out_fma1,
        out_full0,
        out_full1,
        f0.numel(),
        num_warps=1,
    )

    torch.testing.assert_close(out_sum0, f0 + f1, atol=0, rtol=0)
    torch.testing.assert_close(out_sum1, f0 + f1, atol=0, rtol=0)
    torch.testing.assert_close(out_fma0, f0 * f1 + f0, atol=0, rtol=0)
    torch.testing.assert_close(out_fma1, f1 * f0 + f1, atol=0, rtol=0)
    torch.testing.assert_close(out_full0, f0 * 2.0, atol=0, rtol=0)
    torch.testing.assert_close(out_full1, f1 * 2.0, atol=0, rtol=0)


def test_gluon_simulator_runs_float2_pack_unpack_and_reduce_on_cpu():
    values = torch.arange(32, dtype=torch.float32).reshape(4, 8) - 7.5
    roundtrip = torch.empty_like(values)
    even_sum = torch.empty((4,), dtype=torch.float32)
    odd_sum = torch.empty((4,), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_float2_pack_reduce_kernel)

    kernel[(1,)](values, roundtrip, even_sum, odd_sum, num_warps=4)

    torch.testing.assert_close(roundtrip, values, atol=0, rtol=0)
    torch.testing.assert_close(even_sum, values[:, 0::2].sum(dim=1), atol=0, rtol=0)
    torch.testing.assert_close(odd_sum, values[:, 1::2].sum(dim=1), atol=0, rtol=0)


def test_gluon_simulator_runs_float2_constructor_convert_on_cpu():
    values = torch.tensor(
        [[1.0, -2.0, 0.25, 4.0], [3.0, -0.5, 2.5, -1.25]],
        dtype=torch.float32,
    )
    out = torch.empty_like(values)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _float2_constructor_convert_kernel
    )

    kernel[(1,)](values, out, num_warps=4)

    torch.testing.assert_close(out, values * 3.0, atol=0, rtol=0)
