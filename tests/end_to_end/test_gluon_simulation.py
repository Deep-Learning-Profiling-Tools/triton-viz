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


@gluon.jit
def _standard_ops_kernel(out, xnumel: gl.constexpr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    zeros = gl.zeros((BLOCK,), gl.int32, layout=layout)
    total = gl.sum(offsets + zeros, axis=0)
    largest = gl.max(offsets, axis=0)
    blocks = gl.cdiv(xnumel, BLOCK)
    gl.store(out, total + largest + blocks)


@gluon.jit
def _static_print_kernel(in_ptr, out_ptr):
    gl.static_print("triton-viz-static-print", in_ptr.dtype)
    value = gl.load(in_ptr)
    gl.store(out_ptr, value + 1.0)


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


@gluon.aggregate
class _ClcTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    clc_result_buf: gl.shared_memory_descriptor
    barrier: gl.shared_memory_descriptor
    phase: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M, BLOCK_N):
        del M, N, BLOCK_M, BLOCK_N
        has_work = gl.to_tensor(True)
        starting_tile_id = gl.program_id(0)
        barrier = mbarrier.allocate_mbarrier()
        result_buffer = gl.allocate_shared_memory(
            gl.int64,
            [2],
            gl.SwizzledSharedLayout(1, 1, 1, [0]),
        )
        mbarrier.init(barrier, count=1)
        return _ClcTileScheduler(
            has_work,
            starting_tile_id,
            result_buffer,
            barrier,
            gl.to_tensor(0),
        )

    @gluon.jit
    def try_cancel(self):
        clc.try_cancel(self.clc_result_buf, self.barrier)
        mbarrier.expect(self.barrier, 16)

    @gluon.jit
    def advance(self):
        mbarrier.wait(self.barrier, self.phase)
        result = clc.load_result(self.clc_result_buf)
        return _ClcTileScheduler(
            result.is_canceled(),
            result.program_id(0),
            self.clc_result_buf,
            self.barrier,
            self.phase ^ 1,
        )


@gluon.aggregate
class _StaticTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    num_tiles: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M, BLOCK_N):
        starting_tile_id = gl.program_id(0)
        num_tiles = gl.cdiv(M, BLOCK_M) * gl.cdiv(N, BLOCK_N)
        return _StaticTileScheduler(
            starting_tile_id < num_tiles,
            starting_tile_id,
            num_tiles,
        )

    @gluon.jit
    def try_cancel(self):
        pass

    @gluon.jit
    def advance(self):
        next_tile_id = self.tile_id + gl.num_programs(0)
        return _StaticTileScheduler(
            next_tile_id < self.num_tiles,
            next_tile_id,
            self.num_tiles,
        )


@gluon.jit
def _scheduler_helpers_kernel(out, xnumel: gl.constexpr, BLOCK: gl.constexpr):
    pid = gl.program_id(axis=0)
    num_kernels = gl.num_programs(axis=0)
    pid_per_kernel = gl.cdiv(xnumel, num_kernels)
    pid_start = pid * pid_per_kernel
    pid_end = gl.minimum(pid_start + pid_per_kernel, xnumel)
    remaining = gl.maximum(pid_end - pid_start, 0)
    counter = _Counter.create(0, 2)
    counter = counter.next(pred=remaining > BLOCK)
    gl.store(out + pid, remaining + counter.index + counter.phase)


@gluon.jit
def _persistent_tile_scheduler_kernel(
    out,
    M,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    scheduler = _PersistentTileScheduler.initialize(M, N, BLOCK_M, BLOCK_N)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        tile_id = pid_n * num_pid_m + pid_m
        gl.store(out + tile_id, gl.program_id(axis=0) + 1)


@gluon.jit
def _tile_scheduler_loop_kernel(
    out,
    M,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SchedulerImpl: gl.constexpr,
):
    scheduler = SchedulerImpl.initialize(M, N, BLOCK_M, BLOCK_N)
    while scheduler.has_work:
        gl.store(out + scheduler.tile_id, gl.program_id(0) + 1)
        scheduler.try_cancel()
        scheduler = scheduler.advance()


@gluon.jit
def _math_helpers_kernel(out, clamp_out, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    values = gl.exp2(offsets.to(gl.float32))
    logs = gl.log2(values)
    mask = offsets < (BLOCK // 2)
    selected = gl.where(mask, logs, 0.0)
    gl.store(out, gl.sum(selected, axis=0))
    centered = offsets.to(gl.float32) - (BLOCK // 2)
    gl.store(clamp_out + offsets, gl.clamp(centered, -2.0, 2.0))


@gluon.jit
def _libdevice_helpers_kernel(
    x_ptr, y_ptr, exp_out, fast_exp_out, div_out, N: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, N, layout)
    x = gl.load(x_ptr + offsets)
    y = gl.load(y_ptr + offsets)
    gl.store(exp_out + offsets, libdevice.exp(x))
    gl.store(fast_exp_out + offsets, libdevice.fast_expf(x))
    gl.store(div_out + offsets, libdevice.fast_dividef(x, y))


@gluon.jit
def _amd_core_helpers_kernel(values_ptr, abs_out, old_out, lock, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout)
    gl.assume(BLOCK > 0)
    offsets = gl.multiple_of(offsets, [1])
    values = gl.load(values_ptr + offsets)
    gl.store(abs_out + offsets, gl.abs(values))
    old = gl.atomic_cas(lock, 0, 7)
    gl.store(old_out, old)


@gluon.jit
def _amd_tdm_copy_kernel(
    inp,
    copy_out,
    gather_out,
    scatter_out,
    row_indices_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    index_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    desc = amd_tdm.make_tensor_descriptor(
        inp,
        (M, N),
        (N, 1),
        (BLOCK_M, BLOCK_N),
        shared_layout,
    )
    copy_desc = amd_tdm.make_tensor_descriptor(
        copy_out,
        (M, N),
        (N, 1),
        (BLOCK_M, BLOCK_N),
        shared_layout,
    )
    gather_desc = amd_tdm.make_tensor_descriptor(
        gather_out,
        (BLOCK_M, BLOCK_N),
        (BLOCK_N, 1),
        (BLOCK_M, BLOCK_N),
        shared_layout,
    )
    scatter_desc = amd_tdm.make_tensor_descriptor(
        scatter_out,
        (M, N),
        (N, 1),
        (BLOCK_M, BLOCK_N),
        shared_layout,
    )
    smem = gl.allocate_shared_memory(gl.float32, (BLOCK_M, BLOCK_N), shared_layout)
    amd_tdm.async_load(desc, [0, 0], smem)
    amd_tdm.async_wait(0)
    amd_tdm.async_store(copy_desc, [0, 0], smem)
    amd_tdm.prefetch(desc, [0, 0])

    row_offsets = gl.arange(0, BLOCK_M, index_layout)
    row_indices = gl.load(row_indices_ptr + row_offsets)
    gather_smem = gl.allocate_shared_memory(
        gl.float32, (BLOCK_M, BLOCK_N), shared_layout
    )
    amd_tdm.async_gather(desc, row_indices, 0, gather_smem)
    amd_tdm.async_wait(0)
    amd_tdm.async_store(gather_desc, [0, 0], gather_smem)
    amd_tdm.async_scatter(scatter_desc, row_indices, 0, gather_smem)
    amd_tdm.async_wait(0)


@gluon.jit
def _amd_tdm_update_descriptor_kernel(
    inp,
    out_offset,
    out_bounded,
    M: gl.constexpr,
    N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    desc = amd_tdm.make_tensor_descriptor(
        inp,
        [M, N],
        [N, 1],
        [2, N],
        layout,
    )
    offset_desc = amd_tdm.update_tensor_descriptor(desc, add_offsets=[2, 0])
    offset_smem = gl.allocate_shared_memory(gl.float32, [2, N], layout)
    amd_tdm.async_load(offset_desc, [0, 0], offset_smem)
    amd_tdm.async_store(
        amd_tdm.make_tensor_descriptor(out_offset, [2, N], [N, 1], [2, N], layout),
        [0, 0],
        offset_smem,
    )
    bounded_desc = amd_tdm.update_tensor_descriptor(
        desc,
        add_offsets=[M - 1, 0],
        set_bounds=[1, N],
    )
    bounded_smem = gl.allocate_shared_memory(gl.float32, [2, N], layout)
    amd_tdm.async_load(bounded_desc, [0, 0], bounded_smem)
    amd_tdm.async_store(
        amd_tdm.make_tensor_descriptor(out_bounded, [2, N], [N, 1], [2, N], layout),
        [0, 0],
        bounded_smem,
    )


@gluon.jit
def _amd_async_copy_kernel(inp, out, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    offsets = gl.arange(0, BLOCK, layout)
    smem = gl.allocate_shared_memory(gl.float32, (BLOCK,), shared_layout)
    amd_async_copy.global_to_shared(smem, inp + offsets)
    amd_async_copy.commit_group()
    amd_async_copy.wait_group(0)
    amd_cluster.arrive()
    amd_cluster.wait()
    amd_async_copy.shared_to_global(out + offsets, smem)
    amd_async_copy.commit_group()
    amd_async_copy.wait_group(0)


@gluon.jit
def _amd_cdna4_async_copy_kernel(inp, out_global, out_buffer, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout).to(gl.int32)
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK], layout)
    cdna4_async_copy.global_load_to_shared(smem, inp + offsets)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    global_values = cdna4_async_copy.load_shared_relaxed(smem, layout)
    gl.store(out_global + offsets, global_values)

    mask = offsets < (BLOCK - 1)
    cdna4_async_copy.buffer_load_to_shared(smem, inp, offsets, mask, other=-3.0)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    buffer_values = cdna4_async_copy.load_shared_relaxed(smem, layout)
    gl.store(out_buffer + offsets, buffer_values)


@gluon.jit
def _amd_buffer_ops_kernel(inp, out_module, out_direct, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout).to(gl.int32)
    mask = offsets < (BLOCK - 1)
    module_values = gl.amd.gfx1250.buffer_load(inp, offsets, mask=mask, other=-1.0)
    gl.amd.gfx1250.buffer_store(module_values + 1.0, out_module, offsets, mask=mask)
    direct_values = amd_buffer_load(inp, offsets, mask=mask, other=-2.0)
    amd_buffer_store(direct_values + 2.0, out_direct, offsets, mask=mask)


@gluon.jit
def _amd_wmma_kernel(
    a_ptr,
    b_ptr,
    out_module,
    out_direct,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    offs_m = gl.arange(0, M, layout)[:, None]
    offs_n = gl.arange(0, N, layout)[None, :]
    offs_k = gl.arange(0, K, layout)
    a = gl.load(a_ptr + offs_m * K + offs_k[None, :])
    b = gl.load(b_ptr + offs_k[:, None] * N + offs_n)
    acc = gl.zeros((M, N), dtype=gl.float32, layout=layout)
    module_result = gl.amd.gfx1250.wmma(a, b, acc)
    direct_result = amd_wmma(a, b, acc)
    gl.store(out_module + offs_m * N + offs_n, module_result)
    gl.store(out_direct + offs_m * N + offs_n, direct_result)


@gluon.jit
def _amd_rdna_wmma_kernel(
    a_ptr,
    b_ptr,
    out_rdna3,
    out_rdna4,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    offs_m = gl.arange(0, M, layout)[:, None]
    offs_n = gl.arange(0, N, layout)[None, :]
    offs_k = gl.arange(0, K, layout)
    a = gl.load(a_ptr + offs_m * K + offs_k[None, :])
    b = gl.load(b_ptr + offs_k[:, None] * N + offs_n)
    acc = gl.zeros((M, N), dtype=gl.float32, layout=layout)
    rdna3_result = gl.amd.rdna3.wmma(a, b, acc)
    rdna4_result = amd_rdna4.wmma(a, b, acc)
    gl.store(out_rdna3 + offs_m * N + offs_n, rdna3_result)
    gl.store(out_rdna4 + offs_m * N + offs_n, rdna4_result)


@gluon.jit
def _amd_scaled_upcast_kernel(out_cdna3, out_cdna4, out_gfx1250):
    packed_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    unpacked_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    packed = gl.full([2], 0x21, gl.uint8, packed_layout)
    scale = gl.full([4], 0x7F, gl.uint8, unpacked_layout)
    cdna3_values = gl.amd.cdna3.scaled_upcast(packed, scale, gl.float16, axis=0)
    cdna4_values = amd_cdna4.scaled_upcast(packed, scale, gl.float16, axis=0)
    gfx1250_values = gl.amd.gfx1250.scaled_upcast(packed, scale, gl.float16, axis=0)
    offsets = gl.arange(0, 4, unpacked_layout)
    gl.store(out_cdna3 + offsets, cdna3_values)
    gl.store(out_cdna4 + offsets, cdna4_values)
    gl.store(out_gfx1250 + offsets, gfx1250_values)


@gluon.jit
def _fp4_to_fp_kernel(out, BLOCK: gl.constexpr):
    packed_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    unpacked_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    packed = gl.full([2], 0x21, gl.uint8, packed_layout)
    values = gl.fp4_to_fp(packed, gl.float32, axis=0)
    offsets = gl.arange(0, BLOCK, unpacked_layout)
    gl.store(out + offsets, values)


@gluon.jit
def _amd_warp_pipeline_stage_kernel(inp, out, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout)
    with gl.amd.warp_pipeline_stage("load", priority=3):
        values = gl.load(inp + offsets)
    with amd_warp_pipeline_stage("compute"):
        values = (values + 1.0) * 2.0
    gl.store(out + offsets, values)


@gluon.jit
def _amd_cdna_buffer_ops_kernel(
    inp,
    out_cdna3,
    out_cdna4,
    atomic_values,
    old_values,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout).to(gl.int32)
    mask = offsets < (BLOCK - 1)
    cdna3_values = gl.amd.cdna3.buffer_load(inp, offsets, mask=mask, other=-1.0)
    gl.amd.cdna3.buffer_store(cdna3_values + 1.0, out_cdna3, offsets, mask=mask)
    cdna4_values = amd_cdna4.buffer_load(inp, offsets, mask=mask, other=-2.0)
    amd_cdna4.buffer_store(cdna4_values + 2.0, out_cdna4, offsets, mask=mask)
    old = amd_cdna3.buffer_atomic_add(atomic_values, offsets, 3, mask=mask)
    gl.store(old_values + offsets, old, mask=mask)


@gluon.jit
def _amd_cdna_mfma_kernel(
    a_ptr,
    b_ptr,
    out_cdna3,
    out_cdna4,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    offs_m = gl.arange(0, M, layout)[:, None]
    offs_n = gl.arange(0, N, layout)[None, :]
    offs_k = gl.arange(0, K, layout)
    a = gl.load(a_ptr + offs_m * K + offs_k[None, :])
    b = gl.load(b_ptr + offs_k[:, None] * N + offs_n)
    acc = gl.zeros((M, N), dtype=gl.float32, layout=layout)
    cdna3_result = gl.amd.cdna3.mfma(a, b, acc)
    cdna4_result = amd_cdna4.mfma(a, b, acc)
    gl.store(out_cdna3 + offs_m * N + offs_n, cdna3_result)
    gl.store(out_cdna4 + offs_m * N + offs_n, cdna4_result)


@gluon.jit
def _amd_scaled_mma_kernel(
    a_ptr,
    b_ptr,
    out_cdna4,
    out_gfx1250,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [0, 1])
    offs_m = gl.arange(0, M, layout)[:, None]
    offs_n = gl.arange(0, N, layout)[None, :]
    offs_k = gl.arange(0, K, layout)
    a = gl.load(a_ptr + offs_m * K + offs_k[None, :])
    b = gl.load(b_ptr + offs_k[:, None] * N + offs_n)
    acc = gl.zeros((M, N), dtype=gl.float32, layout=layout)
    cdna4_result = amd_cdna4.mfma_scaled(a, 0x7F, "e4m3", b, 0x7F, "e4m3", acc)
    gfx1250_result = gl.amd.gfx1250.wmma_scaled(
        a,
        0x7F,
        "e4m3",
        b,
        0x7F,
        "e4m3",
        acc,
    )
    gl.store(out_cdna4 + offs_m * N + offs_n, cdna4_result)
    gl.store(out_gfx1250 + offs_m * N + offs_n, gfx1250_result)


@gluon.jit
def _layout_anchor_and_barrier_kernel(in_ptr, out_ptr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, N, layout)
    values = gl.load(in_ptr + offsets)
    values = gl.set_auto_layout(values, layout)
    gl.barrier()
    gl.store(out_ptr + offsets, values + 1.0)


@gluon.jit
def _mask_scalar(qk, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return gl.where(mask_i_bit, qk, -float("inf"))


@gluon.jit
def _map_elementwise_mask_kernel(qk_ptr, out_ptr, col_limit_right, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, N, layout)
    qk = gl.load(qk_ptr + offsets)
    s = offsets & ~0xF
    i = offsets & 0xF
    masked = gl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)
    gl.store(out_ptr + offsets, masked)


@gluon.jit
def _inline_asm_helpers_kernel(
    i0_ptr,
    i1_ptr,
    f0_ptr,
    f1_ptr,
    out_i32,
    out_low,
    out_high,
    out_e4,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout)
    i0 = gl.load(i0_ptr + offsets)
    i1 = gl.load(i1_ptr + offsets)
    f0 = gl.load(f0_ptr + offsets)
    f1 = gl.load(f1_ptr + offsets)

    packed_i32 = gl.inline_asm_elementwise(
        "mov.b32 $0, { $1, $2 };",
        "=r,h,h",
        [i0, i1],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )
    gl.store(out_i32 + offsets, packed_i32)

    packed_f32x2 = gl.inline_asm_elementwise(
        "mov.b64 $0, { $1, $2 };",
        "=l,r,r",
        [f0, f1],
        dtype=gl.int64,
        is_pure=True,
        pack=1,
    )
    doubled = gl.inline_asm_elementwise(
        "add.f32x2 $0, $1, $2;",
        "=l,l,l",
        [packed_f32x2, packed_f32x2],
        dtype=gl.int64,
        is_pure=True,
        pack=1,
    )
    low, high = gl.inline_asm_elementwise(
        "mov.b64 { $0, $1 }, $2;",
        "=r,=r,l",
        [doubled],
        dtype=[gl.float32, gl.float32],
        is_pure=True,
        pack=1,
    )
    gl.store(out_low + offsets, low)
    gl.store(out_high + offsets, high)

    packed_e4 = gl.inline_asm_elementwise(
        """
        {
            .reg .f32 lane<2>;
            mov.b64 {lane0, lane1}, $1;
            cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
        }
        """,
        "=h,l",
        [packed_f32x2],
        dtype=gl.int16,
        is_pure=True,
        pack=1,
    )
    gl.store(out_e4 + offsets, packed_e4)


@gluon.jit
def _pointer_bitcast_store_kernel(out_ptr, values_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout)
    values = gl.load(values_ptr + offsets)
    out_i32 = out_ptr.cast(gl.pointer_type(gl.int32), bitcast=True)
    gl.store(out_i32 + offsets, values)


@gluon.jit
def _core_memory_split_ops_kernel(values_ptr, masked_out, split_lhs_out, split_rhs_out):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, 8, layout)
    mask = offsets < 5
    values = gl.load(values_ptr + offsets, mask=mask, other=-7.0)
    lhs, rhs = gl.split(values.reshape((4, 2)))
    gl.store(masked_out + offsets, values, mask=offsets != 6)
    half_offsets = gl.arange(0, 4, layout)
    gl.store(split_lhs_out + half_offsets, lhs)
    gl.store(split_rhs_out + half_offsets, rhs)


@gluon.jit
def _atomic_memory_ops_kernel(counter, addends, add_old, xchg_old, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, BLOCK, layout)
    values = gl.load(addends + offsets)
    mask = offsets != 1
    old = gl.atomic_add(
        counter + offsets, values, mask=mask, sem="relaxed", scope="gpu"
    )
    gl.store(add_old + offsets, old, mask=mask)
    exchanged = gl.atomic_xchg(
        counter + offsets,
        values * 10,
        mask=mask,
        sem="relaxed",
        scope="gpu",
    )
    gl.store(xchg_old + offsets, exchanged, mask=mask)


@gluon.jit
def _moe_pack_e4m3x2(values):
    return gl.inline_asm_elementwise(
        """
        {
            .reg .f32 lane<2>;
            mov.b64 {lane0, lane1}, $1;
            cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
        }
        """,
        "=h,l",
        [values.value],
        dtype=gl.int16,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _moe_pack_u16x2(x0, x1):
    return gl.inline_asm_elementwise(
        """
        mov.b32 $0, { $1, $2 };
        """,
        "=r,h,h",
        [x0, x1],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _moe_pack_fp8x4(values):
    reshaped = values.reshape((values.shape[0], values.shape[1] // 2, 2))
    lhs, rhs = gl.split(reshaped)
    return _moe_pack_u16x2(lhs, rhs)


@gluon.jit
def _moe_epilogue_pack_store_kernel(
    values_ptr,
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 32],
        [1, gl.num_warps()],
        [1, 0],
    )
    rows = gl.arange(0, M, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, N, layout=gl.SliceLayout(0, layout))
    values = gl.load(values_ptr + rows[:, None] * N + cols[None, :])
    fp8_pairs = _moe_pack_e4m3x2(float2_pack(values, axis=1))
    fp8_quads = _moe_pack_fp8x4(fp8_pairs)

    packed_cols = gl.arange(
        0,
        N // 4,
        layout=gl.SliceLayout(0, fp8_quads.type.layout),
    )
    mask_m = gl.expand_dims(rows < M, 1)
    mask_n = gl.expand_dims(packed_cols < N // 4, 0)
    out_i32 = out_ptr.cast(gl.pointer_type(gl.int32), bitcast=True)
    gl.store(
        out_i32 + rows[:, None] * (N // 4) + packed_cols[None, :],
        fp8_quads,
        mask=mask_m & mask_n,
    )


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


@gluon.jit
def _tensor_shape_helpers_kernel(
    values_ptr,
    vector_out,
    split_lhs_out,
    split_rhs_out,
    joined_out,
    expanded_out,
    M: gl.constexpr,
    N: gl.constexpr,
):
    layout_1d: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets_m = gl.arange(0, M, layout_1d)
    values = gl.load(values_ptr + offsets_m)
    broadcasted = values[:, None].broadcast_to(M, N)

    layout_2d: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    offsets_n = gl.arange(0, N, gl.SliceLayout(0, layout_2d))
    rows = gl.arange(0, M, gl.SliceLayout(1, layout_2d))
    gl.store(vector_out + rows[:, None] * N + offsets_n[None, :], broadcasted)
    expanded_rows = gl.expand_dims(rows, 1)
    expanded_cols = gl.expand_dims(offsets_n, 0)
    expanded = gl.load(values_ptr + expanded_rows).broadcast_to(M, N)
    expanded_mask = (expanded_rows < M) & (expanded_cols < N)
    gl.store(
        expanded_out + expanded_rows * N + expanded_cols, expanded, mask=expanded_mask
    )

    reshaped = broadcasted.reshape((M, 2, N // 2)).permute((0, 2, 1))
    lhs, rhs = gl.split(reshaped)
    half_cols = gl.arange(0, N // 2, gl.SliceLayout(0, layout_2d))
    gl.store(split_lhs_out + rows[:, None] * (N // 2) + half_cols[None, :], lhs)
    gl.store(split_rhs_out + rows[:, None] * (N // 2) + half_cols[None, :], rhs)

    joined = gl.join(lhs, rhs).permute((0, 2, 1)).reshape((M, N))
    gl.store(joined_out + rows[:, None] * N + offsets_n[None, :], joined)


@gluon.jit
def _num_ctas_kernel(out):
    gl.store(out, gl.num_ctas())


@gluon.jit
def _mbarrier_batch_shape_kernel(out):
    bars = mbarrier.allocate_mbarrier(batch=3)
    two_cta_bars = mbarrier.allocate_mbarrier(batch=5, two_ctas=True)
    multicta_layout: gl.constexpr = mbarrier.MBarrierLayout.multicta(gl.num_ctas())
    two_cta_layout: gl.constexpr = mbarrier.MBarrierLayout.multicta(
        gl.num_ctas(),
        two_cta=True,
    )
    gl.store(out + 0, bars.shape[0])
    gl.store(out + 1, bars.shape[1])
    gl.store(out + 2, two_cta_bars.shape[0])
    gl.store(out + 3, two_cta_bars.shape[1])
    gl.store(out + 4, len(multicta_layout.cga_layout))
    gl.store(out + 5, multicta_layout.cga_layout[0][0])
    gl.store(out + 6, multicta_layout.cga_layout[1][0])
    gl.store(out + 7, len(two_cta_layout.cga_layout))
    gl.store(out + 8, two_cta_layout.cga_layout[0][0])
    gl.store(out + 9, two_cta_layout.cga_layout[1][0])


@gluon.jit
def _clc_api_kernel(out):
    bar = mbarrier.allocate_mbarrier()
    result_buffer = gl.allocate_shared_memory(
        gl.int64,
        [2],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    mbarrier.init(bar, count=1)
    clc.try_cancel(result_buffer, bar)
    mbarrier.expect(bar, 16)
    mbarrier.wait(bar, phase=0)
    result = clc.load_result(result_buffer)
    gl.store(out, gl.where(result.is_canceled(), 1, 0) + result.program_id(0))


@gluon.jit
def _multicta_softmax_kernel(
    x_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    BLOCK_N: gl.constexpr,
):
    pid = gl.program_id(0)
    cga_layout: gl.constexpr = ((1,), (2,), (4,), (8,), (16,))[
        : gl.num_ctas().bit_length() - 1
    ]
    layout: gl.constexpr = gl.BlockedLayout(
        [4],
        [32],
        [gl.num_warps()],
        [0],
        cga_layout=cga_layout,
    )
    offsets = gl.arange(0, BLOCK_N, layout)
    row_start = pid * x_row_stride
    out_row_start = pid * out_row_stride
    values = gl.load(x_ptr + row_start + offsets)
    row_max = gl.max(values, axis=0)
    exp_values = gl.exp(values - row_max)
    row_sum = gl.sum(exp_values, axis=0)
    gl.store(out_ptr + out_row_start + offsets, exp_values * (1.0 / row_sum))


@gluon.jit
def _warp_specialize_append_partition(out, counter, value: gl.constexpr):
    idx = gl.atomic_add(counter, 1, sem="relaxed", scope="cta")
    gl.store(out + idx, value)
    idx = gl.atomic_add(counter, 1, sem="relaxed", scope="cta")
    gl.store(out + idx, value)


@gluon.jit
def _warp_specialize_helpers_kernel(out, counter):
    gl.store(counter, 0)
    gl.warp_specialize(
        [
            (_warp_specialize_append_partition, (out, counter, 1)),
            (_warp_specialize_append_partition, (out, counter, 2)),
            (_warp_specialize_append_partition, (out, counter, 4)),
        ],
        [1, 1],
        [24, 24],
    )


@gluon.jit
def _ws_elementwise_load_partition(
    descs, barriers, buffers, xoff, ynumel, YBLOCK: gl.constexpr
):
    a_desc, b_desc, _ = descs
    load_empty_bars, load_ready_bars, _, _ = barriers
    a_bufs, b_bufs, _ = buffers
    num_buffers: gl.constexpr = a_bufs.type.shape[0]
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        load_empty_bar = load_empty_bars.index(index)
        load_ready_bar = load_ready_bars.index(index)
        mbarrier.wait(load_empty_bar, phase ^ 1)
        yoff = i * YBLOCK
        mbarrier.expect(
            load_ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes
        )
        tma.async_load(a_desc, [xoff, yoff], load_ready_bar, a_buf)
        tma.async_load(b_desc, [xoff, yoff], load_ready_bar, b_buf)


@gluon.jit
def _ws_elementwise_compute_partition(
    barriers,
    buffers,
    ynumel,
    YBLOCK: gl.constexpr,
    layout: gl.constexpr,
):
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    num_load_buffers: gl.constexpr = a_bufs.type.shape[0]
    num_store_buffers: gl.constexpr = c_bufs.type.shape[0]
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        load_index = i % num_load_buffers
        load_phase = i // num_load_buffers & 1
        a_buf = a_bufs.index(load_index)
        b_buf = b_bufs.index(load_index)
        load_ready_bar = load_ready_bars.index(load_index)
        load_empty_bar = load_empty_bars.index(load_index)
        mbarrier.wait(load_ready_bar, load_phase)
        c_val = a_buf.load(layout) + b_buf.load(layout)
        fence_async_shared()
        mbarrier.arrive(load_empty_bar, count=1)

        store_index = i % num_store_buffers
        store_phase = i // num_store_buffers & 1
        c_buf = c_bufs.index(store_index)
        c_empty_bar = c_empty_bars.index(store_index)
        c_ready_bar = c_ready_bars.index(store_index)
        mbarrier.wait(c_empty_bar, store_phase ^ 1)
        c_buf.store(c_val)
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def _ws_elementwise_store_partition(
    descs, barriers, buffers, xoff, ynumel, YBLOCK: gl.constexpr
):
    _, _, c_desc = descs
    _, _, c_empty_bars, c_ready_bars = barriers
    _, _, c_bufs = buffers
    num_buffers: gl.constexpr = c_bufs.type.shape[0]
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        c_buf = c_bufs.index(index)
        c_ready_bar = c_ready_bars.index(index)
        mbarrier.wait(c_ready_bar, phase)
        yoff = i * YBLOCK
        tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_buf)
        tma.store_wait(0)
        mbarrier.arrive(c_empty_bars.index(index), count=1)


@gluon.jit
def _warp_specialized_elementwise_add_kernel(
    a_desc,
    b_desc,
    c_desc,
    xnumel: gl.constexpr,
    ynumel: gl.constexpr,
    XBLOCK: gl.constexpr,
    YBLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    a_bufs = gl.allocate_shared_memory(
        a_desc.dtype, [2] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        b_desc.dtype, [2] + b_desc.block_type.shape, b_desc.layout
    )
    c_bufs = gl.allocate_shared_memory(
        c_desc.dtype, [2] + c_desc.block_type.shape, c_desc.layout
    )
    load_empty_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    load_ready_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    c_empty_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    c_ready_bars = gl.allocate_shared_memory(
        gl.int64, [2, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(2):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)
        mbarrier.init(c_empty_bars.index(i), count=1)
        mbarrier.init(c_ready_bars.index(i), count=1)
    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    xoff = gl.program_id(0) * XBLOCK
    gl.warp_specialize(
        [
            (
                _ws_elementwise_compute_partition,
                (barriers, buffers, ynumel, YBLOCK, layout),
            ),
            (
                _ws_elementwise_load_partition,
                (descs, barriers, buffers, xoff, ynumel, YBLOCK),
            ),
            (
                _ws_elementwise_store_partition,
                (descs, barriers, buffers, xoff, ynumel, YBLOCK),
            ),
        ],
        [1, 1],
        [24, 24],
    )


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
def _shared_memory_slice_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    offs_m = gl.arange(0, M, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, layout))
    data = gl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :])
    smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [M, N],
        data.dtype,
    )
    smem = gl.allocate_shared_memory(data.dtype, [M, N], smem_layout)
    smem.store(data)
    left = smem.slice(0, N // 2, dim=1)
    right = smem.slice(N // 2, N // 2, dim=1)
    out = left.load(layout) + right.load(layout)
    half_n = gl.arange(0, N // 2, gl.SliceLayout(0, layout))
    gl.store(out_ptr + offs_m[:, None] * (N // 2) + half_n[None, :], out)


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


def test_gluon_simulator_runs_shared_memory_slice_on_cpu():
    inp = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    out = torch.empty((32, 8), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_shared_memory_slice_kernel)

    kernel[(1,)](inp, out, *inp.shape, num_warps=4)

    torch.testing.assert_close(out, inp[:, :8] + inp[:, 8:], atol=0, rtol=0)


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


def test_gluon_simulator_runs_standard_jit_helpers_on_cpu():
    out = torch.empty((1,), dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_standard_ops_kernel)

    kernel[(1,)](out, 17, 8, num_warps=1)

    assert out.item() == sum(range(8)) + 7 + 3


def test_gluon_simulator_runs_static_print_on_cpu(capsys):
    inp = torch.tensor([4.0], dtype=torch.float32)
    out = torch.empty_like(inp)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_static_print_kernel)

    kernel[(1,)](inp, out, num_warps=1)

    assert "triton-viz-static-print" in capsys.readouterr().out
    torch.testing.assert_close(out, inp + 1.0, atol=0, rtol=0)


def test_gluon_simulator_runs_scheduler_style_helpers_on_cpu():
    out = torch.empty((4,), dtype=torch.int64)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_scheduler_helpers_kernel)

    kernel[(4,)](out, 17, 4, num_warps=1)

    # cdiv(17, 4) gives five assigned elements per program. The final program
    # has only two real elements, and Counter.next toggles index when > BLOCK.
    torch.testing.assert_close(out, torch.tensor([6, 6, 6, 2]), atol=0, rtol=0)


def test_gluon_simulator_runs_persistent_tile_scheduler_on_cpu():
    out = torch.full((4,), -1, dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _persistent_tile_scheduler_kernel
    )

    kernel[(2,)](out, 4, 6, 2, 3, num_warps=1)

    torch.testing.assert_close(
        out,
        torch.tensor([1, 1, 2, 2], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_gluon_simulator_runs_static_tile_scheduler_loop_on_cpu():
    out = torch.full((6,), -1, dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tile_scheduler_loop_kernel)

    kernel[(2,)](out, 6, 4, 2, 2, _StaticTileScheduler, num_warps=1)

    torch.testing.assert_close(
        out,
        torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_gluon_simulator_runs_clc_tile_scheduler_loop_on_cpu():
    out = torch.full((6,), -1, dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tile_scheduler_loop_kernel)

    kernel[(6,)](out, 6, 4, 2, 2, _ClcTileScheduler, num_warps=4)

    torch.testing.assert_close(
        out,
        torch.arange(1, 7, dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_gluon_simulator_runs_math_helpers_on_cpu():
    out = torch.empty((1,), dtype=torch.float32)
    clamp_out = torch.empty((8,), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_math_helpers_kernel)

    kernel[(1,)](out, clamp_out, 8, num_warps=1)

    assert out.item() == sum(range(4))
    expected_clamp = torch.clamp(torch.arange(8, dtype=torch.float32) - 4, -2.0, 2.0)
    torch.testing.assert_close(clamp_out, expected_clamp, atol=0, rtol=0)


def test_gluon_simulator_runs_libdevice_helpers_on_cpu():
    x = torch.tensor([-2.0, -0.5, 0.25, 1.5], dtype=torch.float32)
    y = torch.tensor([2.0, -4.0, 0.5, 3.0], dtype=torch.float32)
    exp_out = torch.empty_like(x)
    fast_exp_out = torch.empty_like(x)
    div_out = torch.empty_like(x)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_libdevice_helpers_kernel)

    kernel[(1,)](x, y, exp_out, fast_exp_out, div_out, x.numel(), num_warps=1)

    torch.testing.assert_close(exp_out, torch.exp(x), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_exp_out, torch.exp(x), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(div_out, x / y, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_core_helpers_on_cpu():
    values = torch.tensor([-3, 4, -5, 6], dtype=torch.int32)
    abs_out = torch.empty_like(values)
    old_out = torch.empty((), dtype=torch.int32)
    lock = torch.zeros((), dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_core_helpers_kernel)

    kernel[(1,)](values, abs_out, old_out, lock, values.numel(), num_warps=1)

    torch.testing.assert_close(abs_out, torch.abs(values), atol=0, rtol=0)
    assert old_out.item() == 0
    assert lock.item() == 7


def test_gluon_simulator_runs_amd_tdm_copy_ops_on_cpu():
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    copy_out = torch.full_like(values, -1.0)
    gather_out = torch.full((4, 4), -1.0, dtype=torch.float32)
    scatter_out = torch.full_like(values, -1.0)
    row_indices = torch.tensor([4, 1, 3, 0], dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_tdm_copy_kernel)

    kernel[(1,)](
        values,
        copy_out,
        gather_out,
        scatter_out,
        row_indices,
        values.shape[0],
        values.shape[1],
        row_indices.numel(),
        values.shape[1],
        num_warps=1,
    )

    expected_copy = torch.full_like(values, -1.0)
    expected_copy[: row_indices.numel(), :] = values[: row_indices.numel(), :]
    torch.testing.assert_close(copy_out, expected_copy, atol=0, rtol=0)
    torch.testing.assert_close(gather_out, values[row_indices], atol=0, rtol=0)
    expected_scatter = torch.full_like(values, -1.0)
    expected_scatter[row_indices] = values[row_indices]
    torch.testing.assert_close(scatter_out, expected_scatter, atol=0, rtol=0)


def test_gluon_simulator_updates_amd_tdm_descriptors_on_cpu():
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    out_offset = torch.full((2, 4), -1.0, dtype=torch.float32)
    out_bounded = torch.full((2, 4), -1.0, dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _amd_tdm_update_descriptor_kernel
    )

    kernel[(1,)](
        values,
        out_offset,
        out_bounded,
        values.shape[0],
        values.shape[1],
        num_warps=1,
    )

    torch.testing.assert_close(out_offset, values[2:4], atol=0, rtol=0)
    expected_bounded = torch.zeros_like(out_bounded)
    expected_bounded[0] = values[-1]
    torch.testing.assert_close(out_bounded, expected_bounded, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_async_copy_ops_on_cpu():
    values = torch.tensor([1.0, -2.0, 3.5, 4.25], dtype=torch.float32)
    out = torch.full_like(values, -1.0)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_async_copy_kernel)

    kernel[(1,)](values, out, values.numel(), num_warps=1)

    torch.testing.assert_close(out, values, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_cdna4_async_copy_ops_on_cpu():
    values = torch.tensor([1.0, -2.0, 3.5, 4.25], dtype=torch.float32)
    out_global = torch.full_like(values, -1.0)
    out_buffer = torch.full_like(values, -1.0)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_cdna4_async_copy_kernel)

    kernel[(1,)](values, out_global, out_buffer, values.numel(), num_warps=1)

    torch.testing.assert_close(out_global, values, atol=0, rtol=0)
    expected_buffer = torch.tensor([1.0, -2.0, 3.5, -3.0], dtype=torch.float32)
    torch.testing.assert_close(out_buffer, expected_buffer, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_buffer_ops_on_cpu():
    values = torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32)
    out_module = torch.full_like(values, -9.0)
    out_direct = torch.full_like(values, -7.0)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_buffer_ops_kernel)

    kernel[(1,)](values, out_module, out_direct, values.numel(), num_warps=1)

    expected_module = torch.tensor([3.0, 5.0, 9.0, -9.0], dtype=torch.float32)
    expected_direct = torch.tensor([4.0, 6.0, 10.0, -7.0], dtype=torch.float32)
    torch.testing.assert_close(out_module, expected_module, atol=0, rtol=0)
    torch.testing.assert_close(out_direct, expected_direct, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_wmma_on_cpu():
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2) - 2
    out_module = torch.empty((2, 2), dtype=torch.float32)
    out_direct = torch.empty((2, 2), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_wmma_kernel)

    kernel[(1,)](a, b, out_module, out_direct, 2, 2, 4, num_warps=1)

    expected = a @ b
    torch.testing.assert_close(out_module, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_direct, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_rdna_wmma_on_cpu():
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2) - 2
    out_rdna3 = torch.empty((2, 2), dtype=torch.float32)
    out_rdna4 = torch.empty((2, 2), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_rdna_wmma_kernel)

    kernel[(1,)](a, b, out_rdna3, out_rdna4, 2, 2, 4, num_warps=1)

    expected = a @ b
    torch.testing.assert_close(out_rdna3, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_rdna4, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_scaled_upcast_on_cpu():
    out_cdna3 = torch.empty((4,), dtype=torch.float16)
    out_cdna4 = torch.empty((4,), dtype=torch.float16)
    out_gfx1250 = torch.empty((4,), dtype=torch.float16)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_scaled_upcast_kernel)

    kernel[(1,)](out_cdna3, out_cdna4, out_gfx1250, num_warps=1)

    expected = torch.tensor([0.5, 1.0, 0.5, 1.0], dtype=torch.float16)
    torch.testing.assert_close(out_cdna3, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_cdna4, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_gfx1250, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_fp4_to_fp_on_cpu():
    out = torch.empty((4,), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_fp4_to_fp_kernel)

    kernel[(1,)](out, out.numel(), num_warps=1)

    expected = torch.tensor([0.5, 1.0, 0.5, 1.0], dtype=torch.float32)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_warp_pipeline_stage_on_cpu():
    values = torch.tensor([1.0, -2.0, 3.5, 4.25], dtype=torch.float32)
    out = torch.empty_like(values)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _amd_warp_pipeline_stage_kernel
    )

    kernel[(1,)](values, out, values.numel(), num_warps=1)

    torch.testing.assert_close(out, (values + 1.0) * 2.0, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_cdna_buffer_ops_on_cpu():
    values = torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32)
    out_cdna3 = torch.full_like(values, -9.0)
    out_cdna4 = torch.full_like(values, -7.0)
    atomic_values = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    old_values = torch.full_like(atomic_values, -1)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_cdna_buffer_ops_kernel)

    kernel[(1,)](
        values,
        out_cdna3,
        out_cdna4,
        atomic_values,
        old_values,
        values.numel(),
        num_warps=1,
    )

    expected_cdna3 = torch.tensor([3.0, 5.0, 9.0, -9.0], dtype=torch.float32)
    expected_cdna4 = torch.tensor([4.0, 6.0, 10.0, -7.0], dtype=torch.float32)
    torch.testing.assert_close(out_cdna3, expected_cdna3, atol=0, rtol=0)
    torch.testing.assert_close(out_cdna4, expected_cdna4, atol=0, rtol=0)
    torch.testing.assert_close(
        old_values,
        torch.tensor([10, 20, 30, -1], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        atomic_values,
        torch.tensor([13, 23, 33, 40], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_gluon_simulator_runs_amd_cdna_mfma_on_cpu():
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2) - 2
    out_cdna3 = torch.empty((2, 2), dtype=torch.float32)
    out_cdna4 = torch.empty((2, 2), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_cdna_mfma_kernel)

    kernel[(1,)](a, b, out_cdna3, out_cdna4, 2, 2, 4, num_warps=1)

    expected = a @ b
    torch.testing.assert_close(out_cdna3, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_cdna4, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_amd_scaled_mma_on_cpu():
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2) - 2
    out_cdna4 = torch.empty((2, 2), dtype=torch.float32)
    out_gfx1250 = torch.empty((2, 2), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_amd_scaled_mma_kernel)

    kernel[(1,)](a, b, out_cdna4, out_gfx1250, 2, 2, 4, num_warps=1)

    expected = a @ b
    torch.testing.assert_close(out_cdna4, expected, atol=0, rtol=0)
    torch.testing.assert_close(out_gfx1250, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_layout_anchor_and_barrier_on_cpu():
    values = torch.arange(8, dtype=torch.float32)
    out = torch.empty_like(values)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _layout_anchor_and_barrier_kernel
    )

    kernel[(1,)](values, out, values.numel(), num_warps=1)

    torch.testing.assert_close(out, values + 1.0, atol=0, rtol=0)


def test_gluon_simulator_runs_map_elementwise_causal_mask_on_cpu():
    values = torch.arange(32, dtype=torch.float32)
    out = torch.empty_like(values)
    col_limit_right = 20
    kernel = triton_viz.trace("tracer", frontend="gluon")(_map_elementwise_mask_kernel)

    kernel[(1,)](values, out, col_limit_right, values.numel(), num_warps=1)

    expected = values.clone()
    for offset in range(values.numel()):
        s = offset & ~0xF
        i = offset & 0xF
        mask = -1 << max(col_limit_right - s, 0)
        if (mask & (1 << i)) != 0:
            expected[offset] = -float("inf")
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_gluon_simulator_runs_supported_inline_asm_helpers_on_cpu():
    i0 = torch.tensor([1, 2, 0x1234, -1], dtype=torch.int32)
    i1 = torch.tensor([3, 4, 0x5678, 0], dtype=torch.int32)
    f0 = torch.tensor([1.0, -1.0, 0.0, 448.0], dtype=torch.float32)
    f1 = torch.tensor([2.0, 0.5, -2.0, 500.0], dtype=torch.float32)
    out_i32 = torch.empty_like(i0)
    out_low = torch.empty_like(f0)
    out_high = torch.empty_like(f1)
    out_e4 = torch.empty((4,), dtype=torch.int16)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_inline_asm_helpers_kernel)

    kernel[(1,)](
        i0,
        i1,
        f0,
        f1,
        out_i32,
        out_low,
        out_high,
        out_e4,
        i0.numel(),
        num_warps=1,
    )

    expected_i32 = torch.tensor(
        [0x00030001, 0x00040002, 0x56781234, 0x0000FFFF],
        dtype=torch.int32,
    )
    expected_e4 = torch.tensor([0x4038, 0x30B8, -0x4000, 0x7E7E], dtype=torch.int16)
    torch.testing.assert_close(out_i32, expected_i32, atol=0, rtol=0)
    torch.testing.assert_close(out_low, f0 * 2.0, atol=0, rtol=0)
    torch.testing.assert_close(out_high, f1 * 2.0, atol=0, rtol=0)
    torch.testing.assert_close(out_e4, expected_e4, atol=0, rtol=0)


def test_gluon_simulator_runs_pointer_bitcast_store_on_cpu():
    values = torch.tensor(
        [0x3F800000, 0x40000000, -0x40800000, 0],
        dtype=torch.int32,
    )
    out = torch.empty((4,), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_pointer_bitcast_store_kernel)

    kernel[(1,)](out, values, values.numel(), num_warps=1)

    torch.testing.assert_close(out.view(torch.int32), values, atol=0, rtol=0)


def test_gluon_simulator_runs_core_memory_split_ops_on_cpu():
    values = torch.arange(8, dtype=torch.float32) + 1.0
    masked_out = torch.full((8,), -99.0, dtype=torch.float32)
    split_lhs_out = torch.empty((4,), dtype=torch.float32)
    split_rhs_out = torch.empty((4,), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_core_memory_split_ops_kernel)

    kernel[(1,)](values, masked_out, split_lhs_out, split_rhs_out, num_warps=1)

    expected_masked = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, -7.0, -99.0, -7.0],
        dtype=torch.float32,
    )
    expected_loaded = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, -7.0, -7.0, -7.0],
        dtype=torch.float32,
    )
    torch.testing.assert_close(masked_out, expected_masked, atol=0, rtol=0)
    torch.testing.assert_close(split_lhs_out, expected_loaded.reshape(4, 2)[:, 0])
    torch.testing.assert_close(split_rhs_out, expected_loaded.reshape(4, 2)[:, 1])


def test_gluon_simulator_runs_masked_atomic_memory_ops_on_cpu():
    counter = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    addends = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    add_old = torch.full_like(counter, -1)
    xchg_old = torch.full_like(counter, -1)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_atomic_memory_ops_kernel)

    kernel[(1,)](counter, addends, add_old, xchg_old, counter.numel(), num_warps=1)

    torch.testing.assert_close(
        add_old,
        torch.tensor([5, -1, 7, 8], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        xchg_old,
        torch.tensor([6, -1, 10, 12], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        counter,
        torch.tensor([10, 6, 30, 40], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_gluon_simulator_runs_moe_style_fp8_pack_store_on_cpu():
    values = torch.tensor(
        [
            [0.0, 0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0],
            [0.25, -0.25, 1.5, -1.5, 3.0, -3.0, 6.0, -6.0],
        ],
        dtype=torch.float32,
    )
    out = torch.empty(values.shape, dtype=torch.uint8)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _moe_epilogue_pack_store_kernel
    )

    kernel[(1,)](values, out, *values.shape, num_warps=4)

    expected = values.to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


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


def test_gluon_simulator_runs_tensor_shape_helpers_on_cpu():
    values = torch.arange(4, dtype=torch.float32) + 1
    vector_out = torch.empty((4, 8), dtype=torch.float32)
    split_lhs_out = torch.empty((4, 4), dtype=torch.float32)
    split_rhs_out = torch.empty((4, 4), dtype=torch.float32)
    joined_out = torch.empty((4, 8), dtype=torch.float32)
    expanded_out = torch.empty((4, 8), dtype=torch.float32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_tensor_shape_helpers_kernel)

    kernel[(1,)](
        values,
        vector_out,
        split_lhs_out,
        split_rhs_out,
        joined_out,
        expanded_out,
        values.numel(),
        vector_out.shape[1],
        num_warps=1,
    )

    expected = values[:, None].expand_as(vector_out)
    torch.testing.assert_close(vector_out, expected, atol=0, rtol=0)
    torch.testing.assert_close(split_lhs_out, expected[:, 0::2], atol=0, rtol=0)
    torch.testing.assert_close(split_rhs_out, expected[:, 1::2], atol=0, rtol=0)
    torch.testing.assert_close(joined_out, expected, atol=0, rtol=0)
    torch.testing.assert_close(expanded_out, expected, atol=0, rtol=0)


def test_gluon_simulator_exposes_num_ctas_launch_metadata_on_cpu():
    out = torch.empty((1,), dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_num_ctas_kernel)

    kernel[(1,)](out, num_warps=4, num_ctas=2)

    assert out.item() == 2


def test_gluon_simulator_allocates_mbarrier_batches_with_num_ctas_on_cpu():
    out = torch.full((10,), -1, dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_mbarrier_batch_shape_kernel)

    kernel[(1,)](out, num_warps=4, num_ctas=4)

    torch.testing.assert_close(
        out,
        torch.tensor([3, 4, 5, 2, 2, 1, 2, 2, 0, 1], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_gluon_simulator_runs_multicta_softmax_on_cpu():
    torch.manual_seed(13)
    values = torch.randn((3, 64), dtype=torch.float32)
    out = torch.empty_like(values)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_multicta_softmax_kernel)

    kernel[(values.shape[0],)](
        values,
        out,
        values.stride(0),
        out.stride(0),
        values.shape[1],
        num_warps=4,
        num_ctas=2,
    )

    torch.testing.assert_close(out, torch.softmax(values, dim=1), atol=1e-6, rtol=1e-6)


def test_gluon_simulator_runs_clc_api_miss_on_cpu():
    out = torch.full((1,), -1, dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(_clc_api_kernel)

    kernel[(1,)](out, num_warps=4)

    assert out.item() == 0


def test_gluon_simulator_runs_warp_specialize_fallback_on_cpu():
    out = torch.full((6,), -1, dtype=torch.int32)
    counter = torch.empty((1,), dtype=torch.int32)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _warp_specialize_helpers_kernel
    )

    kernel[(1,)](out, counter, num_warps=4)

    torch.testing.assert_close(
        out,
        torch.tensor([1, 2, 4, 1, 2, 4], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    assert counter.item() == 6


def test_gluon_simulator_runs_warp_specialized_elementwise_pipeline_on_cpu():
    xblock = 4
    yblock = 4
    a = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    b = torch.flip(a, dims=[1]) * 0.5
    c = torch.full_like(a, -1.0)
    layout = gl.NVMMASharedLayout.get_default_for([xblock, yblock], gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, [xblock, yblock], layout)
    b_desc = TensorDescriptor.from_tensor(b, [xblock, yblock], layout)
    c_desc = TensorDescriptor.from_tensor(c, [xblock, yblock], layout)
    kernel = triton_viz.trace("tracer", frontend="gluon")(
        _warp_specialized_elementwise_add_kernel
    )

    kernel[(2,)](a_desc, b_desc, c_desc, *a.shape, xblock, yblock, num_warps=4)

    torch.testing.assert_close(c, a + b, atol=0, rtol=0)


def test_gluon_simulator_keeps_masked_sanitizer_run_concrete_on_cpu():
    inp = torch.arange(4, dtype=torch.float32)
    out = torch.full((8,), -1.0)
    layout = gl.BlockedLayout([1], [32], [1], [0])
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    kernel = triton_viz.trace(sanitizer, frontend="gluon")(_layout_memcpy_1d_kernel)

    kernel[(1,)](inp, out, inp.numel(), 8, layout, num_warps=1)

    torch.testing.assert_close(out[:4], inp, atol=0, rtol=0)
    torch.testing.assert_close(out[4:], torch.full((4,), -1.0), atol=0, rtol=0)
    assert sanitizer.records == []
