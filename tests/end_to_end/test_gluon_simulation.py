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
def _layout_anchor_and_barrier_kernel(in_ptr, out_ptr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offsets = gl.arange(0, N, layout)
    values = gl.load(in_ptr + offsets)
    values = gl.set_auto_layout(values, layout)
    gl.barrier()
    gl.store(out_ptr + offsets, values + 1.0)


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
