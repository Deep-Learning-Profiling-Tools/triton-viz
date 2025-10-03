import math
import tp
import neuronxcc.nki.language as nl
import neuronxcc.nki as nki
import numpy as np
import torch
from triton_viz.clients import Tracer
import triton_viz
from triton_viz.core.trace import launches


def add_kernel(a, b):  # fails @ ix < B
    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    B, D = a.shape

    # memory shared across all SPMD instances
    c_output = nl.ndarray(a.shape, buffer=nl.shared_hbm, dtype=a.dtype)

    ix = pid_x * 128 + nl.arange(128)[:, None]
    iy = pid_y * 512 + nl.arange(512)[None, :]
    mask = (ix < B) & (iy < D)
    a_tmp = nl.load(a[ix, iy], mask=mask)
    b_tmp = nl.load(b[ix, iy], mask=mask)
    c_tmp = a_tmp + b_tmp
    nl.store(c_output[ix, iy], value=c_tmp, mask=mask)
    return c_output

def copy_kernel(a):  # fails @ ix < B
    B, D = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)

    ix = pid_x * 128 + nl.arange(128)[:, None]
    iy = pid_y * 512 + nl.arange(512)[None, :]
    mask = (ix < B) & (iy < D)

    a_tmp = nl.load(a[ix, iy], mask=mask)
    nl.store(out[ix, iy], value=a_tmp, mask=mask)
    return out

def print_kernel():  # works
    a = nl.ndarray([4, 4], dtype=nl.float32, buffer=nl.shared_hbm)
    y = nl.ndarray(
        [4, 4],
        dtype=np.float32,
    )
    nl.store(a, value=y)
    print(a)
    return a

def tmp0_kernel(a):  # works
    B, D = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    nl.device_print(f"pid_x:", pid_x);
    nl.device_print(f"pid_y:", pid_y);

    ix = pid_x * 2 + nl.arange(2)[:, None]
    iy = pid_y * 4 + nl.arange(4)[None, :]
    mask = (ix < B) & (iy < D)
    a_tmp = nl.load(a[ix, iy], mask=mask)
    nl.device_print(f"a_tmp:", a_tmp)

    ix2 = pid_x * 3 + nl.arange(3)[:, None]
    iy2 = pid_y * 4 + nl.arange(4)[None, :]
    mask2 = (ix2 < B) & (iy2 < D)
    a_tmp2 = nl.load(a[ix2, iy2], mask=mask2)
    nl.device_print(f"a_tmp2:", a_tmp2)

    iy3 = nl.arange(3)[None, :] < -1
    a_tmp3 = nl.load(a[80, nl.arange(3)[None, :]], mask=iy3)
    nl.device_print(f"a_tmp3:", a_tmp3)

    nl.store(out[ix, iy], value=a_tmp, mask=mask)
    # for load(src, mask), src.shape, mask.shape need to be same, return shape
    # for store(dst, value, mask), dst.shape, value.shape, mask.shape need to be same, return shape
    return out

def xyz_kernel(a):  # works
    B, T, C, H, W = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    pid_z = nl.program_id(2)

    for i in range(H // 128):
        for j in range(W // 512):
            i_h = pid_x * 128 + nl.arange(128)[:, None]
            i_n = pid_y * 512 + nl.arange(512)[None, :]
            mask = (i_h < H) & (i_n < W)
            a_tmp = nl.load(a[pid_x, pid_y, pid_z, i_h, i_n], mask=mask)
            nl.store(out[pid_x, pid_y, pid_z, i_h, i_n], value=a_tmp, mask=mask)
    return out

def tmp1_kernel(a):
    B, D = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)

    ix = pid_x * 2 + nl.arange(2)[:, None]
    iy = pid_y * 4 + nl.arange(4)[None, :]
    mask = (ix < B) & (iy < D)
    a_tmp = nl.load(a[ix, iy], mask=mask)

    #iy3 = nl.arange(3)[None, :] < -1
    #iy3 = nl.mgrid[80:83, 80:85] < -1
    iy3 = (nl.arange(3)[:, None] < -1) & (nl.arange(5)[None, :] < -1)
    a_tmp3 = nl.load(a[80:83, 80:85], mask=iy3)
    nl.device_print(f"a_tmp3:", a_tmp3)

    nl.store(out[ix, iy], value=a_tmp, mask=mask)
    return out


B, D = 1024, 1024
x = torch.rand((B, D))
y = torch.rand((B, D))

blocks_x = math.ceil(B / 128)
blocks_y = math.ceil(D / 512)

kernel = tmp1_kernel
if kernel == add_kernel:
    kernel_grid = (blocks_x, blocks_y)
    kernel_args = (x.numpy(), y.numpy())
    z1 = x + y
if kernel == copy_kernel:
    kernel_grid = (blocks_x, blocks_y)
    kernel_args = (x.numpy(),)
    z1 = x
elif kernel == print_kernel:
    kernel_grid = (1, 1, 1)
    kernel_args = ()
    z1 = x
if kernel == tmp0_kernel:
    #B, D = 129, 512
    B, D = 3, 5
    #x = torch.rand((B, D))
    #y = torch.rand((B, D))
    x = torch.arange(B*D).int().reshape(B, D)
    y = -torch.arange(B*D).int().reshape(B, D)

    #blocks_x = math.ceil(B / 128)
    blocks_x = math.ceil(B / 2)
    blocks_y = math.ceil(D / 4)

    kernel_grid = (blocks_x, blocks_y)
    kernel_args = (x.numpy(),)
    z1 = x
if kernel == xyz_kernel:
    B, T, C, H, W = 2, 3, 4, 1024, 1024
    x = torch.rand((B, T, C, H, W))
    kernel_grid = (B, T, C)
    kernel_args = (x.numpy(),)
    z1 = x
if kernel == tmp1_kernel:
    B, D = 3, 5
    x = torch.arange(B*D).int().reshape(B, D)
    y = -torch.arange(B*D).int().reshape(B, D)

    blocks_x = math.ceil(B / 2)
    blocks_y = math.ceil(D / 4)

    kernel_grid = (blocks_x, blocks_y)
    kernel_args = (x.numpy(),)
    z1 = x

TRITON_VIZ = False

if TRITON_VIZ:
    kernel = triton_viz.trace(clients=Tracer(), backend="nki")(kernel)
    kk = kernel[kernel_grid]
    z2 = kk(*kernel_args)
    z2 = torch.from_numpy(z2)
    print((z1 - z2).abs().max())

    print(f"Number of launches: {len(launches)}")
    if launches:
        launch = launches[-1]
        print(f"Number of records: {len(launch.records)}")
        for i, record in enumerate(launch.records):
            print(f"Record {i}: {type(record).__name__}")
            if hasattr(record, "ptr"):
                print(f"  ptr: {record.ptr}")
            if hasattr(record, "offsets"):
                print(f"  offsets shape: {record.offsets.shape}")
            if hasattr(record, "masks"):
                print(f"  masks shape: {record.masks.shape}")

    # Try to launch visualization
    try:
        triton_viz.launch(share=False)
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback

        traceback.print_exc()
else:
    kernel = nki.jit(kernel)
    z2 = nki.simulate_kernel(kernel[kernel_grid], *kernel_args)
    z2 = torch.from_numpy(z2)
    print((z1 - z2).abs().max())
