import math
import neuronxcc.nki.language as nl
import neuronxcc.nki as nki
import numpy as np
import torch
from triton_viz.clients import Tracer
import triton_viz

TRITON_VIZ = True

def add_kernel(a, b): # fails @ ix < B
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

def copy_kernel(a): # fails @ ix < B
    B, D = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)

    ix = pid_x * 128 + nl.arange(128)[:, None]
    iy = pid_y * 512 + nl.arange(512)[None, :]
    #mask = (ix < B) & (iy < D)

    #a_tmp = nl.load(a[ix, iy], mask=mask)
    #nl.store(out[ix, iy], value=a_tmp, mask=mask)
    a_tmp = nl.load(a[ix, iy])
    nl.store(out[ix, iy], value=a_tmp)
    return out

def print_kernel(): # works
    a = nl.ndarray([4, 4], dtype=nl.float32, buffer=nl.shared_hbm)
    y = nl.ndarray([4, 4], dtype=np.float32,)
    nl.store(a, value=y) 
    print(a)
    return a

def tmp0_kernel(a): # works
    B, D = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    import tp
    tp.log(f'{pid_x=}, {pid_y=}')

    ix = pid_x * 128 + nl.arange(128)[:, None]
    iy = pid_y * 512 + nl.arange(512)[None, :]
    #mask = (ix < B) & (iy < D)

    #a_tmp = nl.load(a[ix, iy], mask=mask)
    #nl.store(out[ix, iy], value=a_tmp, mask=mask)
    #a_tmp = nl.load(a[:128, :128])
    a_tmp = nl.load(a[ix, iy])
    nl.store(out[ix, iy], value=a_tmp)
    return out

def xyz_kernel(a): # works
    B, T, C, H, W = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    pid_z = nl.program_id(2)


    for i in range(H // 128):
        for j in range(W // 512):
            #i_h = pid_x * 128 + nl.arange(128)[None, None, None, :, None]
            #i_n = pid_y * 512 + nl.arange(512)[None, None, None, None, :]
            i_h = pid_x * 128 + nl.arange(128)[:, None]
            i_n = pid_y * 512 + nl.arange(512)[None, :]
            a_tmp = nl.load(a[pid_x, pid_y, pid_z, i_h, i_n])
            nl.store(out[pid_x, pid_y, pid_z, i_h, i_n], value=a_tmp)
    return out

def xyz_kernel(a): # works
    B, T, C, H, W = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)

    pid_x = nl.program_id(0)
    pid_y = nl.program_id(1)
    pid_z = nl.program_id(2)


    for i in range(H // 128):
        for j in range(W // 512):
            #i_h = pid_x * 128 + nl.arange(128)[None, None, None, :, None]
            #i_n = pid_y * 512 + nl.arange(512)[None, None, None, None, :]
            i_h = pid_x * 128 + nl.arange(128)[:, None]
            i_n = pid_y * 512 + nl.arange(512)[None, :]
            a_tmp = nl.load(a[pid_x, pid_y, pid_z, i_h, i_n])
            nl.store(out[pid_x, pid_y, pid_z, i_h, i_n], value=a_tmp)
    return out

B, D = 1024, 1024
x = torch.rand((B, D))
y = torch.rand((B, D))

blocks_x = math.ceil(B / 128)
blocks_y = math.ceil(D / 512)

kernel = tmp0_kernel
if kernel == add_kernel:
    kernel_grid = (blocks_x, blocks_y)
    kernel_args = (x.numpy(), y.numpy())
    z1 = x + y
if kernel == copy_kernel:
    kernel_grid = (blocks_x, blocks_y)
    kernel_args = (x.numpy(),)
    z1 = x
elif kernel == print_kernel:
    kernel_grid = (1,1,1)
    kernel_args = ()
    z1 = x
if kernel == tmp0_kernel:
    kernel_grid = (blocks_x, blocks_y)
    kernel_args = (x.numpy(),)
    z1 = x
if kernel == xyz_kernel:
    B, T, C, H, W = 2, 3, 4, 1024, 1024
    x = torch.rand((B, T, C, H, W))
    kernel_grid = (B, T, C)
    kernel_args = (x.numpy(),)
    z1 = x

if TRITON_VIZ:
    kernel = triton_viz.trace(clients=Tracer(), backend='nki')(kernel)
    z2 = kernel[kernel_grid](*kernel_args)
    z2 = torch.from_numpy(z2)
    print((z1 - z2).abs().max())
else:
    kernel = nki.jit(kernel)
    z2 = nki.simulate_kernel(kernel[kernel_grid], *kernel_args)
    z2 = torch.from_numpy(z2)
    print((z1 - z2).abs().max())
