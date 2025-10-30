from torch_xla.core import xla_model as xm
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
import math
import torch

@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor):
  # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
  # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
  # and N = a_tensor.shape[1]
  # Reduction (mean) is performed in the free (2nd) dimension
  B, D = a_tensor.shape
  out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Make sure shapes match
  assert D == g_tensor.shape[0]

  # Generate tensor indices to index input tensor
  ix = nl.arange(2)[:, None]
  iw = nl.arange(1)[:, None]
  #iy = nl.arange(a_tensor.shape[1])[None, :]
  iy = nl.arange(8)[None, :]

  # Load RMSNorm weight once, reused by rows/tiles of a_tensor
  g_tile = nl.load(g_tensor.reshape((1, D))[iw, iy], mask=iy < D)

  # Process 2 rows at a time due to 2-partition tile size limitation
  # Since we're not reducing across the first dimension
  # Tiles can be processed independently
  for i in nl.affine_range(math.ceil(B/2)):

    # Load input data from external memory to on-chip memory
    mask = (i * 2 + ix < B) & (iy < D)
    a_tile = nl.load(a_tensor[i * 2 + ix, iy],
                    mask=mask)

    # Compute element-wise square of a_tensor
    in_square = nl.square(a_tile)

    # Calculate sum of squared elements, along last dimension
    square_sum = nl.sum(in_square, axis=[1], mask=mask)

    # Scale and get a reciprocal
    mean = square_sum / D

    # Take square root of mean and then reciprocal with
    # rsqrt API (one ISA instruction)
    rms_reciprocal = nl.rsqrt(mean)

    # Scale the input tensor
    out_tile = nl.multiply(a_tile, rms_reciprocal)

    # Broadcast weight along first axis to match tensor shape
    # B_active = min(B - i * 2, 2)
    g_bcast = g_tile.broadcast_to((2, 8))

    # Multiply with the RMSNorm weight
    out_tile[...] = nl.multiply(out_tile, g_bcast,
                           mask=(i * 2 + ix < B))

    # store the addition results back to external memory (out_tensor)
    nl.store(out_tensor[i * 2 + ix, iy], value=out_tile,
            mask=mask)

  return out_tensor

# ref
def torch_rmsnorm_kernel(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = a_tensor.pow(2)
  # Calculate means in the free dimension
  mean = in_square.mean(dim=1, keepdim=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * torch.rsqrt(mean)

  # Scale the output by the weight
  return tensor * g_tensor

device = 'cpu'

a_tensor = torch.arange(15).float().view(3, 5).to(device=device)
g_tensor = torch.arange(5).float().to(device=device)

output_nki = nki.simulate_kernel(nki_rmsnorm_kernel, a_tensor.numpy(), g_tensor.numpy())

output_torch = torch_rmsnorm_kernel(a_tensor, g_tensor).numpy()

if np.allclose(output_torch, output_nki, atol=1e-5, rtol=1e-3):
  print("NKI and Torch match")
else:
  print("NKI and Torch differ")
