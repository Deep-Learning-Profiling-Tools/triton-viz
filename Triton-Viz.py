# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: tritonviz
#     language: python
#     name: tritonviz
# ---

# # Triton Puzzles
#
#

# +
import torch
import triton
from torch import Tensor
import triton.language as tl
import triton_viz
import inspect
from jaxtyping import Float


def test(puzzle, puzzle_spec, nelem={}, B={"B0": 32}):
    B = dict(B)
    if "N1" in nelem:
        B["B1"] = 32
    if "N2" in nelem:
        B["B2"] = 32

    triton_viz.interpreter.record_builder.reset()
    torch.manual_seed(0)
    signature = inspect.signature(puzzle_spec)
    args = {}
    for n, p in signature.parameters.items():
        args[n + "_ptr"] = [d.size for d in p.annotation.dims]
    args["z_ptr"] = [d.size for d in signature.return_annotation.dims]

    tt_args = []
    for k, v in args.items():
        tt_args.append(torch.rand(*v))
    grid = lambda meta: (
        triton.cdiv(nelem["N0"], meta["B0"]),
        triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
        triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)),
    )

    for k, v in args.items():
        print(k, v)
    triton_viz.trace(puzzle)[grid](*tt_args, **B, **nelem)
    z = tt_args[-1]
    tt_args = tt_args[:-1]
    z_ = puzzle_spec(*tt_args)
    match = torch.allclose(z, z_)
    print("Results match:", match)
    if not match:
        print(z)
        print(z_)
    triton_viz.launch()


# -

# ## Puzzle 1: Constant Add
#
# Add a constant to a vector. Uses one program block. Block size `B0` is always the same as vector length `N0`.
#


# +
def add_spec(x: Float[Tensor, "32"]) -> Float[Tensor, "32"]:
    return x + 10.0


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    range = tl.arange(0, B0)
    x = tl.load(x_ptr + range)
    z = x + 10
    z = tl.store(z_ptr + range, z)


test(add_kernel, add_spec, nelem={"N0": 32})


# -

# ## Puzzle 2: Constant Add Block
#
# Add a constant to a vector. Uses one program block. Block size `B0` is always the same as vector length `N0`.
#
#


# +
def add2_spec(x: Float[Tensor, "200"]) -> Float[Tensor, "200"]:
    return x + 10.0


@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(0)
    range = pid * B0 + tl.arange(0, B0)
    x = tl.load(x_ptr + range, range < N0, 0)
    z = x + 10
    z = tl.store(z_ptr + range, z, range < N0)


test(add_mask2_kernel, add2_spec, nelem={"N0": 200})


# -

# ## Puzzle 3: Outer Vector Add
#
# Add two vectors. Uses one program block. Block size `B0` is always the same as vector `x` length `N0`.
# Block size `B1` is always the same as vector `y` length `N1`.
#


# +
def add_vec_spec(
    x: Float[Tensor, "32"], y: Float[Tensor, "32"]
) -> Float[Tensor, "32 32"]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    i_range = tl.arange(0, B0)[None, :]
    j_range = tl.arange(0, B1)[:, None]

    x = tl.load(x_ptr + i_range)
    y = tl.load(y_ptr + j_range)

    z = x + y
    z = tl.store(z_ptr + i_range + B0 * j_range, z)


test(add_vec_kernel, add_vec_spec, nelem={"N0": 32, "N1": 32})


# -

# ## Puzzle 4: Outer Vector Add Block
#
# Add two vectors. Uses one program block. Block size `B0` is always greater than the vector `x` length `N0`.
# Block size `B1` is always greater than vector `y` length `N1`.
#


# +
def add_vec_block_spec(
    x: Float[Tensor, "100"], y: Float[Tensor, "90"]
) -> Float[Tensor, "90 100"]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    i_range = tl.arange(0, B0)[:, None] + pid_0 * B0
    j_range = tl.arange(0, B1)[None, :] + pid_1 * B1

    x = tl.load(x_ptr + i_range, i_range < N0, 0)
    y = tl.load(y_ptr + j_range, j_range < N1, 0)

    z = x + y
    z = tl.store(z_ptr + i_range + N0 * j_range, z, (i_range < N0) & (j_range < N1))


test(add_vec_block_kernel, add_vec_block_spec, nelem={"N0": 100, "N1": 90})


# -

# ## Puzzle 5: Fused Op


# +
def mul_relu_block_spec(
    x: Float[Tensor, "100"], y: Float[Tensor, "90"]
) -> Float[Tensor, "90 100"]:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    i_range = tl.arange(0, B0)[:, None] + pid_0 * B0
    j_range = tl.arange(0, B1)[None, :] + pid_1 * B1

    x = tl.load(x_ptr + i_range, i_range < N0, 0)
    y = tl.load(y_ptr + j_range, j_range < N1, 0)

    z = x * y
    z = tl.where(z > 0, z, 0)

    z = tl.store(z_ptr + i_range + N0 * j_range, z, (i_range < N0) & (j_range < N1))


test(mul_relu_block_kernel, mul_relu_block_spec, nelem={"N0": 100, "N1": 90})


# -

# ## Puzzle 6: Fused Op Backwards


# +
def mul_relu_block_back_spec(
    x: Float[Tensor, "100"],
    y: Float[Tensor, "90"],
    dz: Float[Tensor, "90 100"],
    dx: Float[Tensor, "100"],
    dy: Float[Tensor, "90"],
) -> Float[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = mul_relu_block_spec(x, y)
    z.backward(dz)
    dx[:] = x.grad
    dy[:] = y.grad
    return z


@triton.jit
def mul_relu_block_back_kernel(
    x_ptr,
    y_ptr,
    dz_ptr,
    dx_ptr,
    dy_ptr,
    z_ptr,
    N0,
    N1,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    i_range = tl.arange(0, B0)[:, None] + pid_0 * B0
    j_range = tl.arange(0, B1)[None, :] + pid_1 * B1

    x = tl.load(x_ptr + i_range, i_range < N0, 0)
    y = tl.load(y_ptr + j_range, j_range < N1, 0)

    # Forward
    z = x * y
    z = tl.where(z > 0, z, 0)
    tl.store(z_ptr + i_range + N0 * j_range, z, (i_range < N0) & (j_range < N1))

    dz = tl.load(dz_ptr + i_range + N0 * j_range, (i_range < N0) & (j_range < N1), 0)
    dr = tl.where(z > 0, dz, 0)
    dx = tl.sum(dr * y, 0, keep_dims=1)
    dy = tl.sum(dr * x, 1, keep_dims=1)

    tl.store(dx_ptr + i_range, (i_range < N0))
    tl.store(dy_ptr + j_range, (j_range < N1))


test(mul_relu_block_back_kernel, mul_relu_block_back_spec, nelem={"N0": 100, "N1": 90})

# -

# ## Puzzle 7: Fused Softmax


# +
def softmax_spec(x: Float[Tensor, "4 80"]) -> Float[Tensor, "4 80"]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, TN1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    x_max = -1e9
    for i in range(0, TN1, B1):
        i_range = tl.arange(0, B1)[None, :] + i
        offset = TN1 * pid_0 + i_range
        x = tl.load(x_ptr + offset, i_range < TN1, -1e9)
        chunk_max = tl.max(x, 1)[:, None]
        x_max = tl.where(chunk_max > x_max, chunk_max, x_max)

    partition = 0
    for i in range(0, TN1, B1):
        i_range = tl.arange(0, B1)[None, :] + i
        offset = TN1 * pid_0 + i_range
        x = tl.load(x_ptr + offset, i_range < TN1, -1e9) - x_max
        partition = partition + tl.sum(tl.exp(x), 1)[:, None]

    for i in range(0, TN1, B1):
        i_range = tl.arange(0, B1)[None, :] + i
        offset = TN1 * pid_0 + i_range
        x = tl.load(x_ptr + offset, i_range < TN1, -1e9) - x_max
        x_exp = tl.exp(x)
        z = x_exp / partition
        tl.store(z_ptr + offset, z, i_range < TN1)


test(
    softmax_kernel,
    softmax_spec,
    B={"B0": 1, "B1": 32},
    nelem={"N0": 4, "N1": 32, "TN1": 80},
)

# -

# ## Puzzle 8: Manual Conv.


# +
def conv2d_spec(
    x: Float[Tensor, "4 8 8"], k: Float[Tensor, "4 4"]
) -> Float[Tensor, "4 8 8"]:
    return x


@triton.jit
def conv2d_kernel(
    x_ptr,
    k_ptr,
    z_ptr,
    N0,
    N1,
    H,
    W,
    KH: tl.constexpr,
    KW: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    kh_range = tl.arange(0, KH)[None, :, None]
    kw_range = tl.arange(0, KW)[None, None, :]
    k = tl.load(k_ptr + kh_range * KW + kw_range)
    for i in range(0, H):
        for j in range(0, W):
            x = tl.load(
                x_ptr + pid_0 * H * W + (kh_range + i) * W + (kw_range + j),
                ((kh_range + i) < H) & ((kw_range + j) < W),
                0,
            )
            out = tl.sum(tl.sum(x * k, 2), 1)
            tl.store(z_ptr + pid_0 * H * W + i * W + j + tl.arange(0, 1), out)


test(
    conv2d_kernel,
    conv2d_spec,
    B={"B0": 1, "B1": 32},
    nelem={"N0": 4, "N1": 32, "H": 8, "W": 8, "KH": 4, "KW": 4},
)


# -

# ## Puzzle 9: Matrix Mult


# +
def dot_spec(
    x: Float[Tensor, "4 32 32"], y: Float[Tensor, "4 32 32"]
) -> Float[Tensor, "4 32 32"]:
    return x @ y


@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    r = tl.program_id(0) * B0
    rows = tl.arange(0, B0)[:, None]
    c = tl.program_id(1) * B1
    cols = tl.arange(0, B1)[None, :]
    mids = tl.arange(0, B0)
    b = tl.program_id(2)
    bid = b * N0 * N1
    z = 0
    for i in range(0, MID, B0):
        x_val = tl.load(
            x_ptr + bid + (r + rows) * MID + (mids + i)(i + mids < MID)
            & (rows + r < N0),
            0,
        )
        y_val = tl.load(
            y_ptr + bid + (i + mids[:, None]) * N1 + (cols + c),
            (i + mids[:, None] < MID) & (cols + c < N1),
            0,
        )
        z = z + tl.dot(x_val, y_val)
    tl.store(
        z_ptr + (b * N0 * N1) + (r + rows) * N1 + (cols + c),
        z,
        mask=(r + rows) < N0 & (c + cols) < N1,
    )


test(
    dot_kernel,
    dot_spec,
    B={"B0": 16, "B1": 16, "B2": 1},
    nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 32},
)

# -


# ## Puzzle 10: Quantized Matrix Mult
#
# GPT-Q like puzzles

# ## Puzzle 11: Flash Attention
#
# Long reduction.
