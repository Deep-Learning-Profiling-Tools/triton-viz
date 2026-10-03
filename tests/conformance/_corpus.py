"""Kernel + launch corpus of the TTIR reader conformance suite (D10b).

Each :class:`Case` is one kernel launch: ``build(device)`` returns
``(grid, args, kwargs)`` and ``kernel`` is the ``@triton.jit`` function to
launch (an autotuned kernel contributes one case per config, the config's
kwargs in ``kwargs``). The same module is imported by the TTIR capture
child (real compile), by the interpreter child (``TRITON_INTERPRET=1``, so
the decorators build InterpretedFunctions there) and by the test module
(case names only), so it must import neither tilelens nor anything that
depends on how ``@triton.jit`` resolved.

Every tensor argument owns its storage, zero-padded by ``PAD`` elements on
each side: the interpreter child attributes an address to the argument
whose storage holds it, and a moderately out-of-bounds lane (an unmasked
ragged tile, ...) still lands inside its own argument's padding.

Rules for kernels here: every memory-op call on ONE source line, at most
one per line (sites are keyed by line; for a call wrapped over several
lines Triton 3.6 locates the op at the line of the argument its code
generator visited last while the interpreter's frame reports the call's,
so the interpreter child refuses such a call; 3.8 locates it at the
call's first line), and launches small enough for the interpreter. The audit
regression kernels are the goldens' own
(``tests/golden/ir/reader_kernels.py``), imported by path.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNELS = os.path.join(HERE, "..", "golden", "ir", "reader_kernels.py")
PAD = 1 << 12  # elements of zero padding on each side of every tensor


def _load_reader_kernels():
    name = "tilelens_conformance_reader_kernels"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, os.path.abspath(READER_KERNELS)
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # Triton reads the jit fn's module
        spec.loader.exec_module(module)
    return module


RK = _load_reader_kernels()


def buf(
    numel: int, dtype=torch.float32, dev: str = "cpu", init: Any = "arange"
) -> torch.Tensor:
    """A 1-D tensor of ``numel`` elements inside its own zero-padded storage."""
    base = torch.zeros(numel + 2 * PAD, dtype=dtype, device=dev)
    t = base[PAD : PAD + numel]
    if init == "arange" and numel:
        t.copy_(torch.arange(numel, device=dev).to(dtype))
    elif init is not None and init != "arange":
        t.copy_(torch.as_tensor(init, dtype=dtype, device=dev))
    return t


def T(
    *shape: int, dtype=torch.float32, dev: str = "cpu", init: Any = "arange"
) -> torch.Tensor:
    return buf(math.prod(shape), dtype, dev, init).view(*shape)


@dataclass(frozen=True)
class Case:
    name: str
    kernel: Any  # the JITFunction (InterpretedFunction under TRITON_INTERPRET=1)
    build: Callable[[str], tuple]  # device -> (grid, args, kwargs)
    group: str
    note: str = ""


CASES: list[Case] = []


def case(name: str, kernel: Any, group: str, note: str = ""):
    def deco(build):
        if any(c.name == name for c in CASES):
            raise ValueError(f"duplicate case {name!r}")
        CASES.append(Case(name, kernel, build, group, note))
        return build

    return deco


def by_name() -> dict[str, Case]:
    return {c.name: c for c in CASES}


# ════════════════════════════ a: 1-D ════════════════════════════


@triton.jit
def k_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@case("a_add_masked", k_add, "a")
def _(dev):
    n = 300
    return (3,), (T(n, dev=dev), T(n, dev=dev), T(n, dev=dev), n), {"BLOCK": 128}


@triton.jit
def k_copy(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, v)


@case("a_copy_exact", k_copy, "a")
def _(dev):
    return (4,), (T(256, dev=dev), T(256, dev=dev)), {"BLOCK": 64}


@case(
    "a_copy_ragged",
    k_copy,
    "a",
    "unmasked ragged tail: out-of-bounds lanes land in the padding",
)
def _(dev):
    return (3,), (T(150, dev=dev), T(150, dev=dev)), {"BLOCK": 64}


@triton.jit
def k_mask_le(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs <= n
    v = tl.load(x_ptr + offs, mask=m)
    tl.store(out_ptr + offs, v, mask=m)


@case("a_mask_le", k_mask_le, "a")
def _(dev):
    return (4,), (T(100, dev=dev), T(100, dev=dev), 99), {"BLOCK": 32}


@triton.jit
def k_mask_or_pid0(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = (offs < n) | (pid == 0)
    v = tl.load(x_ptr + offs, mask=m)
    tl.store(out_ptr + offs, v, mask=m)


@case("a_mask_or_pid0", k_mask_or_pid0, "a")
def _(dev):
    return (2,), (T(10, dev=dev), T(10, dev=dev), 10), {"BLOCK": 16}


@triton.jit
def k_mask_window(x_ptr, lo, hi, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = (offs >= lo) & (offs < hi)
    tl.store(x_ptr + offs, 1.0, mask=m)


@case("a_mask_window", k_mask_window, "a")
def _(dev):
    return (3,), (T(96, dev=dev), 5, 77), {"BLOCK": 32}


@triton.jit
def k_load_other(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n, other=-1.0)
    tl.store(out_ptr + offs, v)


@case("a_load_other", k_load_other, "a")
def _(dev):
    return (2,), (T(40, dev=dev), T(64, dev=dev), 40), {"BLOCK": 32}


@triton.jit
def k_arange_start(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * 64 + tl.arange(8, 8 + BLOCK)
    tl.store(x_ptr + offs, 2.0)


@case("a_arange_start", k_arange_start, "a", "make_range with a non-zero start")
def _(dev):
    return (3,), (T(192, dev=dev),), {"BLOCK": 32}


@triton.jit
def k_two_ranges_one_dim(x_ptr):
    offs = tl.arange(0, 16) + tl.arange(16, 32)
    tl.store(x_ptr + offs + tl.program_id(0) * 64, 1.0)


@case(
    "a_two_ranges_one_dim",
    k_two_ranges_one_dim,
    "a",
    "two make_ranges share one lane: 2i + 16, not i + j + 16",
)
def _(dev):
    return (2,), (T(128, dev=dev),), {}


@triton.jit
def k_negative_shift(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs - 3, mask=offs >= 3)
    tl.store(out_ptr + offs, v)


@case("a_negative_shift", k_negative_shift, "a")
def _(dev):
    return (2,), (T(64, dev=dev), T(64, dev=dev)), {"BLOCK": 32}


@triton.jit
def k_scalar_per_program(x_ptr, out_ptr):
    pid = tl.program_id(0)
    v = tl.load(x_ptr + pid * 2)
    tl.store(out_ptr + pid, v)


@case("a_scalar_per_program", k_scalar_per_program, "a")
def _(dev):
    return (5,), (T(10, dev=dev), T(5, dev=dev)), {}


@case("a_copy_int8", k_copy, "a", "1-byte elements")
def _(dev):
    return (
        (2,),
        (T(64, dtype=torch.int8, dev=dev), T(64, dtype=torch.int8, dev=dev)),
        {"BLOCK": 32},
    )


@case("a_copy_fp16", k_copy, "a", "2-byte elements")
def _(dev):
    return (
        (2,),
        (T(64, dtype=torch.float16, dev=dev), T(64, dtype=torch.float16, dev=dev)),
        {"BLOCK": 32},
    )


@case("a_copy_bool", k_copy, "a", "i1 pointees, 1 byte each")
def _(dev):
    return (
        (2,),
        (
            T(64, dtype=torch.bool, dev=dev, init=None),
            T(64, dtype=torch.bool, dev=dev, init=None),
        ),
        {"BLOCK": 32},
    )


@case("a_copy_bf16", k_copy, "a", "2-byte bf16 elements")
def _(dev):
    return (
        (2,),
        (T(64, dtype=torch.bfloat16, dev=dev), T(64, dtype=torch.bfloat16, dev=dev)),
        {"BLOCK": 32},
    )


@case("a_copy_f64", k_copy, "a", "8-byte float elements")
def _(dev):
    return (
        (2,),
        (T(64, dtype=torch.float64, dev=dev), T(64, dtype=torch.float64, dev=dev)),
        {"BLOCK": 32},
    )


@case("a_copy_i64", k_copy, "a", "8-byte integer elements")
def _(dev):
    return (
        (2,),
        (T(64, dtype=torch.int64, dev=dev), T(64, dtype=torch.int64, dev=dev)),
        {"BLOCK": 32},
    )


@triton.jit
def k_same_width_ptr_bitcast(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr.to(tl.pointer_type(tl.int32)) + offs)
    tl.store(out_ptr + offs, v)


@case(
    "a_same_width_ptr_bitcast",
    k_same_width_ptr_bitcast,
    "a",
    "an f32 argument read through an i32 pointer: the element width stays",
)
def _(dev):
    return (1,), (T(16, dev=dev), T(16, dtype=torch.int32, dev=dev)), {"BLOCK": 16}


@triton.jit
def k_i64_limit(x_ptr, big, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + tl.minimum(offs, big - 4294967296), 1.0, mask=offs < big)


@case("a_i64_param", k_i64_limit, "a", "an i64 scalar argument (2**32 + 40)")
def _(dev):
    return (2,), (T(64, dev=dev), 2**32 + 40), {"BLOCK": 32}


@triton.jit
def k_int64_offsets(x_ptr, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * stride + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, 3.0)


@case("a_int64_offsets", k_int64_offsets, "a", "extsi to i64 before the row stride")
def _(dev):
    return (4,), (T(4 * 40, dev=dev), 40), {"BLOCK": 32}


@triton.jit
def k_stride_param(x_ptr, s, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs * s, mask=offs < n)
    tl.store(x_ptr + offs * s + 1, v, mask=offs < n)


@case("a_stride_param", k_stride_param, "a")
def _(dev):
    return (2,), (T(3 * 50, dev=dev), 3, 50), {"BLOCK": 32}


@triton.jit
def k_hints(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    tl.assume(n > 0)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, v, mask=offs < n)


@case("a_hints", k_hints, "a", "multiple_of / max_contiguous / assume")
def _(dev):
    return (2,), (T(50, dev=dev), T(50, dev=dev), 50), {"BLOCK": 32}


@triton.jit
def _store_helper(p, offs, n):
    tl.store(p + offs, 1.0, mask=offs < n)


@triton.jit
def _load_then_store_helper(p, offs, n):
    v = tl.load(p + offs, mask=offs < n)
    _store_helper(p + 64, offs, n)
    return v


@triton.jit
def k_inlined_helpers(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = _load_then_store_helper(x_ptr, offs, n)
    tl.store(out_ptr + offs, v, mask=offs < n)


@case(
    "a_inlined_helpers",
    k_inlined_helpers,
    "a",
    "memory ops in two levels of inlined @triton.jit helpers (callsite locs)",
)
def _(dev):
    return (2,), (T(128, dev=dev), T(64, dev=dev), 50), {"BLOCK": 32}


@triton.jit
def k_debug_ops(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.static_assert(BLOCK % 16 == 0)
    tl.device_assert(n > 0, "n must be positive")
    tl.debug_barrier()
    tl.store(x_ptr + offs, 1.0, mask=offs < n)


@case("a_debug_ops", k_debug_ops, "a", "static_assert / device_assert / debug_barrier")
def _(dev):
    return (2,), (T(64, dev=dev), 40), {"BLOCK": 32}


# ═════════════════ b: min / max / where / integer division ═════════════════


@triton.jit
def k_clamp_min(x_ptr, out_ptr, lim, BLOCK: tl.constexpr):
    offs = tl.minimum(tl.program_id(0) * BLOCK + tl.arange(0, BLOCK), lim)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, v)


@case("b_clamp_min", k_clamp_min, "b")
def _(dev):
    return (4,), (T(100, dev=dev), T(100, dev=dev), 99), {"BLOCK": 32}


@triton.jit
def k_clamp_both(x_ptr, lo, hi, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.maximum(tl.minimum(offs - 4, hi), lo)
    tl.store(x_ptr + c, 1.0)


@case("b_clamp_both", k_clamp_both, "b")
def _(dev):
    return (3,), (T(80, dev=dev), 2, 70), {"BLOCK": 32}


@triton.jit
def k_where_offsets(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.where(offs < n, offs, n - 1)
    tl.store(x_ptr + offs, 1.0)


@case("b_where_offsets", k_where_offsets, "b")
def _(dev):
    return (4,), (T(100, dev=dev), 100), {"BLOCK": 32}


@case(
    "b_where_pointer",
    RK.where_pointer,
    "b",
    "arith.select over two pointers of one base",
)
def _(dev):
    return (1,), (T(128, dev=dev), 9), {}


@triton.jit
def k_modulo(x_ptr, out_ptr, m, BLOCK: tl.constexpr):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)) % m
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, v)


@case("b_modulo", k_modulo, "b")
def _(dev):
    return (4,), (T(100, dev=dev), T(100, dev=dev), 37), {"BLOCK": 32}


@triton.jit
def k_div_mod_2d(x_ptr, W, stride, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offs // W
    col = offs % W
    tl.store(x_ptr + row * stride + col, 1.0, mask=offs < n)


@case("b_div_mod_2d", k_div_mod_2d, "b", "flat index -> (row, col) by runtime divisor")
def _(dev):
    return (3,), (T(8 * 20, dev=dev), 12, 20, 90), {"BLOCK": 32}


@triton.jit
def k_signed_div(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK) - 20
    tl.store(x_ptr + offs // 4 + 8, 1.0)


@case(
    "b_signed_div_trunc",
    k_signed_div,
    "b",
    "divsi truncates toward zero on negative dividends",
)
def _(dev):
    return (1,), (T(64, dev=dev),), {"BLOCK": 64}


@triton.jit
def k_signed_rem(x_ptr, d, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK) - 20
    tl.store(x_ptr + offs % d + 10, 1.0)


@case("b_signed_rem", k_signed_rem, "b", "remsi takes the dividend's sign")
def _(dev):
    return (1,), (T(64, dev=dev), 7), {"BLOCK": 64}


@case("b_unsigned_index", RK.unsigned_index, "b", "divui, cmpi ult, extui")
def _(dev):
    return (7,), (T(8, dev=dev), 5), {}


@triton.jit
def k_bool_extui(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs * 2 + (offs > 5).to(tl.int32), 1.0)


@case("b_bool_extui", k_bool_extui, "b", "extui of an i1 compare")
def _(dev):
    return (1,), (T(64, dev=dev),), {"BLOCK": 32}


@triton.jit
def k_unsigned_min(x_ptr, lim, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + tl.minimum(offs.to(tl.uint32), lim.to(tl.uint32)), 1.0)


@case("b_unsigned_min", k_unsigned_min, "b", "minui, then extui to i64 for the address")
def _(dev):
    return (1,), (T(64, dev=dev), 20), {"BLOCK": 32}


@triton.jit
def k_unsigned_cmp(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, 1.0, mask=offs.to(tl.uint32) < n.to(tl.uint32))


@case(
    "b_unsigned_cmp_highbit",
    k_unsigned_cmp,
    "b",
    "cmpi ult with n = -1 (2**32 - 1 unsigned): the unsigned-operand obligation fails",
)
def _(dev):
    return (1,), (T(16, dev=dev), -1), {"BLOCK": 16}


@triton.jit
def k_nested_select(x_ptr, a, b, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    o = tl.where(offs < a, offs, tl.where(offs < b, offs * 2, 0))
    tl.store(x_ptr + o, 1.0)


@case("b_nested_select", k_nested_select, "b")
def _(dev):
    return (1,), (T(128, dev=dev), 5, 20), {"BLOCK": 32}


# ══════════════════════ c: 2-D tiles and tensor views ══════════════════════


@triton.jit
def k_tile2d(
    x_ptr, out_ptr, M, N, sxm, sxn, sym, syn, BM: tl.constexpr, BN: tl.constexpr
):
    # x: the input view, y: the output
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    v = tl.load(x_ptr + rm[:, None] * sxm + rn[None, :] * sxn, mask=m)
    tl.store(out_ptr + rm[:, None] * sym + rn[None, :] * syn, v, mask=m)


@case("c_tile2d", k_tile2d, "c")
def _(dev):
    M, N = 20, 24
    x, out = T(M, N, dev=dev), T(M, N, dev=dev)
    return (2, 2), (x, out, M, N, *x.stride(), *out.stride()), {"BM": 16, "BN": 16}


@case(
    "c_strided_view", k_tile2d, "c", "x[:, ::2]: a non-contiguous view, strides (2N, 2)"
)
def _(dev):
    M, N = 12, 10
    x = T(M, 2 * N, dev=dev)[:, ::2]
    out = T(M, N, dev=dev)
    return (1, 1), (x, out, M, N, *x.stride(), *out.stride()), {"BM": 16, "BN": 16}


@case("c_transposed_view", k_tile2d, "c", "x.t(): strides (1, M)")
def _(dev):
    M, N = 12, 20
    x = T(N, M, dev=dev).t()
    out = T(M, N, dev=dev)
    return (1, 2), (x, out, M, N, *x.stride(), *out.stride()), {"BM": 16, "BN": 16}


@case("c_expand_view_stride0", k_tile2d, "c", "x.expand(M, N): a stride-0 dim")
def _(dev):
    M, N = 6, 10
    x = T(N, dev=dev).expand(M, N)
    out = T(M, N, dev=dev)
    return (1, 1), (x, out, M, N, *x.stride(), *out.stride()), {"BM": 8, "BN": 16}


@triton.jit
def k_block_ptr(x_ptr, out_ptr, M, N, sm, sn, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    src = tl.make_block_ptr(x_ptr, (M, N), (sm, sn), (pid * BM, 0), (BM, BN), (1, 0))
    v = tl.load(src, boundary_check=(0, 1), padding_option="zero")
    dst = tl.make_block_ptr(out_ptr, (M, N), (sm, sn), (pid * BM, 0), (BM, BN), (1, 0))
    tl.store(dst, v, boundary_check=(0, 1))


@case(
    "c_block_ptr",
    k_block_ptr,
    "c",
    "block pointers, rewritten to pointer tiles + masks in TTIR",
)
def _(dev):
    M, N = 20, 12
    x, out = T(M, N, dev=dev), T(M, N, dev=dev)
    return (3,), (x, out, M, N, *x.stride()), {"BM": 8, "BN": 16}


@triton.jit
def k_block_ptr_loop(x_ptr, out_ptr, K, BK: tl.constexpr):
    src = tl.make_block_ptr(x_ptr, (K,), (1,), (0,), (BK,), (0,))
    acc = tl.zeros((BK,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        acc += tl.load(src, boundary_check=(0,), padding_option="zero")
        src = tl.advance(src, (BK,))
    tl.store(out_ptr + tl.arange(0, BK), acc)


@case(
    "c_block_ptr_loop",
    k_block_ptr_loop,
    "c",
    "an advanced block pointer: the loop carries integer offsets",
)
def _(dev):
    return (1,), (T(40, dev=dev), T(16, dev=dev), 40), {"BK": 16}


@triton.jit
def k_broadcast_row(x_ptr, b_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    bias = tl.load(b_ptr + rn, mask=rn < N)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=m)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], v + bias[None, :], mask=m)


@case("c_broadcast_row", k_broadcast_row, "c")
def _(dev):
    M, N = 6, 12
    return (
        (1,),
        (T(M, N, dev=dev), T(N, dev=dev), T(M, N, dev=dev), M, N),
        {"BM": 8, "BN": 16},
    )


@triton.jit
def k_broadcast_col(x_ptr, out_ptr, M, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    col = tl.load(x_ptr + rm[:, None] + rn[None, :] * 0, mask=rm[:, None] < M)
    tl.store(out_ptr + rm[:, None] * BN + rn[None, :], col, mask=rm[:, None] < M)


@case(
    "c_broadcast_col",
    k_broadcast_col,
    "c",
    "one column broadcast along dim 1 (stride 0)",
)
def _(dev):
    return (1,), (T(8, dev=dev), T(8 * 8, dev=dev), 7), {"BM": 8, "BN": 8}


@case(
    "c_tile3d_shared_arange",
    RK.tile3d_shared_arange,
    "c",
    "one make_range on all three dims",
)
def _(dev):
    return (1,), (T(64, dev=dev),), {"N": 4}


@triton.jit
def k_one_range_two_dims(x_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    tl.store(x_ptr + r[:, None] * (2 * N) + r[None, :] + tl.program_id(0) * N, 1.0)


@case("c_one_range_two_dims", k_one_range_two_dims, "c")
def _(dev):
    return (2,), (T(8 * 16, dev=dev),), {"N": 8}


@triton.jit
def k_softmax_rows(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride + cols, mask=cols < n_cols, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(out_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=cols < n_cols)


@case("c_softmax_rows", k_softmax_rows, "c")
def _(dev):
    R, C = 5, 13
    return (R,), (T(R, 16, dev=dev), T(R, 16, dev=dev), C, 16), {"BLOCK": 16}


@triton.jit
def k_transpose_store(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    src = x_ptr + rm[:, None] * N + rn[None, :]
    dst = out_ptr + rn[:, None] * M + rm[None, :]
    v = tl.load(src, mask=(rm[:, None] < M) & (rn[None, :] < N))
    tl.store(dst, tl.trans(v), mask=(rn[:, None] < N) & (rm[None, :] < M))


@case("c_transpose_store", k_transpose_store, "c")
def _(dev):
    M, N = 6, 10
    return (1,), (T(M, N, dev=dev), T(N, M, dev=dev), M, N), {"BM": 8, "BN": 16}


@triton.jit
def k_row_sum(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=m, other=0.0)
    tl.store(out_ptr + rm, tl.sum(v, axis=1), mask=rm < M)


@case("c_row_sum", k_row_sum, "c", "a reduction's 1-D result stored by the 1-D range")
def _(dev):
    M, N = 11, 12
    return (2,), (T(M, N, dev=dev), T(M, dev=dev), M, N), {"BM": 8, "BN": 16}


# ═══════════════════════ d: program ids and the grid ═══════════════════════


@triton.jit
def k_pid3d(x_ptr, BLOCK: tl.constexpr):
    base = tl.program_id(0) * 100 + tl.program_id(1) * 20 + tl.program_id(2) * 7
    tl.store(x_ptr + base + tl.arange(0, BLOCK), 1.0)


@case("d_pid3d", k_pid3d, "d", "program ids on axes 0, 1 and 2")
def _(dev):
    return (2, 3, 2), (T(200, dev=dev),), {"BLOCK": 4}


@triton.jit
def k_num_programs(x_ptr, out_ptr):
    p0 = tl.program_id(0)
    p1 = tl.program_id(1)
    v = tl.load(x_ptr + p1 * tl.num_programs(0) + p0)
    tl.store(out_ptr + p0 * tl.num_programs(1) + p1, v)


@case("d_num_programs", k_num_programs, "d")
def _(dev):
    return (3, 2), (T(6, dev=dev), T(6, dev=dev)), {}


@triton.jit
def k_num_programs_3d(x_ptr, BLOCK: tl.constexpr):
    np0 = tl.num_programs(0)
    flat = (
        tl.program_id(2) * tl.num_programs(1) + tl.program_id(1)
    ) * np0 + tl.program_id(0)
    tl.store(x_ptr + flat * BLOCK + tl.arange(0, BLOCK), 1.0)


@case("d_num_programs_3d", k_num_programs_3d, "d", "num_programs on axes 0, 1 and 2")
def _(dev):
    return (2, 2, 3), (T(12 * 4, dev=dev),), {"BLOCK": 4}


@triton.jit
def k_num_programs_unread_axis(x_ptr, BLOCK: tl.constexpr):
    # graph.pid_axes must list axis 1 though no program_id reads it
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs * tl.num_programs(1), 1.0)


@case(
    "d_num_programs_unread_axis",
    k_num_programs_unread_axis,
    "d",
    "num_programs(1) without program_id(1): every program on axis 1 stores alike",
)
def _(dev):
    return (2, 3), (T(24, dev=dev),), {"BLOCK": 4}


@triton.jit
def k_grid_stride(x_ptr, out_ptr, n, n_tiles, BLOCK: tl.constexpr):
    for tile in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n)
        tl.store(out_ptr + offs, v * 2, mask=offs < n)


@case(
    "d_grid_stride_loop",
    k_grid_stride,
    "d",
    "for tile in range(pid, n_tiles, num_programs)",
)
def _(dev):
    n, B = 150, 16
    return (3,), (T(n, dev=dev), T(n, dev=dev), n, triton.cdiv(n, B)), {"BLOCK": B}


@triton.jit
def k_grouped_order(
    c_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], 1.0, mask=m)


@case(
    "d_grouped_order",
    k_grouped_order,
    "d",
    "grouped (swizzled) tile order: div / mod / min on pid",
)
def _(dev):
    M, N = 40, 24
    grid = (triton.cdiv(M, 8) * triton.cdiv(N, 8),)
    return grid, (T(M, N, dev=dev), M, N), {"BM": 8, "BN": 8, "GROUP_M": 2}


@triton.jit
def k_pid_axis1_rows(x_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + row * stride + cols, 1.0, mask=cols < n_cols)


@case("d_pid_axis1_rows", k_pid_axis1_rows, "d")
def _(dev):
    return (2, 4), (T(4, 40, dev=dev), 40, 27), {"BLOCK": 16}


# ═══════════════════ e: loops and loop-carried pointers ═══════════════════


@triton.jit
def k_loop_ptr_advance(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs
    for k in range(0, n):
        v = tl.load(p)
        tl.store(out_ptr + k * BLOCK + offs, v)
        p += BLOCK


@case("e_loop_ptr_advance", k_loop_ptr_advance, "e")
def _(dev):
    return (1,), (T(5 * 16, dev=dev), T(5 * 16, dev=dev), 5), {"BLOCK": 16}


@triton.jit
def k_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=c_mask)


@case(
    "e_matmul",
    k_matmul,
    "e",
    "two 2-D pointer tiles advanced by BK * stride, K-tail masks",
)
def _(dev):
    M, N, K = 20, 18, 40
    a, b, c = T(M, K, dev=dev), T(K, N, dev=dev), T(M, N, dev=dev)
    args = (a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    return (2, 2), args, {"BM": 16, "BN": 16, "BK": 16}


@triton.jit
def k_loop_iv_offset(x_ptr, lo, hi):
    for k in range(lo, hi):
        tl.store(x_ptr + k * 2 + tl.program_id(0), 1.0)


@case(
    "e_loop_iv_offset",
    k_loop_iv_offset,
    "e",
    "runtime lower bound, the induction variable in the address",
)
def _(dev):
    return (2,), (T(40, dev=dev), 3, 11), {}


@triton.jit
def k_loop_step(x_ptr, lo, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for k in range(lo, n, 3):
        tl.store(x_ptr + k * BLOCK + offs, 1.0)


@case("e_loop_step3", k_loop_step, "e")
def _(dev):
    return (1,), (T(20 * 4, dev=dev), 2, 17), {"BLOCK": 4}


@triton.jit
def k_zero_trip(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), 0.0)
    p = x_ptr + offs
    for k in range(0, n):
        tl.store(p, 1.0)
        p += BLOCK


@case("e_zero_trip", k_zero_trip, "e", "n = 0: the loop's store has no footprint")
def _(dev):
    return (2,), (T(64, dev=dev), T(2, dev=dev), 0), {"BLOCK": 16}


@case("e_some_trips", k_zero_trip, "e")
def _(dev):
    return (2,), (T(64, dev=dev), T(2, dev=dev), 3), {"BLOCK": 16}


@case(
    "e_expand_iterarg_3d",
    RK.expand_iterarg_3d,
    "e",
    "a loop-carried [N, N] tile expanded to 3-D",
)
def _(dev):
    x = buf(64, dev=dev)
    return (1,), (x, T(64, dev=dev), 2), {"N": 4}


@case(
    "e_expand_iterarg_mask",
    RK.expand_iterarg_mask,
    "e",
    "a loop-carried 1-D tile expanded to 2-D, masked",
)
def _(dev):
    return (1,), (T(64, dev=dev), T(16, dev=dev), 3, 2), {"N": 4}


@case("e_two_step_advance", RK.loop_two_step_advance, "e", "two addptrs per iteration")
def _(dev):
    return (1,), (T(64, dev=dev), 5, 4), {}


@triton.jit
def k_if_in_loop(x_ptr, n):
    for k in range(0, n):
        if k % 2 == 0:
            tl.store(x_ptr + k, 1.0)


@case("e_if_in_loop", k_if_in_loop, "e", "an scf.if on the induction variable")
def _(dev):
    return (1,), (T(16, dev=dev), 9), {}


@triton.jit
def k_iv_mask(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs
    for k in range(0, tl.cdiv(n, BLOCK)):
        tl.store(p, 1.0, mask=offs < n - k * BLOCK)
        p += BLOCK


@case("e_iv_mask", k_iv_mask, "e", "a mask on the induction variable and the lane")
def _(dev):
    return (1,), (T(64, dev=dev), 45), {"BLOCK": 16}


@triton.jit
def k_static_range_in_loop(x_ptr, n):
    p = x_ptr
    for k in range(0, n):
        for j in tl.static_range(3):
            tl.store(p + j, 1.0)
        p += 4


@case(
    "e_static_range_in_loop",
    k_static_range_in_loop,
    "e",
    "an unrolled inner loop: three stores on one line",
)
def _(dev):
    return (1,), (T(32, dev=dev), 5), {}


@triton.jit
def k_pid_trips(x_ptr):
    pid = tl.program_id(0)
    for k in range(0, pid + 1):
        tl.store(x_ptr + pid * 8 + k, 1.0)


@case(
    "e_pid_dependent_trips", k_pid_trips, "e", "a trip count that differs per program"
)
def _(dev):
    return (5,), (T(40, dev=dev),), {}


@triton.jit
def k_unsigned_loop(x_ptr, n):
    for k in range(0, n.to(tl.uint32)):
        tl.store(x_ptr + k, 1.0)


@case("e_unsigned_loop", k_unsigned_loop, "e", "an unsigned induction variable")
def _(dev):
    return (1,), (T(16, dev=dev), 7), {}


@triton.jit
def k_tl_range(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for k in tl.range(0, n, num_stages=2):
        tl.store(x_ptr + k * BLOCK + offs, 1.0)


@case("e_tl_range", k_tl_range, "e", "tl.range with num_stages")
def _(dev):
    return (1,), (T(6 * 8, dev=dev), 6), {"BLOCK": 8}


@triton.jit
def k_static_unroll(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for j in tl.static_range(4):
        tl.store(x_ptr + j * 2 * BLOCK + offs, 1.0)


@case(
    "e_static_unroll",
    k_static_unroll,
    "e",
    "four unrolled stores on one line, no scf.for",
)
def _(dev):
    return (1,), (T(8 * 8, dev=dev),), {"BLOCK": 8}


@triton.jit
def k_negative_delta(x_ptr, out_ptr, n):
    p = x_ptr + 60
    for k in range(0, n):
        v = tl.load(p)
        tl.store(out_ptr + k, v)
        p -= 3


@case("e_negative_delta", k_negative_delta, "e")
def _(dev):
    return (1,), (T(64, dev=dev), T(16, dev=dev), 12), {}


@triton.jit
def k_per_lane_delta(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs
    for k in range(0, n):
        tl.store(p, 1.0)
        p += offs + 1


@case("e_per_lane_delta", k_per_lane_delta, "e", "a loop-invariant per-lane advance")
def _(dev):
    return (1,), (T(64, dev=dev), 4), {"BLOCK": 8}


@triton.jit
def k_expand_axis1(x_ptr, out_ptr, n, N: tl.constexpr):
    r = tl.arange(0, N)
    p = x_ptr + r * N
    for k in range(0, n):
        v = tl.load(p[:, None] + r[None, :])
        tl.store(out_ptr + r[:, None] * N + r[None, :] + k * N * N, v)
        p += 1


@case(
    "e_expand_axis1", k_expand_axis1, "e", "a loop-carried 1-D tile expanded at axis 1"
)
def _(dev):
    return (1,), (T(40, dev=dev), T(3 * 16, dev=dev), 3), {"N": 4}


@triton.jit
def k_expand_per_lane_delta(x_ptr, out_ptr, n, N: tl.constexpr):
    r = tl.arange(0, N)
    p = x_ptr + r
    for k in range(0, n):
        v = tl.load(p[None, :] + r[:, None] * N)
        tl.store(out_ptr + r[:, None] * N + r[None, :] + k * N * N, v)
        p += r + 1


@case(
    "e_expand_per_lane_delta",
    k_expand_per_lane_delta,
    "e",
    "a loop-carried 1-D tile expanded to 2-D whose delta is per lane",
)
def _(dev):
    return (1,), (T(64, dev=dev), T(3 * 16, dev=dev), 3), {"N": 4}


@triton.jit
def k_negative_step(x_ptr, out_ptr, n):
    for k in range(n - 1, -1, -1):
        v = tl.load(x_ptr + k)
        tl.store(out_ptr + (n - 1 - k), v)


@case("e_negative_step", k_negative_step, "e", "range(n - 1, -1, -1)")
def _(dev):
    return (1,), (T(16, dev=dev), T(16, dev=dev), 9), {}


@triton.jit
def k_negative_lower(x_ptr, n):
    for k in range(-3, n):
        tl.store(x_ptr + k + 3, 1.0)


@case("e_negative_lower", k_negative_lower, "e", "a negative lower bound")
def _(dev):
    return (1,), (T(16, dev=dev), 6), {}


@triton.jit
def k_zero_trip_invariant(x_ptr, out_ptr, n):
    tl.store(out_ptr + tl.program_id(0), 0.0)
    for k in range(0, n):
        tl.store(x_ptr + tl.program_id(0), 1.0)


@case(
    "e_zero_trip_invariant",
    k_zero_trip_invariant,
    "e",
    "n = 0: an in-loop store that reads no loop value still has no footprint",
)
def _(dev):
    return (2,), (T(4, dev=dev), T(2, dev=dev), 0), {}


# ═══════════════════════════ f: structured if ═══════════════════════════


@triton.jit
def k_if_else_pid(a_ptr, b_ptr):
    pid = tl.program_id(0)
    if pid % 2 == 0:
        tl.store(a_ptr + pid, 1.0)
    else:
        tl.store(b_ptr + pid // 2, 2.0)


@case("f_if_else_pid", k_if_else_pid, "f")
def _(dev):
    return (5,), (T(5, dev=dev), T(5, dev=dev)), {}


@triton.jit
def k_if_param(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < n:
        tl.store(x_ptr + pid * BLOCK + tl.arange(0, BLOCK), 1.0)


@case("f_if_param", k_if_param, "f")
def _(dev):
    return (4,), (T(4 * 8, dev=dev), 3), {"BLOCK": 8}


@triton.jit
def k_nested_if(x_ptr, lo, hi):
    pid = tl.program_id(0)
    if pid > lo:
        if pid < hi:
            tl.store(x_ptr + pid, 1.0)
        else:
            tl.store(x_ptr + pid + 10, 2.0)


@case("f_nested_if", k_nested_if, "f")
def _(dev):
    return (6,), (T(16, dev=dev), 1, 4), {}


@triton.jit
def k_if_pointer_result(x_ptr, n):
    pid = tl.program_id(0)
    if pid == 0:
        p = x_ptr + 1
    else:
        p = x_ptr + pid * 3 + n
    tl.store(p, 1.0)


@case(
    "f_if_pointer_result",
    k_if_pointer_result,
    "f",
    "an scf.if yielding a pointer of one base",
)
def _(dev):
    return (3,), (T(16, dev=dev), 2), {}


# ═════════════════════════ g: atomics (results unused) ═════════════════════════


@triton.jit
def k_atomic_add_masked(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.atomic_add(x_ptr + offs, 1.0, mask=offs < n)


@case("g_atomic_add_masked", k_atomic_add_masked, "g")
def _(dev):
    return (3,), (T(70, dev=dev), 70), {"BLOCK": 32}


@triton.jit
def k_atomic_counter(cnt_ptr, x_ptr):
    tl.atomic_add(cnt_ptr, 1)
    tl.store(x_ptr + tl.program_id(0), 1.0)


@case("g_atomic_counter", k_atomic_counter, "g", "a scalar counter every program bumps")
def _(dev):
    return (4,), (T(1, dtype=torch.int32, dev=dev, init=None), T(4, dev=dev)), {}


@triton.jit
def k_atomic_max_int(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.atomic_max(x_ptr + offs % 8, offs - 10, mask=offs < n)


@case("g_atomic_max_int", k_atomic_max_int, "g")
def _(dev):
    return (1,), (T(8, dtype=torch.int32, dev=dev, init=None), 20), {"BLOCK": 32}


@triton.jit
def k_atomic_cas(lock_ptr):
    tl.atomic_cas(lock_ptr + tl.program_id(0) * 2, 0, 1)


@case("g_atomic_cas", k_atomic_cas, "g")
def _(dev):
    return (4,), (T(8, dtype=torch.int32, dev=dev, init=None),), {}


@triton.jit
def k_atomic_xchg_2d(x_ptr, M, N: tl.constexpr):
    rm = tl.arange(0, 8)
    rn = tl.arange(0, N)
    tl.atomic_xchg(x_ptr + rm[:, None] * N + rn[None, :], 5, mask=rm[:, None] < M)


@case("g_atomic_xchg_2d", k_atomic_xchg_2d, "g")
def _(dev):
    return (1,), (T(8 * 4, dtype=torch.int32, dev=dev, init=None), 6), {"N": 4}


@triton.jit
def k_atomic_histogram(h_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.atomic_add(h_ptr + offs % 8, 1)


@case("g_atomic_histogram", k_atomic_histogram, "g")
def _(dev):
    return (2,), (T(8, dtype=torch.int32, dev=dev, init=None),), {"BLOCK": 16}


@triton.jit
def k_atomic_in_loop(x_ptr, n):
    for k in range(0, n):
        tl.atomic_add(x_ptr + k * 2 + tl.program_id(0), 1.0)


@case("g_atomic_in_loop", k_atomic_in_loop, "g")
def _(dev):
    return (2,), (T(16, dev=dev, init=None), 6), {}


@triton.jit
def k_atomic_result_value(cnt_ptr, out_ptr):
    old = tl.atomic_add(cnt_ptr, 1)
    tl.store(out_ptr + tl.program_id(0), old)


@case(
    "g_atomic_result_as_value",
    k_atomic_result_value,
    "g",
    "an observation stored as data, not an address",
)
def _(dev):
    return (
        (4,),
        (T(1, dtype=torch.int32, dev=dev, init=None), T(4, dtype=torch.int32, dev=dev)),
        {},
    )


# ═════════════════════════ h: autotune-style configs ═════════════════════════


@triton.autotune(
    configs=[triton.Config({"BLOCK": b}) for b in (16, 32, 64)],
    key=["n"],
)
@triton.jit
def k_autotuned(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, v + 1.0, mask=offs < n)


def _autotune_case(cfg_kwargs: dict):
    def build(dev):
        n = 100
        return (
            (triton.cdiv(n, cfg_kwargs["BLOCK"]),),
            (T(n, dev=dev), T(n, dev=dev), n),
            dict(cfg_kwargs),
        )

    return build


for _i, _cfg in enumerate(k_autotuned.configs):
    case(f"h_autotune_cfg{_i}", k_autotuned.fn, "h", f"config {_cfg.kwargs}")(
        _autotune_case(dict(_cfg.kwargs))
    )


# ═══════════ i: accesses the model over-approximates (skipped, not compared) ═══════════


@triton.jit
def k_datadep_mask(flags_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    f = tl.load(flags_ptr + offs)
    tl.store(out_ptr + offs, 1.0, mask=f != 0)


@case(
    "i_datadep_mask",
    k_datadep_mask,
    "i",
    "a mask from loaded data: dropped, the store skipped",
)
def _(dev):
    return (
        (1,),
        (
            T(16, dtype=torch.int32, dev=dev, init=[i % 3 for i in range(16)]),
            T(16, dev=dev),
        ),
        {"BLOCK": 16},
    )


@triton.jit
def k_atomic_max_float(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.atomic_max(x_ptr + offs % 8, offs.to(tl.float32) - 10.0, mask=offs < n)
    tl.store(out_ptr + offs, 1.0, mask=offs < n)


@case(
    "i_atomic_max_float",
    k_atomic_max_float,
    "i",
    "float max: two integer atomics masked by the value's sign",
)
def _(dev):
    return (1,), (T(8, dev=dev, init=None), T(32, dev=dev), 20), {"BLOCK": 32}


@triton.jit
def k_guarded_branch(x_ptr, out_ptr):
    pid = tl.program_id(0)
    v = tl.load(x_ptr + pid)
    if v > 2.0:
        tl.store(out_ptr + pid, v)


@case(
    "i_guarded_branch",
    k_guarded_branch,
    "i",
    "a branch on loaded data: the store is guarded, skipped",
)
def _(dev):
    return (5,), (T(5, dev=dev), T(5, dev=dev)), {}


@triton.jit
def k_observed_mask_path(cnt_ptr, out_ptr):
    old = tl.atomic_add(cnt_ptr, 1)
    tl.store(out_ptr + tl.program_id(0), 1, mask=old < 2)
    if old == 0:
        tl.store(out_ptr + 8, 2)


@case(
    "i_observed_mask_path",
    k_observed_mask_path,
    "i",
    "an observation in a mask and in a branch condition",
)
def _(dev):
    return (
        (4,),
        (
            T(1, dtype=torch.int32, dev=dev, init=None),
            T(16, dtype=torch.int32, dev=dev),
        ),
        {},
    )


# ═══════════ j: shapes outside the reader's model (refuse OR conform) ═══════════


@triton.jit
def k_two_loops(x_ptr, n):
    for k in range(0, n):
        tl.store(x_ptr + k, 1.0)
    for j in range(0, 2 * n):
        tl.store(x_ptr + 32 + j, 2.0)


@case("j_two_loops", k_two_loops, "j", "two sequential scf.for")
def _(dev):
    return (1,), (T(64, dev=dev), 5), {}


@triton.jit
def k_nested_loops(x_ptr, n):
    for k in range(0, n):
        for j in range(0, 3):
            tl.store(x_ptr + k * 4 + j, 1.0)


@case("j_nested_loops", k_nested_loops, "j", "an scf.for in an scf.for")
def _(dev):
    return (1,), (T(32, dev=dev), 5), {}


@triton.jit
def k_loop_under_if(x_ptr, n):
    if tl.program_id(0) == 0:
        for k in range(0, n):
            tl.store(x_ptr + k, 1.0)


@case("j_loop_under_if", k_loop_under_if, "j", "an scf.for under an scf.if")
def _(dev):
    return (2,), (T(16, dev=dev), 5), {}


@triton.jit
def k_while(x_ptr, n):
    k = 0
    while k < n:
        tl.store(x_ptr + k, 1.0)
        k += 1


@case("j_while", k_while, "j", "an scf.while")
def _(dev):
    return (1,), (T(16, dev=dev), 5), {}


@triton.jit
def k_csr_bound(rowptr_ptr, x_ptr, out_ptr):
    pid = tl.program_id(0)
    lo = tl.load(rowptr_ptr + pid)
    hi = tl.load(rowptr_ptr + pid + 1)
    acc = 0.0
    for k in range(lo, hi):
        acc += tl.load(x_ptr + k)
    tl.store(out_ptr + pid, acc)


@case("j_csr_bound", k_csr_bound, "j", "CSR rows: loop bounds loaded from memory")
def _(dev):
    rowptr = T(4, dtype=torch.int32, dev=dev, init=[0, 2, 5, 9])
    return (3,), (rowptr, T(9, dev=dev), T(3, dev=dev)), {}


@triton.jit
def k_gather(x_ptr, idx_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    v = tl.load(x_ptr + i)
    tl.store(out_ptr + offs, v)


@case("j_gather", k_gather, "j", "a gather: the index is loaded from memory")
def _(dev):
    idx = T(16, dtype=torch.int32, dev=dev, init=[(i * 5) % 16 for i in range(16)])
    return (1,), (T(16, dev=dev), idx, T(16, dev=dev)), {"BLOCK": 16}


# ═══════════════ r: audit regression kernels (refuse OR conform) ═══════════════
# The shapes #361's reader misread (ir_mode_audit/probes/), from the goldens.


@case("r_p1_variant_delta", RK.p1_variant_delta, "r", "p1: p += k")
def _(dev):
    return (1,), (T(100, dev=dev), T(1, dev=dev), 4), {}


@case("r_p2_swap", RK.p2_swap, "r", "p2: p, q = q, p")
def _(dev):
    return (1,), (T(128, dev=dev), T(128, dev=dev), 4), {}


@case("r_p3_call_guarded", RK.p3_call_guarded, "r", "p3: noinline call under if")
def _(dev):
    return (8,), (T(8, dev=dev), 4), {}


@case("r_p3_call_offset", RK.p3_call_offset, "r", "p3: noinline call, actual != formal")
def _(dev):
    return (4,), (T(128, dev=dev), 4), {}


@case(
    "r_p3_call_formals",
    RK.p3_call_formals,
    "r",
    "p3: noinline call, formals match no caller name",
)
def _(dev):
    return (4,), (T(128, dev=dev), 4), {}


@case(
    "r_p4_observed_direct",
    RK.p4_observed_direct,
    "r",
    "p4: an address from an atomic observation",
)
def _(dev):
    return (3,), (T(1, dtype=torch.int32, dev=dev, init=None), T(16, dev=dev), 8), {}


@case(
    "r_p4_observed_loop",
    RK.p4_observed_loop,
    "r",
    "p4: the observation in a loop-carried offset0",
)
def _(dev):
    return (3,), (T(1, dtype=torch.int32, dev=dev, init=None), T(16, dev=dev), 8), {}


@case(
    "r_p4_observed_delta",
    RK.p4_observed_delta,
    "r",
    "p4: the observation in the loop delta",
)
def _(dev):
    return (3,), (T(1, dtype=torch.int32, dev=dev, init=None), T(64, dev=dev), 8), {}


@case(
    "r_trunci_alias_pid0",
    RK.rv_trunci_alias,
    "r",
    "trunci of pid * 2**32 with pid 0 only: fits",
)
def _(dev):
    return (1,), (T(4, dtype=torch.int32, dev=dev),), {}


@case(
    "r_trunci_alias_wrap",
    RK.rv_trunci_alias,
    "r",
    "trunci of pid * 2**32: every program stores x[0]",
)
def _(dev):
    return (3,), (T(4, dtype=torch.int32, dev=dev),), {}


@case("r_i32_wrap_small", RK.rv_i32_wrap, "r", "(pid * S) * S without a wrap")
def _(dev):
    return (4,), (T(64, dtype=torch.int32, dev=dev), 3), {}


@case("r_i32_wrap_wrap", RK.rv_i32_wrap, "r", "(pid * 65536) * 65536 wraps to 0 in i32")
def _(dev):
    return (2,), (T(4, dtype=torch.int32, dev=dev), 65536), {}


@triton.jit
def k_iv_wrap(x_ptr, lo, n, STEP: tl.constexpr):
    # the induction variable's increment wraps in i32 when n is near INT32_MAX
    for k in range(lo, n, STEP):
        tl.store(x_ptr + (k - lo) // STEP, 1.0)


@case("r_iv_wrap_small", k_iv_wrap, "r", "the loop increment stays in i32")
def _(dev):
    return (1,), (T(16, dev=dev), 0, 10 << 20), {"STEP": 1 << 20}


@case("r_iv_wrap_wrap", k_iv_wrap, "r", "upper - 1 + step overflows i32")
def _(dev):
    step = 1 << 20
    return (1,), (T(16, dev=dev), 2**31 - 3 * step - 7, 2**31 - 1), {"STEP": step}


@case("r_inline_asm_store", RK.rv_inline_asm_store, "r", "an impure asm st.global")
def _(dev):
    return (1,), (T(8, dtype=torch.int32, dev=dev),), {"OFF": 4096}


@case(
    "r_pure_asm_int_addr",
    RK.pure_asm_int_addr,
    "r",
    "a pure asm handed the address as an integer",
)
def _(dev):
    return (1,), (T(8, dtype=torch.int32, dev=dev),), {}


@case(
    "r_loop_observed_advance",
    RK.loop_observed_advance,
    "r",
    "the advance is an atomic observed in the loop",
)
def _(dev):
    return (1,), (T(1, dtype=torch.int32, dev=dev, init=None), T(64, dev=dev), 4), {}


@case(
    "r_int_iterarg_offset",
    RK.int_iterarg_offset,
    "r",
    "an integer offset carried by the loop",
)
def _(dev):
    return (1,), (T(64, dev=dev), 3), {"B": 8}


@case(
    "r_observed_lanes",
    RK.observed_lanes,
    "r",
    "two lanes of one tensor atomic's old values",
)
def _(dev):
    return (1,), (T(4, dtype=torch.int32, dev=dev, init=None), T(64, dev=dev)), {"N": 4}
