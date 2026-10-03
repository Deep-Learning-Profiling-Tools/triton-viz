"""Regenerate the kernel-derived TTIR goldens of the installed Triton release.

    python tests/golden/ir/generate_ttir.py [NAME ...]

Host-compiles each kernel below (ASTSource + ``triton.compile``, no GPU
needed, a throwaway Triton cache) into the installed release's directory:
``ttir/`` for 3.6, the base release, ``ttir_<release>/`` for any other. The
goldens' locs name the kernels' lines: a line added above the first kernel
moves every loc, and the base goldens no longer regenerate byte for byte,
so notes (BASE_RELEASE, the copies, RESPELLED) go below the kernels.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))


@triton.jit
def dot_precisions(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    # fp32 inputs: `ieee` is the printer-elided default, tf32x3 prints
    offs = tl.arange(0, BLOCK)
    idx = offs[:, None] * BLOCK + offs[None, :]
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    c = tl.dot(a, b, input_precision="ieee")
    d = tl.dot(a, b, input_precision="tf32x3")
    tl.store(c_ptr + idx, c + d)


@triton.jit
def eps_consts(x_ptr, s_ptr, out_ptr, BLOCK: tl.constexpr):
    # uppercase-E float literals: a scalar constant and a dense splat
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.load(s_ptr) + 1e-6
    tl.store(out_ptr + offs, x * s + 1e-12)


@triton.jit
def unicode_msgs(x_ptr, BLOCK: tl.constexpr):
    # a non-ASCII tt.assert message (debug=True); device_print prefixes must be ASCII
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.device_assert(x > 0, "错误: π must be > 0")
    tl.device_print("x=", x)
    tl.store(x_ptr + offs, x + 1)


@triton.jit
def deep_chain(out_ptr, s, N: tl.constexpr):
    # a def chain deeper than Python's default recursion limit
    pid = tl.program_id(0)
    off = pid
    for _ in tl.static_range(N):
        off = off * s + pid
    tl.store(out_ptr + off, 1.0)


@triton.jit
def dot_scaled_k(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):  # fmt: skip
    # tt.dot_scaled prints `%a scale %as, %b scale %bs, %c`: ODS (a, b, c, as, bs)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    rs = tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    a_scale = tl.load(as_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b_scale = tl.load(bs_ptr + rn[:, None] * (K // 32) + rs[None, :])
    c = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


# ── the release directories, the copied goldens, RESPELLED ──
#
# BASE_RELEASE printed ttir/, whose goldens are read under every release; a
# later release's ttir_<release>/ (e.g. ttir_3.8/) holds that release's own
# printing of the same goldens, which shadows the base copy of the same name
# under that release.
#
# ``SPECS`` are the ``kernel_<name>`` goldens. The other base goldens are
# copies: ``golden_*`` from #361's ``tests/golden/ttgir/*.ttir``, ``spike_*``
# from the D10a spike corpus, ``adv_*`` / ``nat_*`` from its independent
# review, and ``crafted_*`` are hand-written TTIR for shapes no kernel prints
# reliably (empty region bodies, generic form, quoted symbols, loc forms, cf
# edge cases). ``RESPELLED`` rebuilds from their kernels, for every release
# but the base one, the copies a later release prints differently in a way
# its tests must see: a syntax the base copy does not parse under (3.8: the
# ``!tt.tensordesc`` type), or an op that release's reader reads differently
# (3.8: ``tl.debug_barrier()`` is ``ttg.barrier all``, not ``gpu.barrier``).

BASE_RELEASE = "3.6"  # the release that printed ttir/

# The SPECS kernels keep the lines ttir/'s goldens were printed from (their
# locs), dot_scaled_k's one-line signature included (hence ``fmt: skip``).


# ── RESPELLED: the kernels behind base copies a later printer respells ──


@triton.jit
def descs(a_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # adv_descs (the D10a review's adv_kernels.py): device-side tensor
    # descriptors, descriptor_load / store / reduce / gather / scatter
    d = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [BM, BN])
    x = d.load([0, BN])
    d.store([BM, 0], x)
    d.atomic_add([BM, BN], x)
    d1 = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [1, BN])
    rows = tl.arange(0, BM)
    g = d1.gather(rows, 0)
    d1.scatter(g, rows, BN)


@triton.jit
def matmul_tma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # golden_matmul_tma_s{1,3}_sm90 (#361's tests/golden/ttgir/generate_golden.py)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
        acc += tl.dot(a, b)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.float16))


@triton.jit
def matmul_tma_ws_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # golden_matmul_tma_ws_s3_sm90 (#361): the warp-specialized loop
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), warp_specialize=True):
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
        acc += tl.dot(a, b)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.float16))


@triton.jit
def zero_result(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # adv_zero_result (the D10a review's adv_kernels.py): zero-result ops
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.device_assert(offs < n + BLOCK, "offs {bad} } { loc(")
    tl.device_print("pid=", pid, offs, hex=True)
    tl.debug_barrier()
    if pid == 0:
        tl.atomic_add(out_ptr, 1)
    tl.atomic_max(out_ptr + offs, v.to(tl.int32), mask=offs < n)
    tl.store(out_ptr + offs, v.to(tl.int32), mask=offs < n)


@triton.jit
def misc(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # spike_misc (the D10a spike's corpus_kernels.py)
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n, other=-1.5)
    tl.debug_barrier()
    tl.device_print("v{x} loc(", npg)
    tl.store(out_ptr + offs, tl.where(v > 0, v, 0.0), mask=offs < n)


# (kernel, signature, constexprs, compute capability, compile options,
# ASTSource attrs)
_Spec = tuple[Any, dict[str, str], dict[str, int], int, dict[str, Any], dict]

SPECS: dict[str, _Spec] = {
    "dot_precisions": (
        dot_precisions,
        {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 16},
        80,
        {},
        {},
    ),
    "eps_consts": (
        eps_consts,
        {"x_ptr": "*fp32", "s_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {},
        {},
    ),
    "unicode_msgs": (
        unicode_msgs,
        {"x_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {"debug": True},
        {},
    ),
    "deep_chain": (
        deep_chain,
        {"out_ptr": "*fp32", "s": "i32", "N": "constexpr"},
        {"N": 600},
        80,
        {},
        {},
    ),
    "dot_scaled": (
        dot_scaled_k,
        {
            "a_ptr": "*fp8e4nv",
            "as_ptr": "*u8",
            "b_ptr": "*fp8e4nv",
            "bs_ptr": "*u8",
            "c_ptr": "*fp32",
            "M": "constexpr",
            "N": "constexpr",
            "K": "constexpr",
        },
        {"M": 128, "N": 128, "K": 64},
        100,
        {},
        {},
    ),
}


_TMA_SIG = {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
    "BLOCK_K": "constexpr",
}
_TMA_CONST = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
# divisibility 16 on the pointers and M, N, K, as #361's generator set it
_TMA_ATTRS = {(i,): [["tt.divisibility", 16]] for i in range(6)}

RESPELLED: dict[str, _Spec] = {
    "adv_zero_result": (
        zero_result,
        {"x_ptr": "*fp32", "out_ptr": "*i32", "n": "i32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {},
        {},
    ),
    "spike_misc": (
        misc,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "adv_descs": (
        descs,
        {
            "a_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "BM": "constexpr",
            "BN": "constexpr",
        },
        {"BM": 32, "BN": 32},
        100,
        {},
        {},
    ),
    "golden_matmul_tma_s1_sm90": (
        matmul_tma_kernel,
        _TMA_SIG,
        _TMA_CONST,
        90,
        {"num_stages": 1},
        _TMA_ATTRS,
    ),
    "golden_matmul_tma_s3_sm90": (
        matmul_tma_kernel,
        _TMA_SIG,
        _TMA_CONST,
        90,
        {"num_stages": 3},
        _TMA_ATTRS,
    ),
    "golden_matmul_tma_ws_s3_sm90": (
        matmul_tma_ws_kernel,
        _TMA_SIG,
        _TMA_CONST,
        90,
        {"num_stages": 3},
        _TMA_ATTRS,
    ),
}


def release() -> str:
    """The installed Triton's minor release ("3.6")."""
    return ".".join(triton.__version__.split(".")[:2])


def out_dir(rel: str) -> str:
    """The golden directory of release ``rel``."""
    return os.path.join(HERE, "ttir" if rel == BASE_RELEASE else f"ttir_{rel}")


def jobs(rel: str) -> dict[str, _Spec]:
    """Golden name -> spec of what release ``rel`` prints into out_dir(rel)."""
    todo = {f"kernel_{name}": spec for name, spec in SPECS.items()}
    if rel != BASE_RELEASE:  # the base release holds these as copies
        todo.update(RESPELLED)
    return todo


# A loc's path up to this repository's tests/ or Triton's package directory:
# where the checkout and Triton happen to be on the machine that printed it.
_REPO_LOC = r'loc\("[^"]*/tests/(?=golden/)'
_TRITON_LOC = r'loc\("[^"]*/triton/'


def portable(text: str) -> str:
    """``text`` with each loc naming a file of this repository from
    ``tests/`` and one of Triton's own sources (tl.cdiv, ...) from
    ``triton/``, so a golden prints the same on every machine."""
    import re  # not above the kernels: a line added there moves their locs

    return re.sub(_TRITON_LOC, 'loc("triton/', re.sub(_REPO_LOC, 'loc("tests/', text))


def ttir(spec: _Spec) -> str:
    """The TTIR the installed Triton prints for ``spec`` (a host compile)."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn, sig, consts, cc, options, attrs = spec
    src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs)
    k = triton.compile(
        src, target=GPUTarget("cuda", cc, 32), options={"num_warps": 4, **options}
    )
    return k.asm["ttir"]


def main() -> int:
    only = set(sys.argv[1:])
    rel = release()
    out = out_dir(rel)
    failed = 0
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        for name, spec in jobs(rel).items():
            if only and name not in only and name.removeprefix("kernel_") not in only:
                continue
            try:
                text = ttir(spec)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(
                    f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}",
                    file=sys.stderr,
                )
                continue
            path = os.path.join(out, f"{name}.ttir")
            with open(path, "w", encoding="utf-8") as f:
                f.write(portable(text))
            print(f"[{name}] wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
