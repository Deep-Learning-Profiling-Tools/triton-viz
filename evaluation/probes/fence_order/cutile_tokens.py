"""Probe: how cuda.tile 1.5.0 threads memory-ordering tokens for
intra-instance conflicts.  Compiles each kernel and prints the FULL final IR."""
from __future__ import annotations

import sys
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
MO = ct.MemoryOrder
MS = ct.MemoryScope
BLOCK = 64


def compile_ir(kernel, args):
    from cuda.tile import compilation
    from cuda.tile._bytecode.version import BytecodeVersion
    from cuda.tile._compile import compile_tile

    cap = torch.cuda.get_device_capability()
    cc = compilation.CallingConvention.cutile_python_v2
    if callable(cc):
        cc = cc()
    sig = compilation.KernelSignature.from_kernel_args(kernel, args, cc)
    res = compile_tile(
        kernel._annotated_function,
        [sig],
        sm_arch=f"sm_{cap[0]}{cap[1]}",
        bytecode_version=BytecodeVersion.V_13_3,
        return_final_ir=True,
        return_bytecode=False,
        return_cubin=False,
    )
    return "\n".join(blk.to_string() for blk in res.final_ir)


# K1a: Figure 1 shape, pointer path (scatter then gather, same array)
@ct.kernel
def k1a_scatter_then_gather(hist, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    zero = ct.arange(1, dtype=np.int32)
    ct.scatter(hist, offs, offs)
    v = ct.gather(hist, zero + pid)
    ct.scatter(out, zero + pid, v)


# K1b: Figure 1 shape, view path (store then load, same array)
@ct.kernel
def k1b_store_then_load(hist, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    ct.store(hist, index=(0,), tile=offs)
    v = ct.load(hist, index=(pid,), shape=(1,))
    ct.store(out, index=(pid,), tile=v)


# K2: store to A then load from a DIFFERENT array B
@ct.kernel
def k2_store_a_load_b(a, b, out, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    ct.scatter(a, offs, offs)
    v = ct.gather(b, offs)
    ct.scatter(out, offs, v)


# K3: store to data, then atomic_xchg RELEASE on flag
@ct.kernel
def k3_store_then_release_xchg(data, flag, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    zero = ct.arange(1, dtype=np.int32)
    ct.scatter(data, offs, offs)
    ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)


# K3b (control): store to data, then atomic_xchg RELAXED on flag
@ct.kernel
def k3b_store_then_relaxed_xchg(data, flag, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    zero = ct.arange(1, dtype=np.int32)
    ct.scatter(data, offs, offs)
    ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)


# K4: atomic_add ACQUIRE on flag, then load of data
@ct.kernel
def k4_acquire_add_then_load(flag, data, out, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    zero = ct.arange(1, dtype=np.int32)
    f = ct.atomic_add(flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE)
    v = ct.gather(data, offs)
    ct.scatter(out, offs, v + f)


# K4b (control): atomic_add RELAXED on flag, then load of data
@ct.kernel
def k4b_relaxed_add_then_load(flag, data, out, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    zero = ct.arange(1, dtype=np.int32)
    f = ct.atomic_add(flag, zero, 0, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)
    v = ct.gather(data, offs)
    ct.scatter(out, offs, v + f)


# K5: two scatters to the same array in sequence
@ct.kernel
def k5_two_scatters_same_array(out, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    ct.scatter(out, offs, offs)
    ct.scatter(out, offs, offs + 1)


# K6 (control): two gathers from the same array, then a store
@ct.kernel
def k6_two_gathers_same_array(x, out, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    a = ct.gather(x, offs)
    b = ct.gather(x, offs + 1)
    ct.scatter(out, offs, a + b)


# K7: gather then scatter on the same array (WAR)
@ct.kernel
def k7_gather_then_scatter_same_array(x, BLOCK: ConstInt):
    offs = ct.arange(BLOCK, dtype=np.int32)
    a = ct.gather(x, offs)
    ct.scatter(x, offs + 1, a)


def t(n=4 * BLOCK):
    return torch.zeros(n, dtype=torch.int32, device="cuda")


CASES = [
    (
        "K1a scatter->gather same array (Figure 1, pointer path)",
        k1a_scatter_then_gather,
        (t(), t(), BLOCK),
    ),
    (
        "K1b store->load same array (Figure 1, view path)",
        k1b_store_then_load,
        (t(), t(), BLOCK),
    ),
    (
        "K2 store A -> load B (different arrays)",
        k2_store_a_load_b,
        (t(), t(), t(), BLOCK),
    ),
    (
        "K3 store data -> atomic_xchg RELEASE flag",
        k3_store_then_release_xchg,
        (t(), t(1), BLOCK),
    ),
    (
        "K3b store data -> atomic_xchg RELAXED flag (control)",
        k3b_store_then_relaxed_xchg,
        (t(), t(1), BLOCK),
    ),
    (
        "K4 atomic_add ACQUIRE flag -> load data",
        k4_acquire_add_then_load,
        (t(1), t(), t(), BLOCK),
    ),
    (
        "K4b atomic_add RELAXED flag -> load data (control)",
        k4b_relaxed_add_then_load,
        (t(1), t(), t(), BLOCK),
    ),
    ("K5 two scatters same array", k5_two_scatters_same_array, (t(), BLOCK)),
    (
        "K6 two gathers same array (control)",
        k6_two_gathers_same_array,
        (t(), t(), BLOCK),
    ),
    (
        "K7 gather -> scatter same array (WAR)",
        k7_gather_then_scatter_same_array,
        (t(), BLOCK),
    ),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, k, args in CASES:
        if only and not any(o in name for o in only):
            continue
        print("=" * 100)
        print("##", name)
        print("=" * 100)
        try:
            print(compile_ir(k, args))
        except Exception:
            import traceback

            traceback.print_exc()
