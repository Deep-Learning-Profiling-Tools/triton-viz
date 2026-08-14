"""Engine-local instruction microbenchmarks.

Dependent-chain modes estimate consumer-visible latency / dependency overhead.
Independent-stream modes estimate initiation interval and saturated throughput.
"""

from __future__ import annotations

import numpy as np

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

from microbench.inf2_nki.common.nki_utils import dtype_for_load


def vector_add_factory(*, p: int, f: int, repeat: int, mode: str, dtype_name: str = "float32"):
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        out = nl.ndarray((p, f), dtype=kdtype, buffer=nl.shared_hbm)
        base = nl.load(x, dtype=kdtype)
        if mode == "dependent_chain":
            acc = base
            for i in nl.static_range(repeat):
                # A compile-time scalar +1 chain is algebraically folded by
                # the compiler. A runtime tile operand preserves one true
                # loop-carried VectorE dependency per source operation.
                acc = nisa.tensor_tensor(acc, base, op=nl.add, dtype=kdtype, engine=nisa.engine.vector, name=f"vec_dep_{i}")
            nl.store(out, acc)
        elif mode == "independent_stream":
            out = nl.ndarray((repeat, p, f), dtype=kdtype, buffer=nl.shared_hbm)
            outs = nl.ndarray((repeat, nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
            for i in nl.static_range(repeat):
                outs[i] = nisa.tensor_scalar(data=base, op0=nl.add, operand0=float(i + 1), dtype=kdtype, engine=nisa.engine.vector, name=f"vec_ind_{i}")
            for i in nl.static_range(repeat):
                nl.store(out[i], outs[i])
        elif mode == "independent_two_input":
            # Retain every identical two-stream operation in a distinct SBUF
            # destination/HBM output. Explorer instruction counts are required
            # to confirm the compiler did not CSE the named operations.
            out = nl.ndarray((repeat, p, f), dtype=kdtype, buffer=nl.shared_hbm)
            outs = nl.ndarray((repeat, nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
            for i in nl.static_range(repeat):
                outs[i] = nisa.tensor_tensor(
                    base, base, op=nl.add, dtype=kdtype,
                    engine=nisa.engine.vector, name=f"vec_two_input_{i}",
                )
            for i in nl.static_range(repeat):
                nl.store(out[i], outs[i])
        else:
            raise ValueError("unknown vector mode")
        return out

    return kernel, [(p, f)], (1,)


def scalar_exp_factory(*, p: int, f: int, repeat: int, mode: str, dtype_name: str = "float32"):
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(x):
        kdtype = dtype_for_load(kernel_dtype_name, x.dtype)
        out = nl.ndarray((p, f), dtype=kdtype, buffer=nl.shared_hbm)
        if mode == "dependent_chain":
            base = nl.load(x, dtype=kdtype)
            acc = base
            for i in nl.static_range(repeat):
                # Keep a long EXP dependency chain numerically finite. Scaling
                # is fused into ScalarE activation at no additional cost.
                acc = nisa.activation(np.exp, data=acc, scale=0.001, dtype=kdtype, name=f"sca_dep_{i}")
            nl.store(out, acc)
        elif mode == "independent_stream":
            base = nl.load(x, dtype=kdtype)
            outs = nl.ndarray((repeat, nl.par_dim(p), f), dtype=kdtype, buffer=nl.sbuf)
            for i in nl.static_range(repeat):
                # ScalarE applies scale before EXP at no extra cost. Distinct
                # scales prevent CSE while all EXP instructions remain mutually
                # independent and reuse one SBUF input tile.
                outs[i] = nisa.activation(np.exp, data=base, scale=float(i + 1), dtype=kdtype, name=f"sca_ind_{i}")
            # Make every EXP observable with one HBM store. The VectorE fold is
            # accounted separately from ScalarE active time in profile analysis.
            keepalive = outs[0]
            for i in nl.static_range(1, repeat):
                keepalive = nisa.tensor_tensor(keepalive, outs[i], op=nl.add, dtype=kdtype, engine=nisa.engine.vector, name=f"sca_keepalive_{i}")
            nl.store(out, keepalive)
        else:
            raise ValueError("unknown scalar mode")
        return out

    return kernel, [(p, f)], (1,)


def tensor_matmul_factory(*, m: int, k: int, n: int, repeat: int, mode: str, dtype_name: str = "float32"):
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(lhs, rhs):
        kdtype = dtype_for_load(kernel_dtype_name, lhs.dtype)
        out = nl.ndarray((m, n), dtype=kdtype, buffer=nl.shared_hbm)
        lhs_s = nl.load(lhs, dtype=kdtype)
        rhs_s = nl.load(rhs, dtype=kdtype)
        if mode == "dependent_accumulate":
            acc = nl.zeros((nl.par_dim(m), n), dtype=nl.float32, buffer=nl.psum)
            for i in nl.static_range(repeat):
                acc += nisa.nc_matmul(stationary=lhs_s, moving=rhs_s, name=f"mm_dep_{i}")
            sb = nisa.tensor_copy(acc, dtype=kdtype, engine=nisa.engine.vector, name="evict_psum")
            nl.store(out, sb)
        elif mode == "independent_stream":
            out = nl.ndarray((repeat, m, n), dtype=kdtype, buffer=nl.shared_hbm)
            psums = nl.ndarray((repeat, nl.par_dim(m), n), dtype=nl.float32, buffer=nl.psum)
            for i in nl.static_range(repeat):
                psums[i] = nisa.nc_matmul(stationary=lhs_s, moving=rhs_s, name=f"mm_ind_{i}")
            for i in nl.static_range(repeat):
                sb = nisa.tensor_copy(psums[i], dtype=kdtype, engine=nisa.engine.vector, name=f"evict_{i}")
                nl.store(out[i], sb)
        else:
            raise ValueError("unknown tensor mode")
        return out

    return kernel, [(k, m), (k, n)], (1,)


def work_units(*, p: int | None = None, f: int | None = None, m: int | None = None, k: int | None = None, n: int | None = None, repeat: int, mode: str = "", **_: object) -> dict[str, int]:
    if m is not None and k is not None and n is not None:
        return {"matmul_flops": 2 * m * n * k * repeat, "logical_instructions": repeat}
    if p is not None and f is not None:
        streams = 2 if mode in ("dependent_chain", "independent_two_input") else 1
        return {
            "elements": p * f * repeat,
            "logical_instructions": repeat,
            "input_stream_count": streams,
        }
    return {"logical_instructions": repeat}
