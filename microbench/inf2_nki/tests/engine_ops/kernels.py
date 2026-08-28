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


def tensor_matmul_tiled_factory(
    *,
    m: int,
    k: int,
    n: int,
    mode: str = "independent",
    dtype_name: str = "float32",
):
    """Tilebench-shaped TensorE pipeline control.

    This is intentionally not a copy of the Tilebench source: it is an
    independent control kernel with the same TensorE tile geometry
    (128x128 stationary, 128x512 moving), the same block loops, and the same
    on-chip ``nc_transpose`` staging pattern. It is used only for calibration;
    Tilebench's ``matmul_fp32_fp16_fp8`` kernel remains the holdout. The fit
    is dtype-keyed throughput (startup + flops / throughput): there is no
    per-shape lookup table.
    """
    kernel_dtype_name = dtype_name
    tile_m = tile_k = 128
    tile_n = 512
    tib_m = m // tile_m
    tib_k = k // tile_k
    tib_n = n // tile_n

    @nki.jit
    def kernel(lhs, rhs):
        kdtype = dtype_for_load(kernel_dtype_name, lhs.dtype)
        out = nl.ndarray((m, n), dtype=kdtype, buffer=nl.shared_hbm)

        for bm in nl.affine_range(tib_m):
            lhs_tiles = nl.ndarray(
                (tib_k, nl.par_dim(tile_k), tile_m),
                dtype=kdtype,
                buffer=nl.sbuf,
            )
            i_lhs = nl.mgrid[0:tile_m, 0:tile_k]
            for bk in nl.affine_range(tib_k):
                lhs_tile = nl.ndarray(
                    (tile_m, tile_k), dtype=kdtype, buffer=nl.sbuf
                )
                lhs_tile[...] = nl.load(
                    lhs[
                        bm * tile_m + i_lhs.p,
                        bk * tile_k + i_lhs.x,
                    ]
                )
                lhs_tiles[bk] = nisa.nc_transpose(lhs_tile)

            for bn in nl.affine_range(tib_n):
                rhs_tiles = nl.ndarray(
                    (tib_k, nl.par_dim(tile_k), tile_n),
                    dtype=kdtype,
                    buffer=nl.sbuf,
                )
                i_rhs = nl.mgrid[0:tile_k, 0:tile_n]
                for bk in nl.affine_range(tib_k):
                    rhs_tiles[bk] = nl.load(
                        rhs[
                            bk * tile_k + i_rhs.p,
                            bn * tile_n + i_rhs.x,
                        ]
                    )

                res_psum = nl.zeros(
                    (tile_m, tile_n), dtype=nl.float32, buffer=nl.psum
                )
                for bk in nl.affine_range(tib_k):
                    res_psum += nisa.nc_matmul(
                        stationary=lhs_tiles[bk],
                        moving=rhs_tiles[bk],
                        name=f"tile_mm_{bm}_{bn}_{bk}",
                    )

                res_sbuf = nisa.tensor_copy(
                    res_psum, dtype=kdtype, engine=nisa.engine.vector
                )
                i_out = nl.mgrid[0:tile_m, 0:tile_n]
                nl.store(
                    out[
                        bm * tile_m + i_out.p,
                        bn * tile_n + i_out.x,
                    ],
                    value=res_sbuf,
                )
        return out

    return kernel, [(m, k), (k, n)], (1,)


def tensor_matmul_tiled_work_units(
    *,
    m: int,
    k: int,
    n: int,
    mode: str = "independent",
    dtype_name: str = "float32",
) -> dict[str, int]:
    return {
        "matmul_flops": 2 * m * n * k,
        "logical_instructions": (m // 128) * (n // 512) * (k // 128),
        "dot_count": (m // 128) * (n // 512) * (k // 128),
    }


def tensor_matmul_small_factory(
    *,
    m: int,
    k: int,
    n: int,
    repeat: int,
    mode: str = "independent",
    dtype_name: str = "float32",
):
    """Small single-tile TensorE calibration control.

    This independent control measures the TensorE ``startup + flops /
    throughput`` regime for small tiles. It is deliberately disjoint from the
    attention holdout geometry: the pipeline config uses
    ``m=64, k=64, n={96,192,320,448}`` while the tiled-attention holdout uses
    ``m=128, k=128, n={64,128,256,512}``. The fit remains keyed by operand
    dtype only -- never by tile shape -- and ``repeat`` varies the number of
    retained independent Dots so the per-kernel startup can be separated from
    the steady-state slope. There is no softmax and no cross-Dot dependency:
    every Dot writes a distinct output.
    """
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(lhs, rhs):
        kdtype = dtype_for_load(kernel_dtype_name, lhs.dtype)
        out = nl.ndarray((repeat, m, n), dtype=kdtype, buffer=nl.shared_hbm)
        lhs_tiles = nl.ndarray(
            (repeat, nl.par_dim(k), m), dtype=kdtype, buffer=nl.sbuf
        )
        rhs_tiles = nl.ndarray(
            (repeat, nl.par_dim(k), n), dtype=kdtype, buffer=nl.sbuf
        )
        for i in nl.static_range(repeat):
            lhs_tiles[i] = nl.load(lhs[i])
            rhs_tiles[i] = nl.load(rhs[i])
            res_psum = nl.zeros((m, n), dtype=nl.float32, buffer=nl.psum)
            res_psum += nisa.nc_matmul(
                stationary=lhs_tiles[i],
                moving=rhs_tiles[i],
                name=f"small_mm_{i}",
            )
            res_sbuf = nisa.tensor_copy(
                res_psum, dtype=kdtype, engine=nisa.engine.vector
            )
            nl.store(out[i], value=res_sbuf)
        return out

    return kernel, [(repeat, k, m), (repeat, k, n)], (1,)


def tensor_matmul_small_work_units(
    *,
    m: int,
    k: int,
    n: int,
    repeat: int,
    mode: str = "independent",
    dtype_name: str = "float32",
) -> dict[str, int]:
    return {
        "matmul_flops": 2 * m * n * k * repeat,
        "logical_instructions": repeat,
        "dot_count": repeat,
    }


def tensor_matmul_transpose_pipeline_factory(
    *, m: int, k: int, n: int, repeat: int = 1,
    mode: str = "independent", dtype_name: str = "float32",
):
    """Disjoint primitive control for mixed TRANSPOSE/REGULAR TensorE work.

    This deliberately contains no attention normalization or application
    dataflow.  Three stationary tiles are transposed and two independent dots
    are retained, isolating the compiler's mixed TensorE instruction stream.
    """
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(lhs, rhs):
        kdtype = dtype_for_load(kernel_dtype_name, lhs.dtype)
        out = nl.ndarray((3, m, n), dtype=kdtype, buffer=nl.shared_hbm)
        stationary = nl.ndarray(
            (3, nl.par_dim(k), m), dtype=kdtype, buffer=nl.sbuf
        )
        for index in nl.static_range(3):
            tile = nl.load(lhs[index])
            stationary[index] = nisa.nc_transpose(tile)
        moving = nl.load(rhs)
        for index in nl.static_range(3):
            result = nisa.nc_matmul(
                stationary=stationary[index], moving=moving,
                name=f"transpose_pipeline_mm_{index}",
            )
            sbuf = nisa.tensor_copy(
                result, dtype=kdtype, engine=nisa.engine.vector
            )
            nl.store(out[index], sbuf)
        return out

    return kernel, [(3, m, k), (k, n)], (1,)


def tensor_matmul_transpose_pipeline_work_units(
    *, m: int, k: int, n: int, repeat: int = 1,
    mode: str = "independent", dtype_name: str = "float32",
) -> dict[str, int]:
    return {
        "matmul_flops": 6 * m * n * k,
        "logical_instructions": 6,
        "dot_count": 3,
    }


def tensor_attention_pipeline_factory(
    *, dv: int, repeat: int = 1, mode: str = "dependent",
    dtype_name: str = "float32",
):
    """Independent QK-normalize-PV control for attention resource behavior.

    The control preserves the two dependent Dot phases and the intervening
    normalization/reuse edge, but owns its implementation and uses widths
    disjoint from the attention holdout.  It is a compiler-behavior control,
    not an operator replay or target-derived label.
    """
    del repeat, mode
    kernel_dtype_name = dtype_name

    @nki.jit
    def kernel(q, k, v):
        kdtype = dtype_for_load(kernel_dtype_name, q.dtype)
        q_tile = nl.load(q)
        k_tile = nl.load(k)
        v_tile = nl.load(v)

        q_t = nisa.tensor_copy(nisa.nc_transpose(q_tile), dtype=kdtype)
        k_t = nisa.tensor_copy(nisa.nc_transpose(k_tile), dtype=kdtype)

        scores_psum = nisa.nc_matmul(stationary=q_t, moving=k_t)
        scores = nisa.tensor_copy(scores_psum, dtype=nl.float32)
        row_max = nl.max(scores, axis=1, keepdims=True)
        exp_scores = nl.exp(nl.subtract(scores, row_max))
        row_sum = nl.sum(exp_scores, axis=1, keepdims=True)
        probs_t = nisa.tensor_copy(nisa.nc_transpose(exp_scores), dtype=kdtype)

        acc_psum = nisa.nc_matmul(stationary=probs_t, moving=v_tile)
        acc = nisa.tensor_copy(acc_psum, dtype=nl.float32)
        normalized = nl.divide(acc, row_sum)
        out = nl.ndarray((128, dv), dtype=kdtype, buffer=nl.shared_hbm)
        nl.store(out, value=nisa.tensor_copy(normalized, dtype=kdtype))
        return out

    return kernel, [(128, 128), (128, 128), (128, dv)], (1,)


def tensor_attention_pipeline_work_units(
    *, dv: int, repeat: int = 1, mode: str = "dependent",
    dtype_name: str = "float32",
) -> dict[str, int]:
    return {
        "matmul_flops": 2 * 128 * 128 * (128 + dv),
        "logical_instructions": 12,
        "dot_count": 2,
        "attention_value_width": dv,
    }


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
