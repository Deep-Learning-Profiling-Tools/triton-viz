#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 NKI beginner tutorial + Triton-Viz trace end-to-end demo (single runnable file)
================================================================================

This script targets readers who have never used NKI before. After reading the
comments and running it once, you should understand:

  1. What NKI is and what its programming model looks like
     (tile / SBUF / PSUM / HBM / multiple engines).
  2. The most common NKI syntax: program_id / arange / load / store / mask /
     matmul / loops.
  3. How Triton-Viz "intercepts" every operator in an NKI kernel to build a
     trace (a record of what executed).
  4. How to persist a trace to a file and turn it into a "performance event
     stream + predicted timeline".

About hardware: this uses the Triton-Viz CPU functional interpreter (backed by
  NumPy). It can "run" an NKI kernel and produce a trace WITHOUT a real AWS
  Trainium / Inferentia chip. The trade-off: the computed values are correct,
  but the timings are model estimates (real latency must be calibrated with the
  AWS neuron-profile tool on real hardware).

--------------------------------------------------------------------------------
 Background: the NKI mental model (30-second version)
--------------------------------------------------------------------------------
NKI = Neuron Kernel Interface, a tile-level Python DSL for AWS Trainium /
Inferentia chips. It reads a lot like Triton: you program over a tile (a small
matrix) rather than over individual scalars.

Key structure of a NeuronCore chip (keep this picture in mind when writing a
kernel):

    HBM  (off-chip, large, slow)    <-- input/output tensors start here
     |  ^
     |  |  the DMA engine moves data (HBM <-> SBUF), overlapping with compute
     v  |
    SBUF (on-chip, fast cache)      <-- a tile must be moved here before compute
     |  ^
     v  |  compute engines read/write
    PSUM (matmul-only accumulation) <-- TensorE matmul results land here first

    There are 4 compute engines (this tutorial mainly uses the first two ideas):
      - TensorE  : matrix multiply (nl.matmul); ~90% of the chip's FLOPs are here
      - VectorE  : element-wise vector ops (add/sub/mul, reduce, ...)
      - ScalarE  : activation functions, etc.
      - GpSimdE  : general-purpose / miscellaneous

In one sentence: move data HBM -> SBUF -> compute on an engine -> move the
result back to HBM. That is exactly what the two kernels below do.

--------------------------------------------------------------------------------
 How to run
--------------------------------------------------------------------------------
  # Option A (recommended): the one-shot script creates the venv, installs deps,
  # and runs this file automatically:
  bash examples/nki_tutorial/run_tutorial.sh

  # Option B (you already activated a venv that has nki installed):
  python examples/nki_tutorial/nki_trace_tutorial.py

After running, these files are produced under ./nki_tutorial_out/:
  - vector_add.tvz       : trace archive of the vector-add kernel (openable in
                           the triton-viz visualizer)
  - matmul.tvz           : trace archive of the matmul kernel
  - matmul_events.jsonl  : the trace converted into a "unified performance event
                           stream" (one operator event per line)
  - matmul_timeline.json : a "per-engine predicted timeline" estimated with a
                           placeholder cost model
================================================================================
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------
# Dependency imports. A ModuleNotFoundError here means the current environment
# does not have nki / triton-viz installed. Prefer the one-shot script
# run_tutorial.sh, which prepares the environment for you.
# ------------------------------------------------------------------------------
try:
    import triton_viz
    from triton_viz.clients import Tracer
    from triton_viz.core.trace import launches
    # Imported for teaching: these are the normalized trace record types.
    from triton_viz.core.data import Dot, Grid, Load, Store  # noqa: F401

    # We use the official NKI language namespace neuronxcc.nki.language (aliased
    # as nl). This is the most common form in AWS's official tutorials and is the
    # most beginner-friendly syntax.
    import neuronxcc.nki.language as nl

    # NDArray is the tensor container used by the Triton-Viz NKI interpreter
    # (backed by a numpy array). On real NKI you would pass PyTorch/XLA tensors;
    # in CPU simulation we use NDArray instead.
    from triton_viz.core.simulation.nki import NDArray
except ModuleNotFoundError as exc:  # pragma: no cover - friendly error message
    raise SystemExit(
        "\n[缺少依赖] 没有找到 nki / triton-viz。\n"
        "请用一键脚本运行（它会自动建虚拟环境并安装依赖）：\n"
        "    bash examples/nki_tutorial/run_tutorial.sh\n"
        f"原始错误: {exc}\n"
    )


OUT_DIR = Path(__file__).resolve().parent.parent.parent / "nki_tutorial_out"


# ==============================================================================
# Tutorial kernel 1: element-wise vector add  out = x + y
# ------------------------------------------------------------------------------
# Goal: use the simplest possible example to explain the NKI "four basics":
#       program_id / arange / load / store, plus the very important mask
#       (out-of-bounds protection).
#
# Key concept: SPMD + grid
#   Like CUDA/Triton, an NKI kernel is launched as many instances (the grid
#   decides how many). Each instance is a "program", and program_id tells each
#   one which slice of data it owns.
#   Example: grid=(3,) launches 3 instances; program_id(0) is 0, 1, 2.
# ==============================================================================
#
# What the @triton_viz.trace(...) decorator does:
#   - client=Tracer()  : the "recorder" client that logs each operator as one
#                        trace record.
#   - frontend="nki"   : tells Triton-Viz this is an NKI kernel (executed by the
#                        NKI interpreter).
# Once wrapped, every nl.load / nl.store / ... inside the kernel is intercepted
# and recorded.
@triton_viz.trace(client=Tracer(), frontend="nki")
def vector_add_kernel(x_ptr, y_ptr, out_ptr):
    """Add x and y element-wise into out; each program handles BLOCK_SIZE elems."""
    BLOCK_SIZE = 4  # number of elements each program handles (the tile size)

    # 1) Which program am I? (along grid axis 0)
    pid = nl.program_id(axis=0)

    # 2) What are the global indices of my BLOCK_SIZE elements?
    #    nl.arange(BLOCK_SIZE) = [0,1,2,3]; adding pid*BLOCK_SIZE yields this
    #    program's absolute indices.
    #    e.g. pid=2, BLOCK_SIZE=4 -> offsets = [8,9,10,11]
    offsets = pid * BLOCK_SIZE + nl.arange(BLOCK_SIZE)

    # 3) mask: out-of-bounds protection. The array length may not be divisible by
    #    BLOCK_SIZE, so the last program may address elements "past the end". Only
    #    positions where mask=True are actually read/written.
    mask = offsets < x_ptr.shape[0]

    # 4) load: read data from HBM at the given offsets (masked-out positions are
    #    not read).
    x = nl.load(x_ptr[offsets], mask=mask)
    y = nl.load(y_ptr[offsets], mask=mask)

    # 5) Element-wise compute (this maps to VectorE on hardware).
    #    The current Triton-Viz NKI interpreter may compute unused placeholder
    #    values at masked positions, which can make NumPy emit an overflow
    #    warning; this does not affect the real results inside the mask. main()
    #    filters out that tutorial-irrelevant warning.
    result = x + y

    # 6) store: write results back to HBM at the given offsets (masked-out
    #    positions are not written).
    nl.store(out_ptr[offsets], result, mask=mask)


# ==============================================================================
# Tutorial kernel 2: tiled matrix multiply  result = lhs @ rhs
# ------------------------------------------------------------------------------
# Goal: explain the core NKI pattern -- tiling + PSUM accumulation + matmul --
#       which is the foundation of real AI kernels (GEMM/Attention).
#
# Idea: cut the large matrices into small tiles, multiply tile by tile, and
#       accumulate.
#   result[M,N] = lhs[M,K] @ rhs[K,N]
#   Tile the output along M and N; accumulate along K (the "contraction" dim).
# ==============================================================================
@triton_viz.trace(client=Tracer(), frontend="nki")
def matmul_kernel(lhs, rhs, result):
    """Compute result = lhs @ rhs; uses tiny tiles to demo tiling + PSUM accum."""
    # Tiles are intentionally tiny so you can follow the math by hand.
    TILE_M, TILE_K, TILE_N = 2, 2, 4

    M, K = lhs.shape
    K_, N = rhs.shape
    assert K == K_, "lhs columns must equal rhs rows"

    # Two outer loops: iterate over every [TILE_M, TILE_N] block of the output.
    # nl.affine_range is NKI's loop construct (a for-range the compiler analyzes).
    for m in nl.affine_range(math.ceil(M / TILE_M)):
        for n in nl.affine_range(math.ceil(N / TILE_N)):
            # PSUM is the matmul-only accumulation buffer. A single output block
            # sums the partial products from all K sub-tiles, so start by zeroing
            # it in PSUM.
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            # Inner loop: tile along the contraction dim K; multiply each pair of
            # sub-tiles and accumulate into PSUM.
            for k in nl.affine_range(math.ceil(K / TILE_K)):
                # ---- Build the row/col indices of this lhs sub-tile (both 2-D) --
                # [:, None] and [None, :] are numpy-style broadcasting used to
                # form a 2-D index grid.
                lhs_rows = nl.arange(TILE_M)[:, None] + m * TILE_M   # shape [TILE_M, 1]
                lhs_cols = nl.arange(TILE_K)[None, :] + k * TILE_K   # shape [1, TILE_K]
                lhs_mask = (lhs_rows < M) & (lhs_cols < K)
                lhs_tile = nl.load(lhs[lhs_rows, lhs_cols], mask=lhs_mask)

                # ---- rhs sub-tile ----
                rhs_rows = nl.arange(TILE_K)[:, None] + k * TILE_K
                rhs_cols = nl.arange(TILE_N)[None, :] + n * TILE_N
                rhs_mask = (rhs_rows < K) & (rhs_cols < N)
                rhs_tile = nl.load(rhs[rhs_rows, rhs_cols], mask=rhs_mask)

                # ---- Core: tile-level matrix multiply (runs on TensorE) ----
                # Accumulate into PSUM: res_psum += lhs_tile @ rhs_tile
                res_psum += nl.matmul(lhs_tile[...], rhs_tile[...], transpose_x=False)

            # Copy the accumulated PSUM result into SBUF, casting to the output
            # dtype.
            res_sb = nl.copy(res_psum, dtype=result.dtype)

            # Write back into the matching block of result (mask guards the ragged
            # edges when sizes are not divisible by the tile sizes).
            out_rows = nl.arange(TILE_M)[:, None] + m * TILE_M
            out_cols = nl.arange(TILE_N)[None, :] + n * TILE_N
            nl.store(
                result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
                value=res_sb,
                mask=(out_rows < M) & (out_cols < N),
            )


# ==============================================================================
# Below is the "driver code": prepare data -> run kernel -> check correctness ->
# save/show the trace.
# ==============================================================================
def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _summarize_records(records) -> dict:
    """Count how many times each operator type appears in the trace.

    Gives an at-a-glance view of what the kernel actually did.
    """
    counts: dict[str, int] = {}
    for r in records:
        counts[type(r).__name__] = counts.get(type(r).__name__, 0) + 1
    return counts


def run_vector_add() -> None:
    _print_header("第 1 步 / Kernel 1：向量加法 out = x + y  ->  生成 trace")

    # Deliberately use 10 (not a multiple of 4) so you can see how the mask
    # handles the final partial block.
    n = 10
    x = NDArray(value=np.arange(n, dtype=np.float32))          # [0,1,...,9]
    y = NDArray(value=np.arange(n, dtype=np.float32) * 10.0)   # [0,10,...,90]
    out = NDArray(value=np.empty(n, dtype=np.float32))

    BLOCK_SIZE = 4
    grid = (math.ceil(n / BLOCK_SIZE),)  # need ceil(10/4)=3 programs
    print(f"输入 x = {x.data}")
    print(f"输入 y = {y.data}")
    print(f"grid = {grid}  (启动 {grid[0]} 个 program，每个处理 {BLOCK_SIZE} 个元素)")

    # Actually execute: launch with the kernel[grid](args) syntax, like Triton.
    vector_add_kernel[grid](x, y, out)

    expected = x.data + y.data
    ok = np.allclose(out.data, expected)
    print(f"输出 out = {out.data}")
    print(f"正确性检查: {'通过 OK' if ok else '失败 FAIL'} (与 numpy x+y 对比)")
    assert ok, "vector add result is incorrect!"

    records = launches[-1].records
    print(f"这次执行记录了 {len(records)} 条 trace 记录，分类统计: {_summarize_records(records)}")
    print("上面每个 program 都产生了 Grid/Load/Load/Store，一共 3 个 program。")

    out_path = OUT_DIR / "vector_add.tvz"
    triton_viz.save(out_path)
    print(f"[saved] trace 已保存: {out_path}")


def run_matmul():
    _print_header("第 2 步 / Kernel 2：矩阵乘法 result = lhs @ rhs  ->  生成 trace")

    lhs = np.arange(16, dtype=np.float32).reshape(4, 4)
    rhs = np.arange(32, dtype=np.float32).reshape(4, 8)
    result = np.empty((4, 8), dtype=np.float32)
    print(f"lhs 形状 {lhs.shape}, rhs 形状 {rhs.shape}, result 形状 {result.shape}")

    # The matmul kernel does not split data via program_id (it tiles with inner
    # loops), so a grid of (1,1,1) is enough.
    matmul_kernel[(1, 1, 1)](lhs, rhs, result)

    expected = lhs @ rhs
    ok = np.allclose(result, expected)
    print(f"正确性检查: {'通过 OK' if ok else '失败 FAIL'} (与 numpy lhs@rhs 对比)")
    assert ok, "matmul result is incorrect!"

    records = launches[-1].records
    n_dot = sum(isinstance(r, Dot) for r in records)
    n_load = sum(isinstance(r, Load) for r in records)
    n_store = sum(isinstance(r, Store) for r in records)
    print(f"trace 统计: {_summarize_records(records)}")
    print(f"其中矩阵乘法算子(Dot)={n_dot}, Load={n_load}, Store={n_store}")
    print("  （输出 4x8 切成 2x4 的块 = 2*2=4 个输出块，每块沿 K 累加 2 次 -> 8 次 matmul）")

    out_path = OUT_DIR / "matmul.tvz"
    triton_viz.save(out_path)
    print(f"[saved] trace 已保存: {out_path}")
    return records


def run_perf_pipeline(records) -> None:
    """Advanced (optional): turn the trace into an event stream + timeline.

    This step shows a downstream use of the trace: besides visualization, it can
    feed a performance model. If the helper modules do not exist (older repo
    checkout), it is skipped automatically without affecting the main flow.
    """
    _print_header("第 3 步（进阶）：trace -> 性能事件流 -> 预测每引擎时间线")
    try:
        from triton_viz.tools.nki_trace_dump import write_jsonl, summarize_events
        from triton_viz.tools.nki_cost_model import simulate_jsonl
    except ModuleNotFoundError:
        print("（本仓库暂无 triton_viz.tools 性能工具，跳过这一步。主流程已完成。）")
        return

    events_path = OUT_DIR / "matmul_events.jsonl"
    events = write_jsonl(records, events_path)
    summary = summarize_events(events)
    print(f"[saved] 性能事件流已保存: {events_path}")
    print(f"事件摘要: {json.dumps(summary, ensure_ascii=False)}")
    print("事件流把每个算子标注成 引擎/内存搬运/字节数/FLOPs，是性能模拟器的输入。")
    if summary.get("bytes_by_edge"):
        print("说明：本教程使用 frontend=\"nki\"，它记录 Load/Store 的 offsets/mask，")
        print("      bytes 按有效 mask 元素数 × dtype 字节数统计，是逻辑访存量；")
        print("      它不等于考虑 cache line、合并访存后的实际 HBM 事务字节数。")

    result = simulate_jsonl(events_path)
    timeline_path = OUT_DIR / "matmul_timeline.json"
    timeline_path.write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] 预测时间线已保存: {timeline_path}")
    print(f"预测总延迟(占位常数, ns): {result.as_dict()['predicted_latency_ns']}")
    print(f"各引擎利用率(占位): {result.as_dict()['engine_utilization']}")
    print("注意：这里的时间是【占位常数】估算，不是真机数字；")
    print("   真实延迟需要 AWS neuron-profile 在 Trainium/Inferentia 上标定。")


def main() -> None:
    # Filter one known, tutorial-irrelevant warning: placeholder computation at
    # masked positions can trigger a NumPy overflow warning, but the real output
    # inside the mask is already validated by the numpy correctness checks.
    warnings.filterwarnings("ignore", message="overflow encountered in add", category=RuntimeWarning)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Triton-Viz 版本:", getattr(triton_viz, "__version__", "unknown"))
    print("输出目录:", OUT_DIR)

    run_vector_add()
    mm_records = run_matmul()
    run_perf_pipeline(mm_records)

    _print_header("全部完成")
    print("你已经跑通了 NKI kernel 的 trace 生成全流程。生成的文件在:")
    print(f"  {OUT_DIR}")
    print("\n可选：用可视化器打开 trace（会启动一个本地网页）：")
    print("  # 在已激活的虚拟环境里：")
    print("  python -c \"import triton_viz; triton_viz.load('nki_tutorial_out/matmul.tvz'); triton_viz.launch(share=False)\"")
    print("\n想继续学 NKI，可修改本文件里的 kernel（改 tile 大小 / 换算子）再跑一遍看看 trace 变化。")


if __name__ == "__main__":
    main()
