#!/usr/bin/env python
"""Sanitizer performance benchmark for CI.

Runs a set of Triton kernels through the SymbolicSanitizer and records
wall-clock timing.  Three modes:

  Run (default)          → execute benchmarks, write JSON
  Compare                → compare two JSON files, emit markdown table
  Compare-single         → emit markdown table from a single JSON file

Usage:
  python benchmarks/bench_sanitizer.py --output results.json
  python benchmarks/bench_sanitizer.py --compare base.json pr.json
  python benchmarks/bench_sanitizer.py --compare-single pr.json
"""

from __future__ import annotations

import argparse
import json
import platform
import signal
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicSanitizer,
    _fn_symbolic_cache_set,
)

# ---------------------------------------------------------------------------
# Benchmark timeout helper
# ---------------------------------------------------------------------------

BENCH_TIMEOUT = 180  # seconds per benchmark


class BenchmarkTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise BenchmarkTimeout("Benchmark exceeded timeout")


# ---------------------------------------------------------------------------
# Sanitizer instances (one per kernel, following test_sanitizer.py pattern)
# ---------------------------------------------------------------------------

simple_sanitizer = SymbolicSanitizer(abort_on_error=False)
gemm_sanitizer = SymbolicSanitizer(abort_on_error=False)
gemm_oob_sanitizer = SymbolicSanitizer(abort_on_error=False)
indirect_sanitizer = SymbolicSanitizer(abort_on_error=False)
nested_sanitizer = SymbolicSanitizer(abort_on_error=False)
block_ptr_sanitizer = SymbolicSanitizer(abort_on_error=False)
jsd_sanitizer = SymbolicSanitizer(abort_on_error=False)
element_mul_sanitizer = SymbolicSanitizer(abort_on_error=False)
flaggems_sanitizer = SymbolicSanitizer(abort_on_error=False)

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton_viz.trace(client=simple_sanitizer)
@triton.jit
def simple_load_store_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


@triton_viz.trace(client=gemm_sanitizer)
@triton.jit
def gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    range_m = tl.arange(0, TILE_SIZE)
    range_n = tl.arange(0, TILE_SIZE)
    range_k = tl.arange(0, TILE_SIZE)
    range_m_block = TILE_SIZE * m_block + range_m[:, None]
    range_n_block = TILE_SIZE * n_block + range_n[None, :]
    accum = tl.zeros((TILE_SIZE, TILE_SIZE), dtype=tl.float32)
    for k_block in range(K // TILE_SIZE):
        range_k_block = TILE_SIZE * k_block + range_k
        A_off = K * range_m_block + range_k_block[None, :]
        A_tile = tl.load(A + A_off)
        B_off = N * range_k_block[:, None] + range_n_block
        B_tile = tl.load(B + B_off)
        accum += tl.dot(A_tile, B_tile, allow_tf32=False)
    C_off = N * range_m_block + range_n_block
    tl.store(C + C_off, accum)


@triton_viz.trace(client=gemm_oob_sanitizer)
@triton.jit
def gemm_oob_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    range_m = tl.arange(0, TILE_SIZE)
    range_n = tl.arange(0, TILE_SIZE)
    range_k = tl.arange(0, TILE_SIZE)
    range_m_block = TILE_SIZE * m_block + range_m[:, None]
    range_n_block = TILE_SIZE * n_block + range_n[None, :]
    accum = tl.zeros((TILE_SIZE, TILE_SIZE), dtype=tl.float32)
    for k_block in range(K // TILE_SIZE):
        range_k_block = TILE_SIZE * k_block + range_k
        A_off = K * range_m_block + range_k_block[None, :]
        A_tile = tl.load(A + A_off + 1)  # intentional OOB
        B_off = N * range_k_block[:, None] + range_n_block
        B_tile = tl.load(B + B_off)
        accum += tl.dot(A_tile, B_tile, allow_tf32=False)
    C_off = N * range_m_block + range_n_block
    tl.store(C + C_off, accum)


@triton_viz.trace(client=indirect_sanitizer)
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    indices = tl.load(idx_ptr + offs)
    vals = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offs, vals)


@triton_viz.trace(client=nested_sanitizer)
@triton.jit
def nested_loop_kernel(out_ptr):
    for i in range(0, 4):
        for j in range(0, 8):
            idx = i * 8 + j
            tl.store(out_ptr + idx, idx)


@triton_viz.trace(client=block_ptr_sanitizer)
@triton.jit
def block_pointer_loop_advance_kernel(ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    for _ in range(N // BLOCK):
        tl.load(block_ptr, boundary_check=(0,))
        block_ptr = tl.advance(block_ptr, (BLOCK,))


@triton_viz.trace(client=jsd_sanitizer)
@triton.jit
def jsd_kernel(
    X_ptr,  # input in logspace, X = log Q
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log P
    Y_stride,
    loss_ptr,
    loss_stride,
    dX_ptr,
    dX_stride,
    label_ptr,
    beta,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        Q = tl.exp(X)
        P = tl.exp(Y)
        M = beta * P + (1 - beta) * Q
        log_M = tl.log(M)

        loss = beta * P * Y + (1 - beta) * Q * X - M * log_M
        loss = loss / n_non_ignore
        tl.store(loss_ptr + offsets, loss, mask=mask)

        dX = (1 - beta) * Q * (X - log_M) / n_non_ignore
        tl.store(dX_ptr + offsets, dX, mask=mask)


@triton_viz.trace(client=element_mul_sanitizer)
@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride

    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


# ---------------------------------------------------------------------------
# FlagGems LayerNorm kernels
# Adapted from FlagGems/src/flag_gems/ops/layernorm.py
# ---------------------------------------------------------------------------


@triton_viz.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)
    var = sum_square / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b
    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@triton_viz.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_multiline_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)
    var = sum_square / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = (x - m[:, None]) * rstd[:, None] * w + b
    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@triton_viz.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    # Welford online mean/variance
    m = tl.zeros((TILE_N,), dtype=tl.float32)
    s = tl.zeros((TILE_N,), dtype=tl.float32)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # last step with masking
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    m = final_m
    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    # Normalize - reverse sweep (inlined prev_multiple_of)
    prev_multiple = tl.cdiv(N, TILE_N) * TILE_N - TILE_N
    # first step with masking
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out)


@triton_viz.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = (
        tl.program_id(0).to(tl.int64) * BLOCK_ROW_SIZE
        + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    )
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean).to(tl.float32)
    rstd = tl.load(Rstd).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@triton_viz.trace(client=flaggems_sanitizer)
@triton.jit
def flaggems_ln_wb_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = (
        tl.program_id(0).to(tl.int64) * BLOCK_COL_SIZE
        + tl.arange(0, BLOCK_COL_SIZE)[None, :]
    )
    col_mask = pid < N
    dY += pid
    X += pid
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)
        row_mask = rows[:, None] < M
        mask = row_mask & col_mask
        dy = tl.load(dY + rows[:, None] * N, mask).to(tl.float32)
        x = tl.load(X + rows[:, None] * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M)[:, None].to(tl.float32)
        x = tl.where(col_mask, x - mean, 0.0)
        x_hat = x * rstd
        accW += dy * x_hat
        accB += dy
    dw = tl.sum(accW, axis=0)
    tl.store(dW + pid, dw[None, :], mask=col_mask)
    db = tl.sum(accB, axis=0)
    tl.store(dB + pid, db[None, :], mask=col_mask)


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


def _reset_sanitizer(san: SymbolicSanitizer):
    """Clear sanitizer state and the global cache so re-runs are not skipped."""
    _fn_symbolic_cache_set.clear()
    san.records.clear()


BENCHMARKS: dict[str, dict[str, Any]] = {
    "simple_load_store": {
        "setup": lambda: {
            "inp": torch.randn(512, dtype=torch.float32),
            "out": torch.empty(512, dtype=torch.float32),
        },
        "run": lambda d: simple_load_store_kernel[(4,)](
            d["inp"], d["out"], N=512, BLOCK=128
        ),
        "sanitizer": simple_sanitizer,
    },
    "gemm": {
        "setup": lambda: {
            "A": torch.randn(64, 64, dtype=torch.float32),
            "B": torch.randn(64, 64, dtype=torch.float32),
            "C": torch.empty(64, 64, dtype=torch.float32),
        },
        "run": lambda d: gemm_kernel[(4, 4)](
            d["A"], d["B"], d["C"], M=64, N=64, K=64, TILE_SIZE=16
        ),
        "sanitizer": gemm_sanitizer,
    },
    "gemm_oob": {
        "setup": lambda: {
            "A": torch.randn(64, 64, dtype=torch.float32),
            "B": torch.randn(64, 64, dtype=torch.float32),
            "C": torch.empty(64, 64, dtype=torch.float32),
        },
        "run": lambda d: gemm_oob_kernel[(4, 4)](
            d["A"], d["B"], d["C"], M=64, N=64, K=64, TILE_SIZE=16
        ),
        "sanitizer": gemm_oob_sanitizer,
    },
    "indirect_load": {
        "setup": lambda: {
            "idx": torch.arange(256, dtype=torch.int32),
            "src": torch.randn(256, dtype=torch.float32),
            "dst": torch.empty(256, dtype=torch.float32),
        },
        "run": lambda d: indirect_load_kernel[(4,)](
            d["idx"], d["src"], d["dst"], BLOCK=64
        ),
        "sanitizer": indirect_sanitizer,
    },
    "nested_loop": {
        "setup": lambda: {
            "out": torch.empty(32, dtype=torch.int32),
        },
        "run": lambda d: nested_loop_kernel[(1,)](d["out"]),
        "sanitizer": nested_sanitizer,
    },
    "block_pointer_loop_advance": {
        "setup": lambda: {
            "data": torch.randn(256, dtype=torch.float32),
        },
        "run": lambda d: block_pointer_loop_advance_kernel[(1,)](
            d["data"], N=256, BLOCK=64
        ),
        "sanitizer": block_ptr_sanitizer,
    },
}

# ---------------------------------------------------------------------------
# Liger-Kernel fused_linear_jsd: 12 parameter combinations
# Matches test_correctness_functional parameter space exactly
# ---------------------------------------------------------------------------

_LIGER_SHAPES = [
    (2, 4, 2048, 3200),  # small
    (2, 2048, 4096, 32000),  # llama2/mistral
    (4, 423, 8192, 32000),  # random shape
]
_LIGER_DTYPES = [
    (0.5, torch.bfloat16),
    (0.5, torch.float32),
]
_LIGER_PARAMS = [
    (1.0, 0.5, -100),  # default
    (2.0, 0.1, 42),  # custom
]

MAX_FUSED_SIZE = 65536 // 2


def _make_liger_setup(BT, H, V, scalar, dtype, beta, ignore_index):
    def setup():
        BLOCK_SIZE_JSD = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
        student_log_probs = torch.log_softmax(
            torch.randn(BT, V, dtype=dtype) * scalar, dim=-1
        ).contiguous()
        teacher_log_probs = torch.log_softmax(
            torch.randn(BT, V, dtype=dtype) * scalar, dim=-1
        ).contiguous()
        loss = torch.zeros(BT, V, dtype=torch.float32)
        dX = torch.empty(BT, V, dtype=dtype)
        labels = torch.randint(0, V, (BT,), dtype=torch.long)
        labels[: max(1, BT // 2)] = ignore_index
        n_non_ignore = int((labels != ignore_index).sum().item())

        H_student = H // 2
        BLOCK_SIZE_MUL = min(MAX_FUSED_SIZE, triton.next_power_of_2(H_student))
        grad_input = torch.randn(BT, H_student, dtype=dtype).contiguous()
        grad_output = torch.tensor(0.5, dtype=dtype)

        return {
            "BT": BT,
            "V": V,
            "student_log_probs": student_log_probs,
            "teacher_log_probs": teacher_log_probs,
            "loss": loss,
            "dX": dX,
            "labels": labels,
            "BLOCK_SIZE_JSD": BLOCK_SIZE_JSD,
            "n_non_ignore": n_non_ignore,
            "beta": beta,
            "ignore_index": ignore_index,
            "grad_input": grad_input,
            "grad_output": grad_output,
            "H_student": H_student,
            "BLOCK_SIZE_MUL": BLOCK_SIZE_MUL,
        }

    return setup


def _liger_run(d):
    jsd_kernel[(d["BT"],)](
        X_ptr=d["student_log_probs"],
        X_stride=d["student_log_probs"].stride(0),
        Y_ptr=d["teacher_log_probs"],
        Y_stride=d["teacher_log_probs"].stride(0),
        loss_ptr=d["loss"],
        loss_stride=d["loss"].stride(0),
        dX_ptr=d["dX"],
        dX_stride=d["dX"].stride(0),
        label_ptr=d["labels"],
        beta=d["beta"],
        n_non_ignore=d["n_non_ignore"],
        ignore_index=d["ignore_index"],
        n_cols=d["V"],
        BLOCK_SIZE=d["BLOCK_SIZE_JSD"],
        HAS_LABEL=True,
    )
    element_mul_kernel[(d["BT"],)](
        d["grad_input"],
        d["grad_input"].stride(0),
        d["grad_output"],
        d["H_student"],
        BLOCK_SIZE=d["BLOCK_SIZE_MUL"],
    )


for _B, _T, _H, _V in _LIGER_SHAPES:
    for _scalar, _dtype in _LIGER_DTYPES:
        for _temperature, _beta, _ignore_index in _LIGER_PARAMS:
            _BT = _B * _T
            _dtype_str = "bf16" if _dtype == torch.bfloat16 else "f32"
            _param_str = "default" if _beta == 0.5 else "custom"
            _name = f"liger_jsd_{_BT}x{_V}_{_dtype_str}_{_param_str}"
            BENCHMARKS[_name] = {
                "setup": _make_liger_setup(
                    _BT, _H, _V, _scalar, _dtype, _beta, _ignore_index
                ),
                "run": _liger_run,
                "sanitizer": jsd_sanitizer,
            }

# ---------------------------------------------------------------------------
# FlagGems LayerNorm: 30 parameter combinations
# Matches test_accuracy_layernorm parameter space exactly
# ---------------------------------------------------------------------------

_FLAGGEMS_LN_SHAPES = [
    (200, 36),
    (4096, 100),
    (1, 40999),
    (100, 40499),
    (4096, 256),
]
_FLAGGEMS_LN_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
_FLAGGEMS_LN_WB = [False, True]  # wb_none values

_LN_BWD_BLOCK_ROW = 4
_LN_BWD_BLOCK_COL = 1024
_LN_WB_BWD_BLOCK_ROW = 8
_LN_WB_BWD_BLOCK_COL = 1024
_LN_LOOP_TILE_N = 1024


def _make_flaggems_ln_setup(M, N, dtype, wb_none):
    def setup():
        inp = torch.randn(M, N, dtype=dtype)
        out = torch.empty_like(inp)
        mean = torch.empty(M, dtype=torch.float32)
        rstd = torch.empty(M, dtype=torch.float32)
        weight = None if wb_none else torch.randn(N, dtype=dtype)
        bias = None if wb_none else torch.randn(N, dtype=dtype)

        # Backward tensors
        out_grad = torch.randn(M, N, dtype=dtype)
        in_grad = torch.empty_like(inp)
        weight_grad = None if wb_none else torch.empty(N, dtype=dtype)
        bias_grad = None if wb_none else torch.empty(N, dtype=dtype)

        # Forward grid and tile sizes
        if N <= 128:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = triton.cdiv(1024, TILE_N)
            fwd_grid = (triton.cdiv(M, TILE_M),)
            kernel_variant = "multiline"
        elif N <= 4096:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = 0
            fwd_grid = (M,)
            kernel_variant = "persistent"
        else:
            TILE_N = _LN_LOOP_TILE_N
            TILE_M = 0
            fwd_grid = (M,)
            kernel_variant = "loop"

        bwd_grid = (triton.cdiv(M, _LN_BWD_BLOCK_ROW),)
        wb_bwd_grid = (triton.cdiv(N, _LN_WB_BWD_BLOCK_COL),)

        return {
            "M": M,
            "N": N,
            "inp": inp,
            "out": out,
            "mean": mean,
            "rstd": rstd,
            "weight": weight,
            "bias": bias,
            "out_grad": out_grad,
            "in_grad": in_grad,
            "weight_grad": weight_grad,
            "bias_grad": bias_grad,
            "kernel_variant": kernel_variant,
            "TILE_N": TILE_N,
            "TILE_M": TILE_M,
            "fwd_grid": fwd_grid,
            "bwd_grid": bwd_grid,
            "wb_bwd_grid": wb_bwd_grid,
            "wb_none": wb_none,
        }

    return setup


def _flaggems_ln_run(d):
    M, N = d["M"], d["N"]
    eps = 1e-5

    # Forward
    if d["kernel_variant"] == "multiline":
        flaggems_ln_multiline_kernel[d["fwd_grid"]](
            d["inp"],
            d["out"],
            d["weight"],
            d["bias"],
            d["mean"],
            d["rstd"],
            M,
            N,
            eps,
            d["TILE_M"],
            d["TILE_N"],
        )
    elif d["kernel_variant"] == "persistent":
        flaggems_ln_persistent_kernel[d["fwd_grid"]](
            d["inp"],
            d["out"],
            d["weight"],
            d["bias"],
            d["mean"],
            d["rstd"],
            M,
            N,
            eps,
            d["TILE_N"],
        )
    else:
        flaggems_ln_loop_kernel[d["fwd_grid"]](
            d["inp"],
            d["out"],
            d["weight"],
            d["bias"],
            d["mean"],
            d["rstd"],
            M,
            N,
            eps,
            d["TILE_N"],
        )

    # Backward
    flaggems_ln_backward_kernel[d["bwd_grid"]](
        d["out_grad"],
        d["inp"],
        d["weight"],
        d["mean"],
        d["rstd"],
        d["in_grad"],
        M,
        N,
        BLOCK_ROW_SIZE=_LN_BWD_BLOCK_ROW,
        BLOCK_COL_SIZE=_LN_BWD_BLOCK_COL,
    )
    if not d["wb_none"]:
        flaggems_ln_wb_backward_kernel[d["wb_bwd_grid"]](
            d["out_grad"],
            d["inp"],
            d["mean"],
            d["rstd"],
            d["weight_grad"],
            d["bias_grad"],
            M,
            N,
            BLOCK_ROW_SIZE=_LN_WB_BWD_BLOCK_ROW,
            BLOCK_COL_SIZE=_LN_WB_BWD_BLOCK_COL,
        )


_DTYPE_NAMES = {torch.float16: "f16", torch.float32: "f32", torch.bfloat16: "bf16"}

for _M, _N in _FLAGGEMS_LN_SHAPES:
    for _dtype in _FLAGGEMS_LN_DTYPES:
        for _wb_none in _FLAGGEMS_LN_WB:
            _dtype_str = _DTYPE_NAMES[_dtype]
            _wb_str = "no_wb" if _wb_none else "wb"
            _name = f"flaggems_ln_{_M}x{_N}_{_dtype_str}_{_wb_str}"
            BENCHMARKS[_name] = {
                "setup": _make_flaggems_ln_setup(_M, _N, _dtype, _wb_none),
                "run": _flaggems_ln_run,
                "sanitizer": flaggems_sanitizer,
            }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmarks(
    warmup: int = 1,
    iterations: int = 5,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "iterations": iterations,
        "warmup_iterations": warmup,
        "benchmarks": {},
    }

    for name, bench in BENCHMARKS.items():
        print(f"  [{name}] ", end="", flush=True)

        data = bench["setup"]()
        san = bench["sanitizer"]

        # Set per-benchmark timeout (Unix only)
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(BENCH_TIMEOUT)

        try:
            # Warmup
            for _ in range(warmup):
                _reset_sanitizer(san)
                bench["run"](data)

            # Measured iterations
            times: list[float] = []
            for _ in range(iterations):
                _reset_sanitizer(san)
                t0 = time.perf_counter()
                bench["run"](data)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            results["benchmarks"][name] = {
                "times": times,
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "stddev": statistics.stdev(times) if len(times) > 1 else 0.0,
            }
            median_str = f"{statistics.median(times):.3f}s"
            print(f"{median_str}")

        except BenchmarkTimeout:
            print(f"TIMEOUT (>{BENCH_TIMEOUT}s)")
            results["benchmarks"][name] = {
                "times": [],
                "mean": None,
                "median": None,
                "min": None,
                "stddev": None,
                "error": f"Timeout after {BENCH_TIMEOUT}s",
            }

        except Exception as e:
            print(f"ERROR: {e}")
            results["benchmarks"][name] = {
                "times": [],
                "mean": None,
                "median": None,
                "min": None,
                "stddev": None,
                "error": str(e),
            }

        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

REGRESSION_THRESHOLD = 0.15  # 15%


def _fmt_time(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.3f}s"


def _fmt_change(base_val: float | None, pr_val: float | None) -> str:
    if base_val is None or pr_val is None:
        return "N/A"
    if base_val == 0:
        return "N/A"
    pct = (pr_val - base_val) / base_val
    sign = "+" if pct >= 0 else ""
    flag = " :warning:" if pct > REGRESSION_THRESHOLD else ""
    return f"{sign}{pct:.1%}{flag}"


def generate_comparison(base: dict, pr: dict) -> str:
    lines = [
        "## Sanitizer Performance Benchmark",
        "",
        "| Benchmark | main (median) | PR (median) | Change |",
        "|-----------|---------------|-------------|--------|",
    ]

    all_names = list(pr.get("benchmarks", {}).keys())
    base_total = 0.0
    pr_total = 0.0
    base_total_valid = True
    pr_total_valid = True

    for name in all_names:
        pr_bench = pr["benchmarks"].get(name, {})
        base_bench = base.get("benchmarks", {}).get(name, {})
        pr_med = pr_bench.get("median")
        base_med = base_bench.get("median")

        if pr_med is not None:
            pr_total += pr_med
        else:
            pr_total_valid = False
        if base_med is not None:
            base_total += base_med
        else:
            base_total_valid = False

        lines.append(
            f"| {name} | {_fmt_time(base_med)} | {_fmt_time(pr_med)} "
            f"| {_fmt_change(base_med, pr_med)} |"
        )

    # Total row
    bt = base_total if base_total_valid else None
    pt = pr_total if pr_total_valid else None
    lines.append(
        f"| **Total** | **{_fmt_time(bt)}** | **{_fmt_time(pt)}** "
        f"| {_fmt_change(bt, pt)} |"
    )

    lines.append("")
    lines.append(
        f"_Threshold: >{REGRESSION_THRESHOLD:.0%} regression flagged with :warning:_"
    )
    lines.append(
        f"_Iterations: {pr.get('warmup_iterations', '?')} warmup + {pr.get('iterations', '?')} measured_"
    )
    return "\n".join(lines)


def generate_single(pr: dict) -> str:
    lines = [
        "## Sanitizer Performance Benchmark",
        "",
        "_No baseline available (benchmark script not present on main)._",
        "",
        "| Benchmark | PR (median) |",
        "|-----------|-------------|",
    ]

    pr_total = 0.0
    pr_total_valid = True

    for name, bench in pr.get("benchmarks", {}).items():
        med = bench.get("median")
        if med is not None:
            pr_total += med
        else:
            pr_total_valid = False
        lines.append(f"| {name} | {_fmt_time(med)} |")

    pt = pr_total if pr_total_valid else None
    lines.append(f"| **Total** | **{_fmt_time(pt)}** |")

    lines.append("")
    lines.append(
        f"_Iterations: {pr.get('warmup_iterations', '?')} warmup + {pr.get('iterations', '?')} measured_"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Sanitizer performance benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE_JSON", "PR_JSON"),
        help="Compare two result files and print markdown",
    )
    parser.add_argument(
        "--compare-single",
        metavar="PR_JSON",
        help="Print markdown table from a single result file (no baseline)",
    )

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        with open(args.compare[0]) as f:
            base = json.load(f)
        with open(args.compare[1]) as f:
            pr = json.load(f)
        print(generate_comparison(base, pr))
        return

    # Single mode
    if args.compare_single:
        with open(args.compare_single) as f:
            pr = json.load(f)
        print(generate_single(pr))
        return

    # Run mode
    print("Running sanitizer benchmarks...")
    results = run_benchmarks(warmup=args.warmup, iterations=args.iterations)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
