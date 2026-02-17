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
