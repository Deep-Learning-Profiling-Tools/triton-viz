"""M5 — the shared-memory (Track 1) evaluation sweep (plan Part II §7).

Sweeps the pipelined tutorial kernels across ``num_stages ∈ {1..4}`` at
sm80, recording per cell: the TTGIR verdict (proof / reports /
unsupported+kind), analyze wall-time, and async-op counts. Every PROVED
cell then enters the MUTATION-DETECTION MATRIX: the three pipeliner-bug
mutations (weakened wait, deleted wait, single-buffered rotation) are
applied to its TTGIR and the verdict must flip to RAW reports.

The two M5 case studies fall out of the matrix and are narrated with
their solver witnesses:

  CS1 "missing async_wait"      — the deleted-wait mutation: the loop's
                                  local_loads run with no wait coverage
                                  at all (the classic forgotten-wait bug).
  CS2 "insufficient buffering"  — the single-buffer mutation: rotation
                                  depth 1 under a 2-deep prefetch, i.e. a
                                  producer cp.async overwrites the slot a
                                  consumer is still reading (the
                                  insufficient-num_stages bug class).

sm90 is GATED on M4 (advisor Q5); this sweep is the sm80 half the paper's
Compiled-Mode Evaluation placeholder consumes.

Usage:  uv run python -m evaluation.shared_track
Writes results/SHARED_TRACK.md.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch  # noqa: F401  (imported for parity with the harness env)
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_viz.clients.race_detector.compiled.smt_encoder import analyze_ttgir

RESULTS_DIR = Path(__file__).parent / "results"
STAGES = (1, 2, 3, 4)


# ── kernels (the pipelined tutorials; vendored shapes from tutorials.py) ──


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_bk, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    """The tutorial matmul with the INNER strides folded to 1 (row-major),
    mirroring the real JIT's specialization of contiguous tensors — a
    runtime inner stride defeats the contiguity proof the pipeliner needs
    to emit cp.async, and the sweep would silently measure unpipelined
    code. Matches tests/golden/ttgir/generate_golden.py."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr,
):  # fmt: skip
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


_MATMUL_SIG = {
    "a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
    "M": "i32", "N": "i32", "K": "i32",
    "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32",
    "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
}  # fmt: skip
_SOFTMAX_SIG = {
    "output_ptr": "*fp32", "input_ptr": "*fp32",
    "input_row_stride": "i32", "output_row_stride": "i32",
    "n_rows": "i32", "n_cols": "i32",
    "BLOCK_SIZE": "constexpr", "num_stages": "constexpr",
}  # fmt: skip


# divisibility-16 on pointers + shape/stride scalars: mirrors the real
# JIT's specialization of well-aligned tensors — REQUIRED for the
# vectorized loads the pipeliner turns into cp.async (without it the
# sweep silently measures unpipelined code: 0 async copies everywhere).
_MATMUL_ATTRS = {(i,): [["tt.divisibility", 16]] for i in range(9)}
_SOFTMAX_ATTRS = {(i,): [["tt.divisibility", 16]] for i in range(6)}


def _kernels(stages: int):
    return (
        (
            "tut03_matmul",
            matmul_kernel,
            _MATMUL_SIG,
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            {"num_stages": stages, "num_warps": 4},
            _MATMUL_ATTRS,
        ),
        (
            "tut02_softmax",
            softmax_kernel,
            _SOFTMAX_SIG,
            {"BLOCK_SIZE": 128, "num_stages": stages},
            {"num_stages": stages, "num_warps": 4},
            _SOFTMAX_ATTRS,
        ),
    )


def _ttgir(fn, sig, consts, opts, attrs) -> str:
    src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs)
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32), options=opts)
    return k.asm["ttgir"]


# ── the pipeliner-bug mutations (matching the e2e mutation tests) ──

_RE_WAIT_NUM = re.compile(r"\{num = (\d+) : i32\}")


def _mut_weaken_wait(ttgir: str) -> str | None:
    """async_wait tolerating one MORE outstanding group than the rotation
    provides — the off-by-one pipeliner bug."""

    def repl(m: re.Match) -> str:
        return f"{{num = {int(m.group(1)) + 1} : i32}}"

    new, n = _RE_WAIT_NUM.subn(repl, ttgir)
    return new if n else None


def _mut_delete_wait(ttgir: str) -> str | None:
    """CS1 — the forgotten async_wait: loop local_loads run uncovered."""
    lines = ttgir.splitlines()
    kept = [ln for ln in lines if "ttg.async_wait %" not in ln]
    return "\n".join(kept) if len(kept) != len(lines) else None


def _mut_single_buffer(ttgir: str) -> str | None:
    """CS2 — insufficient buffering: shrink the rotation to depth 1 under
    the same prefetch distance, so the producer's next cp.async targets
    the very slot the consumer still reads (the insufficient-num_stages
    bug class). Well-formed shrink (generic form of the golden test's
    _shrink_to_single_buffer): buffer memdesc depth D→1, the rotation-wrap
    compare's D→1, every constant slot index →0. n/a when the pipeline is
    already single-buffered (num_stages=2 ⇒ depth 1)."""
    m = re.search(r"ttg\.local_alloc[^\n]*!ttg\.memdesc<(\d+)x", ttgir)
    if not m:
        return None
    depth = int(m.group(1))
    if depth < 2:
        return None
    out = ttgir.replace(f"memdesc<{depth}x", "memdesc<1x")
    # rotation wrap: `cmpi sge, %idx, %cD_i32` guards the modular reset
    out = re.sub(rf"(arith\.cmpi sge, %[\w.#]+, %c){depth}(_i32)", r"\g<1>1\g<2>", out)
    # prologue prefetches into slots 1..D-1: a depth-1 buffer has only 0
    out = re.sub(r"\[%c[1-9]\d*_i32\]", "[%c0_i32]", out)
    return out if out != ttgir else None


_MUTATIONS = (
    ("weaken_wait", _mut_weaken_wait),
    ("delete_wait", _mut_delete_wait),
    ("single_buffer", _mut_single_buffer),
)


# ── the sweep ─────────────────────────────────────────────────────


def _analyze(ttgir: str) -> tuple[str, int, str | None, float, list]:
    t0 = time.perf_counter()
    r = analyze_ttgir(ttgir)
    dt = time.perf_counter() - t0
    return r.status, len(r.reports), r.unsupported_reason, dt, r.reports


def sweep() -> str:
    lines = [
        "# M5 — shared-memory track evaluation (sm80)",
        "",
        "Track 1 (`analyze_ttgir`) over the pipelined tutorials at",
        "`num_stages ∈ {1..4}`, GPUTarget(cuda, 80). sm90 is gated on M4.",
        "",
        "## Sweep",
        "",
        "| kernel | stages | async copies | verdict | reports | analyze s |",
        "|---|---|---|---|---|---|",
    ]
    proved: list[tuple[str, int, str]] = []
    for stages in STAGES:
        for name, fn, sig, consts, opts, attrs in _kernels(stages):
            try:
                ttgir = _ttgir(fn, sig, consts, opts, attrs)
            except Exception as e:  # noqa: BLE001
                lines.append(
                    f"| {name} | {stages} | - | compile-error "
                    f"({type(e).__name__}) | - | - |"
                )
                continue
            n_async = ttgir.count("ttg.async_copy_global_to_local")
            status, n_reports, reason, dt, _ = _analyze(ttgir)
            verdict = (
                status
                if status != "unsupported"
                else f"unsupported: {(reason or '')[:60]}"
            )
            lines.append(
                f"| {name} | {stages} | {n_async} | {verdict} | {n_reports} "
                f"| {dt:.3f} |"
            )
            if status == "ok" and n_reports == 0 and n_async > 0:
                proved.append((name, stages, ttgir))

    lines += [
        "",
        "## Mutation-detection matrix (every proved pipelined cell)",
        "",
        "| kernel | stages | " + " | ".join(n for n, _ in _MUTATIONS) + " |",
        "|---|---|" + "---|" * len(_MUTATIONS),
    ]
    matrix_ok = True
    case_studies: dict[str, tuple[str, int, list]] = {}
    for name, stages, ttgir in proved:
        row = [name, str(stages)]
        for mut_name, mut in _MUTATIONS:
            mutated = mut(ttgir)
            if mutated is None:
                row.append("n/a")
                continue
            status, n_reports, reason, _, reports = _analyze(mutated)
            if status == "ok" and n_reports > 0:
                row.append(f"detected ({n_reports})")
                if mut_name == "delete_wait" and "CS1" not in case_studies:
                    case_studies["CS1"] = (name, stages, reports)
                if mut_name == "single_buffer" and "CS2" not in case_studies:
                    case_studies["CS2"] = (name, stages, reports)
            elif status == "ok":
                row.append("MISSED")
                matrix_ok = False
            else:
                row.append(f"abstained ({(reason or '')[:24]})")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        f"Matrix: {'every applicable mutation DETECTED' if matrix_ok else 'MISSES present — investigate'}.",
        "",
    ]

    # ── case studies ──
    lines += ["## Case studies (historical pipeliner bug classes)", ""]
    narr = {
        "CS1": (
            "Missing `async_wait` (the forgotten-wait bug): every loop "
            "`local_load` runs with no commit-group coverage at all — each "
            "prefetch's cp.async may still be in flight when its slot is "
            "read."
        ),
        "CS2": (
            "Insufficient buffering (the insufficient-`num_stages` bug "
            "class): the rotation is shrunk to a single slot under an "
            "unchanged prefetch distance, so the producer's next cp.async "
            "targets the very slot the consumer is still reading."
        ),
    }
    for cs in ("CS1", "CS2"):
        if cs not in case_studies:
            lines += [f"### {cs}: NOT CAPTURED — investigate", ""]
            continue
        name, stages, reports = case_studies[cs]
        rep = reports[0]
        w = rep.witness
        lines += [
            f"### {cs} — {name} @ num_stages={stages}",
            "",
            narr[cs],
            "",
            f"- verdict: RAW race, {len(reports)} report(s)",
            f"- witness: copy "
            f"{'prologue prefetch' if w.get('k_copy', -1) < 0 else 'iteration k_copy=' + str(w['k_copy'])}, "
            f"load iteration k_load={w['k_load']}, shared-memory slot "
            f"{w['slot']}"
            + (
                f", byte offset {rep.byte_offset}"
                if getattr(rep, "byte_offset", None) is not None
                else ""
            ),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    out = sweep()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "SHARED_TRACK.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
