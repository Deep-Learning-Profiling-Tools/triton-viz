"""One-time GPU launch capture for the tilebench corpus
(Deep-Learning-Profiling-Tools/Tilebench — the group's own multi-backend
tile-DSL benchmark; we sweep its Triton twin implementations, one per
operator, as the baseline for the future cuTile frontend).

TileBench is analyzed AS A LOCAL GIT CHECKOUT (it has no packaging
metadata, so it cannot be pip-pinned like the other corpora):
``TILEBENCH_ROOT`` (env override; default ~/workspace/Tilebench) goes on
sys.path and the checkout HEAD commit is recorded as the pin — capture
refuses a tree with tracked modifications, and the corpus module refuses
commit drift at rebuild time.

Capture is harness-driven like tritonbench_meta: each case runs the
suite's own ``core.engine.run_benchmark_suite(op)`` with
``case_indices=[0]`` (first expanded config case, mirroring
tritonbench's ``--input-id 0`` convention), with
``core.engine.report_benchmark`` monkeypatched to a stub — so the ONLY
Triton launch is the engine's verification run on a normal stream; the
Proton/CUDA-graph timing path never executes, which keeps the recorder's
tensor reads (dedup fingerprints, value snapshots) off a graph-capturing
stream. ``autotune`` stays False (the config default): every
``impl_triton.run`` then calls its raw ``@triton.jit`` kernel with the
module's ``_DEFAULT_CONFIG`` — one deterministic launch per kernel, no
autotuner sweep. The engine's cuTile backend import-fails harmlessly
(cuda-tile is not installed in this venv) and the recorder only hooks
Triton's JITFunction, so only the Triton twins are recorded either way.

Usage (GPU machine):
    uv run python -m evaluation.tilebench_capture           # all
    uv run python -m evaluation.tilebench_capture --one <case> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SPECS_PATH = Path(__file__).parent / "kernels" / "tilebench_specs.json"
PER_CASE_TIMEOUT_S = 600
UPSTREAM = "https://github.com/Deep-Learning-Profiling-Tools/Tilebench"

TILEBENCH_ROOT = Path(
    os.environ.get("TILEBENCH_ROOT", str(Path.home() / "workspace" / "Tilebench"))
)


def tilebench_commit() -> str:
    """HEAD of the TileBench checkout; refuses tracked-dirty trees so the
    recorded commit is a real pin (untracked files are fine)."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(TILEBENCH_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(
            f"TileBench checkout {TILEBENCH_ROOT} has tracked modifications; "
            f"commit or stash them so the corpus pin ({head}) is "
            f"meaningful:\n{dirty}"
        )
    return head


# ── case table ───────────────────────────────────────────────────
# Every benchmarks/operators/<op>/ directory that ships an
# impl_triton.py at the pinned commit (the divergence_metric /
# generic_fused_container / matmul_fp16_fp8 / _template dirs do not).

OPS: tuple = (
    "1d_conv",
    "2d_conv",
    "2d_max_pooling",
    "3d_conv",
    "argmax",
    "batch_normalization",
    "batched_matmul",
    "bitonic_sort",
    "block_sparse_attention",
    "cross_entropy",
    "dequantize_rowwise",
    "destindex",
    "dropout",
    "flash_attention",
    "flash_decode",
    "fused_activation",
    "gaussian_blur",
    "histogramming",
    "interleave",
    "jacobi_stencil_2d",
    "kl_divergence",
    "l2_norm",
    "layernorm",
    "leaky_relu",
    "linear_self_attention",
    "matmul_fp32_fp16_fp8",
    "matmul_int8",
    "matrix_copy",
    "matrix_transpose",
    "mean_reduction",
    "moe_topk_gating",
    "mul2",
    "quantize_global",
    "radix_sort",
    "relu",
    "reverse_array",
    "rmsnorm",
    "rope",
    "sigmoid",
    "softmax",
    "streamk_matmul",
    "swiglu",
    "top_k_selection",
    "vector_add",
    "weight_dequant",
)


def _tb_case(op: str):
    def run(torch, device, dtype):
        import sys

        root = str(TILEBENCH_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        # the engine opens config paths relative to the repo root
        os.chdir(root)
        from core import engine as tb_engine

        # skip the Proton/CUDA-graph timing path entirely: the engine's
        # verification run (plain stream) is the launch we record
        tb_engine.report_benchmark = lambda *a, **k: {"mean": float("nan")}
        tb_engine.run_benchmark_suite(
            op,
            benchmark_overrides={"case_indices": [0], "autotune": False},
        )
        return []

    return run


CASES: dict = {op: (op, False, _tb_case(op)) for op in OPS}


def main() -> None:
    from evaluation.capture_common import capture_one_case, run_case_capture

    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        out = args.out.resolve()  # the case fn chdirs into the checkout
        result = capture_one_case(
            CASES,
            args.one,
            dtype_name="float32",
            module_prefix="benchmarks.operators.",
        )
        out.write_text(json.dumps(result, indent=1))
        return

    commit = tilebench_commit()
    run_case_capture(
        "evaluation.tilebench_capture",
        CASES,
        SPECS_PATH,
        payload_meta={
            "upstream": UPSTREAM,
            "tilebench": commit,
            "upstream_commit": commit,
            "tilebench_root": str(TILEBENCH_ROOT),
        },
        per_case_timeout_s=PER_CASE_TIMEOUT_S,
    )


if __name__ == "__main__":
    main()
