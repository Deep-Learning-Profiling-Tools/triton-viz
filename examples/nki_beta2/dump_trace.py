"""Dump a minimal NKI beta2 tile trace to JSONL.

Run from the repository root after installing the NKI extra:

    . .venv/bin/activate
    python examples/nki_beta2/dump_trace.py --output /tmp/nki_trace.jsonl

The output is a deliberately simple, ordered event stream that future performance
models can consume before AWS neuron-profile/NTFF ground truth is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import summarize_events, write_jsonl

try:
    import nki.isa as nisa
    import nki.language as nl
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing guard
    raise SystemExit(
        "NKI packages are missing. Install with: pip install -e '.[test,nki]' "
        "--extra-index-url https://pip.repos.neuron.amazonaws.com"
    ) from exc


def demo_kernel(lhs_t, rhs, out):
    """One-tile GEMM-like kernel that exercises DMA, TensorE, and PSUM/SBUF copies."""
    lhs_tile = nl.ndarray((128, 128), dtype=lhs_t.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    res_psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    out_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)

    nisa.dma_copy(lhs_tile, lhs_t)
    nisa.dma_copy(rhs_tile, rhs)
    nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)
    nisa.tensor_copy(out_tile, res_psum)
    nisa.dma_copy(out, out_tile)


def run(output: Path, *, pre_trace: bool = True) -> dict:
    triton_viz.clear()
    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(demo_kernel)
    lhs_t = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    rhs = np.arange(128 * 512, dtype=np.float32).reshape(128, 512)
    out = np.empty((128, 512), dtype=np.float32)
    traced[(1,)](lhs_t, rhs, out, pre_trace=pre_trace)
    events = write_jsonl(launches[-1].records, output)
    return summarize_events(events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/triton_viz_nki_trace.jsonl")
    parser.add_argument(
        "--no-pre-trace",
        action="store_true",
        help="Skip the initial NKI compiler trace; useful for faster local iteration.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Also run the placeholder cost model and print a predicted per-engine timeline.",
    )
    args = parser.parse_args()
    summary = run(Path(args.output), pre_trace=not args.no_pre_trace)
    payload = {"output": args.output, "summary": summary}
    if args.simulate:
        from triton_viz.tools.nki_cost_model import simulate_jsonl

        payload["simulation"] = simulate_jsonl(args.output).as_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
