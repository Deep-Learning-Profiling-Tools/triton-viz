"""Run compositional NKI region controls through trace, hardware and mapping."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import triton_viz
from microbench.inf2_nki.common.inputs import make_input
from triton_viz.clients.tracer.tracer import Tracer
from triton_viz.core.trace import launches
from triton_viz.core.trace import trace as tv_trace
from triton_viz.tools.nki_explorer import export_parquet
from triton_viz.tools.nki_instruction_source_mapping import write_case
from triton_viz.tools.nki_operator_experiments import _profile_summary, _run_hardware
from triton_viz.tools.nki_provenance import write_experiment_manifest
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature, write_jsonl

KINDS = [
    "elementwise_one",
    "elementwise_two",
    "elementwise_maximum",
    "elementwise_multiply",
    "elementwise_sigmoid",
    "masked_log_reduction",
    "softmax_reduction",
    "elementwise_multiply2",
    "broadcast_multiply2",
    "broadcast_affine",
    "two_pass_reduce_affine",
    "two_pass_reduce_multiply",
    "reduce_broadcast",
    "two_reductions",
    "rsqrt_newton",
    "two_reductions_rsqrt",
    "elementwise_mixed",
    "mask_tail",
]
KINDS += ["two_reductions_rsqrt_masked", "elementwise_mixed_masked"]


def _case_name(kind: str, p: int, f: int, chain: int, dtype: str) -> str:
    return f"control_{kind}__p{p}__f{f}__n{chain}__{dtype}"


def _factory(**kwargs):
    path = Path("microbench/inf2_nki/tests/region_controls/kernels.py").resolve()
    spec = importlib.util.spec_from_file_location("nki_region_controls_dynamic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.region_control_factory(**kwargs)


def _trace(kernel, inputs, path: Path):
    triton_viz.clear()
    tv_trace(client=Tracer(), frontend="nki_beta2")(kernel.func)[(1,)](*inputs)
    return write_jsonl(launches[-1].records, path)


def _declared_trace(kind: str, p: int, f: int, chain: int, dtype: str, path: Path):
    """Exact source-op declaration used when factory AST tracing is unavailable."""
    tile = (
        min(2048, f)
        if kind == "mask_tail" or f > 2048
        else (
            2048 if kind.endswith("_masked") or kind == "two_pass_reduce_affine" else f
        )
    )
    item_bytes = 2 if dtype == "bfloat16" else 4
    events = [{"seq": 0, "op": "grid", "record_type": "Grid", "grid_idx": [0, 0, 0]}]
    if kind == "two_pass_reduce_multiply" and f > 2048:

        def load(load_p):
            events.append(
                {
                    "seq": len(events),
                    "op": "load",
                    "record_type": "Load",
                    "grid_idx": [0, 0, 0],
                    "bytes": load_p * 2048 * item_bytes,
                    "active_lanes": load_p * 2048,
                    "partition_count": load_p,
                    "offsets_shape": [load_p, 2048],
                    "mem_src": "HBM",
                    "mem_dst": "SBUF",
                }
            )

        def ops(tokens, arities, seed):
            previous = seed
            for token, arity in zip(tokens, arities):
                out = previous + 1
                reduction = token == "reduce_sum"
                events.append(
                    {
                        "seq": len(events),
                        "op": "reduce_sum" if reduction else "compute",
                        "api_op": None if reduction else token,
                        "record_type": "ReduceSum" if reduction else "NkiCompute",
                        "grid_idx": [0, 0, 0],
                        "input_ptrs": [previous] if arity == 1 else [previous, 2],
                        "output_ptr": out,
                        "input_shape": [p, 2048],
                        "output_shape": [p, 1 if reduction else 2048],
                        "output_dtype": dtype,
                        "input_dtypes": [dtype] * arity,
                    }
                )
                previous = out

        prefix = ["multiply", "where", "reduce_sum", "add"]
        prefix_a = [2, 3, 1, 2]
        tail = [
            "divide",
            "add",
            "multiply",
            "rsqrt",
            "multiply",
            "multiply",
            "subtract",
            "multiply",
            "multiply",
            "multiply",
            "subtract",
            "multiply",
        ]
        tail_a = [1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2]
        load(p)
        ops(prefix, prefix_a, 1000)
        load(p)
        ops(prefix + tail, prefix_a + tail_a, 2000)
        for block in range(2):
            load(p)
            load(1)
            ops(["broadcast_to", "multiply", "multiply"], [1, 2, 2], 3000 + block * 100)
            events.append(
                {
                    "seq": len(events),
                    "op": "store",
                    "record_type": "Store",
                    "grid_idx": [0, 0, 0],
                    "bytes": p * 2048 * item_bytes,
                    "active_lanes": p * 2048,
                    "partition_count": p,
                    "offsets_shape": [p, 2048],
                    "mem_src": "SBUF",
                    "mem_dst": "HBM",
                }
            )
        _annotate_fusion_signature(events)
        path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
        return events
    if kind == "two_pass_reduce_affine" and f > 2048:

        def load(load_p):
            events.append(
                {
                    "seq": len(events),
                    "op": "load",
                    "record_type": "Load",
                    "grid_idx": [0, 0, 0],
                    "bytes": load_p * 2048 * item_bytes,
                    "active_lanes": load_p * 2048,
                    "partition_count": load_p,
                    "offsets_shape": [load_p, 2048],
                    "mem_src": "HBM",
                    "mem_dst": "SBUF",
                }
            )

        def ops(tokens, arities, seed):
            previous = seed
            for token, arity in zip(tokens, arities):
                out = previous + 1
                red = token == "reduce_sum"
                events.append(
                    {
                        "seq": len(events),
                        "op": "reduce_sum" if red else "compute",
                        "api_op": None if red else token,
                        "record_type": "ReduceSum" if red else "NkiCompute",
                        "grid_idx": [0, 0, 0],
                        "input_ptrs": [previous] if arity == 1 else [previous, 2],
                        "output_ptr": out,
                        "input_shape": [p, 2048],
                        "output_shape": [p, 1 if red else 2048],
                        "output_dtype": dtype,
                        "input_dtypes": [dtype] * arity,
                    }
                )
                previous = out

        prefix = ["where", "reduce_sum", "add", "multiply", "reduce_sum", "add"]
        pa = [3, 1, 2, 2, 1, 2]
        tail = [
            "divide",
            "divide",
            "multiply",
            "subtract",
            "add",
            "multiply",
            "rsqrt",
            "multiply",
            "multiply",
            "subtract",
            "multiply",
            "multiply",
            "multiply",
            "subtract",
            "multiply",
        ]
        ta = [1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2]
        load(p)
        ops(prefix, pa, 1000)
        load(p)
        ops(prefix + tail, pa + ta, 2000)
        for block in range(2):
            load(p)
            load(1)
            load(1)
            ops(
                [
                    "broadcast_to",
                    "broadcast_to",
                    "subtract",
                    "multiply",
                    "multiply",
                    "add",
                ],
                [1, 1, 2, 2, 2, 2],
                3000 + block * 100,
            )
            events.append(
                {
                    "seq": len(events),
                    "op": "store",
                    "record_type": "Store",
                    "grid_idx": [0, 0, 0],
                    "bytes": p * 2048 * item_bytes,
                    "active_lanes": p * 2048,
                    "partition_count": p,
                    "offsets_shape": [p, 2048],
                    "mem_src": "SBUF",
                    "mem_dst": "HBM",
                }
            )
        _annotate_fusion_signature(events)
        path.write_text(
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in events),
            encoding="utf-8",
        )
        return events
    blocks = (f + tile - 1) // tile
    for block in range(blocks):
        active_f = min(tile, f - block * tile)
        load_count = (
            1
            if kind in {"two_pass_reduce_affine", "two_pass_reduce_multiply"}
            else (3 if kind == "broadcast_affine" else 2)
        )
        for load_index in range(load_count):
            load_p = 1 if kind.startswith("broadcast_") and load_index else p
            events.append(
                {
                    "seq": len(events),
                    "op": "load",
                    "record_type": "Load",
                    "grid_idx": [0, 0, 0],
                    "bytes": load_p * active_f * item_bytes,
                    "active_lanes": load_p * active_f,
                    "partition_count": load_p,
                    "offsets_shape": [load_p, tile],
                    "mem_src": "HBM",
                    "mem_dst": "SBUF",
                }
            )
        arities = None
        if f > 2048:
            tokens = ["add"]
        elif kind in {"elementwise_one", "elementwise_two"}:
            tokens = ["add"] * chain
        elif kind == "elementwise_maximum":
            tokens, arities = ["maximum"], [1]
        elif kind == "elementwise_multiply":
            tokens, arities = ["multiply"], [1]
        elif kind == "elementwise_sigmoid":
            tokens, arities = ["sigmoid"], [1]
        elif kind == "masked_log_reduction":
            tokens, arities = (
                ["greater", "log", "where", "subtract", "multiply", "reduce_sum"],
                [1, 1, 3, 2, 2, 1],
            )
        elif kind == "softmax_reduction":
            tokens, arities = (
                ["max", "subtract", "exp", "reduce_sum", "divide", "add"],
                [1, 2, 1, 1, 2, 1],
            )
        elif kind == "elementwise_multiply2":
            tokens, arities = ["multiply", "multiply"], [2, 2]
        elif kind == "broadcast_multiply2":
            tokens, arities = ["broadcast_to", "multiply", "multiply"], [1, 2, 2]
        elif kind == "broadcast_affine":
            tokens, arities = (
                [
                    "broadcast_to",
                    "broadcast_to",
                    "subtract",
                    "multiply",
                    "multiply",
                    "add",
                ],
                [1, 1, 2, 2, 2, 2],
            )
        elif kind == "two_pass_reduce_affine":
            tokens = [
                "where",
                "reduce_sum",
                "add",
                "multiply",
                "reduce_sum",
                "add",
                "divide",
                "divide",
                "multiply",
                "subtract",
                "add",
                "multiply",
                "rsqrt",
                "multiply",
                "multiply",
                "subtract",
                "multiply",
                "multiply",
                "multiply",
                "subtract",
                "multiply",
            ]
            arities = [3, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2]
        elif kind == "two_pass_reduce_multiply":
            tokens = [
                "multiply",
                "reduce_sum",
                "divide",
                "add",
                "multiply",
                "rsqrt",
                "multiply",
                "multiply",
                "subtract",
                "multiply",
                "multiply",
                "multiply",
                "subtract",
                "multiply",
            ]
            arities = [2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2]
        elif kind == "reduce_broadcast":
            tokens, arities = ["multiply", "reduce_sum", "multiply"], [2, 1, 2]
        elif kind == "two_reductions":
            tokens, arities = (
                ["reduce_sum", "multiply", "reduce_sum", "multiply", "multiply", "add"],
                [1, 2, 1, 2, 2, 2],
            )
        elif kind in {
            "rsqrt_newton",
            "two_reductions_rsqrt",
            "two_reductions_rsqrt_masked",
        }:
            base_kind = kind.removesuffix("_masked")
            if base_kind == "two_reductions_rsqrt" and kind.endswith("_masked"):
                prefix = [
                    "where",
                    "reduce_sum",
                    "add",
                    "multiply",
                    "reduce_sum",
                    "add",
                    "divide",
                    "divide",
                    "multiply",
                    "subtract",
                    "add",
                ]
                prefix_arities = [3, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1]
            elif base_kind == "two_reductions_rsqrt":
                prefix = [
                    "reduce_sum",
                    "divide",
                    "multiply",
                    "reduce_sum",
                    "divide",
                    "multiply",
                    "subtract",
                    "add",
                ]
                prefix_arities = [1, 1, 2, 1, 1, 2, 2, 1]
            else:
                prefix, prefix_arities = (
                    ["multiply", "reduce_sum", "divide", "add"],
                    [2, 1, 1, 1],
                )
            tokens = prefix + [
                "multiply",
                "rsqrt",
                "multiply",
                "multiply",
                "subtract",
                "multiply",
                "multiply",
                "multiply",
                "subtract",
                "multiply",
                "multiply",
            ]
            arities = prefix_arities + [1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2]
        elif kind in {"elementwise_mixed", "elementwise_mixed_masked"}:
            tokens, arities = ["subtract", "multiply", "multiply", "add"], [2, 2, 2, 2]
        else:
            tokens = ["add", "where"]
        previous = 1000 + block * 100
        for token_index, token in enumerate(tokens):
            reduction = token == "reduce_sum"
            out = previous + 1
            arity = (
                arities[token_index]
                if arities
                else (
                    1
                    if kind == "elementwise_one" or token in {"rsqrt"} or reduction
                    else 2
                )
            )
            events.append(
                {
                    "seq": len(events),
                    "op": "reduce_sum" if reduction else "compute",
                    "api_op": None if reduction else token,
                    "record_type": "ReduceSum" if reduction else "NkiCompute",
                    "grid_idx": [0, 0, 0],
                    "input_ptrs": [previous] if arity == 1 else [previous, 2],
                    "output_ptr": out,
                    "input_shape": [p, tile],
                    "output_shape": [p, 1 if reduction else tile],
                    "output_dtype": dtype,
                    "input_dtypes": [dtype] * arity,
                }
            )
            previous = out
        if kind in {"two_pass_reduce_affine", "two_pass_reduce_multiply"}:
            epilogue_loads = 3 if kind == "two_pass_reduce_affine" else 2
            for load_index in range(epilogue_loads):
                load_p = p if load_index == 0 else 1
                events.append(
                    {
                        "seq": len(events),
                        "op": "load",
                        "record_type": "Load",
                        "grid_idx": [0, 0, 0],
                        "bytes": load_p * active_f * item_bytes,
                        "active_lanes": load_p * active_f,
                        "partition_count": load_p,
                        "offsets_shape": [load_p, tile],
                        "mem_src": "HBM",
                        "mem_dst": "SBUF",
                    }
                )
            previous = 5000 + block * 100
            epilogue = (
                [
                    ("broadcast_to", 1),
                    ("broadcast_to", 1),
                    ("subtract", 2),
                    ("multiply", 2),
                    ("multiply", 2),
                    ("add", 2),
                ]
                if kind == "two_pass_reduce_affine"
                else [("broadcast_to", 1), ("multiply", 2), ("multiply", 2)]
            )
            for token, arity in epilogue:
                out = previous + 1
                events.append(
                    {
                        "seq": len(events),
                        "op": "compute",
                        "api_op": token,
                        "record_type": "NkiCompute",
                        "grid_idx": [0, 0, 0],
                        "input_ptrs": [previous] if arity == 1 else [previous, 2],
                        "output_ptr": out,
                        "input_shape": [p, tile],
                        "output_shape": [p, tile],
                        "output_dtype": dtype,
                        "input_dtypes": [dtype] * arity,
                    }
                )
                previous = out
        events.append(
            {
                "seq": len(events),
                "op": "store",
                "record_type": "Store",
                "grid_idx": [0, 0, 0],
                "bytes": p * active_f * item_bytes,
                "active_lanes": p * active_f,
                "partition_count": p,
                "offsets_shape": [p, tile],
                "mem_src": "SBUF",
                "mem_dst": "HBM",
            }
        )
    _annotate_fusion_signature(events)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return events


def run_case(
    root: Path,
    kind: str,
    p: int,
    f: int,
    chain: int,
    dtype: str,
    warmup: int,
    iters: int,
    hardware: bool,
) -> dict:
    name = _case_name(kind, p, f, chain, dtype)
    case = (root / name).resolve()
    case.mkdir(parents=True, exist_ok=True)
    kernel, shapes, extras = _factory(
        kind=kind, p=p, f=f, chain=chain, dtype_name=dtype
    )
    inputs = [
        make_input(shape, dtype, seed=index) for index, shape in enumerate(shapes)
    ]
    inputs.extend(extras)
    # Nested nl.* expressions are intentionally flattened by Python syntax but
    # the simulator records only the outer call. Use the exact factory grammar
    # declaration so source-op count matches the written kernel; Penguin still
    # independently audits what the compiler retained/fused.
    events = _declared_trace(kind, p, f, chain, dtype, case / "trace.jsonl")
    trace_source = "source_declaration"
    row = {
        "case": name,
        "kind": kind,
        "p": p,
        "f": f,
        "chain": chain,
        "dtype": dtype,
        "trace_source": trace_source,
        "trace_hbm_read_bytes": sum(
            int(e.get("bytes", 0)) for e in events if e.get("op") == "load"
        ),
        "trace_hbm_write_bytes": sum(
            int(e.get("bytes", 0)) for e in events if e.get("op") == "store"
        ),
    }
    if hardware:
        hardware_dir = case / "hardware"
        if (hardware_dir / "file.neff").is_file() and (
            hardware_dir / "profile.ntff"
        ).is_file():
            profile = _profile_summary(
                hardware_dir / "file.neff",
                hardware_dir / "profile.ntff",
                hardware_dir / "explorer_summary.json",
            )
        else:
            nc_p50_us, profile = _run_hardware(
                "softmax", inputs, hardware_dir, warmup, iters, kernel=kernel
            )
            row["hardware_nc_p50_us"] = nc_p50_us
        export_parquet(hardware_dir)
        row.update(
            vector_active_ns=float(profile.get("vector_engine_active_time", 0)) * 1e9,
            scalar_active_ns=float(profile.get("scalar_engine_active_time", 0)) * 1e9,
        )
        audit = write_case(case)
        row["vector_mapping_coverage_pct"] = audit["engines"]["vector"][
            "mapped_payload_coverage_percent"
        ]
        row["scalar_mapping_coverage_pct"] = audit["engines"]["scalar"][
            "mapped_payload_coverage_percent"
        ]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kinds", nargs="*", choices=KINDS, default=KINDS)
    parser.add_argument(
        "--free-dims", nargs="*", type=int, default=[128, 512, 1024, 2048, 4096]
    )
    parser.add_argument(
        "--dtypes",
        nargs="*",
        choices=["float32", "bfloat16"],
        default=["float32", "bfloat16"],
    )
    parser.add_argument("--chains", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-hardware", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_manifest(
        args.output_dir,
        experiment="nki_region_controls",
        config={key: value for key, value in vars(args).items() if key != "resume"},
        repository_root=Path(__file__).resolve().parents[2],
    )
    results_path = args.output_dir / "control_results.csv"
    rows = []
    if args.resume and results_path.is_file():
        with results_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    completed = {
        row.get("case") for row in rows if row.get("case") and not row.get("error")
    }
    for dtype in args.dtypes:
        dims = (
            args.free_dims
            if dtype == "float32"
            else [f for f in args.free_dims if f in (512, 2048)]
        )
        for kind in args.kinds:
            chains = (
                args.chains if kind in {"elementwise_one", "elementwise_two"} else [1]
            )
            for f in dims:
                if f > 2048 and kind not in {
                    "mask_tail",
                    "two_pass_reduce_affine",
                    "two_pass_reduce_multiply",
                }:
                    continue
                for chain in chains:
                    case_name = _case_name(kind, args.p, f, chain, dtype)
                    if case_name in completed:
                        print(f"SKIP completed {case_name}", flush=True)
                        continue
                    rows = [row for row in rows if row.get("case") != case_name]
                    print(kind, dtype, f, chain, flush=True)
                    try:
                        rows.append(
                            run_case(
                                args.output_dir,
                                kind,
                                args.p,
                                f,
                                chain,
                                dtype,
                                args.warmup,
                                args.iters,
                                not args.no_hardware,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve all cases in the audit CSV
                        print(
                            f"ERROR {kind} {dtype} F={f} chain={chain}: {exc!r}",
                            flush=True,
                        )
                        rows.append({"case": case_name, "error": repr(exc)})
                    fields = sorted({key for row in rows for key in row})
                    with results_path.open("w", newline="", encoding="utf-8") as file:
                        writer = csv.DictWriter(file, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
