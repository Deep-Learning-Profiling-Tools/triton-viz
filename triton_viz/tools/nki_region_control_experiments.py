"""Run compositional NKI region controls through trace, hardware and mapping."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import triton_viz
from microbench.inf2_nki.common.inputs import make_input
from triton_viz.clients.tracer.tracer import Tracer
from triton_viz.core.trace import launches
from triton_viz.core.trace import trace as tv_trace
from triton_viz.tools.nki_operator_experiments import _profile_summary, _run_hardware
from triton_viz.tools.nki_provenance import write_experiment_manifest
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature, write_jsonl

KINDS = [
    "elementwise_one",
    "elementwise_two",
    "elementwise_maximum",
    "elementwise_multiply",
    "elementwise_sigmoid",
    "primitive_divide",
    "primitive_add",
    "primitive_subtract",
    "primitive_multiply",
    "primitive_where_bundle",
    "primitive_exp",
    "primitive_log",
    "primitive_rsqrt",
    "primitive_reduce_sum",
    "sequence_add_multiply",
    "sequence_multiply_add",
    "sequence_subtract_multiply_add",
    "sequence_multiply_subtract_multiply",
    "sequence_exp_multiply_add",
    "sequence_log_add_multiply",
    "sequence_reduce_add_multiply",
    "sequence_reduce_divide_multiply",
    "sequence_two_reduce_add",
    "sequence_two_reduce_multiply",
    "sequence_rsqrt_multiply_add",
    "sequence_add_multiply_wide_memory",
    "sequence_subtract_multiply_add_wide_memory",
    "sequence_exp_multiply_add_wide_memory",
    "sequence_reduce_add_multiply_wide_memory",
    "sequence_two_reduce_multiply_wide_memory",
    "sequence_rsqrt_multiply_add_wide_memory",
    "sequence_multiply_add_wide_memory",
    "sequence_multiply_subtract_multiply_wide_memory",
    "sequence_log_add_multiply_wide_memory",
    "sequence_reduce_divide_multiply_wide_memory",
    "sequence_two_reduce_add_wide_memory",
    "sequence_perm2k_ams", "sequence_perm2k_asm", "sequence_perm2k_mas",
    "sequence_perm2k_msa", "sequence_perm2k_sam", "sequence_perm2k_sma",
    "sequence_perm2k_eam", "sequence_perm2k_aem", "sequence_perm2k_mea",
    "sequence_perm2k_ram", "sequence_perm2k_arm", "sequence_perm2k_mra",
    *[f"sequence_perm2k_long{i:02d}" for i in range(12)],
    "sequence_deep2k_add", "sequence_deep2k_multiply",
    "sequence_deep2k_add_multiply",
    *[f"sequence_deepmixed2k_{i:02d}" for i in range(16)],
    "sequence_randommixed2k",
    "sequence_randomsemantic2k",
    "sequence_randomdag2k",
    "sequence_factorialdag2k",
    "sequence_factorialdagaudit2k",
    "sequence_factorialdaginterleave2k",
    "elementwise_maximum_masked",
    "elementwise_multiply_masked",
    "elementwise_sigmoid_masked",
    "elementwise_maximum_wide_masked",
    "elementwise_multiply_wide_masked",
    "elementwise_sigmoid_wide_masked",
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
KINDS += [
    "padded_add",
    "padded_multiply",
    "padded_sigmoid",
    "padded_mixed",
    "padded_reduce_affine",
    "padded_reduce_transcendental",
    "padded_reduce_pair",
    "padded_reduce_rsqrt",
    "padded_maximum",
    "padded_reduce_maximum",
    "padded_randomdag",
]


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
    tv_trace(client=Tracer(), frontend="nki")(kernel.func)[(1,)](*inputs)
    return write_jsonl(launches[-1].records, path)


def refresh_dependency_trace(root: Path, row: dict) -> int:
    """Recreate only the runtime dependency artifact for an existing case."""
    kind = str(row["kind"])
    p, f, chain = (int(row[name]) for name in ("p", "f", "chain"))
    dtype = str(row["dtype"])
    kernel, shapes, extras = _factory(
        kind=kind, p=p, f=f, chain=chain, dtype_name=dtype
    )
    inputs = [
        make_input(shape, dtype, seed=index)
        for index, shape in enumerate(shapes)
    ]
    inputs.extend(extras)
    case = (root / str(row["case"])).resolve()
    if kind.startswith("padded_"):
        return len(
            _declared_trace(
                kind, p, f, chain, dtype, case / "dependency_trace.jsonl"
            )
        )
    return len(_trace(kernel, inputs, case / "dependency_trace.jsonl"))


def _declared_trace(kind: str, p: int, f: int, chain: int, dtype: str, path: Path):
    """Exact source-op declaration used when factory AST tracing is unavailable."""
    explicit_mask_kinds = {
        "elementwise_maximum_masked",
        "elementwise_multiply_masked",
        "elementwise_sigmoid_masked",
        "elementwise_maximum_wide_masked",
        "elementwise_multiply_wide_masked",
        "elementwise_sigmoid_wide_masked",
        "sequence_add_multiply_wide_memory",
        "sequence_subtract_multiply_add_wide_memory",
        "sequence_exp_multiply_add_wide_memory",
        "sequence_reduce_add_multiply_wide_memory",
        "sequence_two_reduce_multiply_wide_memory",
        "sequence_rsqrt_multiply_add_wide_memory",
        "sequence_multiply_add_wide_memory",
        "sequence_multiply_subtract_multiply_wide_memory",
        "sequence_log_add_multiply_wide_memory",
        "sequence_reduce_divide_multiply_wide_memory",
        "sequence_two_reduce_add_wide_memory",
        "mask_tail",
        "two_pass_reduce_affine",
        "two_pass_reduce_multiply",
        "two_reductions_rsqrt_masked",
        "elementwise_mixed_masked",
        "padded_add",
        "padded_multiply",
        "padded_sigmoid",
        "padded_mixed",
        "padded_reduce_affine",
        "padded_reduce_transcendental",
        "padded_reduce_pair",
        "padded_reduce_rsqrt",
        "padded_maximum",
        "padded_reduce_maximum",
        "padded_randomdag",
    }
    mask_provided = kind in explicit_mask_kinds or kind.startswith(("sequence_perm2k_", "sequence_deep2k_", "sequence_deepmixed2k_", "sequence_randommixed2k", "sequence_randomsemantic2k", "sequence_randomdag2k", "sequence_factorialdag2k", "sequence_factorialdagaudit2k", "sequence_factorialdaginterleave2k"))
    tile = (
        2048
        if kind.startswith(("sequence_perm2k_", "sequence_deep2k_", "sequence_deepmixed2k_", "sequence_randommixed2k", "sequence_randomsemantic2k", "sequence_randomdag2k", "sequence_factorialdag2k", "sequence_factorialdagaudit2k", "sequence_factorialdaginterleave2k"))
        else
        (16384 if "wide_memory" in kind else min(2048, f))
        if kind == "mask_tail" or "wide_masked" in kind or "wide_memory" in kind or f > 2048
        else (
            (16384 if "wide_masked" in kind else 2048)
            if kind.endswith("_masked") or kind == "two_pass_reduce_affine" else f
        )
    )
    item_bytes = 2 if dtype == "bfloat16" else 4
    events = [{"seq": 0, "op": "grid", "record_type": "Grid", "grid_idx": [0, 0, 0]}]
    if kind.startswith("padded_"):
        # These controls deliberately use a PMAX physical partition tile while
        # only ``p`` logical rows are active.  The CPU masked-load simulator
        # cannot index beyond the backing NumPy extent even under a mask, so we
        # write the exact source declaration instead of executing that load.
        physical_p = 128
        events.append(
            {
                "seq": 1,
                "op": "load",
                "record_type": "Load",
                "bytes": p * f * item_bytes,
                "active_lanes": p * f,
                "partition_count": physical_p,
                "offsets_shape": [physical_p, f],
                "mem_src": "HBM",
                "mem_dst": "SBUF",
                "mask_provided": True,
            }
        )
        token_map = {
            "padded_add": [("add", 1)],
            "padded_multiply": [("multiply", 1)],
            "padded_sigmoid": [("sigmoid", 1)],
            "padded_mixed": [("subtract", 2), ("multiply", 2), ("add", 2)],
            "padded_reduce_affine": [
                ("reduce_sum", 1), ("multiply", 1), ("add", 2)
            ],
            "padded_reduce_transcendental": [
                ("multiply", 1), ("exp", 1), ("reduce_sum", 1),
                ("add", 1), ("divide", 2),
            ],
            "padded_reduce_pair": [
                ("reduce_sum", 1), ("multiply", 1), ("reduce_sum", 1),
                ("add", 2), ("multiply", 1), ("add", 2),
            ],
            "padded_reduce_rsqrt": [
                ("multiply", 1), ("reduce_sum", 1), ("add", 1),
                ("rsqrt", 1), ("multiply", 2),
            ],
            "padded_maximum": [("maximum", 1)],
            "padded_reduce_maximum": [
                ("reduce_sum", 1), ("subtract", 2), ("maximum", 1),
            ],
        }
        if kind == "padded_randomdag":
            rng = random.Random(0x4DA6_2100 + int(chain))
            actions = [
                rng.choice((
                    "a_add", "a_multiply", "a_exp", "b_subtract",
                    "b_maximum", "b_rsqrt", "cross_add", "cross_multiply",
                ))
                for _ in range(rng.randint(8, 18))
            ]
            actions.extend(("a_reduce", "b_reduce", "cross_add"))
            rng.shuffle(actions)
            action_tokens = {
                "a_add": ("add", 1), "a_multiply": ("multiply", 1),
                "a_exp": ("exp", 1), "b_subtract": ("subtract", 1),
                "b_maximum": ("maximum", 1), "b_rsqrt": ("rsqrt", 1),
                "cross_add": ("add", 2), "cross_multiply": ("multiply", 2),
                "a_reduce": ("reduce_sum", 1),
                "b_reduce": ("reduce_sum", 1),
            }
            selected_tokens = [action_tokens[action] for action in actions]
        else:
            selected_tokens = token_map[kind]
        previous = 100
        for token, arity in selected_tokens:
            output = previous + 1
            reduction = token == "reduce_sum"
            events.append(
                {
                    "seq": len(events),
                    "op": "reduce_sum" if reduction else "compute",
                    "api_op": None if reduction else token,
                    "record_type": "ReduceSum" if reduction else "NkiCompute",
                    "input_ptrs": [previous] if arity == 1 else [previous, 2],
                    "output_ptr": output,
                    "input_shape": [physical_p, f],
                    "output_shape": [physical_p, 1 if reduction else f],
                    "output_dtype": dtype,
                    "input_dtypes": [dtype] * arity,
                    "mask_provided": True,
                }
            )
            previous = output
        events.append(
            {
                "seq": len(events),
                "op": "store",
                "record_type": "Store",
                "bytes": p * f * item_bytes,
                "active_lanes": p * f,
                "partition_count": physical_p,
                "offsets_shape": [physical_p, f],
                "mem_src": "SBUF",
                "mem_dst": "HBM",
                "mask_provided": True,
            }
        )
        _annotate_fusion_signature(events)
        for event in events:
            if event.get("region_ir") is not None:
                event["region_ir"]["logical_free_dim"] = f
                event["region_ir"]["logical_active_partition_count"] = p
        path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
        return events
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
                    "mask_provided": mask_provided,
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
                    "mask_provided": mask_provided,
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
                    "mask_provided": mask_provided,
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
                    "mask_provided": mask_provided,
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
                    "mask_provided": mask_provided,
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
        elif kind.startswith("primitive_"):
            token = kind.removeprefix("primitive_")
            if token == "where_bundle":
                tokens, arities = ["greater", "where"], [1, 3]
            else:
                tokens, arities = [token], [1 if token in {"exp", "log", "rsqrt", "reduce_sum"} else 2]
        elif kind == "sequence_randommixed2k":
            module_path = Path("microbench/inf2_nki/tests/region_controls/kernels.py").resolve()
            module_spec = importlib.util.spec_from_file_location(
                "nki_region_schedule_dynamic", module_path
            )
            schedule_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(schedule_module)
            tokens, arities = ["add"], [1]
            for token in schedule_module.random_mixed_schedule(chain):
                if token == "reduce":
                    tokens.extend(("reduce_sum", "multiply", "add"))
                    arities.extend((1, 1, 2))
                elif token == "exp":
                    tokens.extend(("multiply", "exp"))
                    arities.extend((1, 1))
                else:
                    tokens.append(token)
                    arities.append(1)
        elif kind == "sequence_randomsemantic2k":
            module_path = Path("microbench/inf2_nki/tests/region_controls/kernels.py").resolve()
            module_spec = importlib.util.spec_from_file_location(
                "nki_region_semantic_schedule_dynamic", module_path
            )
            schedule_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(schedule_module)
            tokens, arities = ["add"], [1]
            for token in schedule_module.random_semantic_schedule(chain):
                if token == "reduce":
                    tokens.extend(("reduce_sum", "multiply", "add"))
                    arities.extend((1, 1, 2))
                elif token == "exp":
                    tokens.extend(("multiply", "exp"))
                    arities.extend((1, 1))
                elif token == "where":
                    tokens.extend(("greater", "multiply", "where"))
                    arities.extend((1, 1, 3))
                else:
                    tokens.append(token)
                    arities.append(1)
        elif kind in {"sequence_randomdag2k", "sequence_factorialdag2k", "sequence_factorialdagaudit2k", "sequence_factorialdaginterleave2k"}:
            module_path = Path("microbench/inf2_nki/tests/region_controls/kernels.py").resolve()
            module_spec = importlib.util.spec_from_file_location(
                "nki_region_dag_schedule_dynamic", module_path
            )
            schedule_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(schedule_module)
            # Emitted below with explicit two-branch pointer identity.
            tokens, arities = [], []
            next_ptr, a_ptr, b_ptr = 1000, None, None

            def dag_op(token, inputs, reduction=False):
                nonlocal next_ptr
                next_ptr += 1
                events.append({
                    "seq": len(events),
                    "op": "reduce_sum" if reduction else "compute",
                    "api_op": None if reduction else token,
                    "record_type": "ReduceSum" if reduction else "NkiCompute",
                    "grid_idx": [0, 0, 0], "input_ptrs": list(inputs),
                    "output_ptr": next_ptr, "input_shape": [p, tile],
                    "output_shape": [p, 1 if reduction else tile],
                    "output_dtype": dtype, "input_dtypes": [dtype] * len(inputs),
                })
                return next_ptr

            a_ptr = dag_op("add", [10])
            b_ptr = dag_op("multiply", [20])
            schedule_fn = {
                "sequence_randomdag2k": schedule_module.random_dag_schedule,
                "sequence_factorialdag2k": schedule_module.factorial_dag_schedule,
                "sequence_factorialdagaudit2k": schedule_module.factorial_dag_audit_schedule,
                "sequence_factorialdaginterleave2k": schedule_module.factorial_dag_interleave_schedule,
            }[kind]
            for action in schedule_fn(chain):
                if action == "a_add": a_ptr = dag_op("add", [a_ptr])
                elif action == "a_multiply": a_ptr = dag_op("multiply", [a_ptr])
                elif action == "a_exp":
                    a_ptr = dag_op("multiply", [a_ptr]); a_ptr = dag_op("exp", [a_ptr])
                elif action == "b_subtract": b_ptr = dag_op("subtract", [b_ptr])
                elif action == "b_maximum": b_ptr = dag_op("maximum", [b_ptr])
                elif action == "b_rsqrt": b_ptr = dag_op("rsqrt", [b_ptr])
                elif action in {"a_reduce", "b_reduce"}:
                    branch = a_ptr if action == "a_reduce" else b_ptr
                    reduced = dag_op("reduce_sum", [branch], reduction=True)
                    scaled = dag_op("multiply", [reduced])
                    joined = dag_op("add", [branch, scaled])
                    if action == "a_reduce": a_ptr = joined
                    else: b_ptr = joined
                elif action == "cross_add": a_ptr = dag_op("add", [a_ptr, b_ptr])
                else: b_ptr = dag_op("multiply", [a_ptr, b_ptr])
            dag_op("add", [a_ptr, b_ptr])
        elif kind.startswith("sequence_deepmixed2k_"):
            specs = {
                "00": [("add",1),("multiply",1),("subtract",1),("exp",1),("multiply",1),("add",1),("reduce_sum",1),("multiply",2),("subtract",1),("multiply",1),("add",1),("multiply",1),("subtract",1)],
                "01": [("multiply",1),("exp",1),("add",1),("subtract",1),("reduce_sum",1),("add",2),("multiply",1),("subtract",1),("add",1),("multiply",1),("add",1),("subtract",1),("multiply",1)],
                "02": [("multiply",1),("add",1),("rsqrt",1),("multiply",1),("subtract",1),("add",1),("reduce_sum",1),("multiply",2),("add",1),("subtract",1),("multiply",1),("add",1),("multiply",1)],
                "03": [("multiply",1),("add",1),("reduce_sum",1),("add",2),("multiply",1),("add",1),("rsqrt",1),("subtract",1),("multiply",1),("add",1),("multiply",1),("subtract",1),("add",1)],
                "04": [("multiply",1),("add",1),("log",1),("multiply",1),("add",1),("reduce_sum",1),("subtract",2),("multiply",1),("add",1),("subtract",1),("multiply",1),("add",1)],
                "05": [("add",1),("multiply",1),("reduce_sum",1),("multiply",2),("add",1),("log",1),("multiply",1),("subtract",1),("add",1),("multiply",1),("subtract",1)],
                "06": [("multiply",1),("reduce_sum",1),("reduce_sum",1),("multiply",2),("add",2),("subtract",1),("multiply",1),("add",1),("multiply",1),("subtract",1),("add",1),("multiply",1),("add",1)],
                "07": [("add",1),("multiply",1),("reduce_sum",1),("multiply",2),("reduce_sum",1),("add",2),("multiply",1),("subtract",1),("add",1),("multiply",1),("subtract",1),("add",1),("multiply",1)],
                "08": [("add",1),("multiply",1),("reduce_sum",1),("add",2),("subtract",1),("exp",1),("multiply",1),("add",1),("subtract",1),("multiply",1),("add",1),("multiply",1)],
                "09": [("subtract",1),("add",1),("multiply",1),("exp",1),("subtract",1),("multiply",1),("reduce_sum",1),("multiply",2),("add",1),("subtract",1),("multiply",1),("add",1)],
                "10": [("add",1),("reduce_sum",1),("multiply",2),("subtract",1),("multiply",1),("add",1),("rsqrt",1),("multiply",1),("add",1),("subtract",1),("multiply",1),("add",1)],
                "11": [("multiply",1),("add",1),("rsqrt",1),("subtract",1),("add",1),("multiply",1),("reduce_sum",1),("add",2),("multiply",1),("subtract",1),("add",1),("multiply",1)],
                "12": [("add",1),("reduce_sum",1),("multiply",2),("multiply",1),("add",1),("log",1),("subtract",1),("multiply",1),("add",1),("subtract",1),("multiply",1),("add",1)],
                "13": [("multiply",1),("add",1),("log",1),("multiply",1),("subtract",1),("reduce_sum",1),("add",2),("multiply",1),("add",1),("subtract",1),("multiply",1),("add",1)],
                "14": [("reduce_sum",1),("add",2),("multiply",1),("subtract",1),("multiply",1),("reduce_sum",1),("multiply",2),("add",1),("subtract",1),("multiply",1),("add",1),("multiply",1)],
                "15": [("multiply",1),("reduce_sum",1),("subtract",1),("multiply",2),("add",1),("reduce_sum",1),("add",2),("multiply",1),("subtract",1),("add",1),("multiply",1),("add",1)],
            }[kind.rsplit("_", 1)[1]]
            tokens, arities = map(list, zip(*specs))
        elif kind.startswith("sequence_deep2k_"):
            token = kind.removeprefix("sequence_deep2k_")
            if token == "add_multiply":
                tokens, arities = [value for _ in range(chain) for value in ("add", "multiply")], [1] * (2 * chain)
            else:
                tokens, arities = [token] * chain, [1] * chain
        elif kind.startswith("sequence_"):
            sequence = kind.removeprefix("sequence_").removesuffix("_wide_memory")
            specs = {
                "add_multiply": [("add", 2), ("multiply", 2)],
                "multiply_add": [("multiply", 2), ("add", 2)],
                "subtract_multiply_add": [("subtract", 2), ("multiply", 2), ("add", 2)],
                "multiply_subtract_multiply": [("multiply", 2), ("subtract", 2), ("multiply", 2)],
                "exp_multiply_add": [("exp", 1), ("multiply", 2), ("add", 2)],
                "log_add_multiply": [("log", 1), ("add", 2), ("multiply", 2)],
                "reduce_add_multiply": [("reduce_sum", 1), ("add", 1), ("multiply", 2)],
                "reduce_divide_multiply": [("reduce_sum", 1), ("divide", 1), ("multiply", 2)],
                "two_reduce_add": [("reduce_sum", 1), ("reduce_sum", 1), ("add", 2), ("multiply", 2)],
                "two_reduce_multiply": [("multiply", 2), ("reduce_sum", 1), ("reduce_sum", 1), ("multiply", 2), ("multiply", 2)],
                "rsqrt_multiply_add": [("rsqrt", 1), ("multiply", 2), ("add", 2)],
                "perm2k_ams": [("add", 1), ("multiply", 1), ("subtract", 1)],
                "perm2k_asm": [("add", 1), ("subtract", 1), ("multiply", 1)],
                "perm2k_mas": [("multiply", 1), ("add", 1), ("subtract", 1)],
                "perm2k_msa": [("multiply", 1), ("subtract", 1), ("add", 1)],
                "perm2k_sam": [("subtract", 1), ("add", 1), ("multiply", 1)],
                "perm2k_sma": [("subtract", 1), ("multiply", 1), ("add", 1)],
                "perm2k_eam": [("exp", 1), ("add", 1), ("multiply", 1)],
                "perm2k_aem": [("add", 1), ("exp", 1), ("multiply", 1)],
                "perm2k_mea": [("multiply", 1), ("exp", 1), ("add", 1)],
                "perm2k_ram": [("reduce_sum", 1), ("add", 1), ("multiply", 2)],
                "perm2k_arm": [("add", 1), ("reduce_sum", 1), ("multiply", 2)],
                "perm2k_mra": [("multiply", 1), ("reduce_sum", 1), ("add", 2)],
                "perm2k_long00": [("add", 1), ("multiply", 1), ("subtract", 1), ("add", 1), ("multiply", 1), ("subtract", 1)],
                "perm2k_long01": [("multiply", 1), ("subtract", 1), ("add", 1), ("multiply", 1), ("subtract", 1), ("add", 1)],
                "perm2k_long02": [("subtract", 1), ("add", 1), ("multiply", 1), ("subtract", 1), ("add", 1), ("multiply", 1)],
                "perm2k_long03": [("exp", 1), ("add", 1), ("multiply", 1), ("subtract", 1), ("multiply", 1), ("add", 1)],
                "perm2k_long04": [("add", 1), ("exp", 1), ("add", 1), ("multiply", 1), ("subtract", 1), ("multiply", 1)],
                "perm2k_long05": [("multiply", 1), ("exp", 1), ("add", 1), ("multiply", 1), ("subtract", 1), ("add", 1), ("multiply", 1)],
                "perm2k_long06": [("reduce_sum", 1), ("add", 1), ("multiply", 1), ("subtract", 1), ("multiply", 1), ("add", 1)],
                "perm2k_long07": [("add", 1), ("multiply", 1), ("reduce_sum", 1), ("add", 1), ("subtract", 1), ("multiply", 1)],
                "perm2k_long08": [("multiply", 1), ("subtract", 1), ("add", 1), ("reduce_sum", 1), ("multiply", 1), ("add", 1)],
                "perm2k_long09": [("reduce_sum", 1), ("multiply", 1), ("exp", 1), ("add", 1), ("subtract", 1), ("multiply", 1)],
                "perm2k_long10": [("add", 1), ("reduce_sum", 1), ("multiply", 1), ("subtract", 1), ("exp", 1), ("multiply", 1)],
                "perm2k_long11": [("multiply", 1), ("add", 1), ("reduce_sum", 1), ("subtract", 1), ("add", 1), ("multiply", 1)],
            }[sequence]
            tokens, arities = map(list, zip(*specs))
        elif kind == "elementwise_maximum_masked":
            tokens, arities = ["maximum"], [1]
        elif kind == "elementwise_multiply_masked":
            tokens, arities = ["multiply"], [1]
        elif kind == "elementwise_sigmoid_masked":
            tokens, arities = ["sigmoid"], [1]
        elif kind == "elementwise_maximum_wide_masked":
            tokens, arities = ["maximum"], [1]
        elif kind == "elementwise_multiply_wide_masked":
            tokens, arities = ["multiply"], [1]
        elif kind == "elementwise_sigmoid_wide_masked":
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
                        "mask_provided": mask_provided,
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
                "mask_provided": mask_provided,
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
    postcompile_audit: bool = False,
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
    # Keep the exact declared source grammar for compiler-lowering calibration,
    # and a separate runtime trace carrying physical SBUF dependency identity.
    # The two artifacts answer different questions and must not overwrite one
    # another.
    if kind.startswith(("sequence_perm2k_", "sequence_deep2k_", "sequence_deepmixed2k_", "sequence_randommixed2k", "sequence_randomsemantic2k", "sequence_randomdag2k", "sequence_factorialdag2k", "sequence_factorialdagaudit2k", "sequence_factorialdaginterleave2k", "padded_")):
        dependency_events = _declared_trace(
            kind, p, f, chain, dtype, case / "dependency_trace.jsonl"
        )
    else:
        dependency_events = _trace(kernel, inputs, case / "dependency_trace.jsonl")
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
        "dependency_trace_source": "runtime_nki",
        "dependency_trace_events": len(dependency_events),
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
        row.update(
            vector_active_ns=float(profile.get("vector_engine_active_time", 0)) * 1e9,
            scalar_active_ns=float(profile.get("scalar_engine_active_time", 0)) * 1e9,
            gpsimd_active_ns=float(profile.get("gpsimd_engine_active_time", 0)) * 1e9,
            # Aggregate Explorer counters are labels for independent controls,
            # not compiler instruction or DMA-packet features.  Keeping them
            # in the control manifest lets strict source-only models be fit
            # without opening any post-compile table.
            dynamic_dma_active_ns=float(
                profile.get(
                    "software_dynamic_dma_active_time",
                    profile.get("dynamic_dma_active_time", 0),
                )
            )
            * 1e9,
            static_dma_active_ns=float(profile.get("static_dma_active_time", 0))
            * 1e9,
        )
        if postcompile_audit:
            # Optional control-only compiler audit. Strict source-only Stage-2
            # collection leaves this disabled and records aggregate counters only.
            from triton_viz.tools.nki_explorer import export_parquet
            from triton_viz.tools.nki_instruction_source_mapping import write_case

            export_parquet(hardware_dir)
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
    parser.add_argument("--p", type=int, nargs="*", default=[128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-hardware", action="store_true")
    parser.add_argument(
        "--postcompile-audit",
        action="store_true",
        help="Control-only diagnostic; forbidden for strict source-only collection.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-dependency-traces",
        action="store_true",
        help="With --resume, retrace completed cases without rerunning hardware.",
    )
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
    for p in args.p:
        for dtype in args.dtypes:
            dims = args.free_dims
            for kind in args.kinds:
                chains = (
                    args.chains if kind in {"elementwise_one", "elementwise_two", "sequence_deep2k_add", "sequence_deep2k_multiply", "sequence_deep2k_add_multiply", "sequence_randommixed2k", "sequence_randomsemantic2k", "sequence_randomdag2k", "sequence_factorialdag2k", "sequence_factorialdagaudit2k", "sequence_factorialdaginterleave2k", "padded_randomdag"} else [1]
                )
                for f in dims:
                    if f > 2048 and kind not in {
                        "mask_tail",
                        "two_pass_reduce_affine",
                        "two_pass_reduce_multiply",
                    } and not kind.startswith("padded_"):
                        continue
                    for chain in chains:
                        case_name = _case_name(kind, p, f, chain, dtype)
                        if case_name in completed:
                            if args.refresh_dependency_traces:
                                existing = next(
                                    row for row in rows if row.get("case") == case_name
                                )
                                count = refresh_dependency_trace(args.output_dir, existing)
                                existing["dependency_trace_source"] = "runtime_nki"
                                existing["dependency_trace_events"] = count
                                print(
                                    f"REFRESH dependency trace {case_name} ({count} events)",
                                    flush=True,
                                )
                                continue
                            print(f"SKIP completed {case_name}", flush=True)
                            continue
                        rows = [row for row in rows if row.get("case") != case_name]
                        print(kind, dtype, p, f, chain, flush=True)
                        try:
                            rows.append(
                                run_case(
                                    args.output_dir,
                                    kind,
                                    p,
                                    f,
                                    chain,
                                    dtype,
                                    args.warmup,
                                    args.iters,
                                    not args.no_hardware,
                                    args.postcompile_audit,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 - preserve all cases in the audit CSV
                            print(
                                f"ERROR {kind} {dtype} p={p} F={f} chain={chain}: {exc!r}",
                                flush=True,
                            )
                            rows.append({"case": case_name, "error": repr(exc)})
                        fields = sorted({key for row in rows for key in row})
                        with results_path.open("w", newline="", encoding="utf-8") as file:
                            writer = csv.DictWriter(file, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(rows)
    return 1 if any(row.get("error") for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
