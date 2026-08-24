"""Source-only whole-program context features for NKI lowering controls.

The Region IR describes one contiguous compute region.  Compiler routing and
materialization may also depend on source-visible state spanning the whole
kernel: live-value pressure, producer/consumer distance, buffer reuse, memory
boundaries, and precision roles.  This module summarizes those facts without
exporting storage identities, pointers, case names, operator names, or any
post-compile artifact.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import math
from typing import Any

PROGRAM_CONTEXT_FEATURE_NAMES = frozenset(
    {
        "program_event_count",
        "program_compute_event_count",
        "program_memory_event_count",
        "program_load_count",
        "program_store_count",
        "program_transfer_count",
        "program_region_count",
        "program_storage_count",
        "program_storage_reuse_count",
        "program_reused_value_count",
        "program_peak_live_values",
        "program_peak_live_bytes",
        "program_live_distance_mean",
        "program_live_distance_max",
        "program_total_transfer_bytes",
        "program_hbm_read_bytes",
        "program_hbm_write_bytes",
        "program_dependency_edge_count",
        "program_dependency_distance_mean",
        "program_dependency_distance_max",
        "program_cross_region_dependency_count",
        "program_materialization_boundary_count",
        "program_logical_free_dim",
        "program_physical_free_dim",
        "program_allocation_to_logical_ratio",
        "program_partition_count",
        "program_masked_event_count",
        "program_input_bf16_region_count",
        "program_input_fp32_region_count",
        "program_accumulator_bf16_region_count",
        "program_accumulator_fp32_region_count",
        "program_mixed_precision_region_count",
        "program_dag_join_count",
        "program_dag_join_add_count",
        "program_dag_join_multiply_count",
        "program_dag_join_other_count",
        "program_dag_join_position_fraction_mean",
        "program_dag_join_branch_depth_min",
        "program_dag_join_branch_depth_max",
        "program_dag_join_branch_depth_imbalance_max",
        "program_dag_join_branch_reduction_min",
        "program_dag_join_branch_reduction_max",
        "program_dag_join_branch_reduction_imbalance_max",
        "program_dag_reduction_to_join_distance_mean",
        "program_dag_reduction_to_join_distance_max",
        "program_dag_reduction_before_join_count",
        "program_dag_reduction_after_join_count",
        "program_dag_post_join_event_count",
        "program_dag_post_join_transcendental_count",
        "program_dag_branch_source_interleave_count",
        "program_dag_branch_run_count_min",
        "program_dag_branch_run_count_max",
        "program_dag_branch_token_change_count_min",
        "program_dag_branch_token_change_count_max",
        "program_dag_branch_add_multiply_order_min",
        "program_dag_branch_add_multiply_order_max",
        "program_dag_branch_subtract_maximum_order_min",
        "program_dag_branch_subtract_maximum_order_max",
        "program_dag_branch_reduction_position_fraction_min",
        "program_dag_branch_reduction_position_fraction_max",
    }
)


def _range_bytes(value: Any) -> int:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return max(0, int(value[1]) - int(value[0]))
    return 0


def _dtype_bytes(dtype: Any) -> int:
    normalized = str(dtype or "").lower()
    if normalized in {"bool", "boolean", "int8", "uint8"}:
        return 1
    if normalized in {"float16", "fp16", "bfloat16", "bf16", "int16", "uint16"}:
        return 2
    if normalized in {"float64", "fp64", "int64", "uint64"}:
        return 8
    return 4


def _shape_bytes(shape: Any, dtype: Any) -> int:
    if not isinstance(shape, (list, tuple)) or not shape:
        return 0
    return int(math.prod(max(0, int(value)) for value in shape)) * _dtype_bytes(
        dtype
    )


def _items(event: dict[str, Any], prefix: str) -> list[tuple[Any, Any, Any]]:
    """Return storage/version/range triples for one event side."""
    plural_storages = list(event.get(f"{prefix}_storages") or ())
    plural_versions = list(event.get(f"{prefix}_versions") or ())
    plural_ranges = list(event.get(f"{prefix}_ranges") or ())
    result = []
    for index, storage in enumerate(plural_storages):
        result.append(
            (
                storage,
                plural_versions[index] if index < len(plural_versions) else None,
                plural_ranges[index] if index < len(plural_ranges) else None,
            )
        )
    storage = event.get(f"{prefix}_storage")
    if storage is not None:
        item = (
            storage,
            event.get(f"{prefix}_version"),
            event.get(f"{prefix}_range"),
        )
        if item not in result:
            result.append(item)
    if result:
        return result

    # Declared source controls intentionally omit runtime storage allocation
    # identities. Pointer-like source value IDs remain valid transient evidence
    # for DAG liveness as long as they are never exported as features.
    pointers = list(event.get(f"{prefix}_ptrs") or ())
    if not pointers:
        pointer = event.get(f"{prefix}_ptr")
        if pointer is not None:
            pointers = [pointer]
    versions = list(event.get(f"{prefix}_versions") or ())
    ranges = list(event.get(f"{prefix}_ranges") or ())
    dtypes = list(event.get(f"{prefix}_dtypes") or ())
    shape = event.get(f"{prefix}_shape")
    singular_dtype = event.get(f"{prefix}_dtype")
    for index, pointer in enumerate(pointers):
        value_range = ranges[index] if index < len(ranges) else None
        if value_range is None:
            dtype = dtypes[index] if index < len(dtypes) else singular_dtype
            width = _shape_bytes(shape, dtype)
            value_range = [0, width] if width else None
        result.append(
            (
                pointer,
                versions[index] if index < len(versions) else None,
                value_range,
            )
        )
    return result


def _dtype_role_counts(regions: list[dict[str, Any]]) -> dict[str, float]:
    result = {
        "program_input_bf16_region_count": 0.0,
        "program_input_fp32_region_count": 0.0,
        "program_accumulator_bf16_region_count": 0.0,
        "program_accumulator_fp32_region_count": 0.0,
        "program_mixed_precision_region_count": 0.0,
    }
    for region in regions:
        input_dtype = str(
            region.get("input_dtype") or region.get("dtype") or ""
        ).lower()
        accumulator_dtype = str(
            region.get("accumulator_dtype") or region.get("dtype") or ""
        ).lower()
        if input_dtype in {"bfloat16", "bf16"}:
            result["program_input_bf16_region_count"] += 1.0
        elif input_dtype in {"float32", "fp32", "f32"}:
            result["program_input_fp32_region_count"] += 1.0
        if accumulator_dtype in {"bfloat16", "bf16"}:
            result["program_accumulator_bf16_region_count"] += 1.0
        elif accumulator_dtype in {"float32", "fp32", "f32"}:
            result["program_accumulator_fp32_region_count"] += 1.0
        result["program_mixed_precision_region_count"] += float(
            input_dtype != accumulator_dtype
        )
    return result


def _region_dag_topology(region: dict[str, Any]) -> dict[str, float]:
    tokens = [str(token) for token in region.get("tokens") or ()]
    node_count = len(tokens)
    edges = [
        (int(source), int(target))
        for source, target in region.get("dag_edges") or ()
        if 0 <= int(source) < node_count and 0 <= int(target) < node_count
    ]
    predecessors = {index: [] for index in range(node_count)}
    successors = {index: [] for index in range(node_count)}
    for source, target in edges:
        predecessors[target].append(source)
        successors[source].append(target)

    depths = [0] * node_count
    ancestor_reductions: list[set[int]] = [set() for _ in range(node_count)]
    source_roots: list[set[int]] = [set() for _ in range(node_count)]
    for index in range(node_count):
        if not predecessors[index]:
            source_roots[index].add(index)
        for source in predecessors[index]:
            depths[index] = max(depths[index], depths[source] + 1)
            ancestor_reductions[index].update(ancestor_reductions[source])
            source_roots[index].update(source_roots[source])
            if tokens[source] in {"reduce_sum", "max", "min", "mean"}:
                ancestor_reductions[index].add(source)

    # A reduction broadcast (value + reduced(value)) is a local fan-in, not a
    # cross-branch routing join. Keep only fan-ins whose predecessors carry at
    # least two distinct source roots.
    joins = [
        index
        for index in range(node_count)
        if len(predecessors[index]) >= 2
        and len(
            set().union(*(source_roots[source] for source in predecessors[index]))
        )
        >= 2
    ]
    branch_depths = []
    branch_reductions = []
    reduction_distances = []
    branch_depth_imbalances = []
    branch_reduction_imbalances = []
    for join in joins:
        join_depths = [depths[source] for source in predecessors[join]]
        join_reductions = [
            len(
                ancestor_reductions[source]
                | (
                    {source}
                    if tokens[source] in {"reduce_sum", "max", "min", "mean"}
                    else set()
                )
            )
            for source in predecessors[join]
        ]
        branch_depths.extend(join_depths)
        branch_reductions.extend(join_reductions)
        if join_depths:
            branch_depth_imbalances.append(max(join_depths) - min(join_depths))
        if join_reductions:
            branch_reduction_imbalances.append(
                max(join_reductions) - min(join_reductions)
            )
        reduction_distances.extend(
            join - reduction
            for reduction in ancestor_reductions[join]
            if reduction < join
        )

    first_join = min(joins, default=node_count)
    reductions = [
        index
        for index, token in enumerate(tokens)
        if token in {"reduce_sum", "max", "min", "mean"}
    ]
    post_join_tokens = tokens[first_join + 1 :] if joins else []
    transcendental = {
        "exp",
        "log",
        "rsqrt",
        "sqrt",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
    }

    branch_sequences: list[list[tuple[int, str]]] = []
    source_interleave_count = 0
    if joins:
        first_predecessors = predecessors[first_join]
        branch_nodes = []
        for predecessor in first_predecessors:
            nodes = set()
            stack = [predecessor]
            while stack:
                current = stack.pop()
                if current in nodes or current >= first_join:
                    continue
                nodes.add(current)
                stack.extend(predecessors[current])
            branch_nodes.append(nodes)
        ownership = {}
        for branch_index, nodes in enumerate(branch_nodes):
            for node in nodes:
                ownership.setdefault(node, []).append(branch_index)
        branch_sequences = [
            [
                (node, tokens[node])
                for node in sorted(nodes)
                if len(ownership.get(node, ())) == 1
            ]
            for nodes in branch_nodes
        ]
        source_owners = [
            ownership[node][0]
            for node in sorted(ownership)
            if len(ownership[node]) == 1
        ]
        source_interleave_count = sum(
            left != right
            for left, right in zip(source_owners, source_owners[1:])
        )

    branch_run_counts = []
    branch_change_counts = []
    add_multiply_orders = []
    subtract_maximum_orders = []
    reduction_positions = []
    for sequence in branch_sequences:
        branch_tokens = [token for _index, token in sequence]
        if not branch_tokens:
            continue
        branch_run_counts.append(
            1
            + sum(
                left != right
                for left, right in zip(branch_tokens, branch_tokens[1:])
            )
        )
        branch_change_counts.append(
            sum(
                left != right
                for left, right in zip(branch_tokens, branch_tokens[1:])
            )
        )

        def order_score(first: str, second: str) -> float | None:
            first_positions = [
                index for index, token in enumerate(branch_tokens) if token == first
            ]
            second_positions = [
                index for index, token in enumerate(branch_tokens) if token == second
            ]
            if not first_positions or not second_positions:
                return None
            first_mean = sum(first_positions) / len(first_positions)
            second_mean = sum(second_positions) / len(second_positions)
            return (second_mean - first_mean) / max(1, len(branch_tokens) - 1)

        score = order_score("add", "multiply")
        if score is not None:
            add_multiply_orders.append(score)
        score = order_score("subtract", "maximum")
        if score is not None:
            subtract_maximum_orders.append(score)
        branch_reductions = [
            index
            for index, token in enumerate(branch_tokens)
            if token in {"reduce_sum", "max", "min", "mean"}
        ]
        if branch_reductions:
            reduction_positions.append(
                sum(branch_reductions)
                / len(branch_reductions)
                / max(1, len(branch_tokens) - 1)
            )
    return {
        "program_dag_join_count": float(len(joins)),
        "program_dag_join_add_count": float(
            sum(tokens[index] == "add" for index in joins)
        ),
        "program_dag_join_multiply_count": float(
            sum(tokens[index] == "multiply" for index in joins)
        ),
        "program_dag_join_other_count": float(
            sum(tokens[index] not in {"add", "multiply"} for index in joins)
        ),
        "program_dag_join_position_fraction_mean": (
            float(sum(joins) / len(joins) / max(1, node_count - 1))
            if joins
            else 0.0
        ),
        "program_dag_join_branch_depth_min": float(
            min(branch_depths, default=0)
        ),
        "program_dag_join_branch_depth_max": float(
            max(branch_depths, default=0)
        ),
        "program_dag_join_branch_depth_imbalance_max": float(
            max(branch_depth_imbalances, default=0)
        ),
        "program_dag_join_branch_reduction_min": float(
            min(branch_reductions, default=0)
        ),
        "program_dag_join_branch_reduction_max": float(
            max(branch_reductions, default=0)
        ),
        "program_dag_join_branch_reduction_imbalance_max": float(
            max(branch_reduction_imbalances, default=0)
        ),
        "program_dag_reduction_to_join_distance_mean": (
            float(sum(reduction_distances) / len(reduction_distances))
            if reduction_distances
            else 0.0
        ),
        "program_dag_reduction_to_join_distance_max": float(
            max(reduction_distances, default=0)
        ),
        "program_dag_reduction_before_join_count": float(
            sum(index < first_join for index in reductions)
        ),
        "program_dag_reduction_after_join_count": float(
            sum(index > first_join for index in reductions)
        ),
        "program_dag_post_join_event_count": float(len(post_join_tokens)),
        "program_dag_post_join_transcendental_count": float(
            sum(token in transcendental for token in post_join_tokens)
        ),
        "program_dag_branch_source_interleave_count": float(
            source_interleave_count
        ),
        "program_dag_branch_run_count_min": float(
            min(branch_run_counts, default=0)
        ),
        "program_dag_branch_run_count_max": float(
            max(branch_run_counts, default=0)
        ),
        "program_dag_branch_token_change_count_min": float(
            min(branch_change_counts, default=0)
        ),
        "program_dag_branch_token_change_count_max": float(
            max(branch_change_counts, default=0)
        ),
        "program_dag_branch_add_multiply_order_min": float(
            min(add_multiply_orders, default=0.0)
        ),
        "program_dag_branch_add_multiply_order_max": float(
            max(add_multiply_orders, default=0.0)
        ),
        "program_dag_branch_subtract_maximum_order_min": float(
            min(subtract_maximum_orders, default=0.0)
        ),
        "program_dag_branch_subtract_maximum_order_max": float(
            max(subtract_maximum_orders, default=0.0)
        ),
        "program_dag_branch_reduction_position_fraction_min": float(
            min(reduction_positions, default=0.0)
        ),
        "program_dag_branch_reduction_position_fraction_max": float(
            max(reduction_positions, default=0.0)
        ),
    }


def program_context_features(
    events: Iterable[dict[str, Any]],
) -> dict[str, float]:
    """Summarize source-visible liveness, geometry, and dependency context.

    Storage identities and versions are used transiently to reconstruct source
    dependence and lifetimes.  They are never returned as feature names or
    values, preventing the context vector from becoming a target identity key.
    """
    source = list(events)
    leaders: dict[Any, dict[str, Any]] = {}
    for event in source:
        region = event.get("region_ir")
        group = event.get("fusion_group")
        if region is not None and group not in leaders:
            leaders[group] = region
    regions = list(leaders.values())

    produced_at: dict[tuple[Any, Any], int] = {}
    last_used_at: dict[tuple[Any, Any], int] = {}
    value_bytes: dict[tuple[Any, Any], int] = {}
    base_versions: dict[Any, set[Any]] = {}
    dependency_distances = []
    dependency_edges = 0
    consumers = Counter()
    cross_region_dependencies = 0

    for index, event in enumerate(source):
        consumer_group = event.get("fusion_group")
        for storage, version, value_range in _items(event, "input"):
            identity = (storage, version)
            last_used_at[identity] = index
            value_bytes[identity] = max(
                value_bytes.get(identity, 0), _range_bytes(value_range)
            )
            consumers[identity] += 1
            if identity in produced_at:
                producer_index = produced_at[identity]
                dependency_edges += 1
                dependency_distances.append(index - producer_index)
                producer_group = source[producer_index].get("fusion_group")
                cross_region_dependencies += int(
                    producer_group is not None
                    and consumer_group is not None
                    and producer_group != consumer_group
                )
        for storage, version, value_range in _items(event, "output"):
            identity = (storage, version)
            produced_at.setdefault(identity, index)
            last_used_at.setdefault(identity, index)
            value_bytes[identity] = max(
                value_bytes.get(identity, 0), _range_bytes(value_range)
            )
            base_versions.setdefault(storage, set()).add(version)

    peak_live_values = 0
    peak_live_bytes = 0
    lifetimes = []
    for index in range(len(source)):
        live = [
            identity
            for identity, birth in produced_at.items()
            if birth <= index < last_used_at.get(identity, birth)
        ]
        peak_live_values = max(peak_live_values, len(live))
        peak_live_bytes = max(
            peak_live_bytes, sum(value_bytes.get(identity, 0) for identity in live)
        )
    for identity, birth in produced_at.items():
        lifetimes.append(max(0, last_used_at.get(identity, birth) - birth))

    memory_events = [
        event
        for event in source
        if event.get("op") in {"load", "store", "transfer"}
    ]
    op_counts = Counter(str(event.get("op") or "") for event in source)
    total_transfer_bytes = sum(int(event.get("bytes") or 0) for event in memory_events)
    hbm_read_bytes = sum(
        int(event.get("bytes") or 0)
        for event in memory_events
        if str(event.get("mem_src") or "").upper() == "HBM"
    )
    hbm_write_bytes = sum(
        int(event.get("bytes") or 0)
        for event in memory_events
        if str(event.get("mem_dst") or "").upper() == "HBM"
    )
    logical_free = max(
        (
            int(region.get("logical_free_dim") or region.get("free_dim") or 1)
            for region in regions
        ),
        default=1,
    )
    physical_free = max(
        (int(region.get("free_dim") or 1) for region in regions), default=1
    )
    partitions = max(
        (int(region.get("partition_count") or 1) for region in regions),
        default=1,
    )

    # Memory operations occurring between two compute regions are explicit
    # source materialization boundaries. Consecutive memory events count once
    # per event because each can independently alter compiler allocation.
    compute_indices = [
        index
        for index, event in enumerate(source)
        if event.get("op") in {"compute", "reduce_sum", "binary", "dot"}
    ]
    first_compute = min(compute_indices, default=0)
    last_compute = max(compute_indices, default=-1)
    boundary_memory_events = sum(
        first_compute < index < last_compute
        and event.get("op") in {"load", "store", "transfer"}
        for index, event in enumerate(source)
    )

    result = {
        "program_event_count": float(len(source)),
        "program_compute_event_count": float(
            sum(
                event.get("op") in {"compute", "reduce_sum", "binary", "dot"}
                for event in source
            )
        ),
        "program_memory_event_count": float(len(memory_events)),
        "program_load_count": float(op_counts["load"]),
        "program_store_count": float(op_counts["store"]),
        "program_transfer_count": float(op_counts["transfer"]),
        "program_region_count": float(len(regions)),
        "program_storage_count": float(len({item[0] for item in produced_at})),
        "program_storage_reuse_count": float(
            sum(max(0, len(versions) - 1) for versions in base_versions.values())
        ),
        "program_reused_value_count": float(
            sum(count > 1 for count in consumers.values())
        ),
        "program_peak_live_values": float(peak_live_values),
        "program_peak_live_bytes": float(peak_live_bytes),
        "program_live_distance_mean": (
            float(sum(lifetimes) / len(lifetimes)) if lifetimes else 0.0
        ),
        "program_live_distance_max": float(max(lifetimes, default=0)),
        "program_total_transfer_bytes": float(total_transfer_bytes),
        "program_hbm_read_bytes": float(hbm_read_bytes),
        "program_hbm_write_bytes": float(hbm_write_bytes),
        "program_dependency_edge_count": float(dependency_edges),
        "program_dependency_distance_mean": (
            float(sum(dependency_distances) / len(dependency_distances))
            if dependency_distances
            else 0.0
        ),
        "program_dependency_distance_max": float(
            max(dependency_distances, default=0)
        ),
        "program_cross_region_dependency_count": float(
            cross_region_dependencies
        ),
        "program_materialization_boundary_count": float(boundary_memory_events),
        "program_logical_free_dim": float(logical_free),
        "program_physical_free_dim": float(physical_free),
        "program_allocation_to_logical_ratio": float(
            physical_free / max(1, logical_free)
        ),
        "program_partition_count": float(partitions),
        "program_masked_event_count": float(
            sum(
                bool(
                    event.get("mask_provided")
                    or event.get("compute_mask_provided")
                )
                for event in source
            )
        ),
    }
    result.update(_dtype_role_counts(regions))
    topology = Counter()
    for region in regions:
        topology.update(_region_dag_topology(region))
    result.update(
        {
            name: float(topology.get(name, 0.0))
            for name in PROGRAM_CONTEXT_FEATURE_NAMES
            if name.startswith("program_dag_")
        }
    )
    if set(result) != PROGRAM_CONTEXT_FEATURE_NAMES:
        missing = sorted(PROGRAM_CONTEXT_FEATURE_NAMES - set(result))
        unexpected = sorted(set(result) - PROGRAM_CONTEXT_FEATURE_NAMES)
        raise AssertionError(
            f"ProgramContextIR feature schema drift: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return result
