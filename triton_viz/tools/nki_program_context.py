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
        "program_dag_branch0_event_count",
        "program_dag_branch1_event_count",
        "program_dag_branch0_depth_max",
        "program_dag_branch1_depth_max",
        "program_dag_branch0_reduction_count",
        "program_dag_branch1_reduction_count",
        "program_dag_branch0_add_count",
        "program_dag_branch1_add_count",
        "program_dag_branch0_multiply_count",
        "program_dag_branch1_multiply_count",
        "program_dag_branch0_subtract_count",
        "program_dag_branch1_subtract_count",
        "program_dag_branch0_maximum_count",
        "program_dag_branch1_maximum_count",
        "program_dag_branch0_transcendental_count",
        "program_dag_branch1_transcendental_count",
        "program_dag_branch0_root_add",
        "program_dag_branch1_root_add",
        "program_dag_branch0_root_multiply",
        "program_dag_branch1_root_multiply",
        "program_dag_branch_event_count_signed_difference",
        "program_dag_branch_depth_signed_difference",
        "program_dag_branch_reduction_signed_difference",
        "program_dag_branch_add_signed_difference",
        "program_dag_branch_multiply_signed_difference",
        "program_dag_join_update_branch0_count",
        "program_dag_join_update_branch1_count",
        "program_dag_join_update_ambiguous_count",
        "program_dag_join_update_signed_difference",
        "program_dag_join_update_transition_count",
        "program_dag_first_join_update_branch0",
        "program_dag_first_join_update_branch1",
        "program_dag_last_join_update_branch0",
        "program_dag_last_join_update_branch1",
        "program_dag_branch0_local_fanout_count",
        "program_dag_branch1_local_fanout_count",
        "program_dag_branch0_local_rejoin_count",
        "program_dag_branch1_local_rejoin_count",
        "program_dag_local_fanout_count",
        "program_dag_local_rejoin_count",
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
    owners: list[int | None] = [None] * node_count
    root_indices = [index for index in range(node_count) if not predecessors[index]]
    for owner, root in enumerate(root_indices[:2]):
        owners[root] = owner
    join_update_owners = []
    for index in range(node_count):
        if owners[index] is not None:
            continue
        predecessor_owners = {
            owners[source]
            for source in predecessors[index]
            if owners[source] is not None
        }
        if len(predecessor_owners) == 1:
            owners[index] = next(iter(predecessor_owners))
            continue
        if len(predecessor_owners) < 2:
            continue
        reused = [
            source
            for source in predecessors[index]
            if any(successor > index for successor in successors[source])
        ]
        update_owner = None
        if len(reused) == 1 and owners[reused[0]] is not None:
            preserved_owner = owners[reused[0]]
            replaced = predecessor_owners - {preserved_owner}
            if len(replaced) == 1:
                update_owner = next(iter(replaced))
        owners[index] = update_owner
        if index in joins:
            join_update_owners.append(update_owner)
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
        branch_sequences.sort(
            key=lambda sequence: (
                min((index for index, _token in sequence), default=node_count),
                len(sequence),
            )
        )
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

    branch_descriptors = []
    for sequence in branch_sequences[:2]:
        indices = [index for index, _token in sequence]
        index_set = set(indices)
        branch_tokens = [token for _index, token in sequence]
        local_fanout_count = sum(
            len([successor for successor in successors[index] if successor in index_set])
            > 1
            and not any(
                tokens[successor] in {"reduce_sum", "max", "min", "mean"}
                for successor in successors[index]
                if successor in index_set
            )
            for index in indices
        )
        local_rejoin_count = sum(
            len([source for source in predecessors[index] if source in index_set])
            > 1
            and not any(
                tokens[source] in {"reduce_sum", "max", "min", "mean"}
                for source in predecessors[index]
                if source in index_set
            )
            for index in indices
        )
        branch_descriptors.append(
            {
                "event_count": len(sequence),
                "depth_max": max((depths[index] for index in indices), default=0),
                "reduction_count": sum(
                    token in {"reduce_sum", "max", "min", "mean"}
                    for token in branch_tokens
                ),
                "add_count": branch_tokens.count("add"),
                "multiply_count": branch_tokens.count("multiply"),
                "subtract_count": branch_tokens.count("subtract"),
                "maximum_count": branch_tokens.count("maximum"),
                "transcendental_count": sum(
                    token in transcendental for token in branch_tokens
                ),
                "root_add": int(bool(branch_tokens) and branch_tokens[0] == "add"),
                "root_multiply": int(
                    bool(branch_tokens) and branch_tokens[0] == "multiply"
                ),
                "local_fanout_count": local_fanout_count,
                "local_rejoin_count": local_rejoin_count,
            }
        )
    while len(branch_descriptors) < 2:
        branch_descriptors.append(
            {
                "event_count": 0,
                "depth_max": 0,
                "reduction_count": 0,
                "add_count": 0,
                "multiply_count": 0,
                "subtract_count": 0,
                "maximum_count": 0,
                "transcendental_count": 0,
                "root_add": 0,
                "root_multiply": 0,
                "local_fanout_count": 0,
                "local_rejoin_count": 0,
            }
        )
    branch0, branch1 = branch_descriptors
    known_join_update_owners = [
        owner for owner in join_update_owners if owner in {0, 1}
    ]
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
        **{
            f"program_dag_branch{branch}_{name}": float(descriptor[name])
            for branch, descriptor in enumerate(branch_descriptors)
            for name in (
                "event_count",
                "depth_max",
                "reduction_count",
                "add_count",
                "multiply_count",
                "subtract_count",
                "maximum_count",
                "transcendental_count",
                "root_add",
                "root_multiply",
                "local_fanout_count",
                "local_rejoin_count",
            )
        },
        "program_dag_branch_event_count_signed_difference": float(
            branch0["event_count"] - branch1["event_count"]
        ),
        "program_dag_branch_depth_signed_difference": float(
            branch0["depth_max"] - branch1["depth_max"]
        ),
        "program_dag_branch_reduction_signed_difference": float(
            branch0["reduction_count"] - branch1["reduction_count"]
        ),
        "program_dag_branch_add_signed_difference": float(
            branch0["add_count"] - branch1["add_count"]
        ),
        "program_dag_branch_multiply_signed_difference": float(
            branch0["multiply_count"] - branch1["multiply_count"]
        ),
        "program_dag_join_update_branch0_count": float(
            known_join_update_owners.count(0)
        ),
        "program_dag_join_update_branch1_count": float(
            known_join_update_owners.count(1)
        ),
        "program_dag_join_update_ambiguous_count": float(
            len(join_update_owners) - len(known_join_update_owners)
        ),
        "program_dag_join_update_signed_difference": float(
            known_join_update_owners.count(0)
            - known_join_update_owners.count(1)
        ),
        "program_dag_join_update_transition_count": float(
            sum(
                left != right
                for left, right in zip(
                    known_join_update_owners,
                    known_join_update_owners[1:],
                )
            )
        ),
        "program_dag_first_join_update_branch0": float(
            bool(known_join_update_owners)
            and known_join_update_owners[0] == 0
        ),
        "program_dag_first_join_update_branch1": float(
            bool(known_join_update_owners)
            and known_join_update_owners[0] == 1
        ),
        "program_dag_last_join_update_branch0": float(
            bool(known_join_update_owners)
            and known_join_update_owners[-1] == 0
        ),
        "program_dag_last_join_update_branch1": float(
            bool(known_join_update_owners)
            and known_join_update_owners[-1] == 1
        ),
        "program_dag_local_fanout_count": float(
            branch0["local_fanout_count"] + branch1["local_fanout_count"]
        ),
        "program_dag_local_rejoin_count": float(
            branch0["local_rejoin_count"] + branch1["local_rejoin_count"]
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


def source_routing_state(features: dict[str, float]) -> str:
    """Classify a reusable source phase from low-dimensional DAG topology.

    The state is intentionally coarser than a structural key. It captures only
    branch interleaving, post-join topology, and local add/multiply order; it
    does not use case names, exact token strings, pointers, or hashes.
    """
    join_count = float(features.get("program_dag_join_count", 0.0))
    if join_count <= 0:
        return "other"
    if float(
        features.get("program_dag_branch_source_interleave_count", 0.0)
    ) >= 5.0:
        return "interleaved"
    if (
        join_count >= 4.0
        or float(features.get("program_dag_post_join_event_count", 0.0)) >= 6.0
    ):
        return "blocked"
    if float(
        features.get("program_dag_branch_add_multiply_order_max", 0.0)
    ) < 0.1:
        return "reversed"
    return "canonical"


def source_join_ownership_state(features: dict[str, float]) -> str:
    """Classify which oriented source branch receives cross-join results."""
    branch0 = int(features.get("program_dag_join_update_branch0_count", 0.0))
    branch1 = int(features.get("program_dag_join_update_branch1_count", 0.0))
    if branch0 > 0 and branch1 == 0:
        return "branch0_only"
    if branch1 > 0 and branch0 == 0:
        return "branch1_only"
    if branch0 <= 0 or branch1 <= 0:
        return "ambiguous"
    first = (
        "branch0"
        if features.get("program_dag_first_join_update_branch0", 0.0)
        else (
            "branch1"
            if features.get("program_dag_first_join_update_branch1", 0.0)
            else "unknown"
        )
    )
    transitions = min(
        2, int(features.get("program_dag_join_update_transition_count", 0.0))
    )
    return f"mixed_{first}_t{transitions}"


def source_routing_regime(features: dict[str, float]) -> str:
    """Return the factored source phase × join-ownership regime."""
    return (
        f"{source_routing_state(features)}:"
        f"{source_join_ownership_state(features)}"
    )


def source_local_topology_state(features: dict[str, float]) -> str:
    """Classify non-reduction branch-local fanout/rejoin topology."""
    branch0 = (
        features.get("program_dag_branch0_local_fanout_count", 0.0) > 0
        or features.get("program_dag_branch0_local_rejoin_count", 0.0) > 0
    )
    branch1 = (
        features.get("program_dag_branch1_local_fanout_count", 0.0) > 0
        or features.get("program_dag_branch1_local_rejoin_count", 0.0) > 0
    )
    if branch0 and branch1:
        return "fanout_both"
    if branch0:
        return "fanout_branch0"
    if branch1:
        return "fanout_branch1"
    return "linear"


def source_full_routing_regime(features: dict[str, float]) -> str:
    """Return source phase × join ownership × local topology."""
    return (
        f"{source_routing_regime(features)}:"
        f"{source_local_topology_state(features)}"
    )


def source_branch_orientation_state(features: dict[str, float]) -> str:
    """Return the ordered source-root primitive orientation."""
    if (
        features.get("program_dag_branch0_root_add", 0.0) > 0
        and features.get("program_dag_branch1_root_multiply", 0.0) > 0
    ):
        return "add_mul"
    if (
        features.get("program_dag_branch0_root_multiply", 0.0) > 0
        and features.get("program_dag_branch1_root_add", 0.0) > 0
    ):
        return "mul_add"
    return "other"


def source_complete_routing_regime(features: dict[str, float]) -> str:
    """Return phase × ownership × local topology × root orientation."""
    return (
        f"{source_full_routing_regime(features)}:"
        f"{source_branch_orientation_state(features)}"
    )
