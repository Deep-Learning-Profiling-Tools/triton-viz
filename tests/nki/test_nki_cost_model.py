import numpy as np
import pytest
import triton_viz
import csv

try:
    import nki.isa as nisa
    import nki.language as nl
    from triton_viz.clients import Tracer
    from triton_viz.core.trace import launches
    from triton_viz.tools.nki_trace_dump import records_to_events
    from triton_viz.tools.nki_cost_model import (
        CostModel,
        ComputeCalibration,
        DmaCalibrationSurface,
        LoweringExpansionCalibration,
        RuntimeOverheadCalibration,
        StaticDmaCalibrationSurface,
        StridedDmaCalibration,
        StructuredControlCalibration,
        TensorCalibrationSurface,
        simulate,
    )
    from triton_viz.tools.nki_region_ir import structural_calibration_key
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki


def _kernel(lhs_t, rhs, out):
    lhs_tile = nl.ndarray((128, 128), dtype=lhs_t.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    res_psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    out_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(lhs_tile, lhs_t)
    nisa.dma_copy(rhs_tile, rhs)
    nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)
    nisa.tensor_copy(out_tile, res_psum)
    nisa.dma_copy(out, out_tile)


def _events():
    triton_viz.clear()
    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(_kernel)
    lhs_t = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    rhs = np.arange(128 * 512, dtype=np.float32).reshape(128, 512)
    out = np.empty((128, 512), dtype=np.float32)
    traced[(1,)](lhs_t, rhs, out)
    return records_to_events(launches[-1].records)


def test_cost_model_costs_are_positive():
    model = CostModel()
    dma = {"op": "transfer", "engine": "dma", "bytes": 65536}
    dot = {"op": "dot", "engine": "tensor", "flops": 2 * 128 * 512 * 128}
    grid = {"op": "grid"}
    assert model.cost_ns(dma) > 0
    assert model.cost_ns(dot) > 0
    assert model.cost_ns(grid) == 0.0
    assert model.cost_ns({"op": "load", "engine": "dma", "bytes": 8}) < (
        model.cost_ns({"op": "load", "engine": "dma", "bytes": 16})
    )


def test_dma_cost_uses_partition_geometry_and_calibration():
    model = CostModel(dma_startup_ns=0)
    one_engine = {"op": "transfer", "engine": "dma", "bytes": 8192,
                  "partition_count": 8, "free_bytes_per_partition": 1024}
    two_engines = {**one_engine, "partition_count": 16, "bytes": 16384}
    assert model.cost_ns(one_engine) == pytest.approx(model.cost_ns(two_engines))

    calibration = DmaCalibrationSurface({(8, 1024): 10.0, (16, 1024): 18.0})
    calibrated = CostModel(dma_calibration=calibration)
    assert calibrated.cost_ns(one_engine) == pytest.approx(819.2)

    transpose_calibration = DmaCalibrationSurface({(8, 1024): 5.0})
    dual = CostModel(dma_calibration=calibration, dma_transpose_calibration=transpose_calibration)
    transpose_event = {**one_engine, "dma_pattern": "transpose"}
    assert dual.cost_ns(transpose_event) == pytest.approx(1638.4)

    write_calibration = DmaCalibrationSurface({(8, 1024): 20.0})
    directional = CostModel(
        dma_calibration=calibration, dma_write_calibration=write_calibration
    )
    store_event = {**one_engine, "mem_src": "sbuf", "mem_dst": "hbm"}
    load_event = {**one_engine, "mem_src": "hbm", "mem_dst": "sbuf"}
    assert directional.cost_ns(store_event) == pytest.approx(409.6)
    assert directional.cost_ns(load_event) == pytest.approx(819.2)


def test_dma_surface_is_the_only_calibrated_dma_path():
    surface = DmaCalibrationSurface({(8, 1024): 10.0})
    events = [
        {
            "seq": index,
            "op": "load",
            "engine": "dma",
            "mem_src": "hbm",
            "mem_dst": "sbuf",
            "bytes": 100,
            "partition_count": 8,
            "free_bytes_per_partition": 1024,
            "src_ptr": index,
            "dst_ptr": index + 100,
        }
        for index in (1, 2)
    ]
    result = simulate(
        events,
        CostModel(
            dma_calibration=surface,
            dma_resource_count=0,
        ),
    )
    assert result.engine_busy_ns["dma"] == pytest.approx(20.0)
    assert result.components_ns["dma_surface_exact_count"] == 2


def test_simulate_produces_timeline_and_latency():
    result = simulate(_events())
    # Positive predicted latency.
    assert result.predicted_latency_ns > 0
    # Both DMA and TensorE engines must appear in the timeline.
    assert "dma" in result.timeline
    assert "tensor" in result.timeline
    # Exactly one TensorE dot.
    assert len(result.timeline["tensor"]) == 1
    # Utilization is a fraction in [0, 1] for every engine.
    for util in result.as_dict()["engine_utilization"].values():
        assert 0.0 <= util <= 1.0


def test_simulate_models_dependency_and_overlap():
    result = simulate(_events())
    dot = result.timeline["tensor"][0]
    dma_entries = sorted(result.timeline["dma"], key=lambda e: e.start)
    # The matmul must not start before its two input DMAs both finish (data dep).
    input_dma_end = max(e.end for e in dma_entries[:2])
    assert dot.start >= input_dma_end - 1e-6
    # This kernel (load lhs -> load rhs -> matmul -> psum copy -> store) is a
    # genuine serial true-dependency chain now that TensorE records its output
    # pointer, so the makespan legitimately equals the serial sum. The scheduler
    # must never *exceed* the serial sum (that would be double counting).
    serial_sum = sum(
        e.end - e.start
        for entries in result.timeline.values()
        for e in entries
    )
    assert result.predicted_latency_ns <= serial_sum + 1e-6


def test_simulate_overlaps_independent_cross_engine_work():
    # Two independent chains: a DMA load feeds a VectorE op, while a second,
    # unrelated DMA load runs on the DMA engine concurrently with that VectorE
    # op. Genuine cross-engine overlap must make the makespan strictly smaller
    # than the fully serial sum.
    model = CostModel(
        dma_startup_ns=0, dma_bytes_per_ns=1,
        vector_startup_ns=0, vector_elements_per_ns=1,
    )
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "dst_ptr": 101},
        {"seq": 1, "op": "binary", "engine": "vector", "elements": 10,
         "input_ptrs": [101], "output_ptr": 102},
        {"seq": 2, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 2, "dst_ptr": 201},
    ]
    result = simulate(events, model)
    serial_sum = sum(
        e.end - e.start for entries in result.timeline.values() for e in entries
    )
    # seq1 (vector, 10->20) overlaps seq2 (dma, 10->20): makespan 20 < serial 30.
    assert result.predicted_latency_ns == pytest.approx(20)
    assert result.predicted_latency_ns < serial_sum


def test_simulate_models_write_after_read_and_write_after_write():
    # Buffer-reuse anti-hazards across engines. Buffer 101 is filled by a DMA
    # load (seq0), read by a VectorE op (seq1), then *overwritten* by a second
    # DMA load (seq2). The overwrite must wait for the reader to finish (WAR),
    # not float back to the producer's completion.
    model = CostModel(
        dma_startup_ns=0, dma_bytes_per_ns=1,
        vector_startup_ns=0, vector_elements_per_ns=1,
    )
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "dst_ptr": 101},           # write 101: 0->10
        {"seq": 1, "op": "binary", "engine": "vector", "elements": 20,
         "input_ptrs": [101], "output_ptr": 102},  # read 101: 10->30
        {"seq": 2, "op": "transfer", "engine": "dma", "bytes": 5,
         "src_ptr": 2, "dst_ptr": 101},           # WAR: must wait for reader end 30
    ]
    result = simulate(events, model)
    overwrite = [e for e in result.timeline["dma"] if e.seq == 2][0]
    assert overwrite.start == pytest.approx(30)
    assert overwrite.end == pytest.approx(35)


def test_cross_engine_sync_latency_only_charged_across_engines():
    # A cross-engine RAW edge pays the sync latency once; a same-engine
    # dependency does not (that is already captured by program-order queueing).
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "dst_ptr": 101},
        {"seq": 1, "op": "binary", "engine": "vector", "elements": 10,
         "input_ptrs": [101], "output_ptr": 102},
    ]
    base = CostModel(dma_startup_ns=0, dma_bytes_per_ns=1,
                     vector_startup_ns=0, vector_elements_per_ns=1)
    synced = CostModel(dma_startup_ns=0, dma_bytes_per_ns=1,
                       vector_startup_ns=0, vector_elements_per_ns=1,
                       cross_engine_sync_ns=7)
    base_res = simulate(events, base)
    synced_res = simulate(events, synced)
    base_binary = base_res.timeline["vector"][0]
    synced_binary = synced_res.timeline["vector"][0]
    # DMA finishes at 10; VectorE consumer starts at 10 (no sync) or 17 (sync).
    assert base_binary.start == pytest.approx(10)
    assert synced_binary.start == pytest.approx(17)
    # Engine busy times are unchanged by sync latency.
    assert synced_res.engine_busy_ns == base_res.engine_busy_ns


def test_parallel_dma_queues_overlap_independent_transfers():
    # Two independent loads (distinct buffers) should overlap across DMA queues,
    # while a single-queue model serializes them.
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "dst_ptr": 101},
        {"seq": 1, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 2, "dst_ptr": 102},
    ]
    serial = CostModel(
        dma_startup_ns=0,
        dma_bytes_per_ns=1,
        dma_queue_count=1,
        dma_resource_count=0,
    )
    parallel = CostModel(
        dma_startup_ns=0,
        dma_bytes_per_ns=1,
        dma_queue_count=2,
        dma_resource_count=0,
    )
    serial_res = simulate(events, serial)
    parallel_res = simulate(events, parallel)
    # Serial: 10 + 10 = 20. Parallel: both start at 0, finish at 10.
    assert serial_res.predicted_latency_ns == pytest.approx(20)
    assert parallel_res.predicted_latency_ns == pytest.approx(10)
    # Engine busy time (sum of durations) is unchanged; only the makespan moves.
    assert parallel_res.engine_busy_ns["dma"] == pytest.approx(20)


def test_parallel_dma_queues_still_serialize_dependent_transfers():
    # A store that reads a buffer a prior load wrote must wait for that load even
    # with multiple queues (the RAW hazard is queue-independent).
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "dst_ptr": 101},
        {"seq": 1, "op": "transfer", "engine": "dma", "bytes": 5,
         "src_ptr": 101, "dst_ptr": 2},
    ]
    model = CostModel(dma_startup_ns=0, dma_bytes_per_ns=1, dma_queue_count=4)
    result = simulate(events, model)
    dependent = [e for e in result.timeline["dma"] if e.seq == 1][0]
    assert dependent.start == pytest.approx(10)
    assert result.predicted_latency_ns == pytest.approx(15)


def test_binary_event_waits_for_both_inputs_and_store_waits_for_output():
    model = CostModel(
        dma_startup_ns=0,
        dma_bytes_per_ns=1,
        dma_resource_count=0,
        vector_startup_ns=0,
        vector_elements_per_ns=1,
    )
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "dst_ptr": 101},
        {"seq": 1, "op": "transfer", "engine": "dma", "bytes": 20,
         "src_ptr": 2, "dst_ptr": 102},
        {"seq": 2, "op": "binary", "engine": "vector", "elements": 5,
         "input_ptrs": [101, 102], "output_ptr": 103},
        {"seq": 3, "op": "transfer", "engine": "dma", "bytes": 7,
         "src_ptr": 103, "dst_ptr": 3},
    ]
    result = simulate(events, model)
    binary = result.timeline["vector"][0]
    dma = result.timeline["dma"]
    assert binary.start == pytest.approx(30)
    assert binary.end == pytest.approx(35)
    assert dma[-1].start == pytest.approx(35)
    assert result.predicted_latency_ns == pytest.approx(42)


def test_dma_calibration_marks_ood_clamp_and_rejects_invalid_geometry():
    calibration = DmaCalibrationSurface({(2, 128): 2.0, (8, 512): 8.0})
    assert calibration.bandwidth_gbps(1, 64) == pytest.approx(2.0)
    assert calibration.bandwidth_gbps(16, 1024) == pytest.approx(8.0)
    assert calibration.lookup(1, 64).match == "ood_clamped"
    assert calibration.lookup(4, 256).match == "interpolated"
    assert calibration.lookup(2, 128).match == "exact"
    assert not calibration.in_domain(1, 64)
    assert calibration.in_domain(4, 256)
    with pytest.raises(ValueError, match="must be positive"):
        calibration.bandwidth_gbps(0, 128)


def test_dma_surface_csv_uses_dynamic_time_and_steady_write_rows(tmp_path):
    path = tmp_path / "dma.csv"
    fields = [
        "row_type",
        "status",
        "spec.name",
        "spec.dtype",
        "spec.repeat",
        "work.partition_count",
        "work.free_bytes_per_partition",
        "derived.write_gbps_dma_active",
        "derived.write_gbps_dynamic_dma_active",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "row_type": "benchmark",
            "status": "ok",
            "spec.name": "dma_write_partition_surface",
            "spec.dtype": "float32",
            "spec.repeat": 1,
            "work.partition_count": 8,
            "work.free_bytes_per_partition": 512,
            "derived.write_gbps_dma_active": 1,
            "derived.write_gbps_dynamic_dma_active": 2,
        })
        writer.writerow({
            "row_type": "benchmark",
            "status": "ok",
            "spec.name": "dma_write_partition_surface",
            "spec.dtype": "float32",
            "spec.repeat": 16,
            "work.partition_count": 8,
            "work.free_bytes_per_partition": 512,
            "derived.write_gbps_dma_active": 3,
            "derived.write_gbps_dynamic_dma_active": 4,
        })
    surface = DmaCalibrationSurface.from_csv(
        path,
        "dma_write_partition_surface",
        "derived.write_gbps_dynamic_dma_active",
        "float32",
        required_repeat=16,
    )
    assert surface.points == {(8, 512): 4.0}


def test_dma_surface_does_not_treat_static_dma_as_dynamic(tmp_path):
    path = tmp_path / "dma.csv"
    fields = [
        "row_type",
        "status",
        "spec.name",
        "spec.dtype",
        "work.partition_count",
        "work.free_bytes_per_partition",
        "work.hbm_read_bytes",
        "profile.software_dynamic_dma_active_time",
        "profile.static_dma_active_time",
        "derived.read_gbps_dynamic_dma_active",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "row_type": "benchmark",
            "status": "ok",
            "spec.name": "dma_partition_surface",
            "spec.dtype": "float32",
            "work.partition_count": 1,
            "work.free_bytes_per_partition": 512,
            "work.hbm_read_bytes": 512,
            "profile.software_dynamic_dma_active_time": 0,
            "profile.static_dma_active_time": 256e-9,
            "derived.read_gbps_dynamic_dma_active": "",
        })
    with pytest.raises(ValueError, match="No dma_partition_surface"):
        DmaCalibrationSurface.from_csv(path, dtype_name="float32")


def test_dma_calibration_csv_rejects_empty_and_conflicting_points(tmp_path):
    empty = tmp_path / "empty.csv"
    with empty.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row_type", "status", "spec.name", "work.partition_count",
                "work.free_bytes_per_partition", "derived.read_gbps_dma_active",
            ],
        )
        writer.writeheader()
    with pytest.raises(ValueError, match="No dma_partition_surface"):
        DmaCalibrationSurface.from_csv(
            empty, bandwidth_column="derived.read_gbps_dma_active"
        )

    conflicting = tmp_path / "conflicting.csv"
    with conflicting.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row_type", "status", "spec.name", "work.partition_count",
                "work.free_bytes_per_partition", "derived.read_gbps_dma_active",
            ],
        )
        writer.writeheader()
        for bandwidth in (10.0, 11.0):
            writer.writerow({
                "row_type": "benchmark",
                "status": "ok",
                "spec.name": "dma_partition_surface",
                "work.partition_count": 8,
                "work.free_bytes_per_partition": 1024,
                "derived.read_gbps_dma_active": bandwidth,
            })
    with pytest.raises(ValueError, match="Conflicting calibration rows"):
        DmaCalibrationSurface.from_csv(
            conflicting, bandwidth_column="derived.read_gbps_dma_active"
        )


def test_kernel_overhead_changes_makespan_not_engine_busy_time():
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 100},
    ]
    baseline = simulate(
        events, CostModel(dma_startup_ns=0, dma_bytes_per_ns=10)
    )
    adjusted = simulate(
        events,
        CostModel(
            dma_startup_ns=0,
            dma_bytes_per_ns=10,
            kernel_overhead_ns=8000,
        ),
    )
    assert baseline.predicted_latency_ns == pytest.approx(10)
    assert adjusted.predicted_latency_ns == pytest.approx(8010)
    assert adjusted.engine_busy_ns == baseline.engine_busy_ns


def test_static_dma_surface_loads_paired_incremental_latency(tmp_path):
    path = tmp_path / "static.csv"
    fields = [
        "row_type", "status", "spec.name", "mode",
        "work.partition_count", "work.scatter_rows", "work.scatter_columns",
        "latency.nc_latency.p50_us",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for mode, latency in (
            ("hbm_roundtrip_baseline", 8.0),
            ("sbuf_transpose_scatter", 20.0),
        ):
            writer.writerow({
                "row_type": "benchmark",
                "status": "ok",
                "spec.name": "static_dma_surface",
                "mode": mode,
                "work.partition_count": 8,
                "work.scatter_rows": 4,
                "work.scatter_columns": 8,
                "latency.nc_latency.p50_us": latency,
            })
    surface = StaticDmaCalibrationSurface.from_csv(path)
    assert surface.latency_ns(8, 4, 8) == pytest.approx(12_000)

    model = CostModel(static_dma_calibration=surface)
    event = {
        "op": "transfer", "engine": "static_dma", "bytes": 32,
        "partition_count": 8, "static_dma_group_copies": 32,
        "static_dma_group_x": 4, "static_dma_group_y": 8,
    }
    assert model.cost_ns(event) == pytest.approx(375)


def test_disjoint_ranges_same_base_run_in_parallel():
    # Two writes to the SAME base storage but DISJOINT byte ranges must not
    # serialize (no false hazard), while two DMA queues let them overlap.
    model = CostModel(dma_startup_ns=0, dma_bytes_per_ns=1, dma_queue_count=2)
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "src_storage": 1, "src_range": [0, 10],
         "dst_ptr": 100, "dst_storage": 100, "dst_range": [0, 10]},
        {"seq": 1, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 2, "src_storage": 2, "src_range": [0, 10],
         "dst_ptr": 100, "dst_storage": 100, "dst_range": [10, 20]},
    ]
    result = simulate(events, model)
    # Disjoint -> both start at 0, makespan is a single transfer time.
    assert result.predicted_latency_ns == pytest.approx(10)


def test_overlapping_ranges_same_base_serialize():
    # Two writes to overlapping byte ranges of the same storage must serialize
    # (WAW) even with parallel DMA queues.
    model = CostModel(dma_startup_ns=0, dma_bytes_per_ns=1, dma_queue_count=2)
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "src_storage": 1, "src_range": [0, 10],
         "dst_ptr": 100, "dst_storage": 100, "dst_range": [0, 10]},
        {"seq": 1, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 2, "src_storage": 2, "src_range": [0, 10],
         "dst_ptr": 100, "dst_storage": 100, "dst_range": [5, 15]},
    ]
    result = simulate(events, model)
    assert result.predicted_latency_ns == pytest.approx(20)


def test_partial_overwrite_preserves_uncovered_writer_history():
    model = CostModel(
        dma_startup_ns=0,
        dma_bytes_per_ns=1,
        dma_queue_count=4,
        dma_resource_count=0,
    )
    events = [
        {
            "seq": 0,
            "op": "transfer",
            "engine": "dma",
            "bytes": 100,
            "src_storage": 1,
            "src_range": [0, 100],
            "dst_storage": 100,
            "dst_range": [0, 100],
        },
        {
            "seq": 1,
            "op": "transfer",
            "engine": "dma",
            "bytes": 10,
            "src_storage": 2,
            "src_range": [0, 10],
            "dst_storage": 100,
            "dst_range": [0, 10],
        },
        {
            "seq": 2,
            "op": "transfer",
            "engine": "dma",
            "bytes": 10,
            "src_storage": 100,
            "src_range": [50, 60],
            "dst_storage": 3,
            "dst_range": [0, 10],
        },
    ]
    result = simulate(events, model)
    read = next(entry for entry in result.timeline["dma"] if entry.seq == 2)
    assert read.start == pytest.approx(100)


def test_disjoint_exact_segments_do_not_serialize_despite_overlapping_bounds():
    model = CostModel(
        dma_startup_ns=0,
        dma_bytes_per_ns=1,
        dma_queue_count=2,
        dma_resource_count=0,
    )
    events = [
        {
            "seq": 0,
            "op": "transfer",
            "engine": "dma",
            "bytes": 8,
            "src_storage": 1,
            "src_range": [0, 8],
            "dst_storage": 100,
            "dst_range": [0, 12],
            "dst_ranges": [[0, 4], [8, 12]],
        },
        {
            "seq": 1,
            "op": "transfer",
            "engine": "dma",
            "bytes": 4,
            "src_storage": 2,
            "src_range": [0, 4],
            "dst_storage": 100,
            "dst_range": [4, 8],
            "dst_ranges": [[4, 8]],
        },
    ]
    result = simulate(events, model)
    assert result.predicted_latency_ns == pytest.approx(8)


def test_view_pointer_aliases_parent_storage_via_storage_key():
    # A consumer whose src pointer differs from the producer's dst pointer, but
    # shares a storage id with overlapping range, must still see the RAW edge.
    model = CostModel(dma_startup_ns=0, dma_bytes_per_ns=1, vector_startup_ns=0,
                      vector_elements_per_ns=1)
    events = [
        {"seq": 0, "op": "transfer", "engine": "dma", "bytes": 10,
         "src_ptr": 1, "src_storage": 1, "src_range": [0, 10],
         "dst_ptr": 500, "dst_storage": 500, "dst_range": [0, 10]},
        # Reader uses a different pointer (a view: 508) but same storage 500,
        # overlapping range [8,12) -> must wait for the writer end (10).
        {"seq": 1, "op": "transfer", "engine": "dma", "bytes": 4,
         "src_ptr": 508, "src_storage": 500, "src_range": [8, 12],
         "dst_ptr": 2, "dst_storage": 2, "dst_range": [0, 4]},
    ]
    result = simulate(events, model)
    consumer = [e for e in result.timeline["dma"] if e.seq == 1][0]
    assert consumer.start == pytest.approx(10)


def test_versioned_load_compute_store_forms_cross_engine_critical_path():
    model = CostModel(
        dma_startup_ns=0,
        dma_bytes_per_ns=1,
        vector_startup_ns=0,
        vector_elements_per_ns=1,
        cross_engine_sync_ns=0,
    )
    events = [
        {"seq": 0, "op": "load", "engine": "dma", "bytes": 10,
         "src_storage": 1, "src_range": [0, 10], "src_version": 0,
         "dst_storage": 100, "dst_range": [0, 10], "dst_version": 0},
        {"seq": 1, "op": "compute", "engine": "vector",
         "elements": 10, "input_storages": [100],
         "input_ranges": [[0, 10]], "input_versions": [0],
         "output_storages": [200], "output_ranges": [[0, 10]],
         "output_versions": [0]},
        {"seq": 2, "op": "store", "engine": "dma", "bytes": 10,
         "src_storage": 200, "src_range": [0, 10], "src_version": 0,
         "dst_storage": 2, "dst_range": [0, 10], "dst_version": 1},
    ]
    result = simulate(events, model)
    load = next(e for e in result.timeline["dma"] if e.seq == 0)
    compute = result.timeline["vector"][0]
    store = next(e for e in result.timeline["dma"] if e.seq == 2)

    assert compute.start == pytest.approx(load.end)
    assert store.start == pytest.approx(compute.end)
    assert result.predicted_latency_ns == pytest.approx(30)


def test_strided_dma_interpolates_between_independent_control_sizes():
    calibration = StridedDmaCalibration(
        {("float32", 2, 128): [(512, 600.0, 900.0), (2048, 2600.0, 3300.0)]}
    )
    events = [{
        "op": "store",
        "dma_pattern": "strided",
        "free_stride_items": 2,
        "partition_count": 128,
        "active_access_count": 128 * 1024,
        "item_bytes": 4,
    }]

    assert calibration.predict(events) == pytest.approx((1266.6666667, 1700.0))


def test_strided_dma_uses_explicit_bfloat16_transfer_dtype():
    calibration = StridedDmaCalibration(
        {("bfloat16", 2, 1): [(512, 400.0, 900.0)]}
    )
    events = [{
        "op": "store",
        "mem_src": "SBUF",
        "mem_dst": "HBM",
        "dma_pattern": "strided",
        "free_stride_items": 2,
        "partition_count": 1,
        "active_access_count": 512,
        "item_bytes": 2,
        "src_dtype": "bfloat16",
    }]

    assert calibration.predict(events) == (400.0, 900.0)


def test_structured_multi_reduction_completion_is_operator_agnostic_floor():
    region = {
        "dtype": "float32",
        "free_dim": 512,
        "logical_free_dim": 512,
        "partition_count": 128,
        "reduction_count": 2,
        "reduction_kind": "reduce_sum",
        "op_histogram": {"reduce_sum": 2, "multiply": 3},
        "two_input_elementwise_count": 3,
    }
    key = structural_calibration_key(region)
    calibration = StructuredControlCalibration(
        points={}, completion_points={(key, "float32"): [(512, 53_000.0)]}
    )
    events = [{
        "seq": 1,
        "op": "compute",
        "engine": "vector",
        "elements": 512,
        "region_ir": region,
    }]

    result = simulate(
        events, CostModel(structured_control_lowering=calibration)
    )
    assert result.predicted_latency_ns == 53_000.0

    non_reduction = {**region, "reduction_count": 0}
    assert calibration.predict_completion_ns(non_reduction) == 0.0


def test_tilebench_matmul_builder_uses_rows_as_square_output_and_cols_as_k():
    from triton_viz.tools.nki_operator_experiments import _matmul_inputs

    lhs, rhs, tile_m, tile_n, tile_k, cores, double_row = _matmul_inputs(
        512, 1024, "float32"
    )
    assert lhs.shape == (512, 1024)
    assert rhs.shape == (1024, 512)
    assert (tile_m, tile_n, tile_k, cores, double_row) == (4, 1, 8, 1, False)


def test_operator_agnostic_access_and_compute_feature_schema():
    from triton_viz.tools.nki_features import AccessPattern, ComputeRegion

    access = AccessPattern.from_event({
        "op": "store", "mem_src": "SBUF", "mem_dst": "HBM",
        "bytes": 64, "partition_count": 16, "free_stride_items": 2,
        "active_access_count": 16, "access_span_bytes": 128,
        "item_bytes": 4,
    })
    assert access is not None
    assert access.layout_family == "strided_positive"
    assert access.density == pytest.approx(0.5)

    region = ComputeRegion.from_event({"region_ir": {
        "dtype": "float32", "partition_count": 16,
        "logical_free_dim": 512, "op_histogram": {"add": 2},
        "reduction_count": 1, "broadcast_edge_count": 1,
        "has_mask_or_tail": True,
    }})
    assert region is not None
    assert region.op_histogram == (("add", 2),)
    assert region.partition_count == 16


def test_runtime_overhead_is_a_concurrent_path_and_reports_control_domain():
    calibration = RuntimeOverheadCalibration(
        sequencer_base_ns=100,
        vector_activation_ns=20,
        partition_log2_ns=10,
        dma_packet_log2_ns=5,
        partition_min=1,
        partition_max=128,
        free_access_min=128,
        free_access_max=2048,
    )
    events = [{
        "seq": 0, "op": "compute", "engine": "vector", "elements": 500,
        "partition_count": 16, "free_dim": 128,
    }]
    result = simulate(
        events,
        CostModel(
            vector_startup_ns=0,
            vector_elements_per_ns=1,
            runtime_overhead_calibration=calibration,
        ),
    )
    # Runtime setup overlaps engine work: it is max(500, 100+20+4*10),
    # never an additive residual on top of the scheduler result.
    assert result.predicted_latency_ns == pytest.approx(500)
    assert result.components_ns["runtime_control_in_domain"] == 0.0
    assert calibration.in_domain(16, 512)
    assert not calibration.in_domain(16, 4096)


def test_lowering_calibration_expands_one_fusion_group_across_engines():
    level_b = ComputeCalibration({
        ("vector", "float32", 2): (10.0, 1.0),
        ("scalar", "float32", 1): (5.0, 0.5),
    })
    level_a = LoweringExpansionCalibration({
        ("subtract_exp", "float32", "vector", 100): (2.0, 2),
        ("subtract_exp", "float32", "scalar", 100): (3.0, 1),
    })
    events = [
        {"seq": 1, "op": "compute", "api_op": "subtract", "engine": "vector",
         "input_shape": [128, 100], "output_shape": [128, 100],
         "output_dtype": "float32", "fusion_signature": "subtract_exp",
         "fusion_group": 0, "input_ptrs": [10, 11], "output_ptr": 12},
        {"seq": 2, "op": "compute", "api_op": "exp", "engine": "scalar",
         "input_shape": [128, 100], "output_shape": [128, 100],
         "output_dtype": "float32", "fusion_signature": "subtract_exp",
         "fusion_group": 0, "input_ptrs": [12], "output_ptr": 13},
    ]
    result = simulate(events, CostModel(
        compute_calibration=level_b, lowering_calibration=level_a
    ))
    assert result.engine_busy_ns["vector"] == pytest.approx(220.0)
    assert result.engine_busy_ns["scalar"] == pytest.approx(165.0)
    assert result.predicted_latency_ns == pytest.approx(220.0)


def test_lowering_calibration_adds_instruction_audited_fixed_work():
    level_b = ComputeCalibration({("vector", "float32", 2): (10.0, 1.0)})
    level_a = LoweringExpansionCalibration(
        {("add", "float32", "vector", 100): (2.0, 2)},
        {("add", "float32", "vector", 100): 30.0},
    )
    event = {"op": "compute", "api_op": "add", "engine": "vector",
             "input_shape": [128, 100], "output_shape": [128, 100],
             "output_dtype": "float32", "fusion_signature": "add", "fusion_group": 0}
    result = simulate([event], CostModel(compute_calibration=level_b, lowering_calibration=level_a))
    assert result.engine_busy_ns["vector"] == pytest.approx(250.0)


def test_strict_compute_calibration_uses_value_dtype_for_bool_predicates():
    calibration = ComputeCalibration({("vector", "float32", 2): (10.0, 1.0)})
    result = simulate(
        [
            {
                "op": "compute",
                "api_op": "greater",
                "engine": "vector",
                "input_shape": [128, 100],
                "output_shape": [128, 100],
                "input_dtypes": ["float32"],
                "output_dtype": "bool",
                "region_ir": {"dtype": "float32"},
            }
        ],
        CostModel(compute_calibration=calibration, strict_calibration=True),
    )
    assert result.engine_busy_ns["vector"] == pytest.approx(110.0)


def test_compute_calibration_strict_dtype_rejects_silent_fp32_fallback():
    calibration = ComputeCalibration({("vector", "float32", 1): (10.0, 1.0)})
    assert calibration.instruction_ns("vector", "bfloat16", 1, 100) == 110.0
    assert (
        calibration.instruction_ns(
            "vector", "bfloat16", 1, 100, strict_dtype=True
        )
        is None
    )
    with pytest.raises(ValueError, match="Missing exact compute calibration"):
        simulate(
            [
                {
                    "op": "compute",
                    "engine": "vector",
                    "input_shape": [128, 100],
                    "output_shape": [128, 100],
                    "output_dtype": "bfloat16",
                }
            ],
            CostModel(
                compute_calibration=calibration,
                strict_calibration=True,
            ),
        )


def test_compositional_lowering_uses_region_features_without_signature_row():
    from triton_viz.tools.nki_cost_model import CompositionalLoweringCalibration
    level_b = ComputeCalibration({("vector", "float32", 2): (10.0, 1.0)})
    structured = CompositionalLoweringCalibration({
        ("vector", "float32", "effective_count"): {"intercept": 1.0, "two_input_elementwise_count": 2.0},
        ("vector", "float32", "fixed_ns"): {"intercept": 7.0},
    })
    event = {"op": "compute", "api_op": "add", "input_shape": [128, 100], "output_shape": [128, 100],
             "output_dtype": "float32", "fusion_signature": "unseen", "fusion_group": 0,
             "region_ir": {"dtype": "float32", "logical_free_dim": 100,
                           "two_input_elementwise_count": 2}}
    result = simulate([event], CostModel(compute_calibration=level_b, compositional_lowering=structured))
    assert result.engine_busy_ns["vector"] == pytest.approx(557.0)


def test_tensor_calibration_is_dtype_throughput_without_shape_lookup(tmp_path):
    path = tmp_path / "tensor.csv"
    fields = [
        "row_type",
        "status",
        "kind",
        "spec.dtype",
        "work.matmul_flops",
        "profile.tensor_engine_active_time",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for dtype, startup_ns, flops_per_ns in (
            ("float32", 1000.0, 20_000.0),
            ("bfloat16", 500.0, 80_000.0),
        ):
            for flops in (5e8, 1e9, 2e9, 4e9):
                active_ns = startup_ns + flops / flops_per_ns
                writer.writerow({
                    "row_type": "benchmark",
                    "status": "ok",
                    "kind": "tensor_matmul_tiled",
                    "spec.dtype": dtype,
                    "work.matmul_flops": flops,
                    "profile.tensor_engine_active_time": active_ns * 1e-9,
                })

    calibration = TensorCalibrationSurface.from_csv(
        path, benchmark_name="tensor_matmul_tiled"
    )
    assert calibration.flops_per_ns("float32") == pytest.approx(20_000.0)
    assert calibration.flops_per_ns("bf16") == pytest.approx(80_000.0)
    assert calibration.startup_ns("float32") == pytest.approx(1000.0)
    assert calibration.startup_ns("bf16") == pytest.approx(500.0)
    # Throughput-only: no per-dot table and no tile-shape key may exist.
    assert not hasattr(calibration, "ns_per_dot")
    assert not hasattr(calibration, "shape_points")
    assert calibration.domain_match("float32", 5e8) == "in_domain"
    assert calibration.domain_match("float32", 4e6) == "below_domain"
    assert calibration.domain_match("float32", 5e9) == "above_domain"

    model = CostModel(tensor_calibration=calibration)
    small_attention_dot = {
        "op": "dot",
        "engine": "tensor",
        "flops": 2 * 128 * 128 * 128,
        "input_dtypes": ["float32", "float32"],
        "output_dtype": "float32",
    }
    assert model.cost_ns(small_attention_dot) == pytest.approx(
        small_attention_dot["flops"] / 20_000.0
    )
