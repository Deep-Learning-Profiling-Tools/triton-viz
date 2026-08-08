import csv

import numpy as np
import pytest

from triton_viz.tools.nki_cost_model import (
    CostModel,
    DmaAffineCalibration,
    NcLatencyCalibration,
    StridedDmaCalibration,
    eliminate_redundant_hbm_loads,
    simulate,
)
from triton_viz.tools.nki_region_ir import structural_calibration_key
from triton_viz.tools.nki_trace_dump import _offset_geometry


def _load(seq, *, grid=(0,), storage=10, span=(0, 1024), shape=(128, 2)):
    return {
        "seq": seq,
        "op": "load",
        "mem_src": "HBM",
        "mem_dst": "SBUF",
        "grid_idx": list(grid),
        "src_storage": storage,
        "src_range": list(span),
        "offsets_shape": list(shape),
        "bytes": span[1] - span[0],
    }


def test_exact_repeated_hbm_load_is_eliminated_and_audited():
    events, audit = eliminate_redundant_hbm_loads([_load(1), _load(2)])
    assert [event["seq"] for event in events] == [1]
    assert audit == {"eliminated_load_count": 1, "eliminated_load_bytes": 1024}


def test_load_cse_does_not_cross_grid_programs_or_guess_partial_overlap():
    events, audit = eliminate_redundant_hbm_loads(
        [_load(1), _load(2, grid=(1,)), _load(3, span=(0, 512))]
    )
    assert len(events) == 3
    assert audit["eliminated_load_count"] == 0


def test_overlapping_hbm_store_invalidates_cached_load():
    store = {
        "seq": 2,
        "op": "store",
        "mem_src": "SBUF",
        "mem_dst": "HBM",
        "dst_storage": 10,
        "dst_range": [512, 768],
        "bytes": 256,
    }
    events, audit = eliminate_redundant_hbm_loads([_load(1), store, _load(3)])
    assert len(events) == 3
    assert audit["eliminated_load_count"] == 0


def _write_dma_fit_csv(path, name, dtype, byte_column, samples):
    fields = [
        "row_type",
        "status",
        "spec.name",
        "spec.dtype",
        "work.partition_count",
        byte_column,
        "profile.software_dynamic_dma_active_time",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for nbytes, time_ns in samples:
            writer.writerow(
                {
                    "row_type": "benchmark",
                    "status": "ok",
                    "spec.name": name,
                    "spec.dtype": dtype,
                    "work.partition_count": 128,
                    byte_column: nbytes,
                    "profile.software_dynamic_dma_active_time": time_ns / 1e9,
                }
            )


def test_affine_dma_charges_kernel_startup_once_and_directional_slopes(tmp_path):
    read = tmp_path / "read.csv"
    write = tmp_path / "write.csv"
    _write_dma_fit_csv(
        read,
        "dma_partition_surface",
        "float32",
        "work.hbm_read_bytes",
        [(100, 70), (200, 120)],
    )
    _write_dma_fit_csv(
        write,
        "dma_write_partition_surface",
        "float32",
        "work.hbm_write_bytes",
        [(100, 30), (200, 50)],
    )
    calibration = DmaAffineCalibration.from_csvs(read, write, "float32")
    assert calibration.startup_ns == pytest.approx(20)
    assert calibration.read_ns_per_byte == pytest.approx(0.5)
    assert calibration.write_ns_per_byte == pytest.approx(0.2)
    events = [
        {
            "seq": 1,
            "op": "load",
            "engine": "dma_or_vector_load",
            "mem_src": "HBM",
            "mem_dst": "SBUF",
            "bytes": 100,
        },
        {
            "seq": 2,
            "op": "store",
            "engine": "dma_or_vector_store",
            "mem_src": "SBUF",
            "mem_dst": "HBM",
            "bytes": 100,
        },
    ]
    result = simulate(events, CostModel(dma_affine_calibration=calibration))
    assert result.engine_busy_ns["dma"] == pytest.approx(90)


def test_nc_latency_calibration_adds_dispatch_residual_to_engine_busy():
    region = {
        "schema_version": 2,
        "dtype": "float32",
        "free_dim": 128,
        "logical_free_dim": 128,
        "reduction_count": 0,
        "op_histogram": {"maximum": 1},
        "one_input_elementwise_count": 1,
        "two_input_elementwise_count": 0,
    }
    calibration = NcLatencyCalibration(
        {(structural_calibration_key(region), "float32"): [(128, 7000.0)]}
    )
    result = simulate(
        [
            {
                "seq": 1,
                "op": "compute",
                "api_op": "maximum",
                "engine": "vector",
                "output_shape": [128, 128],
                "output_dtype": "float32",
                "region_ir": region,
            }
        ],
        CostModel(nc_latency_calibration=calibration),
    )
    assert result.predicted_latency_ns == pytest.approx(
        result.engine_busy_ns["vector"] + 7000.0
    )


def test_offset_geometry_uses_active_mask_and_detects_stride_two():
    offsets = np.array([[0, 8, 16, 24], [64, 72, 80, 88]])
    masks = np.array([[True, True, False, False], [True, True, False, False]])
    geometry = _offset_geometry(offsets, masks, nbytes=16)
    assert geometry["dma_pattern"] == "strided"
    assert geometry["item_bytes"] == 4
    assert geometry["free_stride_items"] == 2
    assert geometry["active_access_count"] == 4


def test_dma_resource_tokens_serialize_full_width_but_overlap_narrow_transfers():
    def transfers(partitions):
        return [
            {
                "seq": index,
                "op": "load",
                "engine": "dma_or_vector_load",
                "partition_count": partitions,
                "bytes": 100,
                "src_ptr": index,
            }
            for index in (1, 2)
        ]

    model = CostModel(
        dma_startup_ns=0,
        dma_bytes_per_ns=1,
        dma_resource_count=16,
    )
    full = simulate(transfers(128), model)
    narrow = simulate(transfers(1), model)
    assert full.predicted_latency_ns == pytest.approx(200)
    assert narrow.predicted_latency_ns == pytest.approx(100)
    assert full.engine_busy_ns["dma"] == narrow.engine_busy_ns["dma"] == 200


def test_strided_dma_calibration_overrides_packet_train_and_completion():
    calibration = StridedDmaCalibration(
        {("float32", 2, 128): [(512, 600_000.0, 90_000.0)]}
    )
    events = [
        {
            "seq": index,
            "op": "store",
            "engine": "dma_or_vector_store",
            "mem_src": "SBUF",
            "mem_dst": "HBM",
            "bytes": 128 * 512 * 4,
            "partition_count": 128,
            "active_access_count": 128 * 512,
            "free_stride_items": 2,
            "item_bytes": 4,
            "dma_pattern": "strided",
        }
        for index in (1, 2)
    ]
    result = simulate(events, CostModel(strided_dma_calibration=calibration))
    assert result.engine_busy_ns["dma"] == 600_000.0
    assert result.predicted_latency_ns == 690_000.0
