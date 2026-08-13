import numpy as np
import pytest

from triton_viz.tools.nki_cost_model import (
    CostModel,
    DmaCalibrationSurface,
    StridedDmaCalibration,
    eliminate_redundant_hbm_loads,
    simulate,
)
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


def test_directional_surfaces_price_read_and_write_without_affine_state():
    events = [
        {
            "seq": 1,
            "op": "load",
            "engine": "dma_or_vector_load",
            "mem_src": "HBM",
            "mem_dst": "SBUF",
            "bytes": 800,
            "partition_count": 8,
            "free_bytes_per_partition": 100,
        },
        {
            "seq": 2,
            "op": "store",
            "engine": "dma_or_vector_store",
            "mem_src": "SBUF",
            "mem_dst": "HBM",
            "bytes": 800,
            "partition_count": 8,
            "free_bytes_per_partition": 100,
        },
    ]
    result = simulate(
        events,
        CostModel(
            dma_calibration=DmaCalibrationSurface({(8, 100): 10.0}),
            dma_write_calibration=DmaCalibrationSurface({(8, 100): 20.0}),
        ),
    )
    assert result.engine_busy_ns["dma"] == 120.0


def test_offset_geometry_uses_active_mask_and_detects_stride_two():
    offsets = np.array([[0, 8, 16, 24], [64, 72, 80, 88]])
    masks = np.array([[True, True, False, False], [True, True, False, False]])
    geometry = _offset_geometry(offsets, masks, nbytes=16)
    assert geometry["dma_pattern"] == "strided"
    assert geometry["item_bytes"] == 4
    assert geometry["free_stride_items"] == 2
    assert geometry["active_access_count"] == 4


def test_access_pattern_preserves_reverse_stride_and_distinguishes_stride_zero():
    from triton_viz.tools.nki_features import AccessPattern

    reverse = AccessPattern.from_event(
        {
            "op": "load",
            "free_stride_items": -1,
            "active_access_count": 4,
            "access_span_bytes": 16,
            "bytes": 16,
            "item_bytes": 4,
        }
    )
    broadcast = AccessPattern.from_event(
        {
            "op": "load",
            "free_stride_items": 0,
            "active_access_count": 4,
            "access_span_bytes": 4,
            "bytes": 16,
            "item_bytes": 4,
        }
    )
    irregular = AccessPattern.from_event(
        {
            "op": "load",
            "free_stride_items": None,
            "active_access_count": 4,
            "access_span_bytes": 20,
            "bytes": 16,
            "item_bytes": 4,
        }
    )
    assert reverse.layout_family == "reverse"
    assert broadcast.layout_family == "broadcast_stride0"
    assert irregular.layout_family == "irregular"


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


def test_strided_dma_calibration_sets_busy_time_and_completion_floor():
    calibration = StridedDmaCalibration(
        {("float32", 2, 128): [(512, 600_000.0, 700_000.0)]}
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
    assert result.predicted_latency_ns == 700_000.0
