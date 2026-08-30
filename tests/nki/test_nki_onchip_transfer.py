"""Measured on-chip PSUM/SBUF copy latency replaces the hardcoded placeholder."""

import pytest

from triton_viz.tools.nki_cost_model import (
    CostModel,
    OnChipTransferCalibration,
    simulate,
)

CSV = """engine,dtype,input_stream_count,startup_ns,ns_per_free_elem,domain_min_free,domain_max_free
vector,float32,1,65.0,1.5,48,320
vector,bfloat16,1,66.0,1.5,48,320
"""


def _calibration(tmp_path):
    path = tmp_path / "onchip.csv"
    path.write_text(CSV)
    return OnChipTransferCalibration.from_csv(path)


def _copy(free_bytes, item_bytes=4, engine="vector", seq=0):
    return {
        "seq": seq,
        "op": "transfer",
        "engine": engine,
        "mem_src": "psum",
        "mem_dst": "sbuf",
        "dma_pattern": "copy",
        "bytes": 128 * free_bytes,
        "partition_count": 128,
        "free_bytes_per_partition": free_bytes,
        "item_bytes": item_bytes,
        "output_dtype": "float32",
        "src_storage": 1,
        "src_range": [0, 128 * free_bytes],
        "src_version": 0,
        "dst_storage": 2,
        "dst_range": [0, 128 * free_bytes],
        "dst_version": 1,
    }


def test_latency_grows_with_free_width(tmp_path):
    calibration = _calibration(tmp_path)
    assert calibration.latency_ns("vector", "float32", 128) == (
        pytest.approx(65.0 + 1.5 * 128),
        "in_domain",
    )
    assert calibration.latency_ns("vector", "bfloat16", 320) == (
        pytest.approx(66.0 + 1.5 * 320),
        "in_domain",
    )


def test_free_width_outside_the_measured_domain_is_reported(tmp_path):
    calibration = _calibration(tmp_path)
    value, match = calibration.latency_ns("vector", "float32", 1)
    assert match == "ood_extrapolated"
    assert value == pytest.approx(66.5)
    _value, match = calibration.latency_ns("vector", "float32", 4096)
    assert match == "ood_extrapolated"


def test_unmeasured_engine_reuses_the_measured_copy_path(tmp_path):
    calibration = _calibration(tmp_path)
    value, match = calibration.latency_ns("static_dma", "float32", 128)
    assert match == "in_domain"
    assert value == pytest.approx(65.0 + 1.5 * 128)


def test_onchip_copy_replaces_the_placeholder_and_occupies_its_engine(tmp_path):
    calibration = _calibration(tmp_path)
    events = [_copy(512)]
    placeholder = simulate(events, CostModel(cross_engine_sync_ns=0.0))
    measured = simulate(
        events,
        CostModel(
            cross_engine_sync_ns=0.0, onchip_transfer_calibration=calibration
        ),
    )
    assert measured.engine_busy_ns["vector"] == pytest.approx(65.0 + 1.5 * 128)
    assert measured.engine_busy_ns["vector"] != placeholder.engine_busy_ns["vector"]
    assert measured.components_ns["onchip_transfer_count"] == 1.0
    assert measured.components_ns["onchip_transfer_ood"] == 0.0


def test_hbm_transfers_are_untouched_by_the_onchip_surface(tmp_path):
    calibration = _calibration(tmp_path)
    event = dict(_copy(512))
    event.update(mem_src="hbm", mem_dst="sbuf", engine="dma", active_access_count=128 * 128)
    model_args = dict(
        cross_engine_sync_ns=0.0, dma_bytes_per_ns=100.0, dma_startup_ns=0.0
    )
    plain = simulate([event], CostModel(**model_args))
    bounded = simulate(
        [event], CostModel(**model_args, onchip_transfer_calibration=calibration)
    )
    assert bounded.engine_busy_ns == plain.engine_busy_ns
    assert bounded.components_ns["onchip_transfer_count"] == 0.0
