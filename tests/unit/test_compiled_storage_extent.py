"""Allocation bounds for compiled races across tensor views and aliases."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from z3 import Solver, sat, simplify, unsat

from triton_viz.clients.common.ttir_reader import UnsupportedTTIR, parse_ttir
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    _written_load_sources,
    encode_graph,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_route2_snapshot_select import SCATTER
from .test_t1_rmw_static import _module


def _capture(**values):
    detector = CompiledRaceDetector(confirm_races=False, ladder_level="L2")
    detector._capture_launch(
        SimpleNamespace(arg_names=list(values)), tuple(values.values()), {"grid": (2,)}
    )
    assert detector._capture_error is None
    return detector._launch_tensors


def _store(offset):
    return parse_ttir(
        _module(
            "%out_ptr: !tt.ptr<i32>",
            f"%offset = arith.constant {offset} : i32",
            "%value = arith.constant 0 : i32",
            "%ptr = tt.addptr %out_ptr, %offset : !tt.ptr<i32>, i32",
            "tt.store %ptr, %value : !tt.ptr<i32>",
        )
    )


def _store_result(meta, offset):
    enc = encode_graph(_store(offset), {}, {"out_ptr": meta})
    check = Solver()
    check.add(*enc.records[0].local_constraints)
    return (
        enc.records[0],
        check.check(),
        TwoCopySymbolicHBSolver(
            enc.records, grid=(2, 1, 1), arange_dict=enc.arange_dict
        ).find_races(),
    )


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "gap"])
def test_capture_preserves_view_addresses_strides_and_values(layout):
    base = torch.arange(24, dtype=torch.int32)
    view = {
        "contiguous": base,
        "transpose": base.reshape(4, 6).T,
        "offset": base[4:12],
        "gap": base[3::3],
    }[layout]
    before = (view.data_ptr(), view.storage_offset(), view.shape, view.stride())
    meta = _capture(out_ptr=view)["out_ptr"]
    assert meta.data_ptr == view.data_ptr()
    assert meta.allocation_interval() == (
        base.data_ptr(),
        base.data_ptr() + base.numel() * base.element_size(),
    )
    assert before == (view.data_ptr(), view.storage_offset(), view.shape, view.stride())
    assert torch.equal(base, torch.arange(24, dtype=torch.int32))
    if view.is_contiguous():
        assert meta.snapshot == tuple(view.reshape(-1).tolist())
        assert meta.init_values == meta.snapshot
    else:
        assert meta.snapshot is None
        assert meta.init_values is None
        assert meta.snapshot_reason == "non-contiguous"


def test_gap_access_past_numel_is_kept_and_races():
    base = torch.zeros(12, dtype=torch.int32)
    meta = _capture(out_ptr=base[::3])["out_ptr"]
    assert meta.numel == 4
    record, status, races = _store_result(meta, 9)
    assert simplify(record.addr_expr).as_long() == base.data_ptr() + 9 * 4
    assert status == sat
    assert races
    assert _store_result(meta, 12)[1:] == (unsat, [])


@pytest.mark.parametrize("strided", [False, True])
def test_offset_is_not_added_twice_and_bounds_use_the_allocation(strided):
    base = torch.zeros(12, dtype=torch.int32)
    view = base[4::2] if strided else base[4:8]
    meta = _capture(out_ptr=view)["out_ptr"]
    # Even a contiguous subview may legally address earlier storage via a
    # negative TTIR offset. This intentionally expands the old view bounds.
    record, status, races = _store_result(meta, -4)
    assert simplify(record.addr_expr).as_long() == base.data_ptr()
    assert status == sat
    assert races
    assert _store_result(meta, -5)[1:] == (unsat, [])
    assert _store_result(meta, 8)[1:] == (unsat, [])


def test_transposed_dense_view_reaches_last_storage_element():
    base = torch.zeros((3, 4), dtype=torch.int32)
    meta = _capture(out_ptr=base.T)["out_ptr"]
    assert not meta.contiguous
    assert _store_result(meta, 11)[1] == sat
    assert _store_result(meta, 11)[2]
    assert _store_result(meta, 12)[1:] == (unsat, [])


def test_whole_access_must_fit_in_storage():
    meta = GlobalTensor(
        data_ptr=0x1000,
        elem_size=4,
        numel=3,
        contiguous=False,
        storage_data_ptr=0x1000,
        storage_nbytes=15,
    )
    assert _store_result(meta, 2)[1] == sat
    assert _store_result(meta, 3)[1:] == (unsat, [])


def test_shared_storage_alias_is_not_hidden_by_disjoint_logical_views():
    base = torch.zeros(12, dtype=torch.int32)
    tensors = _capture(x_ptr=base[:4], out_ptr=base[8:])
    graph = parse_ttir(
        _module(
            "%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
            "%c8 = arith.constant 8 : i32",
            "%value = arith.constant 0 : i32",
            "%xp = tt.addptr %x_ptr, %c8 : !tt.ptr<i32>, i32",
            "tt.store %xp, %value : !tt.ptr<i32>",
            "tt.store %out_ptr, %value : !tt.ptr<i32>",
        )
    )
    assert not CompiledRaceDetector._t0_premises_hold_for_launch(graph, tensors)
    enc = encode_graph(graph, {}, tensors)
    races = TwoCopySymbolicHBSolver(
        enc.records, grid=(2, 1, 1), arange_dict=enc.arange_dict
    ).find_races()
    assert any(
        {r.first_record.tensor_name, r.second_record.tensor_name}
        == {"x_ptr", "out_ptr"}
        for r in races
    )


def test_separate_noncontiguous_allocations_can_satisfy_t0_alias_gate():
    tensors = _capture(x_ptr=torch.zeros(12)[::3], out_ptr=torch.zeros(12)[::3])
    graph = parse_ttir(
        _module(
            "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>",
            "%value = tt.load %x_ptr : !tt.ptr<f32>",
            "tt.store %out_ptr, %value : !tt.ptr<f32>",
        )
    )
    assert CompiledRaceDetector._t0_premises_hold_for_launch(graph, tensors)


def test_snapshot_readonly_gate_sees_alias_in_storage_beyond_numel():
    base = torch.arange(32, dtype=torch.int32)
    tensors = _capture(
        idx_ptr=base[:16],
        x_ptr=torch.zeros(16, dtype=torch.int32),
        out_ptr=base[16:],
    )
    graph = parse_ttir(SCATTER, multipath=True)
    assert tensors["idx_ptr"].snapshot is not None
    assert "idx_ptr" in _written_load_sources(graph, tensors)
    with pytest.raises(UnsupportedTTIR, match="overlaps the source"):
        encode_graph(graph, {}, tensors, multipath=True)


def test_partial_storage_snapshot_cannot_hide_valid_unsnapshotted_loads():
    base = torch.arange(32, dtype=torch.int32)
    tensors = _capture(
        idx_ptr=base[:16],
        x_ptr=torch.zeros(16, dtype=torch.int32),
        out_ptr=torch.zeros(64, dtype=torch.int32),
    )
    graph = parse_ttir(SCATTER, multipath=True)
    assert tensors["idx_ptr"].snapshot == tuple(range(16))
    assert not _written_load_sources(graph, tensors)
    with pytest.raises(UnsupportedTTIR, match="snapshot covers only a tensor view"):
        encode_graph(graph, {}, tensors, multipath=True)


def test_missing_storage_metadata_retains_legacy_contiguous_control():
    legacy = GlobalTensor(data_ptr=0x1000, elem_size=4, numel=12)
    assert _store_result(legacy, 11)[2]
    with pytest.raises(UnsupportedTTIR, match="non-contiguous"):
        encode_graph(_store(11), {}, {"out_ptr": replace(legacy, contiguous=False)})


@pytest.mark.parametrize(
    "updates",
    [
        {"storage_data_ptr": 0x1000},
        {"storage_nbytes": 48},
        {"storage_data_ptr": 0x1004, "storage_nbytes": 48},
        {"storage_data_ptr": 0x1000, "storage_nbytes": -1},
        {"storage_data_ptr": 0x1000, "storage_nbytes": 0},
        {"storage_data_ptr": 0x1000, "storage_nbytes": 4},
    ],
)
def test_invalid_explicit_storage_metadata_fails_closed(updates):
    meta = GlobalTensor(data_ptr=0x1000, elem_size=4, numel=12, **updates)
    assert meta.allocation_interval() is None
    with pytest.raises(UnsupportedTTIR, match="allocation extent"):
        encode_graph(_store(0), {}, {"out_ptr": meta})


def test_missing_storage_api_does_not_assume_noncontiguous_numel_extent():
    tensor = SimpleNamespace(
        data_ptr=lambda: 0x1000,
        element_size=lambda: 4,
        numel=lambda: 4,
        is_contiguous=lambda: False,
    )
    meta = _capture(out_ptr=tensor)["out_ptr"]
    assert meta.allocation_interval() is None
    with pytest.raises(UnsupportedTTIR, match="non-contiguous"):
        encode_graph(_store(0), {}, {"out_ptr": meta})
