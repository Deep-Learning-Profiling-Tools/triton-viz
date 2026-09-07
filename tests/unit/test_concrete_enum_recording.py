"""Check recorded byte sets and original lane identities against lane oracles."""

from types import SimpleNamespace

import numpy as np
import pytest

from triton_viz.clients.race_detector import concrete_enum as ce


@pytest.mark.parametrize("fence_order", [False, True])
@pytest.mark.parametrize("kind", [ce._KIND_LOAD, ce._KIND_STORE, ce._KIND_RMW])
@pytest.mark.parametrize("elem", [1, 2, 4, 8])
def test_recorder_preserves_active_bytes_positions_and_duplicates(
    monkeypatch, fence_order, kind, elem
):
    monkeypatch.setattr(
        ce, "capture_current_source_location", lambda: ("case.py", 1, "k")
    )
    rng = np.random.default_rng(809)
    # Include empty masks, holes, duplicate starts, partial-byte overlap,
    # non-monotone layouts, and negative strides. Addresses stay in storage.
    layouts = [
        (np.asarray([64]), np.asarray([True])),
        (np.arange(64, 96, elem), None),
        (np.asarray([80, 68, 64, 68, 65, 96]), None),
        (np.arange(96, 64, -elem), None),
        (np.arange(64, 128, 4).reshape(4, 4).T, np.eye(4, dtype=bool)),
        (np.arange(64, 128, 4).reshape(4, 4), np.zeros((4, 4), dtype=bool)),
    ]
    layouts.extend(
        (rng.integers(64, 128, size=(4, 8)), rng.random((4, 8)) < 0.7)
        for _ in range(12)
    )
    for data, mask in layouts:
        rec = ce.ConcreteFootprintRecorder(bounds=[(32, 256)], fence_order=fence_order)
        rec.grid_idx_callback((0,))
        ptr = SimpleNamespace(
            data=data.astype(np.uint64),
            get_element_ty=lambda: SimpleNamespace(primitive_bitwidth=elem * 8),
        )
        rec._record(kind, ptr, mask)
        active = np.ones(data.shape, dtype=bool) if mask is None else mask
        pairs = [
            (int(address), pos)
            for pos, address in enumerate(data.flat)
            if active.flat[pos]
        ]
        expected_bytes = {
            byte for address, _ in pairs for byte in range(address, address + elem)
        }
        starts, ends, ops = rec.intervals.view()
        assert {
            byte
            for start, end in zip(starts, ends)
            for byte in range(int(start), int(end))
        } == expected_bytes
        assert list(ops) == [0] * len(starts)
        assert rec.op_lanes == [len(pairs)]
        if kind == ce._KIND_RMW:
            assert list(zip(starts, ends)) == [
                (a, a + elem) for a in sorted({a for a, _ in pairs})
            ]
        if fence_order:
            fp = rec.op_positions[0]
            represented_pairs = [
                (address, int(pos) + offset)
                for start, end, pos in zip(fp.starts, fp.ends, fp.positions)
                for offset, address in enumerate(range(int(start), int(end), elem))
            ]
            assert fp.shape == data.shape
            assert represented_pairs == sorted(pairs, key=lambda pair: pair[0])
        overlapping_lanes = any(
            max(a, b) < min(a + elem, b + elem)
            for index, (a, _) in enumerate(pairs)
            for b, _ in pairs[index + 1 :]
        )
        assert bool(rec.intra_dups) == (kind == ce._KIND_STORE and overlapping_lanes)


def test_shared_sort_keeps_original_positions_for_duplicate_addresses(monkeypatch):
    monkeypatch.setattr(ce, "capture_current_source_location", lambda: None)
    rec = ce.ConcreteFootprintRecorder(fence_order=True)
    rec.grid_idx_callback((0,))
    ptr = SimpleNamespace(
        data=np.asarray([40, 32, 32, 36], dtype=np.uint64),
        get_element_ty=lambda: SimpleNamespace(primitive_bitwidth=32),
    )
    rec._record(ce._KIND_STORE, ptr, None)
    fp = rec.op_positions[0]
    assert list(zip(fp.starts, fp.ends, fp.positions)) == [
        (32, 36, 1),
        (32, 40, 2),
        (40, 44, 0),
    ]
    assert rec.intra_dups == [(0, 32)]


def test_handle_class_cache_refreshes_between_recorders(monkeypatch):
    from triton.runtime import interpreter

    original_tensor = ce._tensor_handle_cls()
    original_composites = ce._composite_handle_classes()

    class NewTensor:
        pass

    class NewBlockPointer:
        pass

    with monkeypatch.context() as patch:
        patch.setattr(interpreter, "TensorHandle", NewTensor)
        patch.setattr(interpreter, "BlockPointerHandle", NewBlockPointer)
        ce.ConcreteFootprintRecorder()
        assert ce._tensor_handle_cls() is NewTensor
        assert NewBlockPointer in ce._composite_handle_classes()
        assert ce._handle_of(NewTensor()).__class__ is NewTensor
    ce.ConcreteFootprintRecorder()
    assert ce._tensor_handle_cls() is original_tensor
    assert ce._composite_handle_classes() == original_composites


def test_handle_cache_retains_live_taint_and_nested_descriptor_fields():
    import triton.language as tl
    from triton.runtime.interpreter import BlockPointerHandle, TensorHandle

    ce.ConcreteFootprintRecorder()
    base = TensorHandle(np.asarray([32], dtype=np.uint64), tl.pointer_type(tl.int32))
    offset = TensorHandle(np.asarray([0], dtype=np.int32), tl.int32)
    descriptor = BlockPointerHandle(base, [], [], [offset], (1,), (0,))
    ce._tag(base, frozenset((1,)))
    ce._tag(offset, frozenset((2,)))
    assert ce._collect_taint([[descriptor]]) == (frozenset((1, 2)), False)
    ce._tag(offset, frozenset((3,)))
    assert ce._collect_taint([[descriptor]]) == (frozenset((1, 2, 3)), False)
    offset.attr.pop(ce._TAINT_ATTR)
    assert ce._collect_taint([[descriptor]]) == (frozenset((1,)), True)


def test_store_without_result_keeps_value_taint_and_dependency_roots():
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle

    rec = ce.ConcreteFootprintRecorder(fence_order=True)
    ptr = TensorHandle(np.asarray([32], dtype=np.uint64), tl.pointer_type(tl.int32))
    value = TensorHandle(np.asarray([7], dtype=np.int32), tl.int32)
    ce._tag(value, frozenset((4, 9)))
    owner = SimpleNamespace(create_store=lambda pointer, v: None)
    rec._wrap_attr(owner, "create_store", "store")
    try:
        assert owner.create_store(ptr, value) is None
        assert rec._pending_store_taint == frozenset((4, 9))
        assert rec._pending_dep_inputs == [value]
    finally:
        rec.cleanup()


def test_type_result_skips_only_unused_taint_lookup(monkeypatch):
    rec = ce.ConcreteFootprintRecorder(fence_order=True)
    result_type = object()
    owner = SimpleNamespace(get_block_ty=lambda unused: result_type)
    rec._wrap_attr(owner, "get_block_ty", None)

    def unexpected_taint_lookup(_):
        raise AssertionError("a type result has no value taint to propagate")

    monkeypatch.setattr(ce, "_collect_taint", unexpected_taint_lookup)
    try:
        assert owner.get_block_ty(1) is result_type
    finally:
        rec.cleanup()


def test_nested_handle_results_propagate_all_taints_without_losing_aliases():
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle

    rec = ce.ConcreteFootprintRecorder(fence_order=True)
    left = TensorHandle(np.asarray([1], dtype=np.int32), tl.int32)
    right = TensorHandle(np.asarray([2], dtype=np.int32), tl.int32)
    ce._tag(left, frozenset((2,)))
    ce._tag(right, frozenset((5,)))
    owner = SimpleNamespace(binary_op=lambda a, b: ([a, b], a))
    rec._wrap_attr(owner, "binary_op", None)
    try:
        result = owner.binary_op(left, right)
        assert result[0][0] is left and result[1] is left
        assert ce._taint_of(left) == ce._taint_of(right) == frozenset((2, 5))
    finally:
        rec.cleanup()
