"""Pins for the int/bool value-snapshot sidecar (Hao, 2026-09-04: every
integer and bool tensor is snapshotted, floats stay by descriptor).

Capture side: a small int/bool tensor stays inline; a large one goes to
the content-addressed store and the descriptor carries ``values_ref``;
floats never carry values. Rebuild side: a ``values_ref`` resolves from
the sidecar with a hash check; a missing sidecar, a missing entry, or a
corrupted entry is a hard error (never a random rebuild); old specs
without references keep working unchanged.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.capture_common import (  # noqa: E402
    VALUE_SNAPSHOT_CAP,
    MissingValueSnapshot,
    ValueStore,
    describe_tensor,
    fingerprint,
    make_args_fn,
    make_tensor,
    prune_and_save_sidecar,
    referenced_values,
    values_sidecar_of,
    write_case_result,
)

BIG = VALUE_SNAPSHOT_CAP + 1


def _gen():
    return torch.Generator().manual_seed(0)


def test_small_int_stays_inline_large_int_goes_to_the_store():
    store = ValueStore()
    small = torch.arange(16, dtype=torch.int32)
    d = describe_tensor(small, store)
    assert d["values"] == list(range(16))
    assert "values_ref" not in d
    assert len(store) == 0

    big = torch.randint(0, 1000, (BIG,), dtype=torch.int32, generator=_gen())
    d = describe_tensor(big, store)
    assert "values" not in d
    assert len(d["values_ref"]) == 64
    assert d["init"] == "randint"
    assert len(store) == 1
    rebuilt = make_tensor(d, _gen(), store)
    assert torch.equal(rebuilt, big)


def test_bool_and_int64_and_strided_round_trip():
    store = ValueStore()
    mask = torch.rand(2 * BIG, generator=_gen()) > 0.5
    d = describe_tensor(mask, store)
    assert "values_ref" in d
    assert torch.equal(make_tensor(d, _gen(), store), mask)

    idx = torch.randint(-(2**40), 2**40, (BIG,), dtype=torch.int64, generator=_gen())
    d = describe_tensor(idx, store)
    assert torch.equal(make_tensor(d, _gen(), store), idx)

    base = torch.randint(0, 9, (200, 100), dtype=torch.int32, generator=_gen())
    view = base.t()  # non-contiguous, 20000 elements
    d = describe_tensor(view, store)
    assert d["contiguous"] is False and "values_ref" in d
    out = make_tensor(d, _gen(), store)
    assert out.stride() == view.stride()
    assert torch.equal(out, view)


def test_floats_stay_by_descriptor():
    store = ValueStore()
    x = torch.randn(BIG, generator=_gen())
    d = describe_tensor(x, store)
    assert d["init"] == "randn"
    assert "values" not in d and "values_ref" not in d
    assert len(store) == 0


def test_content_addressing_dedups_identical_tensors():
    store = ValueStore()
    a = torch.randint(0, 5, (BIG,), dtype=torch.int32, generator=_gen())
    da = describe_tensor(a, store)
    db = describe_tensor(a.clone(), store)
    assert da["values_ref"] == db["values_ref"]
    assert len(store) == 1
    # the reference is in the fingerprint: same shape, different contents,
    # different rows (the dedup must not merge them)
    c = a.clone()
    c[0] += 1
    dc = describe_tensor(c, store)
    rec = lambda d: {
        "module": "m",
        "kernel": "k",
        "constexprs": {},
        "grid": [1],
        "args": [d],
        "aliases": {},
    }
    assert fingerprint(rec(da)) != fingerprint(rec(dc))
    assert fingerprint(rec(da)) == fingerprint(rec(db))


def test_capture_without_a_store_marks_the_drop_instead_of_pretending():
    big = torch.randint(0, 3, (BIG,), dtype=torch.int32, generator=_gen())
    d = describe_tensor(big)  # no store: the legacy by-descriptor path
    assert d.get("values_dropped") is True
    assert "values_ref" not in d
    # and the rebuild stays the legacy seeded randint (old specs keep working)
    t = make_tensor(d, _gen())
    assert t.shape == big.shape and t.dtype == big.dtype


def test_sidecar_save_load_and_beside(tmp_path):
    store = ValueStore()
    big = torch.randint(0, 100, (BIG,), dtype=torch.int32, generator=_gen())
    d = describe_tensor(big, store)
    specs = tmp_path / "demo_specs.json"
    payload = {"cases": {"c": {"kernels": {"k": {"args": [dict(d, name="x")]}}}}}
    specs.write_text(json.dumps(payload))
    sidecar = prune_and_save_sidecar(store, payload, ValueStore.beside(specs).path)
    assert sidecar == tmp_path / "demo_values.npz"
    assert sidecar.exists()

    lazy = ValueStore.beside(specs)
    make_args = make_args_fn([dict(d, name="x")], {}, lazy)
    (rebuilt,) = make_args(0)
    assert torch.equal(rebuilt, big)
    assert referenced_values(payload) == {d["values_ref"]}


def test_prune_keeps_only_referenced_snapshots_and_refuses_missing(tmp_path):
    store = ValueStore()
    a = describe_tensor(
        torch.randint(0, 9, (BIG,), dtype=torch.int32, generator=_gen()), store
    )
    describe_tensor(
        torch.randint(0, 9, (BIG,), dtype=torch.int64, generator=_gen()), store
    )
    assert len(store) == 2
    payload = {"kernels": {"k": {"args": [a]}}}
    sidecar = prune_and_save_sidecar(store, payload, tmp_path / "p_values.npz")
    assert len(ValueStore(sidecar)) == 1
    with pytest.raises(MissingValueSnapshot):
        prune_and_save_sidecar(ValueStore(), payload, tmp_path / "q_values.npz")
    # nothing referenced: no sidecar is written (and a stale one is removed)
    stale = tmp_path / "r_values.npz"
    stale.write_bytes(b"stale")
    assert prune_and_save_sidecar(store, {"kernels": {}}, stale) is None
    assert not stale.exists()


def test_missing_sidecar_entry_or_store_is_a_hard_error(tmp_path):
    store = ValueStore()
    d = describe_tensor(
        torch.randint(0, 9, (BIG,), dtype=torch.int32, generator=_gen()), store
    )
    with pytest.raises(MissingValueSnapshot, match="no sidecar store"):
        make_tensor(d, _gen())
    with pytest.raises(MissingValueSnapshot, match="is not in"):
        make_tensor(d, _gen(), ValueStore(tmp_path / "absent_values.npz"))
    # a corrupted entry fails the hash check
    bad = ValueStore()
    bad._arrays[d["values_ref"]] = np.zeros(BIG, dtype=np.int32)
    with pytest.raises(MissingValueSnapshot, match="hash check"):
        make_tensor(d, _gen(), bad)


def test_write_case_result_ships_the_store_beside_the_json(tmp_path):
    store = ValueStore()
    d = describe_tensor(
        torch.randint(0, 9, (BIG,), dtype=torch.int32, generator=_gen()), store
    )
    out = tmp_path / "case.json"
    write_case_result(
        {"case": "c", "kernels": {"k": {"args": [d]}}, "_values": store}, out
    )
    payload = json.loads(out.read_text())
    assert "_values" not in payload
    assert values_sidecar_of(out).exists()
    merged = ValueStore()
    merged.merge(ValueStore(values_sidecar_of(out)))
    assert d["values_ref"] in merged
    # a result without snapshots writes no sidecar
    out2 = tmp_path / "empty.json"
    write_case_result({"case": "e", "kernels": {}, "_values": ValueStore()}, out2)
    assert not values_sidecar_of(out2).exists()


def test_legacy_inline_specs_rebuild_unchanged():
    d = {
        "kind": "tensor",
        "shape": [4],
        "dtype": "torch.int32",
        "contiguous": True,
        "init": "randint",
        "low": 0,
        "high": 3,
        "values": [2, 0, 1, 2],
        "name": "i",
    }
    (t,) = make_args_fn([d], {})(0)
    assert t.tolist() == [2, 0, 1, 2]
