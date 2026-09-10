"""Replay must preserve raw-pointer contents, allocation bounds, and aliases."""

from types import SimpleNamespace

import pytest
import torch

from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.replay import run_replay

from .test_replay_channels import _DD_SIG, _launch, _ttir_of, dd_mask_kernel, waw_kernel


def _snapshot(*args, **kwargs):
    detector = CompiledRaceDetector()
    names = [f"arg{i}" for i in range(len(args))] + list(kwargs)
    detector._snapshot_launch(SimpleNamespace(arg_names=names), args, kwargs)
    return detector


def _assert_unavailable(detector, reason):
    assert reason in detector._snapshot_skipped
    assert detector._replay_jit_fn is None
    assert detector._snapshot_args is None
    assert detector._snapshot_kwargs is None
    assert detector._snapshot_tensors is None
    assert detector._capture_error is None


@pytest.mark.parametrize("live_storage_index", [1, 2])
def test_gapped_flags_cannot_falsely_confirm_or_refute(live_storage_index):
    # The kernel loads raw offsets 0 and 1, not the view's logical offsets
    # 0 and 2. Compaction can introduce OR remove a live store mask while
    # every replay access still fits inside the compact allocation.
    base = torch.zeros(4, dtype=torch.int32)
    base[live_storage_index] = 1
    flags = base[::2]
    assert bool(base[:2].any()) != bool(flags.any())
    ttir = _ttir_of(dd_mask_kernel, _DD_SIG, {"BLOCK": 2})
    detector = CompiledRaceDetector(
        confirm_races=True, differential_check=True, ladder_level="L2"
    )
    _launch(
        detector,
        dd_mask_kernel,
        (flags, torch.arange(4, dtype=torch.float32), torch.zeros(2)),
        {"grid": (2,), "BLOCK": 2},
        ttir,
    )
    assert detector.last_global_status == "unsupported"
    assert detector.last_global_verdict["verdict"] == "potential-race"
    assert detector.last_global_confirmation is None
    assert "full allocation" in detector.last_global_reason
    assert "race-unconfirmed" not in detector.last_global_reason
    assert detector.last_content_hazard == []
    assert detector.last_differential is None
    assert base.tolist() == [int(i == live_storage_index) for i in range(4)]


def test_skipped_replay_keeps_exact_static_race():
    ttir = _ttir_of(
        waw_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 2},
    )
    detector = CompiledRaceDetector(confirm_races=True, differential_check=True)
    _launch(
        detector,
        waw_kernel,
        (torch.arange(8, dtype=torch.float32)[::2], torch.zeros(2)),
        {"grid": (2,), "BLOCK": 2},
        ttir,
    )
    assert detector.last_global_status == "races"
    assert detector.last_global_verdict["race_evidence"] == "exact"
    assert detector.last_global_confirmation is None
    assert detector.last_differential is None
    assert "full allocation" in detector.last_global_reason


@pytest.mark.parametrize("transpose", [False, True])
def test_complete_dense_allocations_preserve_snapshot_layout(transpose):
    value = torch.arange(12).reshape(3, 4)
    if transpose:
        value = value.T
    detector = _snapshot(value, scale=3)
    assert detector._snapshot_skipped is None
    clone = detector._snapshot_args[0]
    assert clone.data_ptr() != value.data_ptr()
    assert clone.shape == value.shape
    assert clone.stride() == value.stride()
    assert torch.equal(clone, value)
    assert detector._snapshot_kwargs == {"scale": 3}
    assert detector._snapshot_tensors["arg0"].contiguous == value.is_contiguous()


@pytest.mark.parametrize("kind", ["prefix", "offset", "gap", "overlap"])
def test_views_not_fully_represented_by_clone_are_unavailable(kind):
    base = torch.arange(4)
    view = {
        "prefix": base[:2],
        "offset": base[2:],
        "gap": base[::2],
        # numel equals the allocation size, but index 1 is repeated and
        # index 3 is absent. Allocation size alone is insufficient.
        "overlap": base.as_strided((2, 2), (1, 1)),
    }[kind]
    detector = _snapshot(view)
    _assert_unavailable(
        detector,
        "dense and non-overlapping" if kind == "overlap" else "full allocation",
    )


def test_shared_allocation_across_args_and_kwargs_is_unavailable():
    base = torch.arange(4)
    detector = _snapshot(base, alias=base.view(2, 2))
    _assert_unavailable(detector, "overlapping allocations")


def test_snapshot_rejection_clears_previous_launch_and_jit_references():
    detector = _snapshot(torch.arange(4))
    assert detector._snapshot_args is not None
    detector._snapshot_launch(None, (torch.arange(4)[::2],), {})
    _assert_unavailable(detector, "full allocation")


def test_snapshot_cap_is_unchanged_and_clears_previous_snapshot():
    detector = _snapshot(torch.arange(4))
    detector.SNAPSHOT_CAP_BYTES = 1
    detector._snapshot_launch(None, (torch.arange(4),), {})
    _assert_unavailable(detector, "tensor snapshot over cap")


def test_missing_tensor_metadata_never_caches_original_pointer_object():
    original = SimpleNamespace(
        data_ptr=lambda: 1234, numel=lambda: 4, element_size=lambda: 4
    )
    detector = _snapshot(original)
    _assert_unavailable(detector, "AttributeError")


def test_clone_retaining_original_allocation_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "clone", lambda self: self)
    detector = _snapshot(torch.arange(4))
    _assert_unavailable(detector, "independent storage")


def test_clone_layout_change_is_unavailable(monkeypatch):
    clone = torch.Tensor.clone
    monkeypatch.setattr(
        torch.Tensor,
        "clone",
        lambda self: clone(self, memory_format=torch.contiguous_format),
    )
    detector = _snapshot(torch.arange(12).reshape(3, 4).T)
    _assert_unavailable(detector, "changed layout")


def test_lazy_value_transform_is_unavailable():
    detector = _snapshot(torch._neg_view(torch.arange(4, dtype=torch.float32)))
    _assert_unavailable(detector, "lazy value transform")


def test_direct_replay_rejects_unfaithful_inputs_before_execution():
    # None cannot execute as a kernel: the layout rejection must occur
    # first, so no original tensor can reach the interpreter as fallback.
    base = torch.arange(4)
    result = run_replay(None, (base[::2],), {}, (1,), None)
    assert "full allocation" in result.error
    assert result.footprints == {}
    assert result.base_map == {}
    assert base.tolist() == list(range(4))


def test_snapshot_tensor_names_skip_middle_keyword_constexpr():
    jit_fn = SimpleNamespace(arg_names=["x_ptr", "BLOCK", "out_ptr"])
    detector = CompiledRaceDetector()
    detector._snapshot_launch(
        jit_fn, (torch.arange(4), torch.zeros(2)), {"BLOCK": 2, "grid": (2,)}
    )
    assert detector._snapshot_skipped is None
    assert set(detector._snapshot_tensors) == {"x_ptr", "out_ptr"}
    for name, clone in zip(("x_ptr", "out_ptr"), detector._snapshot_args):
        assert detector._snapshot_tensors[name].data_ptr == clone.data_ptr()


def test_lazy_integer_transform_cannot_supply_static_value_facts():
    base = torch.arange(1, 5, dtype=torch.int32)
    value = torch._neg_view(base)
    assert value.is_contiguous()
    assert value.data_ptr() == base.data_ptr()
    assert value.tolist() == [-1, -2, -3, -4]
    detector = CompiledRaceDetector(ladder_level="L2")
    detector._capture_launch(SimpleNamespace(arg_names=["p"]), (value,), {"grid": (2,)})
    assert detector._capture_error is None
    meta = detector._launch_tensors["p"]
    assert meta.snapshot is None
    assert meta.snapshot_reason == "lazy value transform"
    assert meta.init_values is None
    _assert_unavailable(detector, "lazy value transform")
    # Ordinary integer sources retain both channels' existing eligibility.
    assert detector._capture_snapshot(base, True) == ((1, 2, 3, 4), "")
    assert detector._capture_init_values(base, True) == (1, 2, 3, 4)
