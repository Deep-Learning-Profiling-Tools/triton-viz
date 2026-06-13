import pytest
import numpy as np
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from triton_viz.core.callbacks import OpCallbacks
from triton_viz.core.client import Client, ClientManager
from triton_viz.core.data import (
    AddPtr,
    BinaryOp,
    CastImpl,
    IntToPtr,
    Load,
    ProgramId,
    PtrToInt,
    ReduceSum,
    Store,
)

from triton_viz.core.frontend.base import AdapterResult, get_frontend
from triton_viz.core.frontend.nki import HAS_NKI
from triton_viz.core.symbolic_metadata import (
    FLOAT32,
    FLOAT8_E4B8,
    FLOAT8_E5B16,
    INT32,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    pointer_type,
)

from triton_viz.core.patch import PatchOp
from triton_viz.core.frontend.gluon import GLUON_CALLABLE_ADAPTERS

TRITON_FRONTEND = get_frontend("triton")
TRITON_ADAPTERS = TRITON_FRONTEND.adapters
NKI_ADAPTERS = get_frontend("nki").adapters
GLUON_FRONTEND = get_frontend("gluon")
GLUON_ADAPTERS = GLUON_FRONTEND.adapters


def test_adapter_result_kwargs_copy():
    """Test that AdapterResult SHALLOW-copies kwargs."""
    kwargs = {"named": "value"}
    result = AdapterResult(1, **kwargs)
    kwargs[
        "named"
    ] = "updated"  # result.kwargs["named"] is immutable str so it doesn't change from this
    assert result.args == (1,)
    assert result.kwargs == {"named": "value"}

    mutable_object = ["before"]
    kwargs = {"named": mutable_object}
    result = AdapterResult(1, **kwargs)
    kwargs["named"][
        0
    ] = "after"  # result.kwargs["named"] is mutable so it DOES change from this
    assert result.args == (1,)
    assert result.kwargs["named"][0] == "after"


def test_patch_op_preserves_descriptor_binding():
    seen = []

    class Target:
        def op(self, value):
            return ("ret", self, value)

    def adapter(instance, value):
        return AdapterResult(instance, value)

    original = Target.op
    callbacks = OpCallbacks(
        before_callback=lambda instance, value: seen.append((instance, value))
    )
    Target.op = PatchOp(
        original,
        ProgramId,
        callbacks,
        adapter,
        run_op_overrider=lambda op, _op_type, _op_overrider, args, kwargs: op(
            *args,
            **kwargs,
        ),
        maybe_yield_for_multism=lambda: None,
    )
    try:
        instance = Target()
        assert instance.op(7) == ("ret", instance, 7)
        assert seen == [(instance, 7)]
    finally:
        Target.op = original


def test_triton_store_adapter_handles_keys():
    """Adapter mirrors tl.store(ptr, value, mask=..., cache_modifier=..., eviction_policy=..., keys=...)."""
    adapter = TRITON_ADAPTERS[Store]
    ptr, value, mask, keys = object(), object(), object(), object()

    cache_modifier = "wb"
    eviction_policy = "evict_last"

    full = adapter(
        ptr,
        value,
        mask,
        cache_modifier=cache_modifier,
        eviction_policy=eviction_policy,
        keys=keys,
    )
    assert isinstance(full, AdapterResult)
    assert full.args == (ptr, mask, keys)
    assert full.kwargs == {}

    minimal = adapter(ptr, value, mask)
    assert minimal.args == (ptr, mask, None)
    assert minimal.kwargs == {}


def test_triton_load_adapter_keys_passthrough():
    """Adapter mirrors tl.load(ptr, mask=..., other=..., keys=...)."""
    adapter = TRITON_ADAPTERS[Load]
    ptr, mask, other, keys = object(), object(), object(), object()

    cache_modifier = "wb"
    eviction_policy = "evict_last"
    is_volatile = False

    full = adapter(
        ptr,
        mask,
        other,
        cache_modifier=cache_modifier,
        eviction_policy=eviction_policy,
        is_volatile=is_volatile,
        keys=keys,
    )
    assert full.args == (ptr, mask, keys)
    assert full.kwargs == {}

    minimal = adapter(ptr, mask, other)
    assert minimal.args == (ptr, mask, None)
    assert minimal.kwargs == {}


def test_triton_reduce_adapter_supports_positional_and_keyword_axis():
    """Adapter retains axis/keep_dims regardless of positional or keyword usage."""
    adapter = TRITON_ADAPTERS[ReduceSum]
    tensor = object()
    axis = 1
    keep_dims = True

    positional = adapter(tensor, axis, keep_dims)
    assert positional.args == (tensor, axis, keep_dims)
    assert positional.kwargs == {}

    keyword = adapter(tensor, axis=2, keep_dims=False)
    assert keyword.args == (tensor, 2, False)
    assert keyword.kwargs == {}


def test_triton_addptr_adapter_orders_arguments():
    """Adapter maps tl.addptr(ptr, offset) to (ptr, offset)."""
    adapter = TRITON_ADAPTERS[AddPtr]
    ptr = object()
    offset = object()
    result = adapter(ptr, offset)
    assert result.args == (ptr, offset)
    assert result.kwargs == {}


def test_program_id_adapter_returns_axis_only():
    """Adapter maps tl.program_id(axis) to (axis,)."""
    adapter = TRITON_ADAPTERS[ProgramId]
    program_id = 0
    result = adapter(program_id)
    assert result.args == (program_id,)
    assert result.kwargs == {}


def test_gluon_descriptor_symbolic_adapters_accept_async_signatures():
    load = GLUON_ADAPTERS[Load]
    store = GLUON_ADAPTERS[Store]
    from triton_viz.core.simulation import gluon as gluon_sim

    normal_descriptor_load = gluon_sim._descriptor_symbolic_load_adapter(
        coord_index=1,
        pred_index=4,
    )
    im2col_descriptor_load = gluon_sim._descriptor_symbolic_load_adapter(
        coord_index=1,
        pred_index=5,
    )
    tdm_descriptor_load = gluon_sim._descriptor_symbolic_load_adapter(
        coord_index=1,
        pred_index=3,
    )
    descriptor_store = gluon_sim._descriptor_symbolic_store_adapter(coord_index=1)
    desc = object()

    def assert_descriptor_access(result, coords, mask=None):
        access, result_mask, other = result.args
        assert access.descriptor is desc
        assert access.coords == coords
        assert access.pred == mask
        assert result_mask is None
        assert other is None

    assert load("ptr", "mask").args == ("ptr", "mask", None)
    assert_descriptor_access(
        normal_descriptor_load(desc, "coord", "bar", "result"),
        "coord",
    )
    assert_descriptor_access(
        normal_descriptor_load(desc, "coord", "bar", "result", "pred"),
        "coord",
        "pred",
    )
    assert_descriptor_access(
        im2col_descriptor_load(desc, "coord", "offsets", "bar", "result", "pred"),
        "coord",
        "pred",
    )
    assert_descriptor_access(
        tdm_descriptor_load(desc, "offsets", "dest"),
        "offsets",
    )
    assert_descriptor_access(
        tdm_descriptor_load(desc, "offsets", "dest", "pred"),
        "offsets",
        "pred",
    )
    assert_descriptor_access(
        normal_descriptor_load(desc, "coord", "bar", "result", pred="pred"),
        "coord",
        "pred",
    )
    assert_descriptor_access(
        normal_descriptor_load(desc, coord="kw_coord"),
        "kw_coord",
    )
    assert_descriptor_access(
        tdm_descriptor_load(desc, coord="kw_offsets"),
        "kw_offsets",
    )

    assert store("ptr", "value", "mask").args == ("ptr", "mask", None)
    store_result = descriptor_store(desc, "coord", "src")
    access, value, mask = store_result.args
    assert access.descriptor is desc
    assert access.coords == "coord"
    assert value == 0
    assert mask is None


def test_gluon_descriptor_symbolic_load_adapters_honor_positional_predicates():
    from triton_viz.core.simulation import gluon as gluon_sim

    desc = object()
    normal_load = gluon_sim._descriptor_symbolic_load_adapter(
        coord_index=1,
        pred_index=4,
    )
    im2col_load = gluon_sim._descriptor_symbolic_load_adapter(
        coord_index=1,
        pred_index=5,
    )
    tdm_load = gluon_sim._descriptor_symbolic_load_adapter(coord_index=1, pred_index=3)

    access, mask, other = normal_load(
        desc,
        "coord",
        "barrier",
        "result",
        False,
    ).args
    assert access.pred is False
    assert mask is None
    assert other is None

    access, mask, other = im2col_load(
        desc,
        "coord",
        "offsets",
        "barrier",
        "result",
    ).args
    assert access.pred is None
    assert mask is None
    assert other is None

    access, mask, other = im2col_load(
        desc,
        "coord",
        "offsets",
        "barrier",
        "result",
        False,
    ).args
    assert access.pred is False
    assert mask is None
    assert other is None

    access, mask, other = tdm_load(desc, "offsets", "dest", False).args
    assert access.pred is False
    assert mask is None
    assert other is None


def test_gluon_shared_memory_descriptor_class_is_not_an_op():
    from triton.experimental.gluon.language import _core as gluon_core

    assert "shared_memory_descriptor" not in GLUON_FRONTEND.namespaces[gluon_core]


def test_gluon_load_store_adapters_keep_first_positional_mask():
    load = GLUON_ADAPTERS[Load]
    store = GLUON_ADAPTERS[Store]

    assert load("ptr", "mask", "other", "boundary_check").args == (
        "ptr",
        "mask",
        None,
    )
    assert store("ptr", "value", "mask", "boundary_check").args == (
        "ptr",
        "mask",
        None,
    )


def test_gluon_load_adapter_does_not_treat_pred_as_mask():
    assert GLUON_ADAPTERS[Load]("ptr", pred="pred").args == (
        "ptr",
        None,
        None,
    )


def test_gluon_run_op_overrider_uses_operation_adapter(monkeypatch):
    calls = []

    def adapter(value):
        return AdapterResult("adapted", value)

    def overrider(lhs, rhs):
        calls.append((lhs, rhs))
        return "overridden"

    monkeypatch.setitem(GLUON_FRONTEND.adapters, Load, adapter)
    result = GLUON_FRONTEND.run_op_overrider(
        lambda value: value,
        Load,
        overrider,
        ("ptr",),
        {},
    )

    assert result == "overridden"
    assert calls == [("adapted", "ptr")]


def test_gluon_run_op_overrider_uses_callable_adapter(monkeypatch):
    calls = []

    def overrider(lhs, rhs, op):
        calls.append((lhs, rhs, op))
        return "overridden"

    def binary_op(lhs, rhs):
        return f"{lhs}+{rhs}"

    monkeypatch.setitem(
        GLUON_CALLABLE_ADAPTERS,
        binary_op,
        lambda lhs, rhs: AdapterResult(lhs, rhs, np.add),
    )

    result = GLUON_FRONTEND.run_op_overrider(
        binary_op,
        BinaryOp,
        overrider,
        ("lhs", "rhs"),
        {},
    )

    assert result == "overridden"
    assert calls == [("lhs", "rhs", np.add)]


def test_gluon_frontend_wraps_symbolic_concrete_fn():
    calls = []

    def concrete_fn(arg, *, named):
        calls.append((arg, named))
        return [arg, named]

    arg = object()
    named = object()
    wrapped = GLUON_FRONTEND.wrap_symbolic_concrete_fn(concrete_fn)

    assert wrapped(arg, named=named) == [arg, named]
    assert calls == [(arg, named)]


def test_gluon_language_ops_patch_and_delegate_to_semantics():
    from triton.experimental.gluon import language as ttgl

    seen_axes = []
    original_program_id = ttgl.program_id

    class RecordingClient(Client):
        NAME = "recording"

        def pre_run_callback(self, fn):
            return True

        def post_run_callback(self, fn):
            return True

        def arg_callback(self, name, arg, arg_cvt):
            pass

        def grid_callback(self, grid):
            pass

        def grid_idx_callback(self, grid_idx):
            pass

        def register_op_callback(self, op_type):
            if op_type is ProgramId:
                return OpCallbacks(before_callback=lambda axis: seen_axes.append(axis))
            return OpCallbacks()

        def register_for_loop_callback(self):
            from triton_viz.core.callbacks import ForLoopCallbacks

            return ForLoopCallbacks()

        def finalize(self):
            return []

        def pre_warmup_callback(self, jit_fn, *args, **kwargs):
            return False

        def post_warmup_callback(self, jit_fn, ret):
            pass

    def kernel():
        pass

    from triton.runtime.interpreter import interpreter_builder

    interpreter_builder.set_grid_dim(1, 1, 1)
    interpreter_builder.set_grid_idx(0, 0, 0)
    manager = ClientManager([RecordingClient()])
    with manager.patch_run(kernel, frontend_name="gluon"):
        assert ttgl.program_id is not original_program_id
        assert ttgl.program_id(0).handle.data == 0

    assert ttgl.program_id is original_program_id
    assert seen_axes == [0]


def test_triton_pointer_cast_aliases_have_distinct_op_types():
    frontend = get_frontend("triton")
    builder_ops = frontend.namespaces[frontend.builder]

    assert builder_ops["cast_impl"] is CastImpl
    assert builder_ops["create_ptr_to_int"] is PtrToInt
    assert builder_ops["create_int_to_ptr"] is IntToPtr


def test_triton_frontend_normalizes_block_pointer_type():
    frontend = get_frontend("triton")

    spec = frontend.normalize_symbolic_value(
        tl.block_type(tl.pointer_type(tl.float32), [1, 16])
    )

    assert isinstance(spec, SymbolicTypeSpec)
    assert spec.dtype == pointer_type(FLOAT32)
    assert spec.shape == (1, 16)


@pytest.mark.parametrize(
    "triton_dtype,symbolic_dtype",
    [
        (tl.float8e4b8, FLOAT8_E4B8),
        (tl.float8e5b16, FLOAT8_E5B16),
    ],
)
def test_triton_frontend_normalizes_other_fp8_dtypes(triton_dtype, symbolic_dtype):
    frontend = get_frontend("triton")

    assert frontend.normalize_symbolic_value(triton_dtype) == symbolic_dtype

    spec = frontend.normalize_symbolic_value(
        tl.block_type(tl.pointer_type(triton_dtype), [2, 4])
    )
    assert isinstance(spec, SymbolicTypeSpec)
    assert spec.dtype == pointer_type(symbolic_dtype)
    assert spec.shape == (2, 4)

    handle = TensorHandle(np.array([1, 2], dtype=np.uint8), triton_dtype)
    normalized = frontend.normalize_symbolic_value(handle)
    assert isinstance(normalized, SymbolicTensorValue)
    assert normalized.dtype == symbolic_dtype
    np.testing.assert_array_equal(normalized.data, handle.data)

    round_tripped = frontend.to_frontend_symbolic_value(normalized)
    assert isinstance(round_tripped, TensorHandle)
    assert round_tripped.dtype == triton_dtype
    np.testing.assert_array_equal(round_tripped.data, handle.data)


def test_triton_frontend_normalizes_tensor_and_round_trips():
    frontend = get_frontend("triton")
    handle = TensorHandle(np.array([7], dtype=np.int32), tl.int32)
    tensor = tl.core.tensor(handle, tl.int32)

    normalized = frontend.normalize_symbolic_value(tensor)

    assert isinstance(normalized, SymbolicTensorValue)
    assert normalized.dtype == INT32
    np.testing.assert_array_equal(normalized.data, np.array([7], dtype=np.int32))

    round_tripped = frontend.to_frontend_symbolic_value(normalized)
    assert isinstance(round_tripped, TensorHandle)
    assert round_tripped.dtype == tl.int32
    np.testing.assert_array_equal(round_tripped.data, handle.data)


def test_triton_frontend_wraps_symbolic_concrete_fn():
    frontend = get_frontend("triton")

    def concrete_fn(value, dtype):
        assert isinstance(value, TensorHandle)
        assert dtype == tl.float32
        return TensorHandle(value.data.astype(np.float32), dtype)

    wrapped = frontend.wrap_symbolic_concrete_fn(concrete_fn)
    result = wrapped(
        SymbolicTensorValue(np.array([3], dtype=np.int32), INT32),
        FLOAT32,
    )

    assert isinstance(result, SymbolicTensorValue)
    assert result.dtype == FLOAT32
    np.testing.assert_array_equal(result.data, np.array([3], dtype=np.float32))


@pytest.mark.skipif(not HAS_NKI, reason="NKI extras not installed")
def test_nki_load_store_adapters_align_with_clients():
    """NKI adapters normalize masked load/store to (tensor, mask, keys)."""
    src = object()
    dst = object()
    keys = object()
    mask = object()
    value = object()

    load_result = NKI_ADAPTERS[Load](src, keys, mask=mask)
    assert load_result.args == (src, mask, keys)
    assert load_result.kwargs == {}

    store_result = NKI_ADAPTERS[Store](dst, keys, value, mask=mask)
    assert store_result.args == (dst, mask, keys)
    assert store_result.kwargs == {}


def test_patchop_uses_adapter_for_callbacks():
    """PatchOp must feed the adapted arguments into before/after callbacks."""
    before_log: list[tuple[str, tuple, dict]] = []
    after_log: list[tuple[str, tuple, dict]] = []

    def before_callback(*args, **kwargs):
        before_log.append(("before", args, kwargs))

    def after_callback(ret, *args, **kwargs):
        after_log.append(("after", args, kwargs))

    adapter = TRITON_ADAPTERS[Store]
    callbacks = OpCallbacks(
        before_callback=before_callback, after_callback=after_callback
    )
    patch_op = PatchOp(
        op=lambda *_a, **_k: "return-value",
        op_type=Store,
        callbacks=callbacks,
        adapter=adapter,
        run_op_overrider=TRITON_FRONTEND.run_op_overrider,
        maybe_yield_for_multism=TRITON_FRONTEND.maybe_yield_for_multism,
    )

    ptr = object()
    mask = object()
    patch_op(ptr, "value", mask)

    expected_args = (ptr, mask, None)
    assert before_log == [("before", expected_args, {})]
    assert after_log == [("after", expected_args, {})]
