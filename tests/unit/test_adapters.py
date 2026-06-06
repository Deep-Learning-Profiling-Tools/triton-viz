import pytest
import numpy as np
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from triton_viz.core.callbacks import OpCallbacks
from triton_viz.core.data import (
    AddPtr,
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
    INT32,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    pointer_type,
)

from triton_viz.core.patch import PatchOp

TRITON_ADAPTERS = get_frontend("triton").adapters
NKI_ADAPTERS = get_frontend("nki").adapters


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
    )

    ptr = object()
    mask = object()
    patch_op(ptr, "value", mask)

    expected_args = (ptr, mask, None)
    assert before_log == [("before", expected_args, {})]
    assert after_log == [("after", expected_args, {})]
