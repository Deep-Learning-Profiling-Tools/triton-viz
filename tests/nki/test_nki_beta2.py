import numpy as np
import pytest

try:
    import nki.isa as nisa
    import neuronxcc.nki.language as nl
    from triton_viz.core.nki_beta2 import (
        NDArray,
        NKIInterpretedFunction,
        dynamic_range,
        dma_copy,
        hbm,
        iota,
        memset,
        nc_matmul,
        nc_matmul_mx,
        nc_transpose,
        nki_builder,
        nki_patch_lang,
        nki_unpatch_lang,
        ndarray,
        quantize_mx,
        reciprocal,
        register_alloc,
        register_load,
        register_move,
        register_store,
        sbuf,
        tensor_copy,
        tensor_tensor,
        store,
    )
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki  # only run at "pytest -m nki"


_MISSING = object()
_LOGICAL_ROUND_SAMPLE = np.array(
    [[0.1, 0.9, 1.24], [1.51, 2.9, -5.2]], dtype=np.float32
)
_ROUND_EXPECTED = {
    "float8_e4m3": np.array(
        [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]], dtype=np.float32
    ),
    "float8_e5m2": np.array(
        [[0.09375, 0.875, 1.25], [1.5, 3.0, -5.0]], dtype=np.float32
    ),
    "float8_e4m3fn": np.array(
        [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]], dtype=np.float32
    ),
    "float8_e5m2fn": np.array(
        [[0.09375, 0.875, 1.25], [1.5, 3.0, -5.0]], dtype=np.float32
    ),
    "float4_e2m1fn": np.array([[0.0, 1.0, 1.0], [1.5, 3.0, -6.0]], dtype=np.float32),
}


class _FakeDType:
    """Minimal dtype-like object for testing unavailable NKI dtypes."""

    def __init__(self, name, itemsize=1):
        self.name = name
        self.itemsize = itemsize


class _PatchScope:
    """Minimal attr patch scope for language/isa monkeypatching."""

    def __init__(self):
        self._changes = []

    def set_attr(self, obj, name, value):
        original = getattr(obj, name, _MISSING)
        self._changes.append((obj, name, original))
        setattr(obj, name, value)

    def restore(self):
        while self._changes:
            obj, name, original = self._changes.pop()
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)


@pytest.mark.parametrize(
    "dtype_name",
    [
        "bfloat16",
        "tfloat32",
        "float8_e4m3",
        "float8_e5m2",
        "float8_e4m3fn",
        "float8_e5m2fn",
        "float4_e2m1fn",
    ],
)
def test_non_np_dtypes_use_float32_backing(dtype_name):
    dtype = getattr(nl, dtype_name, _FakeDType(dtype_name))
    logical_shape = (2, 3)
    expected_shape = logical_shape

    allocated = NDArray(shape=logical_shape, dtype=dtype, buffer="sbuf")
    assert allocated.dtype is dtype
    assert allocated.data.dtype == np.float32
    assert allocated.data.shape == expected_shape

    value = np.arange(np.prod(logical_shape), dtype=np.float32).reshape(logical_shape)
    with_value = NDArray(shape=logical_shape, dtype=dtype, value=value, buffer="sbuf")
    assert with_value.data.dtype == np.float32
    assert with_value.data.shape == expected_shape
    assert with_value.data.shape == value.shape

    via_ndarray = ndarray(logical_shape, dtype, buffer="sbuf")
    assert via_ndarray.data.dtype == np.float32
    assert via_ndarray.data.shape == expected_shape


@pytest.mark.parametrize(
    "dtype_name,expected_name",
    [
        ("float8_e4m3", "float8_e4m3"),
        ("float8_e5m2", "float8_e5m2"),
        ("float8_e4m3fn", "float8_e4m3fn"),
        ("float8_e5m2fn", "float8_e5m2fn"),
        ("float4_e2m1fn", "float4_e2m1fn"),
    ],
)
def test_non_np_dtypes_apply_semantic_rounding(dtype_name, expected_name):
    dtype = getattr(nl, dtype_name, _FakeDType(dtype_name))
    logical = _LOGICAL_ROUND_SAMPLE
    expected = _ROUND_EXPECTED[expected_name]

    created = NDArray(shape=logical.shape, dtype=dtype, value=logical, buffer="sbuf")
    assert np.array_equal(created.data, expected)

    dst = NDArray(shape=logical.shape, dtype=dtype, buffer="sbuf")
    store(dst, value=logical)
    assert np.array_equal(dst.data, expected)


def test_tfloat32_rounds_on_write_paths():
    dtype = getattr(nl, "tfloat32", _FakeDType("tfloat32", itemsize=4))
    values = np.array(
        [1.000123, 0.33333334, -7.12594, np.inf, np.nan], dtype=np.float32
    )

    created = NDArray(shape=values.shape, dtype=dtype, value=values, buffer="sbuf")
    stored = NDArray(shape=values.shape, dtype=dtype, buffer="sbuf")
    store(stored, value=values)
    assert np.array_equal(created.data, stored.data, equal_nan=True)

    finite_mask = np.isfinite(created.data)
    finite_bits = created.data.view(np.uint32)[finite_mask]
    assert np.all((finite_bits & np.uint32(0x1FFF)) == 0)
    assert not np.array_equal(created.data[finite_mask], values[finite_mask])


def test_quickstart_tensor_add_flow_with_dma_and_tensor_tensor():
    a_input = NDArray(value=np.arange(8, dtype=np.float32).reshape(2, 4), buffer="hbm")
    b_input = NDArray(
        value=np.arange(8, dtype=np.float32).reshape(2, 4) + 10, buffer="hbm"
    )

    a_tile = sbuf.view(dtype=a_input.dtype, size=a_input.shape)
    b_tile = sbuf.view(dtype=b_input.dtype, size=b_input.shape)
    c_tile = sbuf.view(dtype=a_input.dtype, size=a_input.shape)
    c_output = hbm.view(dtype=a_input.dtype, size=a_input.shape)

    dma_copy(dst=a_tile, src=a_input)
    dma_copy(dst=b_tile, src=b_input)
    tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)
    dma_copy(dst=c_output, src=c_tile)

    assert np.array_equal(c_output.data, a_input.data + b_input.data)


def test_matmul_and_transpose_match_numpy():
    stationary = NDArray(
        value=np.arange(12, dtype=np.float32).reshape(4, 3), buffer="sbuf"
    )
    moving = NDArray(value=np.arange(8, dtype=np.float32).reshape(4, 2), buffer="sbuf")
    dst = NDArray(value=np.empty((3, 2), dtype=np.float32), buffer="psum")
    nc_matmul(dst, stationary, moving)
    assert np.allclose(dst.data, stationary.data.T @ moving.data)

    data = NDArray(
        value=np.arange(24, dtype=np.float32).reshape(2, 3, 4), buffer="sbuf"
    )
    transposed = NDArray(value=np.empty((12, 2), dtype=np.float32), buffer="psum")
    nc_transpose(transposed, data)
    assert np.array_equal(transposed.data, data.data.reshape(2, -1).T)


def test_ap_offset_pattern_semantics_from_docs():
    t = NDArray(
        value=np.arange(16 * 16, dtype=np.float32).reshape(16, 16), buffer="sbuf"
    )
    access = t.ap(pattern=[[16, 16], [1, 8]], offset=8)

    expected = np.empty((16, 8), dtype=np.float32)
    t_flatten = t.data.flatten()
    for w in range(16):
        for z in range(8):
            idx = 8 + (w * 16) + (1 * z)
            expected[w, z] = t_flatten[idx]

    assert np.array_equal(access.data, expected)


def test_iota_vector_offset_and_dma_copy_dynamic_ap():
    src = NDArray(
        value=np.arange(128 * 512, dtype=np.float32).reshape(128, 512), buffer="hbm"
    )
    dynamic_idx = ndarray((64, 1), np.int32, buffer="sbuf")
    iota(dynamic_idx, [[1, 1]], 0, channel_multiplier=2)
    assert np.array_equal(
        dynamic_idx.data, np.arange(0, 128, 2, dtype=np.int32).reshape(64, 1)
    )

    result = ndarray((64, 512), np.float32, buffer="sbuf")
    src_access = src.ap(
        [[512, 64], [1, 512]], 0, vector_offset=dynamic_idx, indirect_dim=0
    )
    dma_copy(dst=result, src=src_access)
    assert np.array_equal(result.data, src.data[::2, :])

    memset(result, -1)
    assert np.array_equal(result.data, np.full((64, 512), -1, dtype=np.float32))


def test_register_ops_and_dynamic_range():
    reg = register_alloc()
    assert reg == 0
    reg = register_move(reg, 5)
    assert reg == 5

    src = NDArray(value=np.array([[9]], dtype=np.int32), buffer="sbuf")
    reg = register_load(reg, src)
    assert reg == 9

    dst = NDArray(value=np.zeros((1, 1), dtype=np.int32), buffer="sbuf")
    register_store(dst, reg)
    assert int(dst.data[0, 0]) == 9

    assert list(dynamic_range(1, reg, 4)) == [1, 5]


def test_tensor_copy_reciprocal_and_quantize_ops():
    src = NDArray(
        value=np.array([[1.5, -2.0], [3.0, 4.0]], dtype=np.float32), buffer="sbuf"
    )

    copied = NDArray(value=np.empty_like(src.data, dtype=np.float16), buffer="sbuf")
    tensor_copy(copied, src)
    assert copied.data.dtype == np.float16
    assert np.allclose(copied.data, src.data.astype(np.float16))

    rec = NDArray(value=np.empty_like(src.data), buffer="sbuf")
    reciprocal(rec, src)
    assert np.allclose(rec.data, 1.0 / src.data, atol=1e-6)

    quant_dst = NDArray(value=np.empty_like(src.data, dtype=np.int8), buffer="sbuf")
    quant_scale = NDArray(value=np.empty((1,), dtype=np.float32), buffer="sbuf")
    quantize_mx(quant_dst, src, quant_scale)
    assert float(quant_scale.data[0]) > 0

    moving = NDArray(
        value=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), buffer="sbuf"
    )
    moving_q = NDArray(value=np.empty_like(moving.data, dtype=np.int8), buffer="sbuf")
    moving_scale = NDArray(value=np.empty((1,), dtype=np.float32), buffer="sbuf")
    quantize_mx(moving_q, moving, moving_scale)

    mx_out = NDArray(value=np.empty((2, 2), dtype=np.float32), buffer="psum")
    nc_matmul_mx(mx_out, quant_dst, moving_q, quant_scale, moving_scale)
    expected = (quant_dst.data * quant_scale.data[0]).T @ (
        moving_q.data * moving_scale.data[0]
    )
    assert np.allclose(mx_out.data, expected)


def test_nki_patch_lang_maps_and_restores():
    scope = _PatchScope()
    original_ndarray = nl.ndarray
    original_store = nl.store
    original_nc_matmul = nisa.nc_matmul

    nki_patch_lang(scope)
    try:
        assert nl.ndarray is not original_ndarray
        assert nl.store is not original_store
        assert nisa.nc_matmul is not original_nc_matmul
    finally:
        nki_unpatch_lang(scope)

    assert nl.ndarray is original_ndarray
    assert nl.store is original_store
    assert nisa.nc_matmul is original_nc_matmul


def test_private_and_shared_hbm_allocations():
    nki_builder.shared_hbm_arrays = {}

    shared_a = ndarray((4,), np.float32, buffer=nl.shared_hbm, name="shared_vec")
    shared_b = ndarray((4,), np.float32, buffer=nl.shared_hbm, name="shared_vec")
    assert shared_a is shared_b
    assert shared_a.data_ptr() == shared_b.data_ptr()

    private_a = ndarray((4,), np.float32, buffer=nl.private_hbm, name="private_vec")
    private_b = ndarray((4,), np.float32, buffer=nl.private_hbm, name="private_vec")
    assert private_a.data_ptr() != private_b.data_ptr()
    assert private_a.buffer == "hbm"
    assert private_b.buffer == "hbm"


def test_interpreted_function_runs_without_client_manager():
    scope = _PatchScope()
    nki_patch_lang(scope)
    try:

        def copy_kernel(src, dst):
            nisa.dma_copy(dst=dst, src=src)

        fn = NKIInterpretedFunction(copy_kernel)
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        dst = np.empty_like(src)
        fn.run(src, dst, grid=(1,))
        assert np.array_equal(dst, src)
    finally:
        nki_unpatch_lang(scope)
