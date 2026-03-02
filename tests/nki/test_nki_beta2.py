import numpy as np
import pytest

try:
    import nki.isa as nisa
    import nki.language as nl
    from triton_viz.core.nki_beta2 import (
        NDArray,
        NKIBeta2InterpretedFunction,
        dma_copy,
        hbm,
        nc_matmul,
        nc_transpose,
        nki_patch_lang,
        nki_unpatch_lang,
        ndarray,
        reciprocal,
        sbuf,
        tensor_copy,
        tensor_tensor,
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
    dst = NDArray(value=np.zeros((3, 2), dtype=np.float32), buffer="psum")
    nc_matmul(dst, stationary, moving)
    assert np.allclose(dst.data, stationary.data.T @ moving.data)

    data = NDArray(
        value=np.arange(24, dtype=np.float32).reshape(2, 3, 4), buffer="sbuf"
    )
    transposed = NDArray(value=np.empty((12, 2), dtype=np.float32), buffer="psum")
    nc_transpose(transposed, data)
    assert np.array_equal(transposed.data, data.data.reshape(2, -1).T)


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


def test_nki_patch_lang_maps_and_restores():
    scope = _PatchScope()
    original_ndarray = nl.ndarray
    original_program_id = nl.program_id
    original_nc_matmul = nisa.nc_matmul

    nki_patch_lang(scope)
    try:
        assert nl.ndarray is not original_ndarray
        assert nl.program_id is not original_program_id
        assert nisa.nc_matmul is not original_nc_matmul
    finally:
        nki_unpatch_lang(scope)

    assert nl.ndarray is original_ndarray
    assert nl.program_id is original_program_id
    assert nisa.nc_matmul is original_nc_matmul


def test_interpreted_function_runs_without_client_manager():
    scope = _PatchScope()
    nki_patch_lang(scope)
    try:

        def copy_kernel(src, dst):
            nisa.dma_copy(dst=dst, src=src)

        fn = NKIBeta2InterpretedFunction(copy_kernel)
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        dst = np.empty_like(src)
        fn.run(src, dst, grid=(1,))
        assert np.array_equal(dst, src)
    finally:
        nki_unpatch_lang(scope)


def test_trace_backend_nki_beta2_runs():
    import triton_viz
    from triton_viz.clients import Tracer

    triton_viz.clear()

    @triton_viz.trace(client=Tracer(), backend="nki_beta2")
    def copy_kernel(src, dst):
        nisa.dma_copy(dst=dst, src=src)

    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    dst = np.empty_like(src)
    copy_kernel[(1,)](src, dst)
    assert np.array_equal(dst, src)
