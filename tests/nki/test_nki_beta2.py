import inspect

import numpy as np
import pytest

try:
    from triton_viz.core.patch import _LangPatchScope
    import nki.isa as nisa
    import nki.language as nl
    import triton_viz.core.nki_beta2 as b2
    from triton_viz.utils.dtypes import STORAGE_DTYPES
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki


@pytest.fixture
def patched_scope():
    """Patch beta2 interpreter symbols for one test."""
    scope = _LangPatchScope()
    b2.nki_patch_lang(scope)
    yield scope
    b2.nki_unpatch_lang(scope)


def _nd(value, dtype=None, buffer="sbuf"):
    """Create an interpreter NDArray from Python data."""
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return b2.NDArray(value=array, buffer=buffer)


def _zeros(shape, dtype=None, buffer="sbuf"):
    """Create a zero-initialized interpreter NDArray."""
    array = np.zeros(shape, dtype=dtype) if dtype is not None else np.zeros(shape)
    return b2.NDArray(value=array, buffer=buffer)


def _assert_low_precision_dma_copy(dtype_name, expected):
    """Assert dma_copy casts source values to dtype-specific storage."""
    logical = np.array([[0.1, 0.9, 1.24], [1.51, 2.9, -5.2]], dtype=np.float64)
    dtype = getattr(nl, dtype_name, dtype_name)
    dst = b2.NDArray(shape=logical.shape, dtype=dtype, buffer="sbuf")
    b2.dma_copy(dst=dst, src=_nd(logical))
    assert dst.data.dtype == STORAGE_DTYPES[dtype_name]
    assert np.array_equal(dst.data, np.array(expected, dtype=dst.data.dtype))


def _core_inputs():
    """Build canonical tensors used by core dataflow tests."""
    a = _nd(np.arange(8, dtype=float).reshape(2, 4), buffer="hbm")
    b = _nd(np.arange(8, dtype=float).reshape(2, 4) + 10, buffer="hbm")
    out = _zeros((2, 4), buffer="sbuf")
    return a, b, out


def _act_input():
    """Build canonical activation input tensor."""
    return _nd(np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -1.0]]))


def _dtype_token(name):
    """Resolve a dtype name to nl token or raw string fallback."""
    return getattr(nl, name, name)


def _is_float_dtype(dtype):
    """Return whether dtype should be compared as floating-point values."""
    return "float" in str(dtype) or "bfloat" in str(dtype)


def _assert_tensor_equal(actual, expected):
    """Assert equality with float-aware tolerance handling."""
    if _is_float_dtype(actual.dtype):
        assert np.allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=1e-2,
            atol=1e-2,
            equal_nan=True,
        )
        return
    assert np.array_equal(actual, expected)


def _matrix(shape, dist, dtype_name, seed):
    """Generate deterministic matrix inputs across requested distributions."""
    size = int(np.prod(shape))
    if dist == "linspace":
        data = np.linspace(-3.0, 3.0, num=size, dtype=np.float64)
    elif dist == "arange":
        data = np.arange(size, dtype=np.float64) - size // 2
    elif dist == "randn":
        data = np.random.default_rng(seed).standard_normal(size)
    else:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unknown distribution: {dist}")
    if dtype_name == "int16":
        data = np.rint(data * 64.0)
    return np.asarray(data.reshape(shape), dtype=STORAGE_DTYPES[dtype_name])


SHAPE_CASES = ((), (1,), (17,), (19, 23), (3, 5, 7, 11, 13))
NL_DTYPE_CASES = (
    "bool_",
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float16",
    "float32",
    "bfloat16",
    "tfloat32",
    "float8_e4m3",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e5m2fn",
)


def test_dma_copy_bfloat16_quantization():
    """dma_copy should quantize values to bfloat16 precision."""
    _assert_low_precision_dma_copy(
        "bfloat16",
        [[0.10009765625, 0.8984375, 1.2421875], [1.5078125, 2.90625, -5.1875]],
    )


def test_dma_copy_float8_e4m3_quantization():
    """dma_copy should quantize values to float8_e4m3 precision."""
    _assert_low_precision_dma_copy(
        "float8_e4m3",
        [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]],
    )


def test_dma_copy_float8_e5m2_quantization():
    """dma_copy should quantize values to float8_e5m2 precision."""
    _assert_low_precision_dma_copy(
        "float8_e5m2",
        [[0.09375, 0.875, 1.25], [1.5, 3.0, -5.0]],
    )


def test_dma_copy_float8_e4m3fn_quantization():
    """dma_copy should quantize values to float8_e4m3fn precision."""
    _assert_low_precision_dma_copy(
        "float8_e4m3fn",
        [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]],
    )


def test_dma_copy_float4_e2m1fn_quantization():
    """dma_copy should quantize values to float4_e2m1fn precision."""
    _assert_low_precision_dma_copy(
        "float4_e2m1fn",
        [[0.0, 1.0, 1.0], [1.5, 3.0, -6.0]],
    )


def test_unsupported_float8_e5m2fn_dtype_raises():
    """float8_e5m2fn should be unsupported by beta2 dtype mapping."""
    dtype = getattr(nl, "float8_e5m2fn", "float8_e5m2fn")
    with pytest.raises(TypeError, match="Unsupported dtype"):
        b2.NDArray(shape=(1, 1), dtype=dtype, buffer="sbuf")


def test_unsupported_tfloat32_dtype_raises():
    """tfloat32 should be unsupported by beta2 dtype mapping."""
    dtype = getattr(nl, "tfloat32", "tfloat32")
    with pytest.raises(TypeError, match="Unsupported dtype"):
        b2.NDArray(shape=(1, 1), dtype=dtype, buffer="sbuf")


def test_ndarray_has_no_name_attr():
    """Beta2 NDArray should not expose a name attribute."""
    assert not hasattr(b2.NDArray(value=np.array([1])), "name")


def test_patch_surface_and_signatures(patched_scope):
    """Patched surface should match this repository's beta2 dialect."""
    del patched_scope
    nl_callable_names = (
        "ndarray",
        "zeros",
        "add",
        "subtract",
        "multiply",
        "maximum",
        "sqrt",
        "rsqrt",
        "program_id",
        "affine_range",
        "ds",
    )
    nl_value_names = ("tile_size", "sbuf", "hbm", "psum")
    nisa_names = (
        "nc_matmul",
        "nc_transpose",
        "activation",
        "exponential",
        "reciprocal",
        "dma_copy",
        "tensor_copy",
        "tensor_tensor",
        "tensor_reduce",
        "tensor_scalar",
    )

    for name in nl_callable_names:
        assert hasattr(nl, name)
    for name in nl_value_names:
        assert hasattr(nl, name)
    for name in nisa_names:
        assert hasattr(nisa, name)

    for name in nl_callable_names:
        assert str(inspect.signature(getattr(nl, name))) == str(
            inspect.signature(getattr(b2, name))
        )

    for name in nisa_names:
        assert str(inspect.signature(getattr(nisa, name))) == str(
            inspect.signature(getattr(b2, name))
        )


def test_language_helpers_and_ops(patched_scope):
    """Patched language helpers should run with expected semantics."""
    del patched_scope
    arr = nl.ndarray((2, 2), dtype=float, buffer=nl.sbuf)
    z = nl.zeros((2, 2), dtype=float)
    assert arr.shape == (2, 2)
    assert np.array_equal(z.data, np.zeros((2, 2)))

    b2.nki_builder.grid_dims = (2, 3, 4)
    b2.nki_builder.grid_idx = [1, 2, 3]
    assert nl.program_id(0) == 1
    assert list(nl.affine_range(4)) == [0, 1, 2, 3]
    assert list(nl.affine_range(1, 6, 2)) == [1, 3, 5]
    assert list(nl.affine_range(start=1, stop=6, step=2)) == [1, 3, 5]
    assert nl.ds(2, 3) == slice(2, 5, None)
    assert nl.tile_size.pmax == 128

    x = _nd([[1, 2], [3, 4]], dtype=float)
    y = _nd([[4, 3], [2, 1]], dtype=float)
    assert np.array_equal(nl.add(x, y).data, np.array([[5, 5], [5, 5]]))
    assert np.array_equal(nl.subtract(x, y).data, np.array([[-3, -1], [1, 3]]))
    assert np.array_equal(nl.multiply(x, y).data, np.array([[4, 6], [6, 4]]))
    assert np.array_equal(nl.maximum(x, y).data, np.array([[4, 3], [3, 4]]))
    assert np.array_equal(nl.sqrt(_nd([[1.0, 4.0]])).data, np.array([[1.0, 2.0]]))
    assert np.array_equal(nl.rsqrt(_nd([[4.0, 16.0]])).data, np.array([[0.5, 0.25]]))
    exp_dst = _zeros((1, 2))
    nisa.exponential(exp_dst, _nd([[0.0, 1.0]]))
    assert np.allclose(exp_dst.data, np.exp([[0.0, 1.0]]))


def test_dma_copy(patched_scope):
    """dma_copy should copy source data into destination."""
    del patched_scope
    a, _, out = _core_inputs()
    nisa.dma_copy(out, a)
    assert np.array_equal(out.data, a.data)


def test_dma_copy_dst_rmw_op(patched_scope):
    """dma_copy should apply dst_rmw_op when provided."""
    del patched_scope
    a, b, out = _core_inputs()
    nisa.dma_copy(out, a)
    nisa.dma_copy(out, b, dst_rmw_op=nl.add)
    assert np.array_equal(out.data, a.data + b.data)


def test_tensor_copy(patched_scope):
    """tensor_copy should copy tensors elementwise."""
    del patched_scope
    a, _, out = _core_inputs()
    nisa.tensor_copy(out, a)
    assert np.array_equal(out.data, a.data)


def test_tensor_tensor(patched_scope):
    """tensor_tensor should apply the provided binary op."""
    del patched_scope
    a, b, out = _core_inputs()
    nisa.tensor_tensor(out, a, b, nl.subtract)
    assert np.array_equal(out.data, a.data - b.data)


def test_tensor_scalar(patched_scope):
    """tensor_scalar should support chained ops with operand1/op1."""
    del patched_scope
    a, _, out = _core_inputs()
    nisa.tensor_scalar(out, a, nl.multiply, 2.0, op1=nl.add, operand1=1.0)
    assert np.array_equal(out.data, a.data * 2.0 + 1.0)


def test_tensor_scalar_broadcasts_free_dim_only(patched_scope):
    """tensor_scalar should broadcast only along free dimensions."""
    del patched_scope
    data = _nd(np.arange(6, dtype=np.float32).reshape(2, 3))
    dst = _zeros((2, 3))
    operand_free = _nd(np.array([[10.0], [20.0]], dtype=np.float32))
    nisa.tensor_scalar(dst, data, nl.add, operand_free)
    expected = data.data + np.array([[10.0], [20.0]], dtype=np.float32)
    assert np.array_equal(dst.data, expected)

    operand_partition = _nd(np.array([[10.0, 20.0, 30.0]], dtype=np.float32))
    with pytest.raises(
        ValueError, match="tensor_scalar only broadcasts free dimensions"
    ):
        nisa.tensor_scalar(dst, data, nl.add, operand_partition)


def test_tensor_reduce_shape(patched_scope):
    """tensor_reduce should reduce across the selected axis."""
    del patched_scope
    src = _nd(np.array([[1.0, 2.0, 3.0], [5.0, 4.0, 0.0]]))
    dst = _zeros((2, 1))
    nisa.tensor_reduce(dst, nl.maximum, src, axis=1, keepdims=True)
    assert dst.data.shape == (2, 1)
    assert np.array_equal(dst.data.reshape(-1), np.array([3.0, 5.0]))


def test_reciprocal(patched_scope):
    """reciprocal should compute elementwise multiplicative inverse."""
    del patched_scope
    rec = _zeros((2, 4))
    src = _nd(np.array([[1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]]))
    nisa.reciprocal(rec, src)
    assert np.allclose(rec.data[0, :2], np.array([1.0, 0.5]))


def test_nc_transpose(patched_scope):
    """nc_transpose should transpose partition and flattened free axes."""
    del patched_scope
    src = _nd(np.arange(24, dtype=float).reshape(2, 3, 4))
    dst = _zeros((12, 2), buffer="psum")
    nisa.nc_transpose(dst, src)
    assert np.array_equal(dst.data, src.data.reshape(2, -1).T)


def test_nc_matmul(patched_scope):
    """nc_matmul should match transposed stationary matmul semantics."""
    del patched_scope
    stationary = _nd(np.arange(12, dtype=float).reshape(4, 3))
    moving = _nd(np.arange(8, dtype=float).reshape(4, 2))
    dst = _zeros((3, 2), buffer="psum")
    nisa.nc_matmul(dst, stationary, moving)
    assert np.allclose(dst.data, stationary.data.T @ moving.data)


def test_nc_matmul_requires_psum_dst_and_sbuf_inputs(patched_scope):
    """nc_matmul should enforce psum dst and sbuf inputs."""
    del patched_scope
    stationary = _nd(np.arange(12, dtype=float).reshape(4, 3), buffer="sbuf")
    moving = _nd(np.arange(8, dtype=float).reshape(4, 2), buffer="sbuf")
    bad_dst = _zeros((3, 2), buffer="sbuf")
    with pytest.raises(ValueError, match="nc_matmul requires dst in psum"):
        nisa.nc_matmul(bad_dst, stationary, moving)

    dst = _zeros((3, 2), buffer="psum")
    bad_stationary = _nd(np.arange(12, dtype=float).reshape(4, 3), buffer="hbm")
    with pytest.raises(ValueError, match="nc_matmul requires stationary in sbuf"):
        nisa.nc_matmul(dst, bad_stationary, moving)

    bad_moving = _nd(np.arange(8, dtype=float).reshape(4, 2), buffer="hbm")
    with pytest.raises(ValueError, match="nc_matmul requires moving in sbuf"):
        nisa.nc_matmul(dst, stationary, bad_moving)


def test_activation(patched_scope):
    """activation should write outputs into destination."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 3))
    bias = _nd(np.ones((2, 3), dtype=np.float32))
    nisa.activation(dst, np.reciprocal, src, bias=bias, scale=1.0)
    assert dst.data.shape == (2, 3)


def test_activation_rejects_scalar_bias(patched_scope):
    """activation should reject non-tensor bias arguments."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 3))
    with pytest.raises(TypeError, match="activation bias must be a tensor"):
        nisa.activation(dst, np.reciprocal, src, bias=1.0, scale=1.0)


def test_program_id_invalid_axis_raises(patched_scope):
    """program_id should reject out-of-range launch axes."""
    del patched_scope
    b2.nki_builder.grid_idx = [0, 1]
    with pytest.raises(ValueError, match="Invalid axis"):
        nl.program_id(2)


def test_ds_non_scalar_start_raises(patched_scope):
    """ds should reject non-scalar start inputs."""
    del patched_scope
    with pytest.raises(ValueError, match="Expected scalar value"):
        nl.ds(np.array([1, 2]), 3)


def test_interpreted_function_execution(patched_scope):
    """NKIBeta2InterpretedFunction should execute kernels over launch grids."""
    del patched_scope

    class _ClientManager:
        """Minimal client manager used by interpreted function tests."""

        def grid_callback(self, grid):
            """Record the launch grid."""
            self.grid = grid

        def grid_idx_callback(self, idx):
            """Record the active grid index."""
            self.last_idx = idx

        def arg_callback(self, name, arg, ret):
            """Record the latest argument callback payload."""
            self.last_arg = (name, arg, ret)

        def pre_run_callback(self, fn):
            """Allow execution to proceed."""
            self.last_fn = fn
            return True

        def post_run_callback(self, fn):
            """Allow execution to proceed after one program."""
            self.last_post_fn = fn
            return True

    def copy_kernel(src, dst):
        """Copy one source tile into destination."""
        nisa.dma_copy(dst=dst, src=src)

    fn = b2.NKIBeta2InterpretedFunction(copy_kernel)
    src = np.arange(16, dtype=float).reshape(4, 4)
    dst = np.empty_like(src)
    fn.run(src, dst, grid=(1,), client_manager=_ClientManager())
    assert np.array_equal(dst, src)


def test_ndarray_shape_dtype_buffer_matrix(patched_scope):
    """ndarray should match documented shape/dtype/buffer behavior."""
    del patched_scope
    for dtype_name in NL_DTYPE_CASES:
        if not hasattr(nl, dtype_name):
            continue
        dtype = _dtype_token(dtype_name)
        for shape in SHAPE_CASES:
            for buffer in (nl.hbm, nl.sbuf, nl.psum):
                if dtype in STORAGE_DTYPES:
                    out = nl.ndarray(shape, dtype=dtype, buffer=buffer)
                    assert out.shape == shape
                    assert out.buffer == buffer.buffer
                    assert out.data.dtype == STORAGE_DTYPES[dtype]
                else:
                    with pytest.raises(TypeError, match="Unsupported dtype"):
                        nl.ndarray(shape, dtype=dtype, buffer=buffer)
    with pytest.raises(TypeError, match="Unsupported buffer type"):
        nl.ndarray((1,), dtype=_dtype_token("float32"), buffer="private_hbm")


def test_zeros_shape_dtype_matrix(patched_scope):
    """zeros should match shape and dtype behavior."""
    del patched_scope
    for dtype_name in NL_DTYPE_CASES:
        if not hasattr(nl, dtype_name):
            continue
        dtype = _dtype_token(dtype_name)
        for shape in SHAPE_CASES:
            if dtype in STORAGE_DTYPES:
                out = nl.zeros(shape, dtype=dtype)
                assert out.shape == shape
                assert out.buffer == nl.sbuf.buffer
                assert out.data.dtype == STORAGE_DTYPES[dtype]
                assert np.all(out.data == 0)
            else:
                with pytest.raises(TypeError, match="Unsupported dtype"):
                    nl.zeros(shape, dtype=dtype)


def test_zeros_rejects_buffer_kwarg(patched_scope):
    """zeros should reject explicit buffer kwarg."""
    del patched_scope
    with pytest.raises(
        TypeError,
        match="unexpected keyword argument 'buffer' in builtinfunction 'builtin_lang_zeros'",
    ):
        nl.zeros((2, 2), dtype=nl.float32, buffer=nl.sbuf)


@pytest.mark.parametrize(
    "dst_shape,data_shape",
    (((4, 8), (8, 4)), ((35, 3), (3, 5, 7))),
)
@pytest.mark.parametrize(
    "dst_dtype_name,data_dtype_name",
    (("float32", "float32"), ("bfloat16", "bfloat16"), ("bfloat16", "float32")),
)
def test_nc_transpose_shape_dtype_cases(
    patched_scope, dst_shape, data_shape, dst_dtype_name, data_dtype_name
):
    """nc_transpose should match documented shape and dtype combinations."""
    del patched_scope
    src_data = (
        np.arange(np.prod(data_shape), dtype=np.float32).reshape(data_shape) + 0.25
    )
    data = _nd(src_data, dtype=STORAGE_DTYPES[data_dtype_name], buffer="sbuf")
    dst = b2.NDArray(shape=dst_shape, dtype=_dtype_token(dst_dtype_name), buffer="sbuf")
    nisa.nc_transpose(dst, data)
    expected = np.asarray(data.data).reshape(data.shape[0], -1).T
    expected = np.asarray(expected, dtype=dst.data.dtype)
    _assert_tensor_equal(dst.data, expected)


def test_nc_matmul_matches_numpy_matrix(patched_scope):
    """nc_matmul should match numpy across documented shapes/dtypes/distributions."""
    del patched_scope
    shape_cases = (((5, 3), (5, 7)), ((128, 128), (128, 128)), ((129, 127), (129, 125)))
    dtype_names = ("float64", "float32", "bfloat16", "int16")
    distributions = ("linspace", "arange", "randn")
    seed = 17

    for stationary_shape, moving_shape in shape_cases:
        for dtype_name in dtype_names:
            for dist in distributions:
                lhs_np = _matrix(stationary_shape, dist, dtype_name, seed)
                rhs_np = _matrix(moving_shape, dist, dtype_name, seed + 1)
                stationary = _nd(lhs_np, buffer="sbuf")
                moving = _nd(rhs_np, buffer="sbuf")
                dst = b2.NDArray(
                    shape=(stationary_shape[1], moving_shape[1]),
                    dtype=_dtype_token(dtype_name),
                    buffer="psum",
                )
                dst.data.fill(0)
                nisa.nc_matmul(dst, stationary, moving)
                expected = np.zeros(dst.shape, dtype=dst.data.dtype)
                expected += np.asarray(
                    np.asarray(stationary.data).T @ np.asarray(moving.data),
                    dtype=dst.data.dtype,
                )
                _assert_tensor_equal(dst.data, expected)
                seed += 2


def test_nc_matmul_accumulates_across_calls(patched_scope):
    """nc_matmul should accumulate into dst across repeated calls."""
    del patched_scope
    shape_cases = (((5, 3), (5, 7)), ((128, 128), (128, 128)), ((129, 127), (129, 125)))
    dtype_names = ("float64", "float32", "bfloat16", "int16")
    distributions = ("linspace", "arange", "randn")
    seed = 101

    for stationary_shape, moving_shape in shape_cases:
        for dtype_name in dtype_names:
            for dist in distributions:
                lhs0 = _nd(
                    _matrix(stationary_shape, dist, dtype_name, seed), buffer="sbuf"
                )
                rhs0 = _nd(
                    _matrix(moving_shape, dist, dtype_name, seed + 1), buffer="sbuf"
                )
                lhs1 = _nd(
                    _matrix(stationary_shape, dist, dtype_name, seed + 2), buffer="sbuf"
                )
                rhs1 = _nd(
                    _matrix(moving_shape, dist, dtype_name, seed + 3), buffer="sbuf"
                )
                dst = b2.NDArray(
                    shape=(stationary_shape[1], moving_shape[1]),
                    dtype=_dtype_token(dtype_name),
                    buffer="psum",
                )
                dst.data.fill(0)
                nisa.nc_matmul(dst, lhs0, rhs0)
                nisa.nc_matmul(dst, lhs1, rhs1)
                expected = np.zeros(dst.shape, dtype=dst.data.dtype)
                expected += np.asarray(
                    np.asarray(lhs0.data).T @ np.asarray(rhs0.data),
                    dtype=dst.data.dtype,
                )
                expected += np.asarray(
                    np.asarray(lhs1.data).T @ np.asarray(rhs1.data),
                    dtype=dst.data.dtype,
                )
                _assert_tensor_equal(dst.data, expected)
                seed += 4


@pytest.mark.parametrize("dtype_name", ("float64", "float32", "bfloat16", "int16"))
def test_reciprocal_value_dtype_cases(patched_scope, dtype_name):
    """reciprocal should match numpy and overwrite non-zero destination."""
    del patched_scope
    dtype = _dtype_token(dtype_name)
    src = _nd(np.array([[0.0, 23.0, 4.0 / 7.0]], dtype=STORAGE_DTYPES[dtype_name]))
    dst = b2.NDArray(shape=src.shape, dtype=dtype, buffer="sbuf")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nisa.reciprocal(dst, src)
        expected = np.reciprocal(np.asarray(src.data, dtype=np.float32)).astype(
            dst.data.dtype
        )
    _assert_tensor_equal(dst.data, expected)

    dst_overwrite = _nd(np.full(src.shape, 9, dtype=dst.data.dtype))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nisa.reciprocal(dst_overwrite, src)
    _assert_tensor_equal(dst_overwrite.data, expected)


@pytest.mark.parametrize("dtype_name", ("float64", "float32", "bfloat16", "int16"))
@pytest.mark.parametrize(
    "op_name,np_op",
    (
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
        ("maximum", np.maximum),
    ),
)
def test_binary_ops_match_numpy_with_overflow_cases(
    patched_scope, dtype_name, op_name, np_op
):
    """Binary nl ops should match numpy for regular and overflow edge cases."""
    del patched_scope
    lhs = _matrix((19, 23), "linspace", dtype_name, 31)
    rhs = _matrix((19, 23), "randn", dtype_name, 47)
    op = getattr(nl, op_name)
    actual = op(_nd(lhs), _nd(rhs)).data
    expected = np_op(lhs, rhs)
    _assert_tensor_equal(actual, expected)

    if dtype_name == "int16":
        lhs_overflow = np.array([32767, -32768, 30000, -30000], dtype=np.int16)
        rhs_overflow = np.array([1, 1, 10000, -10000], dtype=np.int16)
        actual_overflow = op(_nd(lhs_overflow), _nd(rhs_overflow)).data
        expected_overflow = np_op(lhs_overflow, rhs_overflow)
        assert np.array_equal(actual_overflow, expected_overflow)


@pytest.mark.parametrize("dtype_name", ("float64", "float32", "bfloat16", "int16"))
def test_sqrt_rsqrt_match_numpy_with_edge_values(patched_scope, dtype_name):
    """sqrt/rsqrt should match numpy for regular and edge-value inputs."""
    del patched_scope
    if dtype_name == "int16":
        values = np.array([0, 1, 4, 9, 256, 1024, 32767], dtype=np.int16)
    else:
        values = np.asarray(
            [0.0, 1.0, 4.0, 9.0, 256.0, 1e-12, 1e12], dtype=STORAGE_DTYPES[dtype_name]
        )
    values_nd = _nd(values)
    sqrt_actual = nl.sqrt(values_nd).data
    sqrt_expected = np.sqrt(values)
    _assert_tensor_equal(sqrt_actual, sqrt_expected)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rsqrt_actual = nl.rsqrt(values_nd).data
        rsqrt_expected = 1.0 / np.sqrt(values)
    _assert_tensor_equal(rsqrt_actual, rsqrt_expected)


def test_dma_copy_buffer_dtype_rmw_matrix(patched_scope):
    """dma_copy should match documented buffer/dtype matrix and rmw behavior."""
    del patched_scope
    shape = (19, 23)
    dtype_pairs = (
        ("float32", "float32"),
        ("bfloat16", "bfloat16"),
        ("bfloat16", "float32"),
    )
    for dst_buffer in ("hbm", "sbuf"):
        for src_buffer in ("hbm", "sbuf"):
            for dst_dtype_name, src_dtype_name in dtype_pairs:
                src_np = _matrix(shape, "arange", src_dtype_name, 61)
                src = b2.NDArray(
                    value=src_np,
                    dtype=_dtype_token(src_dtype_name),
                    buffer=src_buffer,
                )
                dst = b2.NDArray(
                    shape=shape,
                    dtype=_dtype_token(dst_dtype_name),
                    buffer=dst_buffer,
                )

                nisa.dma_copy(dst, src, dst_rmw_op=None)
                expected = np.asarray(src.data, dtype=dst.data.dtype)
                _assert_tensor_equal(dst.data, expected)

                dst_init = _matrix(shape, "linspace", dst_dtype_name, 79)
                dst.data[...] = np.asarray(dst_init, dtype=dst.data.dtype)
                nisa.dma_copy(dst, src, dst_rmw_op=nl.add)
                expected_rmw = np.asarray(
                    np.add(np.asarray(dst_init), np.asarray(src.data)),
                    dtype=dst.data.dtype,
                )
                _assert_tensor_equal(dst.data, expected_rmw)

    with pytest.raises(ValueError, match="dma_copy only supports hbm/sbuf"):
        nisa.dma_copy(
            b2.NDArray(shape=(4, 4), dtype=_dtype_token("float32"), buffer="sbuf"),
            b2.NDArray(shape=(4, 4), dtype=_dtype_token("float32"), buffer="psum"),
        )


def test_ndarray_has_no_broadcast_to_method():
    """NDArray should not expose broadcast_to helper."""
    assert not hasattr(b2.NDArray(value=np.array([1])), "broadcast_to")


def test_tensor_binary_operators_raise(patched_scope):
    """Binary operators on tensor objects should raise compiler-like errors."""
    del patched_scope
    tensor = nl.zeros((2, 2), dtype=nl.float32)
    with pytest.raises(
        TypeError,
        match="binary operators on tensors not supported. Use nki.isa directly.",
    ):
        tensor *= 2


def test_tensor_tensor_rejects_scalar_vector_broadcast(patched_scope):
    """tensor_tensor should reject scalar/vector broadcasts."""
    del patched_scope
    dst = _zeros((2, 3))
    lhs = _nd(np.arange(6, dtype=np.float32).reshape(2, 3))
    rhs_vector = _nd(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(
        ValueError,
        match="tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar",
    ):
        nisa.tensor_tensor(dst, lhs, rhs_vector, nl.add)


def test_tensor_tensor_views_second_last_dims_to_dst(patched_scope):
    """tensor_tensor should reshape src free dims to match dst free dims."""
    del patched_scope
    dst = _zeros((2, 3, 2))
    lhs = _nd(np.arange(12, dtype=np.float32).reshape(2, 6))
    rhs = _nd(np.ones((2, 6), dtype=np.float32))
    nisa.tensor_tensor(dst, lhs, rhs, nl.add)
    expected = lhs.data.reshape(dst.shape) + rhs.data.reshape(dst.shape)
    assert np.array_equal(dst.data, expected)


def test_exponential_requires_nc_v4_or_newer(patched_scope):
    """exponential should reject simulated pre-v4 targets."""
    del patched_scope
    prev = b2.nki_builder.nc_version
    b2.nki_builder.nc_version = nisa.nc_version.gen3
    try:
        with pytest.raises(
            RuntimeError, match="exponential only supports neuron-core-v4 or newer"
        ):
            nisa.exponential(_zeros((1, 2)), _nd([[0.0, 1.0]]))
    finally:
        b2.nki_builder.nc_version = prev


def test_tensor_tensor_rejects_tensor_data2_argument(patched_scope):
    """tensor_tensor should reject tensor (non-access) data2 args."""
    del patched_scope
    dst = _zeros((2, 2))
    data1 = _nd(np.ones((2, 2), dtype=np.float32))
    data2 = nl.zeros((2, 2), dtype=nl.float32)
    with pytest.raises(
        TypeError,
        match="failed to resolve an argument 'data2', expecting tensor access, got 'tensor'",
    ):
        nisa.tensor_tensor(dst=dst, data1=data1, data2=data2, op=nl.add)


def test_tensor_tensor_rejects_tensor_dst_argument(patched_scope):
    """tensor_tensor should reject tensor (non-access) dst args."""
    del patched_scope
    dst = nl.zeros((2, 2), dtype=nl.float32)
    data1 = _nd(np.ones((2, 2), dtype=np.float32))
    data2 = _nd(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(
        TypeError,
        match="failed to resolve an argument 'dst', expecting tensor access, got 'tensor'",
    ):
        nisa.tensor_tensor(dst=dst, data1=data1, data2=data2, op=nl.add)


def test_ndarray_mutation_is_not_supported(patched_scope):
    """In-place mutation on tensor objects should raise."""
    del patched_scope
    tensor = nl.ndarray((2, 2), dtype=nl.float32, buffer=nl.sbuf)
    with pytest.raises(RuntimeError, match="mutation not supported"):
        tensor[0, 0] = 1


def test_ndarray_is_zero_initialized(patched_scope):
    """ndarray values should be initialized to zero."""
    del patched_scope
    src = nl.ndarray((2, 2), dtype=nl.float32, buffer=nl.sbuf)
    assert np.array_equal(src.data, np.zeros((2, 2), dtype=src.data.dtype))


def test_ndarray_int_index_keeps_singleton_dim():
    """Integer indexing should keep a singleton dimension."""
    tensor = _nd(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    slice0 = tensor[0]
    assert slice0.shape == (1, 3, 4)


def test_dma_copy_ap_mismatch_error_message(patched_scope):
    """AP mismatch copies should raise the expected compiler-like error."""
    del patched_scope
    rows, cols = 2, 8
    src = _nd(np.arange(rows * cols, dtype=np.float32).reshape(rows, cols))
    dst = _nd(np.zeros((rows, cols // 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="Expect AP same number of elements"):
        nisa.dma_copy(dst=dst[:, :, 0], src=src[:, ::2])


def test_tensor_subscript_is_not_supported(patched_scope):
    """Subscript on tensor objects should raise."""
    del patched_scope
    tensor = nl.zeros((2, 2), dtype=nl.float32)
    with pytest.raises(TypeError, match="subscript not supported, for 'tensor'"):
        _ = tensor[:, 0]


def test_tensor_negation_is_not_supported(patched_scope):
    """Unary negation on tensor objects should raise."""
    del patched_scope
    row_max = nl.ndarray((2, 1), dtype=nl.float32, buffer=nl.sbuf)
    with pytest.raises(TypeError, match="cannot negate values of this type"):
        _ = -row_max
