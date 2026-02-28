import inspect

import numpy as np
import pytest

try:
    from triton_viz.core.patch import _LangPatchScope
    import nki.isa as nisa
    import nki.language as nl
    import triton_viz.core.nki_beta2 as b2
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki

_MISSING = object()


class _PatchScope:
    """Track and restore monkeypatched attributes."""

    def __init__(self):
        self._changes = []

    def set_attr(self, obj, name, value):
        """Patch one attribute and remember prior value."""
        original = getattr(obj, name, _MISSING)
        self._changes.append((obj, name, original))
        setattr(obj, name, value)

    def restore(self):
        """Restore attributes patched through this scope."""
        while self._changes:
            obj, name, original = self._changes.pop()
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)


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


def test_non_numpy_logical_dtypes_have_storage():
    """Logical low-precision dtypes should be backed by float64 storage."""
    logical = np.array([[0.1, 0.9, 1.24], [1.51, 2.9, -5.2]])
    for dtype_name in (
        "bfloat16",
        "tfloat32",
        "float8_e4m3",
        "float8_e5m2",
        "float8_e4m3fn",
        "float8_e5m2fn",
        "float4_e2m1fn",
    ):
        dtype = getattr(nl, dtype_name, None)
        created = b2.NDArray(
            shape=logical.shape, dtype=dtype, value=logical, buffer="sbuf"
        )
        assert created.data.dtype == np.float64
        dst = b2.NDArray(shape=logical.shape, dtype=dtype, buffer="sbuf")
        b2.dma_copy(dst=dst, src=_nd(logical))
        assert dst.data.dtype == np.float64


def test_ndarray_has_no_name_attr():
    """Beta2 NDArray should not expose a name attribute."""
    assert not hasattr(b2.NDArray(value=np.array([1])), "name")


def test_patch_surface_and_signatures(patched_scope):
    """Patched surface should match the new beta2 dialect. Ignoring function signatures of NKIOps like nl.add since they are trivial (only one/two tensor arg)."""
    del patched_scope
    nl_dtype_constants = (
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
    )
    nl_names = (
        "ndarray",
        "zeros",
        "shared_constant",
        "shared_identity_matrix",
        "affine_range",
        "ds",
        "sequential_range",
        "static_range",
        "device_print",
        "num_programs",
        "program_id",
        "program_ndim",
    )
    nisa_names = (
        "activation",
        "activation_reduce",
        "affine_select",
        "bn_aggr",
        "bn_stats",
        "core_barrier",
        "dma_compute",
        "dma_copy",
        "dma_transpose",
        "dropout",
        "iota",
        "local_gather",
        "max8",
        "memset",
        "nc_find_index8",
        "nc_match_replace8",
        "nc_matmul",
        "nc_matmul_mx",
        "nc_n_gather",
        "nc_stream_shuffle",
        "nc_transpose",
        "nonzero_with_count",
        "quantize_mx",
        "rand2",
        "rand_get_state",
        "rand_set_state",
        "range_select",
        "reciprocal",
        "register_alloc",
        "register_load",
        "register_move",
        "register_store",
        "rng",
        "scalar_tensor_tensor",
        "select_reduce",
        "sendrecv",
        "sequence_bounds",
        "set_rng_seed",
        "tensor_copy",
        "tensor_copy_predicated",
        "tensor_partition_reduce",
        "tensor_reduce",
        "tensor_scalar",
        "tensor_scalar_cumulative",
        "tensor_scalar_reduce",
        "tensor_tensor",
        "tensor_tensor_scan",
    )

    for name in nl_names:
        assert hasattr(nl, name)
    for name in nisa_names:
        assert hasattr(nisa, name)
    for name in nl_dtype_constants:
        assert hasattr(nl, name)

    assert nl.bool_ == "bool"
    assert nl.int8 == "int8"
    assert nl.int16 == "int16"
    assert nl.int32 == "int32"
    assert nl.uint8 == "uint8"
    assert nl.uint16 == "uint16"
    assert nl.uint32 == "uint32"
    assert nl.float16 == "float16"
    assert nl.float32 == "float32"
    assert nl.bfloat16 == "bfloat16"
    assert nl.tfloat32 == "tfloat32"
    assert nl.float8_e4m3 == "float8_e4m3"
    assert nl.float8_e5m2 == "float8_e5m2"
    assert nl.float8_e4m3fn == "float8_e4m3fn"

    for name in nl_names:
        assert str(inspect.signature(getattr(nl, name))) == str(
            inspect.signature(getattr(b2, name))
        )

    for name in nisa_names:
        assert str(inspect.signature(getattr(nisa, name))) == str(
            inspect.signature(getattr(b2, name))
        )


def test_language_helpers_and_ops(patched_scope):
    """Language helpers should run with expected semantics."""
    del patched_scope
    arr = nl.ndarray((2, 2), dtype=float, buffer=nl.sbuf)
    z = nl.zeros((2, 2), dtype=float, buffer=nl.sbuf)
    assert arr.shape == (2, 2)
    assert np.array_equal(z.data, np.zeros((2, 2)))

    const = nl.shared_constant(np.array([[1, 2], [3, 4]], dtype=np.int32), dtype=float)
    ident = nl.shared_identity_matrix(4, dtype=nl.uint8)
    assert const.data.shape == (2, 2)
    assert ident.data.shape == (4, 4)
    assert ident.data.dtype == np.uint8

    b2.nki_builder.grid_dims = (2, 3, 4)
    b2.nki_builder.grid_idx = [1, 2, 3]
    assert nl.program_id(0) == 1
    assert nl.program_ndim() == 3
    assert nl.num_programs() == 24
    assert nl.num_programs((0, 2)) == (2, 4)

    assert list(nl.affine_range(1, 6, 2)) == [1, 3, 5]
    assert list(nl.sequential_range(1, 6, 2)) == [1, 3, 5]
    assert list(nl.static_range(1, 6, 2)) == [1, 3, 5]
    assert nl.ds(2, 3) == slice(2, 5, None)
    assert nl.tile_size.pmax == 128

    x = _nd([[1, 2], [3, 4]], dtype=float)
    y = _nd([[4, 3], [2, 1]], dtype=float)
    xi = _nd([[1, 2], [3, 4]], dtype=np.int32)
    yi = _nd([[4, 3], [2, 1]], dtype=np.int32)

    assert np.array_equal(nl.add(x, y).data, np.array([[5, 5], [5, 5]]))
    assert np.array_equal(nl.subtract(x, y).data, np.array([[-3, -1], [1, 3]]))
    assert np.array_equal(nl.multiply(x, y).data, np.array([[4, 6], [6, 4]]))
    assert np.allclose(nl.divide(x, 2.0).data, np.array([[0.5, 1.0], [1.5, 2.0]]))
    assert np.array_equal(nl.maximum(x, y).data, np.array([[4, 3], [3, 4]]))
    assert np.array_equal(nl.minimum(x, y).data, np.array([[1, 2], [2, 1]]))
    assert np.array_equal(nl.power(x, 2).data, np.array([[1, 4], [9, 16]]))
    assert np.array_equal(nl.abs(_nd([[-1, 2]])).data, np.array([[1, 2]]))
    assert np.allclose(nl.reciprocal(_nd([[2.0, 4.0]])).data, np.array([[0.5, 0.25]]))
    assert np.allclose(nl.rsqrt(_nd([[4.0, 16.0]])).data, np.array([[0.5, 0.25]]))

    assert np.array_equal(
        nl.equal(xi, yi).data, np.array([[False, False], [False, False]])
    )
    assert np.array_equal(
        nl.not_equal(xi, yi).data, np.array([[True, True], [True, True]])
    )
    assert np.array_equal(
        nl.less(xi, yi).data, np.array([[True, True], [False, False]])
    )
    assert np.array_equal(
        nl.less_equal(xi, yi).data, np.array([[True, True], [False, False]])
    )
    assert np.array_equal(
        nl.greater(xi, yi).data, np.array([[False, False], [True, True]])
    )
    assert np.array_equal(
        nl.greater_equal(xi, yi).data, np.array([[False, False], [True, True]])
    )
    assert np.array_equal(
        nl.logical_not(_nd([[1, 0], [0, 1]], dtype=np.int32)).data,
        np.array([[False, True], [True, False]]),
    )
    assert np.array_equal(
        nl.logical_and(
            _nd([[1, 0], [1, 0]], dtype=np.int32), _nd([[1, 1], [0, 0]], dtype=np.int32)
        ).data,
        np.array([[True, False], [False, False]]),
    )
    assert np.array_equal(
        nl.logical_or(
            _nd([[1, 0], [1, 0]], dtype=np.int32), _nd([[1, 1], [0, 0]], dtype=np.int32)
        ).data,
        np.array([[True, True], [True, False]]),
    )
    assert np.array_equal(
        nl.logical_xor(
            _nd([[1, 0], [1, 0]], dtype=np.int32), _nd([[1, 1], [0, 0]], dtype=np.int32)
        ).data,
        np.array([[False, True], [True, False]]),
    )

    assert np.array_equal(nl.bitwise_and(xi, yi).data, np.bitwise_and(xi.data, yi.data))
    assert np.array_equal(nl.bitwise_or(xi, yi).data, np.bitwise_or(xi.data, yi.data))
    assert np.array_equal(nl.bitwise_xor(xi, yi).data, np.bitwise_xor(xi.data, yi.data))
    assert np.array_equal(nl.invert(xi).data, np.invert(xi.data))
    assert np.array_equal(nl.left_shift(xi, 1).data, xi.data << 1)
    assert np.array_equal(nl.right_shift(xi, 1).data, xi.data >> 1)

    gelu_x = _nd([[-1.0, 0.0, 1.0]])
    assert nl.gelu_apprx_sigmoid(gelu_x).data.shape == gelu_x.shape
    assert nl.gelu_apprx_sigmoid_dx(gelu_x).data.shape == gelu_x.shape
    nl.device_print("nl_device_print", gelu_x)


def _core_inputs():
    """Build canonical tensors used by core dataflow tests."""
    a = _nd(np.arange(8, dtype=float).reshape(2, 4), buffer="hbm")
    b = _nd(np.arange(8, dtype=float).reshape(2, 4) + 10, buffer="hbm")
    out = _zeros((2, 4), buffer="sbuf")
    return a, b, out


def _act_input():
    """Build canonical activation input tensor."""
    return _nd(np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -1.0]]))


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


def test_dma_compute(patched_scope):
    """dma_compute should combine scaled sources with reduce_op."""
    del patched_scope
    a, b, _ = _core_inputs()
    comp = _zeros((2, 4), buffer="sbuf")
    nisa.dma_compute(comp, [a, b], [1.0, 1.0], nl.add)
    assert np.array_equal(comp.data, a.data + b.data)


def test_tensor_copy(patched_scope):
    """tensor_copy should copy tensors elementwise."""
    del patched_scope
    a, b, out = _core_inputs()
    comp = _zeros((2, 4), buffer="sbuf")
    nisa.dma_compute(comp, [a, b], [1.0, 1.0], nl.add)
    nisa.tensor_copy(out, comp)
    assert np.array_equal(out.data, comp.data)


def test_tensor_tensor(patched_scope):
    """tensor_tensor should apply the provided binary op."""
    del patched_scope
    a, b, out = _core_inputs()
    nisa.tensor_tensor(out, a, b, nl.subtract)
    assert np.array_equal(out.data, a.data - b.data)


def test_scalar_tensor_tensor(patched_scope):
    """scalar_tensor_tensor should combine scalar/tensor operands in order."""
    del patched_scope
    a, b, out = _core_inputs()
    nisa.scalar_tensor_tensor(out, a, nl.add, 2.0, nl.multiply, b)
    assert np.array_equal(out.data, (a.data + 2.0) * b.data)


def test_tensor_scalar(patched_scope):
    """tensor_scalar should support chained ops with operand1/op1."""
    del patched_scope
    a, _, out = _core_inputs()
    nisa.tensor_scalar(out, a, nl.multiply, 2.0, op1=nl.add, operand1=1.0)
    assert np.array_equal(out.data, a.data * 2.0 + 1.0)


def test_tensor_tensor_scan_shape(patched_scope):
    """tensor_tensor_scan should produce the requested destination shape."""
    del patched_scope
    tscan0 = _nd(np.arange(8, dtype=float).reshape(2, 4))
    tscan1 = _nd(np.ones((2, 4)))
    dst = _zeros((2, 4))
    nisa.tensor_tensor_scan(dst, tscan0, tscan1, _nd(np.zeros((2, 1))), nl.add, nl.add)
    assert dst.data.shape == (2, 4)


def test_tensor_scalar_cumulative_shape(patched_scope):
    """tensor_scalar_cumulative should preserve destination shape."""
    del patched_scope
    src = _nd(np.arange(8, dtype=float).reshape(2, 4))
    dst = _zeros((2, 4))
    nisa.tensor_scalar_cumulative(dst, src, nl.add, nl.add, 1.0)
    assert dst.data.shape == (2, 4)


def test_tensor_scalar_reduce_shape(patched_scope):
    """tensor_scalar_reduce should write partition reductions."""
    del patched_scope
    a, _, out = _core_inputs()
    red_dst = _zeros((2, 1))
    nisa.tensor_scalar_reduce(out, a, nl.add, 1.0, nl.add, red_dst)
    assert red_dst.data.shape == (2, 1)


def test_tensor_reduce_shape(patched_scope):
    """tensor_reduce should reduce across the selected axis."""
    del patched_scope
    _, _, out = _core_inputs()
    nisa.memset(out, -3.0)
    red_dst = _zeros((2, 1))
    nisa.tensor_reduce(red_dst, nl.maximum, out, axis=1, keepdims=True)
    assert red_dst.data.shape == (2, 1)


def test_tensor_partition_reduce_shape(patched_scope):
    """tensor_partition_reduce should collapse free axes."""
    del patched_scope
    _, _, out = _core_inputs()
    part_dst = _zeros((1, 4))
    nisa.tensor_partition_reduce(part_dst, nl.add, out)
    assert part_dst.data.shape == (1, 4)


def test_tensor_copy_predicated(patched_scope):
    """tensor_copy_predicated should only write selected elements."""
    del patched_scope
    a, _, out = _core_inputs()
    nisa.memset(out, -1.0)
    pred = _nd(np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8))
    nisa.tensor_copy_predicated(out, a, pred)
    assert out.data[0, 0] == a.data[0, 0]
    assert out.data[0, 1] == -1.0


def test_reciprocal(patched_scope):
    """reciprocal should compute elementwise multiplicative inverse."""
    del patched_scope
    rec = _zeros((2, 4))
    src = _nd(np.array([[1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]]))
    nisa.reciprocal(rec, src)
    assert np.allclose(rec.data[0, :2], np.array([1.0, 0.5]))


def test_memset(patched_scope):
    """memset should fill destination tensor with scalar value."""
    del patched_scope
    _, _, out = _core_inputs()
    nisa.memset(out, -3.0)
    assert np.array_equal(out.data, np.full((2, 4), -3.0))


def test_nc_transpose(patched_scope):
    """nc_transpose should transpose partition and flattened free axes."""
    del patched_scope
    src = _nd(np.arange(24, dtype=float).reshape(2, 3, 4))
    dst = _zeros((12, 2), buffer="psum")
    nisa.nc_transpose(dst, src)
    assert np.array_equal(dst.data, src.data.reshape(2, -1).T)


def test_dma_transpose(patched_scope):
    """dma_transpose should apply the explicit axes permutation."""
    del patched_scope
    src = _nd(np.arange(6, dtype=float).reshape(2, 3))
    dst = _zeros((3, 2), buffer="hbm")
    nisa.dma_transpose(dst, src, axes=(1, 0))
    assert np.array_equal(dst.data, src.data.T)


def test_sendrecv(patched_scope):
    """sendrecv should copy sent values into receive buffer."""
    del patched_scope
    sent = _nd(np.arange(8, dtype=float).reshape(2, 4))
    recv = _zeros((2, 4))
    nisa.sendrecv(sent, recv, send_to_rank=0, recv_from_rank=0, pipe_id=0)
    assert np.array_equal(recv.data, sent.data)


def test_iota(patched_scope):
    """iota should materialize indices from the affine pattern."""
    del patched_scope
    dst = _zeros((4, 8), dtype=np.int32)
    nisa.iota(dst, [[8, 1], [1, 8]], offset=3, channel_multiplier=2)
    assert np.array_equal(dst.data[:, 0], np.array([3, 5, 7, 9], dtype=np.int32))


def test_affine_select(patched_scope):
    """affine_select should route values between true and false branches."""
    del patched_scope
    true_tile = _nd(np.arange(32, dtype=float).reshape(4, 8))
    dst = _zeros((4, 8))
    nisa.affine_select(
        dst,
        [[1, 8]],
        offset=-2,
        channel_multiplier=1,
        on_true_tile=true_tile,
        on_false_value=-1.0,
        cmp_op=nl.greater_equal,
    )
    assert np.any(dst.data == -1.0)


def test_nc_n_gather(patched_scope):
    """nc_n_gather should gather values by per-partition indices."""
    del patched_scope
    data = _nd(np.arange(20, dtype=float).reshape(2, 10))
    idx = _nd(np.array([[0, 3, 9], [1, 4, 8]], dtype=np.uint32))
    dst = _zeros((2, 3))
    nisa.nc_n_gather(dst, data, idx)
    assert np.array_equal(dst.data[0], np.array([0, 3, 9]))


def test_local_gather(patched_scope):
    """local_gather should read indexed values from local source."""
    del patched_scope
    src = _nd(np.arange(64, dtype=float).reshape(16, 4))
    idx = _nd(np.zeros((16, 1), dtype=np.uint16))
    dst = _zeros((16, 1))
    nisa.local_gather(dst, src, idx, num_elem_per_idx=1)
    assert np.array_equal(dst.data, np.zeros((16, 1)))


def test_nc_stream_shuffle(patched_scope):
    """nc_stream_shuffle should reorder rows by provided core map."""
    del patched_scope
    src = _nd(np.arange(32, dtype=float).reshape(32, 1))
    dst = _zeros((32, 1))
    nisa.nc_stream_shuffle(dst, src, list(reversed(range(32))))
    assert np.array_equal(dst.data[:, 0], src.data[::-1, 0])


def test_nonzero_with_count(patched_scope):
    """nonzero_with_count should emit nonzero indices and count."""
    del patched_scope
    src = _zeros((16, 4), dtype=np.int32)
    src.data[0] = np.array([0, 1, 0, 2], dtype=np.int32)
    dst = _nd(np.full((16, 5), -99, dtype=np.int32))
    nisa.nonzero_with_count(dst, src, index_offset=10, padding_val=-1)
    assert np.array_equal(dst.data[0], np.array([11, 13, -1, -1, 2], dtype=np.int32))
    assert np.array_equal(dst.data[1], np.full(5, -99, dtype=np.int32))


def test_sequence_bounds(patched_scope):
    """sequence_bounds should return starts and ends for segment ids."""
    del patched_scope
    seg = _nd(np.array([[1, 1, 2, 2, 0]], dtype=np.int32))
    dst = _zeros((1, 2, 5), dtype=np.int32)
    nisa.sequence_bounds(dst, seg)
    assert np.array_equal(dst.data[0, 0], np.array([0, 0, 2, 2, 5], dtype=np.int32))
    assert np.array_equal(dst.data[0, 1], np.array([1, 1, 3, 3, -1], dtype=np.int32))


def test_nc_matmul(patched_scope):
    """nc_matmul should match transposed stationary matmul semantics."""
    del patched_scope
    stationary = _nd(np.arange(12, dtype=float).reshape(4, 3))
    moving = _nd(np.arange(8, dtype=float).reshape(4, 2))
    dst = _zeros((3, 2), buffer="psum")
    nisa.nc_matmul(dst, stationary, moving)
    assert np.allclose(dst.data, stationary.data.T @ moving.data)


def test_quantize_mx(patched_scope):
    """quantize_mx should emit a positive scaling factor."""
    del patched_scope
    src = _nd(np.array([[1.5, -2.0], [3.0, 4.0]]))
    dst = _zeros((2, 2), dtype=np.int8)
    scale = _zeros((1,))
    nisa.quantize_mx(dst, src, scale)
    assert float(scale.data[0]) > 0


def test_nc_matmul_mx(patched_scope):
    """nc_matmul_mx should match dequantize-then-matmul reference."""
    del patched_scope
    lhs = _zeros((2, 2), dtype=np.int8)
    lhs_scale = _zeros((1,))
    rhs = _zeros((2, 2), dtype=np.int8)
    rhs_scale = _zeros((1,))
    nisa.quantize_mx(lhs, _nd(np.array([[1.5, -2.0], [3.0, 4.0]])), lhs_scale)
    nisa.quantize_mx(rhs, _nd(np.array([[2.0, 1.0], [0.0, -1.0]])), rhs_scale)
    dst = _zeros((2, 2), buffer="psum")
    nisa.nc_matmul_mx(dst, lhs, rhs, lhs_scale, rhs_scale)
    expected = (lhs.data * lhs_scale.data[0]).T @ (rhs.data * rhs_scale.data[0])
    assert np.allclose(dst.data, expected)


def test_max8(patched_scope):
    """max8 should sort values in descending order."""
    del patched_scope
    src = _nd(np.array([[5, 1, 8, 3, 7, 2, 6, 4]], dtype=float))
    dst = _zeros((1, 8))
    nisa.max8(dst, src)
    assert np.array_equal(dst.data[0], np.array([8, 7, 6, 5, 4, 3, 2, 1]))


def test_nc_find_index8(patched_scope):
    """nc_find_index8 should return source indices for each target value."""
    del patched_scope
    src = _nd(np.array([[5, 1, 8, 3, 7, 2, 6, 4]], dtype=float))
    vals = _nd(np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float))
    dst = _zeros((1, 8), dtype=np.uint32)
    nisa.nc_find_index8(dst, src, vals)
    assert np.array_equal(dst.data[0], np.array([2, 4, 6, 0, 7, 3, 5, 1], np.uint32))


def test_nc_match_replace8(patched_scope):
    """nc_match_replace8 should replace matched values and emit indices."""
    del patched_scope
    src = _nd(np.array([[5, 1, 8, 3, 7, 2, 6, 4]], dtype=float))
    vals = _nd(np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float))
    dst = _zeros((1, 8))
    idx = _zeros((1, 8), dtype=np.int32)
    nisa.nc_match_replace8(dst, src, vals, imm=-1.0, dst_idx=idx)
    assert np.all(idx.data[0] >= 0)
    assert np.count_nonzero(dst.data[0] == -1.0) == 8


def test_activation(patched_scope):
    """activation should write outputs and reduce result tensors."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 3))
    red = _zeros((2, 1))
    nisa.activation(
        dst,
        nl.reciprocal,
        src,
        bias=1.0,
        scale=1.0,
        reduce_op=nl.add,
        reduce_res=red,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    assert dst.data.shape == (2, 3)
    assert red.data.shape == (2, 1)


def test_activation_reduce(patched_scope):
    """activation_reduce should combine activation and reduction in one call."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 3))
    red = _zeros((2, 1))
    nisa.activation_reduce(dst, nl.reciprocal, src, nl.add, red, bias=0.0, scale=1.0)
    assert red.data.shape == (2, 1)


def test_range_select(patched_scope):
    """range_select should write reduced values when reduce args are set."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 3))
    red = _zeros((2, 1))
    b0 = _nd(np.array([[0.0], [0.0]]))
    b1 = _nd(np.array([[2.0], [2.0]]))
    nisa.range_select(
        dst,
        src,
        nl.greater_equal,
        nl.less,
        b0,
        b1,
        reduce_res=red,
        reduce_op=nl.maximum,
        range_start=0.0,
    )
    assert red.data.shape == (2, 1)


def test_select_reduce(patched_scope):
    """select_reduce should fill false predicates with immediate value."""
    del patched_scope
    src = _act_input()
    pred = _nd(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8))
    dst = _zeros((2, 3))
    red = _zeros((2, 1))
    nisa.select_reduce(
        dst,
        pred,
        src,
        -5.0,
        reduce_res=red,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    assert dst.data[0, 1] == -5.0


def test_dropout(patched_scope):
    """dropout should preserve data at p=0 and zero it at p=1."""
    del patched_scope
    src = _act_input()
    keep = _zeros((2, 3))
    drop = _zeros((2, 3))
    nisa.dropout(keep, src, 0.0)
    nisa.dropout(drop, src, 1.0)
    assert np.array_equal(keep.data, src.data)
    assert np.array_equal(drop.data, np.zeros_like(src.data))


def test_rand_set_and_get_state(patched_scope):
    """rand_set_state and rand_get_state should round-trip seeds."""
    del patched_scope
    b2.nki_builder.rng_states = {}
    b2.nki_builder.rng_generators = {}
    seeds = _nd(np.arange(6, dtype=np.uint32).reshape(1, 6))
    nisa.rand_set_state(seeds, engine=nisa.engine.unknown)
    dst = _zeros((1, 6), dtype=np.uint32)
    nisa.rand_get_state(dst, engine=nisa.engine.unknown)
    assert np.array_equal(dst.data, seeds.data)


def test_rng(patched_scope):
    """rng should write uint32 random words."""
    del patched_scope
    dst = _zeros((2, 4), dtype=np.uint32)
    nisa.rng(dst, engine=nisa.engine.unknown)
    assert dst.data.dtype == np.uint32


def test_rand2(patched_scope):
    """rand2 should emit values inside the requested interval."""
    del patched_scope
    dst = _zeros((2, 4))
    nisa.rand2(dst, 1.0, 3.0)
    assert np.all(dst.data >= 1.0)
    assert np.all(dst.data <= 3.0)


def test_set_rng_seed(patched_scope):
    """set_rng_seed should accept scalar seed tensors."""
    del patched_scope
    seed = _nd(np.array([[123]], dtype=np.uint32))
    nisa.set_rng_seed(seed)


def test_bn_stats(patched_scope):
    """bn_stats should produce expected tuple layout shape."""
    del patched_scope
    src = _act_input()
    dst = _zeros((2, 6))
    nisa.bn_stats(dst, src)
    assert dst.data.shape == (2, 6)


def test_bn_aggr(patched_scope):
    """bn_aggr should aggregate stats tuples to mean and variance."""
    del patched_scope
    src = _act_input()
    stats = _zeros((2, 6))
    nisa.bn_stats(stats, src)
    dst = _zeros((2, 2))
    nisa.bn_aggr(dst, stats)
    assert dst.data.shape == (2, 2)


def test_core_barrier(patched_scope):
    """core_barrier should be a no-op in interpreter mode."""
    del patched_scope
    src = _act_input()
    assert nisa.core_barrier(src, cores=(0, 1), engine=nisa.engine.unknown) is None


def test_register_alloc_move_load_store(patched_scope):
    """register helpers should allocate, load, move, and store values."""
    del patched_scope
    reg = nisa.register_alloc()
    reg = nisa.register_move(reg, 5)
    assert reg == 5
    reg_src = _nd(np.array([[9]], dtype=np.int32))
    reg = nisa.register_load(reg, reg_src)
    reg_dst = _zeros((1, 1), dtype=np.int32)
    nisa.register_store(reg_dst, reg)
    assert int(reg_dst.data[0, 0]) == 9


def test_interpreted_function_execution(patched_scope):
    """NKIInterpretedFunction should execute kernels over launch grids."""
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

    fn = b2.NKIInterpretedFunction(copy_kernel)
    src = np.arange(16, dtype=float).reshape(4, 4)
    dst = np.empty_like(src)
    fn.run(src, dst, grid=(1,), client_manager=_ClientManager())
    assert np.array_equal(dst, src)


def test_ap_empty_pattern_raises():
    """ap should reject empty patterns."""
    with pytest.raises(ValueError, match="pattern must not be empty"):
        _nd(np.arange(4)).ap()


def test_ap_pair_arity_raises():
    """ap should reject pattern entries that are not [step, count]."""
    with pytest.raises(ValueError, match="pattern pair must have two values"):
        _nd(np.arange(4)).ap([1, 2, 3], offset=0)


def test_ap_unsupported_kwargs_raises():
    """ap should reject unsupported keyword arguments."""
    with pytest.raises(TypeError, match="Unsupported kwargs for ap"):
        _nd(np.arange(4)).ap([1, 4], offset=0, bogus_kw=1)


def test_ap_vector_offset_length_mismatch_raises():
    """ap should validate vector_offset length against indirect dimension."""
    with pytest.raises(ValueError, match="vector_offset length"):
        _nd(np.arange(8)).ap([1, 4], vector_offset=np.array([0, 1, 2]), indirect_dim=0)


def test_shared_constant_requires_dtype(patched_scope):
    """shared_constant should require an explicit dtype argument."""
    del patched_scope
    with pytest.raises(ValueError, match="dtype must be specified"):
        nl.shared_constant(np.array([1, 2, 3]))


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


def test_bn_aggr_non_triplet_input_raises(patched_scope):
    """bn_aggr should reject widths that are not multiples of three."""
    del patched_scope
    data = _nd(np.array([[1.0, 2.0, 3.0, 4.0]]))
    dst = _zeros((1, 2))
    with pytest.raises(ValueError, match="multiple of 3"):
        nisa.bn_aggr(dst, data)


def test_interpreted_function_invalid_grid_type_raises(patched_scope):
    """NKIInterpretedFunction should validate grid type."""
    del patched_scope

    def kernel():
        """No-op kernel used to validate grid argument checking."""
        return None

    fn = b2.NKIInterpretedFunction(kernel)
    with pytest.raises(TypeError, match="grid must be an int or a sequence of ints"):
        fn.run(grid=object())
