import inspect

import numpy as np
import pytest

try:
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
    scope = _PatchScope()
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


def test_core_dataflow_ops(patched_scope):
    """Core movement and tensor math ops should execute correctly."""
    del patched_scope
    a = _nd(np.arange(8, dtype=float).reshape(2, 4), buffer="hbm")
    b = _nd(np.arange(8, dtype=float).reshape(2, 4) + 10, buffer="hbm")
    out = _zeros((2, 4), buffer="sbuf")
    nisa.dma_copy(out, a)
    assert np.array_equal(out.data, a.data)
    nisa.dma_copy(out, b, dst_rmw_op=nl.add)
    assert np.array_equal(out.data, a.data + b.data)

    comp = _zeros((2, 4), buffer="sbuf")
    nisa.dma_compute(comp, [a, b], [1.0, 1.0], nl.add)
    assert np.array_equal(comp.data, a.data + b.data)

    nisa.tensor_copy(out, comp)
    assert np.array_equal(out.data, comp.data)
    nisa.tensor_tensor(out, a, b, nl.subtract)
    assert np.array_equal(out.data, a.data - b.data)
    nisa.scalar_tensor_tensor(out, a, nl.add, 2.0, nl.multiply, b)
    assert np.array_equal(out.data, (a.data + 2.0) * b.data)
    nisa.tensor_scalar(out, a, nl.multiply, 2.0, op1=nl.add, operand1=1.0)
    assert np.array_equal(out.data, a.data * 2.0 + 1.0)

    tscan0 = _nd(np.arange(8, dtype=float).reshape(2, 4))
    tscan1 = _nd(np.ones((2, 4)))
    tscan_dst = _zeros((2, 4))
    nisa.tensor_tensor_scan(
        tscan_dst, tscan0, tscan1, _nd(np.zeros((2, 1))), nl.add, nl.add
    )
    assert tscan_dst.data.shape == (2, 4)

    cum_dst = _zeros((2, 4))
    nisa.tensor_scalar_cumulative(cum_dst, tscan0, nl.add, nl.add, 1.0)
    assert cum_dst.data.shape == (2, 4)

    red_dst = _zeros((2, 1))
    nisa.tensor_scalar_reduce(out, a, nl.add, 1.0, nl.add, red_dst)
    assert red_dst.data.shape == (2, 1)
    nisa.tensor_reduce(red_dst, nl.maximum, out, axis=1, keepdims=True)
    assert red_dst.data.shape == (2, 1)

    part_dst = _zeros((1, 4))
    nisa.tensor_partition_reduce(part_dst, nl.add, out)
    assert part_dst.data.shape == (1, 4)

    pred = _nd(np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8))
    nisa.tensor_copy_predicated(out, a, pred)
    assert out.data[0, 0] == a.data[0, 0]
    assert out.data[0, 1] != a.data[0, 1]

    rec = _zeros((2, 4))
    nisa.reciprocal(rec, _nd(np.array([[1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0]])))
    assert np.allclose(rec.data[0, :2], np.array([1.0, 0.5]))

    nisa.memset(out, -3.0)
    assert np.array_equal(out.data, np.full((2, 4), -3.0))

    t = _nd(np.arange(24, dtype=float).reshape(2, 3, 4))
    t_t = _zeros((12, 2), buffer="psum")
    nisa.nc_transpose(t_t, t)
    assert np.array_equal(t_t.data, t.data.reshape(2, -1).T)

    src_t = _nd(np.arange(6, dtype=float).reshape(2, 3))
    dst_t = _zeros((3, 2), buffer="hbm")
    nisa.dma_transpose(dst_t, src_t, axes=(1, 0))
    assert np.array_equal(dst_t.data, src_t.data.T)

    sent = _nd(np.arange(8, dtype=float).reshape(2, 4))
    recv = _zeros((2, 4))
    nisa.sendrecv(sent, recv, send_to_rank=0, recv_from_rank=0, pipe_id=0)
    assert np.array_equal(recv.data, sent.data)


def test_pattern_and_gather_ops(patched_scope):
    """Pattern generation and gather-like ops should execute."""
    del patched_scope
    iota_dst = _zeros((4, 8), dtype=np.int32)
    nisa.iota(iota_dst, [[8, 1], [1, 8]], offset=3, channel_multiplier=2)
    assert np.array_equal(iota_dst.data[:, 0], np.array([3, 5, 7, 9], dtype=np.int32))

    true_tile = _nd(np.arange(32, dtype=float).reshape(4, 8))
    sel_dst = _zeros((4, 8))
    nisa.affine_select(
        sel_dst,
        [[1, 8]],
        offset=-2,
        channel_multiplier=1,
        on_true_tile=true_tile,
        on_false_value=-1.0,
        cmp_op=nl.greater_equal,
    )
    assert np.any(sel_dst.data == -1.0)

    data = _nd(np.arange(20, dtype=float).reshape(2, 10))
    idx = _nd(np.array([[0, 3, 9], [1, 4, 8]], dtype=np.uint32))
    gathered = _zeros((2, 3))
    nisa.nc_n_gather(gathered, data, idx)
    assert np.array_equal(gathered.data[0], np.array([0, 3, 9]))

    lg_src = _nd(np.arange(64, dtype=float).reshape(16, 4))
    lg_idx = _nd(np.zeros((16, 1), dtype=np.uint16))
    lg_out = _zeros((16, 1))
    nisa.local_gather(lg_out, lg_src, lg_idx, num_elem_per_idx=1)
    assert np.array_equal(lg_out.data, np.zeros((16, 1)))

    shuffle_src = _nd(np.arange(32, dtype=float).reshape(32, 1))
    shuffle_dst = _zeros((32, 1))
    nisa.nc_stream_shuffle(shuffle_dst, shuffle_src, list(reversed(range(32))))
    assert np.array_equal(shuffle_dst.data[:, 0], shuffle_src.data[::-1, 0])

    nz_src = _zeros((16, 4), dtype=np.int32)
    nz_src.data[0] = np.array([0, 1, 0, 2], dtype=np.int32)
    nz_dst = _nd(np.full((16, 5), -99, dtype=np.int32))
    nisa.nonzero_with_count(nz_dst, nz_src, index_offset=10, padding_val=-1)
    assert np.array_equal(nz_dst.data[0], np.array([11, 13, -1, -1, 2], dtype=np.int32))
    assert np.array_equal(nz_dst.data[1], np.full(5, -99, dtype=np.int32))

    seg = _nd(np.array([[1, 1, 2, 2, 0]], dtype=np.int32))
    bounds = _zeros((1, 2, 5), dtype=np.int32)
    nisa.sequence_bounds(bounds, seg)
    assert np.array_equal(bounds.data[0, 0], np.array([0, 0, 2, 2, 5], dtype=np.int32))
    assert np.array_equal(bounds.data[0, 1], np.array([1, 1, 3, 3, -1], dtype=np.int32))


def test_matmul_quantization_and_topk_ops(patched_scope):
    """Matmul family and match/topk ops should run."""
    del patched_scope
    stationary = _nd(np.arange(12, dtype=float).reshape(4, 3))
    moving = _nd(np.arange(8, dtype=float).reshape(4, 2))
    mm_out = _zeros((3, 2), buffer="psum")
    nisa.nc_matmul(mm_out, stationary, moving)
    assert np.allclose(mm_out.data, stationary.data.T @ moving.data)

    src = _nd(np.array([[1.5, -2.0], [3.0, 4.0]]))
    q_dst = _zeros((2, 2), dtype=np.int8)
    q_scale = _zeros((1,))
    nisa.quantize_mx(q_dst, src, q_scale)
    assert float(q_scale.data[0]) > 0

    moving_q = _zeros((2, 2), dtype=np.int8)
    moving_scale = _zeros((1,))
    nisa.quantize_mx(
        moving_q,
        _nd(np.array([[2.0, 1.0], [0.0, -1.0]])),
        moving_scale,
    )
    mx_out = _zeros((2, 2), buffer="psum")
    nisa.nc_matmul_mx(mx_out, q_dst, moving_q, q_scale, moving_scale)
    expected = (q_dst.data * q_scale.data[0]).T @ (moving_q.data * moving_scale.data[0])
    assert np.allclose(mx_out.data, expected)

    top_src = _nd(np.array([[5, 1, 8, 3, 7, 2, 6, 4]], dtype=float))
    top_dst = _zeros((1, 8))
    nisa.max8(top_dst, top_src)
    assert np.array_equal(top_dst.data[0], np.array([8, 7, 6, 5, 4, 3, 2, 1]))

    vals = _nd(np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=float))
    idx_dst = _zeros((1, 8), dtype=np.uint32)
    nisa.nc_find_index8(idx_dst, top_src, vals)
    assert np.array_equal(
        idx_dst.data[0], np.array([2, 4, 6, 0, 7, 3, 5, 1], dtype=np.uint32)
    )

    replace_dst = _zeros((1, 8))
    replace_idx = _zeros((1, 8), dtype=np.int32)
    nisa.nc_match_replace8(replace_dst, top_src, vals, imm=-1.0, dst_idx=replace_idx)
    assert np.all(replace_idx.data[0] >= 0)
    assert np.count_nonzero(replace_dst.data[0] == -1.0) == 8


def test_reduction_activation_and_random_ops(patched_scope):
    """Reduction, activation, rng, and barrier ops should run."""
    del patched_scope
    act_in = _nd(np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -1.0]]))
    act_out = _zeros((2, 3))
    act_red = _zeros((2, 1))
    nisa.activation(
        act_out,
        nl.reciprocal,
        act_in,
        bias=1.0,
        scale=1.0,
        reduce_op=nl.add,
        reduce_res=act_red,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    assert act_out.data.shape == (2, 3)
    assert act_red.data.shape == (2, 1)

    nisa.activation_reduce(
        act_out, nl.reciprocal, act_in, nl.add, act_red, bias=0.0, scale=1.0
    )
    assert act_red.data.shape == (2, 1)

    rs_dst = _zeros((2, 3))
    rs_red = _zeros((2, 1))
    b0 = _nd(np.array([[0.0], [0.0]]))
    b1 = _nd(np.array([[2.0], [2.0]]))
    nisa.range_select(
        rs_dst,
        act_in,
        nl.greater_equal,
        nl.less,
        b0,
        b1,
        reduce_res=rs_red,
        reduce_op=nl.maximum,
        range_start=0.0,
    )
    assert rs_red.data.shape == (2, 1)

    sel_pred = _nd(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8))
    sel_dst = _zeros((2, 3))
    sel_red = _zeros((2, 1))
    nisa.select_reduce(
        sel_dst,
        sel_pred,
        act_in,
        -5.0,
        reduce_res=sel_red,
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )
    assert sel_dst.data[0, 1] == -5.0

    keep = _zeros((2, 3))
    drop = _zeros((2, 3))
    nisa.dropout(keep, act_in, 0.0)
    nisa.dropout(drop, act_in, 1.0)
    assert np.array_equal(keep.data, act_in.data)
    assert np.array_equal(drop.data, np.zeros_like(act_in.data))

    b2.nki_builder.rng_states = {}
    b2.nki_builder.rng_generators = {}
    seeds = _nd(np.arange(6, dtype=np.uint32).reshape(1, 6))
    nisa.rand_set_state(seeds, engine=nisa.engine.unknown)
    state_out = _zeros((1, 6), dtype=np.uint32)
    nisa.rand_get_state(state_out, engine=nisa.engine.unknown)
    assert np.array_equal(state_out.data, seeds.data)

    random_bits = _zeros((2, 4), dtype=np.uint32)
    nisa.rng(random_bits, engine=nisa.engine.unknown)
    assert random_bits.data.dtype == np.uint32

    rand_uniform = _zeros((2, 4))
    nisa.rand2(rand_uniform, 1.0, 3.0)
    assert np.all(rand_uniform.data >= 1.0)
    assert np.all(rand_uniform.data <= 3.0)

    seed_scalar = _nd(np.array([[123]], dtype=np.uint32))
    nisa.set_rng_seed(seed_scalar)

    stats_dst = _zeros((2, 6))
    nisa.bn_stats(stats_dst, act_in)
    aggr_dst = _zeros((2, 2))
    nisa.bn_aggr(aggr_dst, stats_dst)
    assert aggr_dst.data.shape == (2, 2)

    assert nisa.core_barrier(act_in, cores=(0, 1), engine=nisa.engine.unknown) is None


def test_register_and_interpreted_function_execution(patched_scope):
    """Register ops and interpreted function wrapper should run."""
    del patched_scope
    reg = nisa.register_alloc()
    reg = nisa.register_move(reg, 5)
    assert reg == 5
    reg_src = _nd(np.array([[9]], dtype=np.int32))
    reg = nisa.register_load(reg, reg_src)
    reg_dst = _zeros((1, 1), dtype=np.int32)
    nisa.register_store(reg_dst, reg)
    assert int(reg_dst.data[0, 0]) == 9

    class _ClientManager:
        """Minimal client manager used by interpreted function tests."""

        def grid_callback(self, grid):
            """Receive launch grid callback."""
            self.grid = grid

        def grid_idx_callback(self, idx):
            """Receive per-program grid index callback."""
            self.last_idx = idx

        def arg_callback(self, name, arg, ret):
            """Receive argument callback."""
            self.last_arg = (name, arg, ret)

        def pre_run_callback(self, fn):
            """Signal execution should continue."""
            self.last_fn = fn
            return True

        def post_run_callback(self, fn):
            """Signal execution should continue."""
            self.last_post_fn = fn
            return True

    def copy_kernel(src, dst):
        """Kernel that copies one tile."""
        nisa.dma_copy(dst=dst, src=src)

    fn = b2.NKIInterpretedFunction(copy_kernel)
    src = np.arange(16, dtype=float).reshape(4, 4)
    dst = np.empty_like(src)
    fn.run(src, dst, grid=(1,), client_manager=_ClientManager())
    assert np.array_equal(dst, src)
