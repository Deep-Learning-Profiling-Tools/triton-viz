import importlib

import numpy as np
import pytest

try:
    import nki.language as nl
    import nkilib.core.utils.allocator as alloc_mod
    import nkilib.core.subkernels.rmsnorm_tkg as rmsnorm_tkg_mod
    from triton_viz.core.patch import _LangPatchScope
    import nkilib.core.utils.stream_shuffle_broadcast as stream_shuffle_broadcast_mod
    import nkilib.core.utils.tp_broadcast as tp_broadcast_mod
    import triton_viz.core.nki_beta2 as b2
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


def _np_output_projection_tkg(attention, weight, bias):
    """Reference output projection for TRANSPOSE_OUT=False."""
    d_size, b_size, n_size, s_size = attention.shape
    attn = np.transpose(attention, (1, 3, 2, 0)).reshape(
        b_size * s_size, n_size * d_size
    )
    return attn @ weight + bias


def test_stream_shuffle_broadcast_e2e(patched_scope):
    """stream_shuffle_broadcast should handle multi-block dst partition sizes."""
    del patched_scope
    src_row = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    src = _nd(src_row.reshape(1, 4), buffer="sbuf")
    dst = _nd(np.zeros((33, 4), dtype=np.float32), buffer="sbuf")
    stream_shuffle_broadcast_mod.stream_shuffle_broadcast(src, dst)
    expected = np.repeat(src_row.reshape(1, 4), 33, axis=0)
    assert np.array_equal(dst.data, expected)


def test_tp_broadcast_e2e(patched_scope):
    """tp_broadcast should transpose one source column into all destination rows."""
    del patched_scope
    src = _nd(np.arange(12, dtype=np.float32).reshape(4, 3), buffer="sbuf")
    dst = _nd(np.zeros((6, 4), dtype=np.float32), buffer="sbuf")
    tp_broadcast_mod.tp_broadcast(src, dst, src_offset=1)
    expected = np.repeat(np.array([[1.0, 4.0, 7.0, 10.0]], dtype=np.float32), 6, axis=0)
    assert np.array_equal(dst.data, expected)


def test_output_projection_tkg_e2e(patched_scope):
    """output_projection_tkg should run under beta2 interpreter and match reference."""
    del patched_scope
    op_tkg_mod = importlib.import_module(
        "nkilib.core.output_projection.output_projection_tkg"
    )
    op_tkg_mod = importlib.reload(op_tkg_mod)
    kernel_fn = op_tkg_mod.output_projection_tkg.func

    hidden_size = 512
    d_size, b_size, n_size, s_size = 32, 2, 2, 2
    attention_np = (
        np.arange(d_size * b_size * n_size * s_size, dtype=np.float32).reshape(
            d_size, b_size, n_size, s_size
        )
        / 100.0
    )
    weight_np = np.arange((n_size * d_size) * hidden_size, dtype=np.float32).reshape(
        n_size * d_size, hidden_size
    )
    weight_np = weight_np / 1000.0
    bias_np = np.linspace(-1.0, 1.0, hidden_size, dtype=np.float32).reshape(
        1, hidden_size
    )

    attention = _nd(attention_np, buffer="hbm")
    weight = _nd(weight_np, buffer="hbm")
    bias = _nd(bias_np, buffer="hbm")

    out = kernel_fn(attention, weight, bias, TRANSPOSE_OUT=False, OUT_IN_SB=True)
    expected = _np_output_projection_tkg(attention_np, weight_np, bias_np)

    assert out.shape == (b_size * s_size, hidden_size)
    assert out.buffer == "sbuf"
    assert np.allclose(out.data, expected, atol=1e-4, rtol=1e-4)


def test_rmsnorm_tkg_e2e(patched_scope):
    """rmsnorm_tkg should match NumPy RMSNorm for sbuf input path."""

    class _CompatScope:
        """Backwards-compatible scope shim for nkilib allocator dataclass."""

        def __init__(self, starting_addr, num_sections, cur_section_id=0):
            self.starting_addr = starting_addr
            self.num_sections = num_sections
            self.cur_section_id = cur_section_id

    patched_scope.set_attr(alloc_mod, "Scope", _CompatScope)
    patched_scope.set_attr(nl, "square", np.square)

    h0 = 128
    bxs = 3
    h1 = 1
    hidden = h0 * h1

    inp_np = (
        np.arange(h0 * bxs * h1, dtype=np.float32).reshape(h0, bxs, h1) - 30.0
    ) / 20.0
    gamma_np = np.ones((1, hidden), dtype=np.float32)

    inp = _nd(inp_np, buffer="sbuf")
    gamma = _nd(gamma_np, buffer="hbm")

    out = rmsnorm_tkg_mod.rmsnorm_tkg(
        inp=inp, gamma=gamma, eps=1e-6, output_in_sbuf=True
    )

    expected = np.empty_like(inp_np)
    for bs_idx in range(bxs):
        rms = np.sqrt(np.mean(inp_np[:, bs_idx, :] ** 2) + 1e-6)
        expected[:, bs_idx, :] = inp_np[:, bs_idx, :] / rms

    assert out.shape == inp_np.shape
    assert out.buffer == "sbuf"
    assert np.allclose(out.data, expected, atol=1e-6, rtol=1e-6)
