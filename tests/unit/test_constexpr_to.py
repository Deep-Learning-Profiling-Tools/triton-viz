import types

import pytest
import numpy as np
import triton.language as tl

from triton_viz.core.patch import _constexpr_to, _src_np_dtype


@pytest.mark.parametrize(
    "value,triton_dtype,expected_np,expected_val",
    [
        (1.0, tl.float64, np.float64, 1.0),
        (1, tl.uint8, np.uint8, 1),
        (-7, tl.int16, np.int16, -7),
        (1, tl.uint32, np.uint32, 1),
        # Narrowing / signedness-change casts — must wrap, not reject
        (300, tl.uint8, np.uint8, 300 % 256),  # 44
        (-1, tl.uint32, np.uint32, 2**32 - 1),  # 4294967295
        (2**31, tl.int32, np.int32, -(2**31)),  # signed wrap
        (255, tl.int8, np.int8, -1),  # unsigned → signed wrap
    ],
)
def test_constexpr_to_preserves_dtype(value, triton_dtype, expected_np, expected_val):
    mock_self = types.SimpleNamespace(value=value)
    ret = _constexpr_to(mock_self, triton_dtype)
    assert isinstance(ret, tl.core.tensor)
    assert ret.dtype == triton_dtype
    assert ret.handle.data.dtype == np.dtype(expected_np)
    assert ret.handle.data.item() == expected_val


def test_constexpr_bitcast_preserves_dtype():
    mock_self = types.SimpleNamespace(value=42)
    ret = _constexpr_to(mock_self, tl.float32, bitcast=True)
    assert isinstance(ret, tl.core.tensor)
    assert ret.dtype == tl.float32


@pytest.mark.parametrize("bad_mode", ["RTZ", "foo", ""])
def test_constexpr_to_invalid_rounding_mode_raises(bad_mode):
    """Invalid fp_downcast_rounding values must raise ValueError, not silently fall back."""
    mock_self = types.SimpleNamespace(value=1.0)
    with pytest.raises(ValueError, match="fp_downcast_rounding must be one of"):
        _constexpr_to(mock_self, tl.float16, fp_downcast_rounding=bad_mode)


def test_constexpr_to_overflow_raises():
    """Integer literals outside [-2**63, 2**64) must raise OverflowError."""
    mock_self = types.SimpleNamespace(value=-(2**63) - 1)
    with pytest.raises(OverflowError, match="outside the representable range"):
        _constexpr_to(mock_self, tl.int64)


def test_constexpr_bitcast_bool_to_int8_raises():
    """Bitcast bool(int1) -> int8 must raise: primitive_bitwidth 1 != 8."""
    mock_self = types.SimpleNamespace(value=True)
    with pytest.raises(ValueError, match="Cannot bitcast"):
        _constexpr_to(mock_self, tl.int8, bitcast=True)


def test_src_np_dtype_max_uint64():
    """Values in [2**63, 2**64) must map to uint64, not overflow."""
    assert _src_np_dtype(2**64 - 1) == np.dtype(np.uint64)
