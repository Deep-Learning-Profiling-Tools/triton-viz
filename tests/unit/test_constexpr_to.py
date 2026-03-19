import types

import pytest
import numpy as np
import triton.language as tl

from triton_viz.core.patch import _constexpr_to


@pytest.mark.parametrize(
    "value,triton_dtype,expected_np,expected_val",
    [
        (1.0, tl.float64, np.float64, 1.0),  # float64 demoted to float32
        (1, tl.uint8, np.uint8, 1),  # uint8 demoted to int32
        (-7, tl.int16, np.int16, -7),  # int16 demoted to int32
        (1, tl.uint32, np.uint32, 1),  # uint32 demoted to int32
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
