import numpy as np
import pytest
from triton_viz.utils.dtypes import STORAGE_DTYPES

pytest.importorskip("ml_dtypes")


def test_storage_dtypes_has_expected_aliases():
    assert STORAGE_DTYPES[None] is None
    assert STORAGE_DTYPES[int] == np.dtype(np.int64)
    assert STORAGE_DTYPES[float] == np.dtype(np.float64)

    assert STORAGE_DTYPES["int8"] == np.dtype(np.int8)
    assert STORAGE_DTYPES["int16"] == np.dtype(np.int16)
    assert STORAGE_DTYPES["int32"] == np.dtype(np.int32)
    assert STORAGE_DTYPES["uint8"] == np.dtype(np.uint8)
    assert STORAGE_DTYPES["uint16"] == np.dtype(np.uint16)
    assert STORAGE_DTYPES["uint32"] == np.dtype(np.uint32)

    assert STORAGE_DTYPES["half"] == np.dtype(np.float16)
    assert STORAGE_DTYPES["float16"] == np.dtype(np.float16)
    assert STORAGE_DTYPES["float"] == np.dtype(np.float32)
    assert STORAGE_DTYPES["float32"] == np.dtype(np.float32)
    assert STORAGE_DTYPES["double"] == np.dtype(np.float64)
    assert STORAGE_DTYPES["float64"] == np.dtype(np.float64)

    assert STORAGE_DTYPES["bfloat16"].name == "bfloat16"
    assert STORAGE_DTYPES["float8_e4m3"].name == "float8_e4m3"
    assert STORAGE_DTYPES["float8_e5m2"].name == "float8_e5m2"
    assert STORAGE_DTYPES["float8_e4m3fn"].name == "float8_e4m3fn"
    assert STORAGE_DTYPES["float4_e2m1fn"].name == "float4_e2m1fn"


def test_storage_dtypes_round_trips_dtype_objects():
    for alias, dtype in STORAGE_DTYPES.items():
        if alias is None:
            continue
        if isinstance(alias, np.dtype):
            assert STORAGE_DTYPES[alias] == alias
        if isinstance(dtype, np.dtype):
            assert STORAGE_DTYPES[dtype] == dtype


@pytest.mark.parametrize(
    "dtype_name, expected",
    [
        (
            "bfloat16",
            [[0.10009765625, 0.8984375, 1.2421875], [1.5078125, 2.90625, -5.1875]],
        ),
        ("float8_e4m3", [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]]),
        ("float8_e5m2", [[0.09375, 0.875, 1.25], [1.5, 3.0, -5.0]]),
        ("float8_e4m3fn", [[0.1015625, 0.875, 1.25], [1.5, 3.0, -5.0]]),
        ("float4_e2m1fn", [[0.0, 1.0, 1.0], [1.5, 3.0, -6.0]]),
    ],
)
def test_ml_dtypes_cast_matches_expected_quantization(dtype_name: str, expected):
    values = np.array([[0.1, 0.9, 1.24], [1.51, 2.9, -5.2]], dtype=np.float64)
    casted = values.astype(STORAGE_DTYPES[dtype_name])
    assert casted.dtype == STORAGE_DTYPES[dtype_name]
    assert np.array_equal(casted, np.array(expected, dtype=casted.dtype))
