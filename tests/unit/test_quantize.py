import numpy as np
import pytest

from triton_viz.utils.quantize import STORAGE_DTYPES, quantize_float


def _max_finite(exp_bits: int, mant_bits: int, finite_only: bool) -> float:
    """Compute max finite value for the target floating-point format."""
    bias = (1 << (exp_bits - 1)) - 1
    if finite_only:
        emax = (1 << exp_bits) - 1 - bias
        max_sig = 2.0 - 2.0 ** (1 - mant_bits)
    else:
        emax = (1 << exp_bits) - 2 - bias
        max_sig = 2.0 - 2.0 ** (-mant_bits)
    return float(np.ldexp(max_sig, emax))


def test_quantize_float_preserves_nan_and_inf_in_ieee_mode():
    values = np.array([np.nan, np.inf, -np.inf])
    output = quantize_float(values, exp_bits=5, mant_bits=2, finite_only=False)
    assert np.isnan(output[0])
    assert np.isposinf(output[1])
    assert np.isneginf(output[2])


def test_quantize_float_finite_only_clamps_inf_and_overflow():
    exp_bits, mant_bits, finite_only = STORAGE_DTYPES["float8_e4m3fn"]
    max_fin = _max_finite(exp_bits, mant_bits, finite_only)
    values = np.array(
        [np.inf, -np.inf, max_fin * 4.0, -max_fin * 4.0],
    )
    output = quantize_float(values, exp_bits, mant_bits, finite_only=finite_only)
    assert np.array_equal(output, np.array([max_fin, -max_fin, max_fin, -max_fin]))


def test_quantize_float_ieee_mode_overflow_becomes_inf():
    exp_bits, mant_bits, finite_only = STORAGE_DTYPES["float8_e4m3"]
    max_fin = _max_finite(exp_bits, mant_bits, finite_only)
    values = np.array([max_fin * 2.0, -max_fin * 2.0])
    output = quantize_float(values, exp_bits, mant_bits, finite_only=finite_only)
    assert np.isposinf(output[0])
    assert np.isneginf(output[1])


def test_quantize_float_subnormal_rounding_and_signed_zero():
    exp_bits, mant_bits, finite_only = STORAGE_DTYPES["float8_e4m3"]
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    sub_step = np.ldexp(1.0, emin - mant_bits)
    values = np.array(
        [0.49 * sub_step, 0.51 * sub_step, -0.49 * sub_step, -0.51 * sub_step],
    )
    output = quantize_float(values, exp_bits, mant_bits, finite_only=finite_only)
    assert output[0] == 0.0
    assert output[1] == sub_step
    assert output[2] == 0.0
    assert np.signbit(output[2])
    assert output[3] == -sub_step


def test_quantize_float_rounds_normal_values():
    values = np.array([1.3, -1.3])
    output = quantize_float(values, exp_bits=5, mant_bits=2, finite_only=False)
    assert np.array_equal(output, np.array([1.25, -1.25]))


@pytest.mark.parametrize("dtype_name", list(STORAGE_DTYPES))
def test_quantize_float_normal_boundary_rounding(dtype_name: str):
    exp_bits, mant_bits, finite_only = STORAGE_DTYPES[dtype_name]
    step = np.ldexp(1.0, -mant_bits)
    x = 1.0 + 0.5 * step
    eps = step / 16.0
    values = np.array([x, x + eps])
    output = quantize_float(values, exp_bits, mant_bits, finite_only=finite_only)
    assert np.array_equal(output, np.array([1.0, 1.0 + step]))


@pytest.mark.parametrize("dtype_name", list(STORAGE_DTYPES))
def test_quantize_float_subnormal_boundary_rounding(dtype_name: str):
    exp_bits, mant_bits, finite_only = STORAGE_DTYPES[dtype_name]
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    sub_step = np.ldexp(1.0, emin - mant_bits)
    x = 0.5 * sub_step
    eps = sub_step / 16.0
    values = np.array([x, x + eps])
    output = quantize_float(values, exp_bits, mant_bits, finite_only=finite_only)
    assert np.array_equal(output, np.array([0.0, sub_step]))
