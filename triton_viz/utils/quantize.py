import numpy as np

STORAGE_DTYPES = {
    "bfloat16": (8, 7, False),
    "tfloat32": (8, 10, False),
    "float8_e4m3": (4, 3, False),
    "float8_e5m2": (5, 2, False),
    "float8_e4m3fn": (4, 3, True),
    "float8_e5m2fn": (5, 2, True),
    "float4_e2m1fn": (2, 1, True),
}


def quantize_float(value, exp_bits, mant_bits, finite_only=False):
    """Given a float, quantize it as if it only had the float format specified in args. Still returns a float."""
    ax = np.abs(value)

    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    if finite_only:  # no inf; reserve top mantissa at max exp for NaN (FN)
        emax = (1 << exp_bits) - 1 - bias
        max_sig = 2.0 - 2.0 ** (1 - mant_bits)
        has_inf = False
    else:  # IEEE-like: all-ones exp reserved for inf/nan
        emax = (1 << exp_bits) - 2 - bias
        max_sig = 2.0 - 2.0 ** (-mant_bits)
        has_inf = True

    min_norm = np.ldexp(1.0, emin)
    sub_step = np.ldexp(1.0, emin - mant_bits)
    max_fin = np.ldexp(max_sig, emax)

    out = ax.copy()
    fin = np.isfinite(value)
    nz = fin & (ax != 0)

    # quantize finite nonzeros
    sub = nz & (ax < min_norm)
    if np.any(sub):
        out[sub] = np.rint(ax[sub] / sub_step) * sub_step

    nor = nz & ~sub
    if np.any(nor):
        _, e = np.frexp(ax[nor])  # ax = m*2**e, m in [0.5,1)
        step_exp = (e - 1) - mant_bits
        out[nor] = np.ldexp(np.rint(np.ldexp(ax[nor], -step_exp)), step_exp)

    # special inputs
    out[np.isinf(value)] = np.inf if has_inf else max_fin
    # NaNs propagate automatically via abs/copy; keep explicit for clarity:
    out[np.isnan(value)] = np.nan

    # overflow policy after rounding
    if has_inf:
        out[fin & (out > max_fin)] = np.inf
    else:
        out = np.minimum(out, max_fin)

    return np.copysign(out, value)
