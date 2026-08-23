"""Minimal top-level nl.* controls for compositional Level-A calibration."""
import random

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def elementwise_one_kernel(x, y, chain):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value = nl.load(x[pi, fi])
    for index in nl.static_range(chain):
        value = nl.add(value, float(index + 1), dtype=x.dtype)
    nl.store(out[pi, fi], value=value)
    return out


@nki.jit
def elementwise_two_kernel(x, y, chain):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value, rhs = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    for _ in nl.static_range(chain):
        value = nl.add(value, rhs, dtype=x.dtype)
    nl.store(out[pi, fi], value=value)
    return out


@nki.jit
def elementwise_maximum_kernel(x, y):
    """Unary threshold control for activation-style VectorE lowering."""
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value = nl.load(x[pi, fi])
    nl.store(out[pi, fi], value=nl.maximum(value, 0.0))
    return out


@nki.jit
def elementwise_multiply_kernel(x, y):
    """Single scalar multiply control for ScalarE instruction selection."""
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value = nl.load(x[pi, fi])
    nl.store(out[pi, fi], value=nl.multiply(value, 2.0, dtype=x.dtype))
    return out


@nki.jit
def elementwise_sigmoid_kernel(x, y):
    """Unary transcendental control for activation-engine lowering."""
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value = nl.load(x[pi, fi])
    nl.store(out[pi, fi], value=nl.sigmoid(value))
    return out


@nki.jit
def primitive_divide_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    lhs, rhs = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.divide(lhs, rhs))
    return out


@nki.jit
def primitive_add_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    nl.store(out[pi, fi], value=nl.add(nl.load(x[pi, fi]), nl.load(y[pi, fi])))
    return out


@nki.jit
def primitive_subtract_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    nl.store(out[pi, fi], value=nl.subtract(nl.load(x[pi, fi]), nl.load(y[pi, fi])))
    return out


@nki.jit
def primitive_multiply_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    nl.store(out[pi, fi], value=nl.multiply(nl.load(x[pi, fi]), nl.load(y[pi, fi])))
    return out


@nki.jit
def primitive_where_bundle_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    lhs, rhs = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.where(nl.greater(lhs, 0.0), lhs, rhs))
    return out


@nki.jit
def sequence_add_multiply_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.multiply(nl.add(a, b), b))
    return out


@nki.jit
def sequence_multiply_add_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.add(nl.multiply(a, b), b))
    return out


@nki.jit
def sequence_subtract_multiply_add_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.add(nl.multiply(nl.subtract(a, b), b), a))
    return out


@nki.jit
def sequence_multiply_subtract_multiply_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.multiply(nl.subtract(nl.multiply(a, b), b), a))
    return out


@nki.jit
def sequence_exp_multiply_add_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.add(nl.multiply(nl.exp(a), b), a))
    return out


@nki.jit
def sequence_log_add_multiply_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.multiply(nl.add(nl.log(a), b), a))
    return out


@nki.jit
def sequence_reduce_add_multiply_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a = nl.load(x[pi, fi]); reduced = nl.sum(a, axis=1, keepdims=True)
    nl.store(out[pi, fi], value=nl.multiply(a, nl.add(reduced, 1.0)))
    return out


@nki.jit
def sequence_reduce_divide_multiply_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a = nl.load(x[pi, fi]); reduced = nl.sum(a, axis=1, keepdims=True)
    nl.store(out[pi, fi], value=nl.multiply(a, nl.divide(reduced, float(f))))
    return out


@nki.jit
def sequence_two_reduce_add_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    combined = nl.add(nl.sum(a, axis=1, keepdims=True), nl.sum(b, axis=1, keepdims=True))
    nl.store(out[pi, fi], value=nl.multiply(a, combined))
    return out


@nki.jit
def sequence_two_reduce_multiply_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    first = nl.sum(a, axis=1, keepdims=True)
    second = nl.sum(nl.multiply(a, b), axis=1, keepdims=True)
    nl.store(out[pi, fi], value=nl.multiply(a, nl.multiply(first, second)))
    return out


@nki.jit
def sequence_rsqrt_multiply_add_kernel(x, y):
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.add(nl.multiply(nl.rsqrt(a), b), a))
    return out


# Wide physical-tile controls model source kernels that allocate a 16K SBUF
# free dimension while guarding only HBM accesses.  Deliberately do not pass
# ``mask`` to compute primitives: that is a distinct, separately calibrated
# lowering domain from the compute-masked atomic controls above.
@nki.jit
def sequence_add_multiply_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask)
    value = nl.add(a, 1.0); value = nl.multiply(value, 2.0)
    nl.store(out[pi, fi], value, mask=mask); return out


@nki.jit
def sequence_subtract_multiply_add_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask)
    value = nl.subtract(a, 1.0); value = nl.multiply(value, 2.0)
    value = nl.add(value, 1.0)
    nl.store(out[pi, fi], value, mask=mask); return out


@nki.jit
def sequence_exp_multiply_add_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask)
    value = nl.exp(a); value = nl.multiply(value, 2.0); value = nl.add(value, 1.0)
    nl.store(out[pi, fi], value, mask=mask); return out


@nki.jit
def sequence_reduce_add_multiply_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask); reduced = nl.sum(a, axis=1, keepdims=True)
    nl.store(out[pi, fi], nl.multiply(a, nl.add(reduced, 1.0)), mask=mask); return out


@nki.jit
def sequence_two_reduce_multiply_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask)
    product = nl.multiply(a, a); left = nl.sum(a, axis=1, keepdims=True)
    right = nl.sum(product, axis=1, keepdims=True)
    nl.store(out[pi, fi], nl.multiply(nl.multiply(product, left), right), mask=mask); return out


@nki.jit
def sequence_rsqrt_multiply_add_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask)
    value = nl.rsqrt(a); value = nl.multiply(value, 2.0); value = nl.add(value, 1.0)
    nl.store(out[pi, fi], value, mask=mask); return out


@nki.jit
def sequence_multiply_add_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask); value = nl.multiply(a, 2.0); value = nl.add(value, 1.0)
    nl.store(out[pi, fi], value, mask=mask); return out


@nki.jit
def sequence_multiply_subtract_multiply_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask); value = nl.multiply(a, 2.0); value = nl.subtract(value, 1.0); value = nl.multiply(value, 2.0)
    nl.store(out[pi, fi], value, mask=mask); return out


@nki.jit
def sequence_log_add_multiply_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask); value = nl.log(a); value = nl.add(value, 1.0); value = nl.multiply(value, 2.0)
    nl.store(out[pi, fi], value, mask=mask); return out


@nki.jit
def sequence_reduce_divide_multiply_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask); reduced = nl.sum(a, axis=1, keepdims=True)
    scale = nl.divide(reduced, float(f)); nl.store(out[pi, fi], nl.multiply(a, scale), mask=mask); return out


@nki.jit
def sequence_two_reduce_add_wide_memory_kernel(x, y):
    p, f = x.shape; tile_f = 16384; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask); left = nl.sum(a, axis=1, keepdims=True); square = nl.multiply(a, a)
    right = nl.sum(square, axis=1, keepdims=True); nl.store(out[pi, fi], nl.multiply(a, nl.add(left, right)), mask=mask); return out


def _sequence_perm2k_body(x, out, logical_f, pattern):
    """Full-factorial order controls on a fixed 2K physical source tile."""
    p, _ = x.shape; tile_f = 2048
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < logical_f)
    value = nl.load(x[pi, fi], mask=mask)
    if pattern == 0: value = nl.subtract(nl.multiply(nl.add(value, 1.0), 2.0), 1.0)
    elif pattern == 1: value = nl.multiply(nl.subtract(nl.add(value, 1.0), 1.0), 2.0)
    elif pattern == 2: value = nl.subtract(nl.add(nl.multiply(value, 2.0), 1.0), 1.0)
    elif pattern == 3: value = nl.add(nl.subtract(nl.multiply(value, 2.0), 1.0), 1.0)
    elif pattern == 4: value = nl.multiply(nl.add(nl.subtract(value, 1.0), 1.0), 2.0)
    elif pattern == 5: value = nl.add(nl.multiply(nl.subtract(value, 1.0), 2.0), 1.0)
    elif pattern == 6: value = nl.multiply(nl.add(nl.exp(value), 1.0), 2.0)
    elif pattern == 7: value = nl.multiply(nl.exp(nl.add(value, 1.0)), 2.0)
    elif pattern == 8: value = nl.add(nl.exp(nl.multiply(value, 2.0)), 1.0)
    elif pattern == 9:
        reduced = nl.sum(value, axis=1, keepdims=True); value = nl.multiply(nl.add(reduced, 1.0), value)
    elif pattern == 10:
        shifted = nl.add(value, 1.0); reduced = nl.sum(shifted, axis=1, keepdims=True); value = nl.multiply(value, reduced)
    elif pattern == 11:
        scaled = nl.multiply(value, 2.0); reduced = nl.sum(scaled, axis=1, keepdims=True); value = nl.add(value, reduced)
    nl.store(out[pi, fi], value, mask=mask); return out


def _sequence_perm2k_long_body(x, out, logical_f, pattern):
    p, _ = x.shape; tile_f = 2048
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < logical_f)
    value = nl.load(x[pi, fi], mask=mask)
    if pattern == 0: value = nl.subtract(nl.multiply(nl.add(nl.subtract(nl.multiply(nl.add(value, 1.0), 2.0), 1.0), 1.0), 2.0), 1.0)
    elif pattern == 1: value = nl.add(nl.subtract(nl.multiply(nl.add(nl.subtract(nl.multiply(value, 2.0), 1.0), 1.0), 2.0), 1.0), 1.0)
    elif pattern == 2: value = nl.multiply(nl.add(nl.subtract(nl.multiply(nl.add(nl.subtract(value, 1.0), 1.0), 2.0), 1.0), 1.0), 2.0)
    elif pattern == 3: value = nl.add(nl.multiply(nl.subtract(nl.multiply(nl.add(nl.exp(value), 1.0), 2.0), 1.0), 2.0), 1.0)
    elif pattern == 4: value = nl.multiply(nl.subtract(nl.multiply(nl.add(nl.exp(nl.add(value, 1.0)), 1.0), 2.0), 1.0), 2.0)
    elif pattern == 5: value = nl.multiply(nl.add(nl.subtract(nl.multiply(nl.add(nl.exp(nl.multiply(value, 2.0)), 1.0), 2.0), 1.0), 1.0), 2.0)
    elif pattern == 6:
        reduced = nl.sum(value, axis=1, keepdims=True); value = nl.add(reduced, 1.0); value = nl.multiply(value, 2.0); value = nl.subtract(value, 1.0); value = nl.multiply(value, 2.0); value = nl.add(value, 1.0)
    elif pattern == 7:
        value = nl.add(value, 1.0); value = nl.multiply(value, 2.0); reduced = nl.sum(value, axis=1, keepdims=True); value = nl.add(reduced, 1.0); value = nl.subtract(value, 1.0); value = nl.multiply(value, 2.0)
    elif pattern == 8:
        value = nl.multiply(value, 2.0); value = nl.subtract(value, 1.0); value = nl.add(value, 1.0); reduced = nl.sum(value, axis=1, keepdims=True); value = nl.multiply(reduced, 2.0); value = nl.add(value, 1.0)
    elif pattern == 9:
        reduced = nl.sum(value, axis=1, keepdims=True); value = nl.multiply(reduced, 2.0); value = nl.exp(value); value = nl.add(value, 1.0); value = nl.subtract(value, 1.0); value = nl.multiply(value, 2.0)
    elif pattern == 10:
        value = nl.add(value, 1.0); reduced = nl.sum(value, axis=1, keepdims=True); value = nl.multiply(reduced, 2.0); value = nl.subtract(value, 1.0); value = nl.exp(value); value = nl.multiply(value, 2.0)
    else:
        value = nl.multiply(value, 2.0); value = nl.add(value, 1.0); reduced = nl.sum(value, axis=1, keepdims=True); value = nl.subtract(reduced, 1.0); value = nl.add(value, 1.0); value = nl.multiply(value, 2.0)
    nl.store(out[pi, fi], value, mask=mask); return out


def _sequence_deep2k_body(x, out, logical_f, chain, pattern):
    """Long generic chains on a fixed 2K tile; no operator grammar is copied."""
    p, _ = x.shape; tile_f = 2048
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < logical_f)
    value = nl.load(x[pi, fi], mask=mask)
    for index in nl.static_range(chain):
        if pattern == 0:
            value = nl.add(value, float(index + 1))
        elif pattern == 1:
            value = nl.multiply(value, 1.0001)
        else:
            value = nl.multiply(nl.add(value, float(index + 1)), 1.0001)
    nl.store(out[pi, fi], value, mask=mask)
    return out


@nki.jit
def sequence_deep2k_add_kernel(x, y, logical_f, chain):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    return _sequence_deep2k_body(x, out, logical_f, chain, 0)


@nki.jit
def sequence_deep2k_multiply_kernel(x, y, logical_f, chain):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    return _sequence_deep2k_body(x, out, logical_f, chain, 1)


@nki.jit
def sequence_deep2k_add_multiply_kernel(x, y, logical_f, chain):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    return _sequence_deep2k_body(x, out, logical_f, chain, 2)


def _sequence_deepmixed2k_body(x, out, logical_f, pattern):
    """Long mixed grammars with deliberately randomized primitive order."""
    p, _ = x.shape; tile_f = 2048
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < logical_f)
    value = nl.load(x[pi, fi], mask=mask)
    if pattern == 0:
        value = nl.add(value, 1.0); value = nl.multiply(value, 1.01)
        value = nl.subtract(value, .5); value = nl.exp(value)
        value = nl.multiply(value, .25); value = nl.add(value, .75)
        reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.multiply(value, reduced); value = nl.subtract(value, .25)
        value = nl.multiply(value, 1.01); value = nl.add(value, .5)
        value = nl.multiply(value, .99); value = nl.subtract(value, .1)
    elif pattern == 1:
        value = nl.multiply(value, .25); value = nl.exp(value)
        value = nl.add(value, .5); value = nl.subtract(value, .1)
        reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.add(value, reduced); value = nl.multiply(value, 1.01)
        value = nl.subtract(value, .25); value = nl.add(value, .75)
        value = nl.multiply(value, .99); value = nl.add(value, .2)
        value = nl.subtract(value, .1); value = nl.multiply(value, 1.001)
    elif pattern == 2:
        value = nl.add(nl.multiply(value, value), 1.0); value = nl.rsqrt(value)
        value = nl.multiply(value, 1.01); value = nl.subtract(value, .25)
        value = nl.add(value, .75); reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.multiply(value, reduced); value = nl.add(value, .5)
        value = nl.subtract(value, .1); value = nl.multiply(value, .99)
        value = nl.add(value, .2); value = nl.multiply(value, 1.001)
    elif pattern == 3:
        value = nl.multiply(value, .5); value = nl.add(value, 1.0)
        reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.add(value, reduced); value = nl.multiply(value, value)
        value = nl.add(value, 1.0); value = nl.rsqrt(value)
        value = nl.subtract(value, .1); value = nl.multiply(value, 1.01)
        value = nl.add(value, .5); value = nl.multiply(value, .99)
        value = nl.subtract(value, .2); value = nl.add(value, .3)
    elif pattern == 4:
        value = nl.add(nl.multiply(value, value), 1.0); value = nl.log(value)
        value = nl.multiply(value, .5); value = nl.add(value, .75)
        reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.subtract(value, reduced); value = nl.multiply(value, 1.01)
        value = nl.add(value, .5); value = nl.subtract(value, .1)
        value = nl.multiply(value, .99); value = nl.add(value, .2)
    elif pattern == 5:
        value = nl.add(value, 1.0); value = nl.multiply(value, value)
        reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.multiply(value, reduced); value = nl.add(value, 1.0)
        value = nl.log(value); value = nl.multiply(value, .5)
        value = nl.subtract(value, .25); value = nl.add(value, .75)
        value = nl.multiply(value, .99); value = nl.subtract(value, .1)
    elif pattern == 6:
        square = nl.multiply(value, value)
        left = nl.sum(value, axis=1, keepdims=True)
        right = nl.sum(square, axis=1, keepdims=True)
        value = nl.add(nl.multiply(value, left), right)
        value = nl.subtract(value, .25); value = nl.multiply(value, 1.01)
        value = nl.add(value, .5); value = nl.multiply(value, .99)
        value = nl.subtract(value, .1); value = nl.add(value, .2)
        value = nl.multiply(value, 1.001); value = nl.add(value, .3)
    elif pattern == 7:
        value = nl.add(value, .5); square = nl.multiply(value, value)
        right = nl.sum(square, axis=1, keepdims=True)
        value = nl.multiply(value, right); left = nl.sum(value, axis=1, keepdims=True)
        value = nl.add(value, left); value = nl.multiply(value, 1.01)
        value = nl.subtract(value, .25); value = nl.add(value, .75)
        value = nl.multiply(value, .99); value = nl.subtract(value, .1)
        value = nl.add(value, .2); value = nl.multiply(value, 1.001)
    elif pattern == 8:
        value = nl.add(value, .5); value = nl.multiply(value, .25)
        reduced = nl.sum(value, axis=1, keepdims=True); value = nl.add(value, reduced)
        value = nl.subtract(value, .1); value = nl.exp(value)
        value = nl.multiply(value, .25); value = nl.add(value, .75)
        value = nl.subtract(value, .2); value = nl.multiply(value, .99)
        value = nl.add(value, .3); value = nl.multiply(value, 1.001)
    elif pattern == 9:
        value = nl.subtract(value, .1); value = nl.add(value, .5)
        value = nl.exp(nl.multiply(value, .25)); value = nl.subtract(value, .2)
        value = nl.multiply(value, .99); reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.multiply(value, reduced); value = nl.add(value, .75)
        value = nl.subtract(value, .1); value = nl.multiply(value, 1.001)
        value = nl.add(value, .3)
    elif pattern == 10:
        value = nl.add(value, .5); reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.multiply(value, reduced); value = nl.subtract(value, .1)
        value = nl.multiply(value, value); value = nl.add(value, 1.0)
        value = nl.rsqrt(value); value = nl.multiply(value, 1.01)
        value = nl.add(value, .75); value = nl.subtract(value, .2)
        value = nl.multiply(value, .99); value = nl.add(value, .3)
    elif pattern == 11:
        value = nl.multiply(value, value); value = nl.add(value, 1.0)
        value = nl.rsqrt(value); value = nl.subtract(value, .1)
        value = nl.add(value, .5); value = nl.multiply(value, .99)
        reduced = nl.sum(value, axis=1, keepdims=True); value = nl.add(value, reduced)
        value = nl.multiply(value, 1.01); value = nl.subtract(value, .2)
        value = nl.add(value, .3); value = nl.multiply(value, 1.001)
    elif pattern == 12:
        value = nl.add(value, 1.0); reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.multiply(value, reduced); value = nl.multiply(value, value)
        value = nl.add(value, 1.0); value = nl.log(value)
        value = nl.subtract(value, .1); value = nl.multiply(value, .99)
        value = nl.add(value, .75); value = nl.subtract(value, .2)
        value = nl.multiply(value, 1.001); value = nl.add(value, .3)
    elif pattern == 13:
        value = nl.multiply(value, value); value = nl.add(value, 1.0)
        value = nl.log(value); value = nl.multiply(value, .5)
        value = nl.subtract(value, .1); reduced = nl.sum(value, axis=1, keepdims=True)
        value = nl.add(value, reduced); value = nl.multiply(value, .99)
        value = nl.add(value, .75); value = nl.subtract(value, .2)
        value = nl.multiply(value, 1.001); value = nl.add(value, .3)
    elif pattern == 14:
        left = nl.sum(value, axis=1, keepdims=True); value = nl.add(value, left)
        value = nl.multiply(value, .5); value = nl.subtract(value, .1)
        square = nl.multiply(value, value); right = nl.sum(square, axis=1, keepdims=True)
        value = nl.multiply(value, right); value = nl.add(value, .75)
        value = nl.subtract(value, .2); value = nl.multiply(value, .99)
        value = nl.add(value, .3); value = nl.multiply(value, 1.001)
    else:
        square = nl.multiply(value, value); right = nl.sum(square, axis=1, keepdims=True)
        value = nl.subtract(value, .1); value = nl.multiply(value, right)
        value = nl.add(value, .5); left = nl.sum(value, axis=1, keepdims=True)
        value = nl.add(value, left); value = nl.multiply(value, .99)
        value = nl.subtract(value, .2); value = nl.add(value, .75)
        value = nl.multiply(value, 1.001); value = nl.add(value, .3)
    nl.store(out[pi, fi], value, mask=mask)
    return out


def random_mixed_schedule(schedule_id: int) -> tuple[str, ...]:
    """Deterministic, target-independent mixed grammar for control generation."""
    rng = random.Random(0x5A17_2C00 + int(schedule_id))
    schedule = [rng.choice(("add", "subtract", "multiply")) for _ in range(12)]
    schedule.extend((rng.choice(("exp", "log", "rsqrt")), "reduce",
                     rng.choice(("add", "subtract", "multiply")),
                     rng.choice(("add", "subtract", "multiply"))))
    if schedule_id % 3 == 0:
        schedule[rng.randrange(12)] = "reduce"
    rng.shuffle(schedule)
    return tuple(schedule)


def sequence_randommixed2k_factory(schedule_id: int):
    schedule = random_mixed_schedule(schedule_id)

    @nki.jit
    def kernel(x, y, logical_f):
        p, _ = x.shape; tile_f = 2048
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
        mask = (pi < p) & (fi < logical_f)
        value = nl.add(nl.load(x[pi, fi], mask=mask), 2.0)
        for index in nl.static_range(len(schedule)):
            token = schedule[index]
            if token == "add": value = nl.add(value, .1)
            elif token == "subtract": value = nl.subtract(value, .01)
            elif token == "multiply": value = nl.multiply(value, 1.001)
            elif token == "exp": value = nl.exp(nl.multiply(value, .1))
            elif token == "log": value = nl.log(value)
            elif token == "rsqrt": value = nl.rsqrt(value)
            else:
                reduced = nl.sum(value, axis=1, keepdims=True)
                value = nl.add(value, nl.multiply(reduced, .0001))
        nl.store(out[pi, fi], value, mask=mask)
        return out

    return kernel


def random_semantic_schedule(schedule_id: int) -> tuple[str, ...]:
    """Frozen broad grammar: variable length and source-semantic vocabulary."""
    rng = random.Random(0x73C1_9E00 + int(schedule_id))
    schedule = [
        rng.choice(("add", "subtract", "multiply", "divide", "maximum"))
        for _ in range(rng.randint(4, 18))
    ]
    schedule.extend(
        rng.choice(("exp", "log", "rsqrt"))
        for _ in range(1 + schedule_id % 2)
    )
    schedule.extend("reduce" for _ in range(1 + (schedule_id % 3 == 0)))
    if schedule_id % 4 == 0:
        schedule.append("where")
    rng.shuffle(schedule)
    return tuple(schedule)


def sequence_randomsemantic2k_factory(schedule_id: int):
    schedule = random_semantic_schedule(schedule_id)

    @nki.jit
    def kernel(x, y, logical_f):
        p, _ = x.shape; tile_f = 2048
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
        mask = (pi < p) & (fi < logical_f)
        value = nl.add(nl.load(x[pi, fi], mask=mask), 2.0)
        for index in nl.static_range(len(schedule)):
            token = schedule[index]
            if token == "add": value = nl.add(value, .1)
            elif token == "subtract": value = nl.subtract(value, .01)
            elif token == "multiply": value = nl.multiply(value, 1.001)
            elif token == "divide": value = nl.divide(value, 1.001)
            elif token == "maximum": value = nl.maximum(value, 0.0)
            elif token == "exp": value = nl.exp(nl.multiply(value, .1))
            elif token == "log": value = nl.log(value)
            elif token == "rsqrt": value = nl.rsqrt(value)
            elif token == "where":
                value = nl.where(
                    nl.greater(value, 0.0), value, nl.multiply(value, -1.0)
                )
            else:
                reduced = nl.sum(value, axis=1, keepdims=True)
                value = nl.add(value, nl.multiply(reduced, .0001))
        nl.store(out[pi, fi], value, mask=mask)
        return out

    return kernel


def random_dag_schedule(schedule_id: int) -> tuple[str, ...]:
    """Frozen generic two-branch DAG motifs, independent of target programs."""
    rng = random.Random(0x4DA6_2100 + int(schedule_id))
    actions = [
        rng.choice(("a_add", "a_multiply", "a_exp", "b_subtract",
                    "b_maximum", "b_rsqrt", "cross_add", "cross_multiply"))
        for _ in range(rng.randint(8, 18))
    ]
    # Guarantee both branch-local work and a source-visible join/reuse motif.
    actions.extend(("a_reduce", "b_reduce", "cross_add"))
    rng.shuffle(actions)
    return tuple(actions)


def sequence_randomdag2k_factory(schedule_id: int):
    schedule = random_dag_schedule(schedule_id)

    @nki.jit
    def kernel(x, y, logical_f):
        p, _ = x.shape; tile_f = 2048
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
        mask = (pi < p) & (fi < logical_f)
        a = nl.add(nl.load(x[pi, fi], mask=mask), 2.0)
        b = nl.multiply(nl.load(y[pi, fi], mask=mask), 1.001)
        # This is factory-time Python unrolling over a frozen tuple.  Using a
        # normal loop also keeps branch identity explicit in the source DAG.
        for action in schedule:
            if action == "a_add": a = nl.add(a, .1)
            elif action == "a_multiply": a = nl.multiply(a, 1.001)
            elif action == "a_exp": a = nl.exp(nl.multiply(a, .1))
            elif action == "b_subtract": b = nl.subtract(b, .01)
            elif action == "b_maximum": b = nl.maximum(b, 0.0)
            elif action == "b_rsqrt": b = nl.rsqrt(b)
            elif action == "a_reduce":
                a = nl.add(a, nl.multiply(nl.sum(a, axis=1, keepdims=True), .0001))
            elif action == "b_reduce":
                b = nl.add(b, nl.multiply(nl.sum(b, axis=1, keepdims=True), .0001))
            elif action == "cross_add": a = nl.add(a, b)
            else: b = nl.multiply(a, b)
        nl.store(out[pi, fi], nl.add(a, b), mask=mask)
        return out

    return kernel


def factorial_dag_schedule(schedule_id: int) -> tuple[str, ...]:
    """Frozen 3x3x2x3 single-factor DAG design (IDs 3000--3053)."""
    index = int(schedule_id) - 3000
    if not 0 <= index < 54:
        raise ValueError("factorial DAG schedule IDs are frozen to 3000--3053")
    join = ("cross_add", "cross_multiply")[index % 2]
    reduction = ("a", "b", "both")[(index // 2) % 3]
    gap = (0, 4, 8)[(index // 6) % 3]
    depth = (2, 4, 8)[(index // 18) % 3]
    actions = []
    for step in range(depth):
        actions.append("a_add" if step % 2 == 0 else "a_multiply")
    if reduction in {"a", "both"}:
        actions.append("a_reduce")
    for step in range(gap):
        actions.append("b_subtract" if step % 2 == 0 else "b_maximum")
    if reduction in {"b", "both"}:
        actions.append("b_reduce")
    actions.append(join)
    actions.extend(("a_exp", "cross_add"))
    return tuple(actions)


def factorial_dag_audit_schedule(schedule_id: int) -> tuple[str, ...]:
    """Frozen unseen audit grid; same factors, reversed branch-local order."""
    index = int(schedule_id) - 4000
    if not 0 <= index < 54:
        raise ValueError("factorial DAG audit IDs are frozen to 4000--4053")
    join = ("cross_add", "cross_multiply")[index % 2]
    reduction = ("a", "b", "both")[(index // 2) % 3]
    gap = (0, 4, 8)[(index // 6) % 3]
    depth = (2, 4, 8)[(index // 18) % 3]
    actions = []
    for step in range(depth):
        actions.append("a_multiply" if step % 2 == 0 else "a_add")
    if reduction in {"a", "both"}:
        actions.append("a_reduce")
    for step in range(gap):
        actions.append("b_maximum" if step % 2 == 0 else "b_subtract")
    if reduction in {"b", "both"}:
        actions.append("b_reduce")
    actions.extend((join, "a_exp", "cross_add"))
    return tuple(actions)


def factorial_dag_interleave_schedule(schedule_id: int) -> tuple[str, ...]:
    """Frozen third-family audit with interleaved branch-local source order."""
    index = int(schedule_id) - 5000
    if not 0 <= index < 54:
        raise ValueError("factorial DAG interleave IDs are frozen to 5000--5053")
    join = ("cross_add", "cross_multiply")[index % 2]
    reduction = ("a", "b", "both")[(index // 2) % 3]
    gap = (0, 4, 8)[(index // 6) % 3]
    depth = (2, 4, 8)[(index // 18) % 3]
    a_ops = ["a_add"] * (depth // 2) + ["a_multiply"] * (depth // 2)
    b_ops = ["b_maximum", "b_subtract"] + [
        "b_subtract" if step % 2 == 0 else "b_maximum" for step in range(gap)
    ]
    actions = []
    for step in range(max(len(a_ops), len(b_ops))):
        if step < len(a_ops):
            actions.append(a_ops[step])
        if step < len(b_ops):
            actions.append(b_ops[step])
    if reduction in {"a", "both"}:
        actions.append("a_reduce")
    if reduction in {"b", "both"}:
        actions.append("b_reduce")
    actions.extend((join, "a_exp", "cross_add"))
    return tuple(actions)


def sequence_factorialdag2k_factory(schedule_id: int, *, audit: bool = False):
    schedule = (
        factorial_dag_audit_schedule(schedule_id)
        if audit else factorial_dag_schedule(schedule_id)
    )

    @nki.jit
    def kernel(x, y, logical_f):
        p, _ = x.shape; tile_f = 2048
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
        mask = (pi < p) & (fi < logical_f)
        a = nl.add(nl.load(x[pi, fi], mask=mask), 2.0)
        b = nl.multiply(nl.load(y[pi, fi], mask=mask), 1.001)
        for index in nl.static_range(len(schedule)):
            action = schedule[index]
            if action == "a_add": a = nl.add(a, .1)
            elif action == "a_multiply": a = nl.multiply(a, 1.001)
            elif action == "a_exp": a = nl.exp(nl.multiply(a, .1))
            elif action == "b_subtract": b = nl.subtract(b, .01)
            elif action == "b_maximum": b = nl.maximum(b, 0.0)
            elif action == "a_reduce":
                a = nl.add(a, nl.multiply(nl.sum(a, axis=1, keepdims=True), .0001))
            elif action == "b_reduce":
                b = nl.add(b, nl.multiply(nl.sum(b, axis=1, keepdims=True), .0001))
            elif action == "cross_add": a = nl.add(a, b)
            else: b = nl.multiply(a, b)
        nl.store(out[pi, fi], nl.add(a, b), mask=mask)
        return out

    return kernel


def sequence_factorialdaginterleave2k_factory(schedule_id: int):
    # Reuse the exact kernel body by selecting the third frozen schedule here.
    schedule = factorial_dag_interleave_schedule(schedule_id)

    @nki.jit
    def kernel(x, y, logical_f):
        p, _ = x.shape; tile_f = 2048
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
        mask = (pi < p) & (fi < logical_f)
        a = nl.add(nl.load(x[pi, fi], mask=mask), 2.0)
        b = nl.multiply(nl.load(y[pi, fi], mask=mask), 1.001)
        for index in nl.static_range(len(schedule)):
            action = schedule[index]
            if action == "a_add": a = nl.add(a, .1)
            elif action == "a_multiply": a = nl.multiply(a, 1.001)
            elif action == "a_exp": a = nl.exp(nl.multiply(a, .1))
            elif action == "b_subtract": b = nl.subtract(b, .01)
            elif action == "b_maximum": b = nl.maximum(b, 0.0)
            elif action == "a_reduce":
                a = nl.add(a, nl.multiply(nl.sum(a, axis=1, keepdims=True), .0001))
            elif action == "b_reduce":
                b = nl.add(b, nl.multiply(nl.sum(b, axis=1, keepdims=True), .0001))
            elif action == "cross_add": a = nl.add(a, b)
            else: b = nl.multiply(a, b)
        nl.store(out[pi, fi], nl.add(a, b), mask=mask)
        return out

    return kernel


@nki.jit
def sequence_deepmixed2k_00_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 0)
@nki.jit
def sequence_deepmixed2k_01_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 1)
@nki.jit
def sequence_deepmixed2k_02_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 2)
@nki.jit
def sequence_deepmixed2k_03_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 3)
@nki.jit
def sequence_deepmixed2k_04_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 4)
@nki.jit
def sequence_deepmixed2k_05_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 5)
@nki.jit
def sequence_deepmixed2k_06_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 6)
@nki.jit
def sequence_deepmixed2k_07_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 7)
@nki.jit
def sequence_deepmixed2k_08_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 8)
@nki.jit
def sequence_deepmixed2k_09_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 9)
@nki.jit
def sequence_deepmixed2k_10_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 10)
@nki.jit
def sequence_deepmixed2k_11_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 11)
@nki.jit
def sequence_deepmixed2k_12_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 12)
@nki.jit
def sequence_deepmixed2k_13_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 13)
@nki.jit
def sequence_deepmixed2k_14_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 14)
@nki.jit
def sequence_deepmixed2k_15_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_deepmixed2k_body(x, out, logical_f, 15)


@nki.jit
def sequence_perm2k_ams_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 0)
@nki.jit
def sequence_perm2k_asm_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 1)
@nki.jit
def sequence_perm2k_mas_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 2)
@nki.jit
def sequence_perm2k_msa_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 3)
@nki.jit
def sequence_perm2k_sam_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 4)
@nki.jit
def sequence_perm2k_sma_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 5)
@nki.jit
def sequence_perm2k_eam_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 6)
@nki.jit
def sequence_perm2k_aem_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 7)
@nki.jit
def sequence_perm2k_mea_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 8)
@nki.jit
def sequence_perm2k_ram_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 9)
@nki.jit
def sequence_perm2k_arm_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 10)
@nki.jit
def sequence_perm2k_mra_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_body(x, out, logical_f, 11)
@nki.jit
def sequence_perm2k_long00_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 0)
@nki.jit
def sequence_perm2k_long01_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 1)
@nki.jit
def sequence_perm2k_long02_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 2)
@nki.jit
def sequence_perm2k_long03_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 3)
@nki.jit
def sequence_perm2k_long04_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 4)
@nki.jit
def sequence_perm2k_long05_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 5)
@nki.jit
def sequence_perm2k_long06_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 6)
@nki.jit
def sequence_perm2k_long07_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 7)
@nki.jit
def sequence_perm2k_long08_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 8)
@nki.jit
def sequence_perm2k_long09_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 9)
@nki.jit
def sequence_perm2k_long10_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 10)
@nki.jit
def sequence_perm2k_long11_kernel(x, y, logical_f):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm); return _sequence_perm2k_long_body(x, out, logical_f, 11)


@nki.jit
def primitive_exp_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    nl.store(out[pi, fi], value=nl.exp(nl.load(x[pi, fi])))
    return out


@nki.jit
def primitive_log_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    nl.store(out[pi, fi], value=nl.log(nl.load(x[pi, fi])))
    return out


@nki.jit
def primitive_rsqrt_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    nl.store(out[pi, fi], value=nl.rsqrt(nl.load(x[pi, fi])))
    return out


@nki.jit
def primitive_reduce_sum_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray((p, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    nl.store(
        out[pi, nl.arange(1)[None, :]],
        value=nl.sum(nl.load(x[pi, fi]), axis=1, keepdims=True),
    )
    return out


@nki.jit
def elementwise_maximum_masked_kernel(x, y):
    p, f = x.shape
    tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    value = nl.maximum(
        nl.load(x[pi, fi], mask=mask),
        0.0,
    )
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def elementwise_multiply_masked_kernel(x, y):
    p, f = x.shape
    tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    value = nl.multiply(nl.load(x[pi, fi], mask=mask), 2.0)
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def elementwise_sigmoid_masked_kernel(x, y):
    p, f = x.shape
    tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    value = nl.sigmoid(nl.load(x[pi, fi], mask=mask))
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def elementwise_maximum_wide_masked_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    nl.store(out[pi, fi], value=nl.maximum(nl.load(x[pi, fi], mask=mask), 0.0, mask=mask), mask=mask)
    return out


@nki.jit
def elementwise_multiply_wide_masked_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    nl.store(out[pi, fi], value=nl.multiply(nl.load(x[pi, fi], mask=mask), 2.0, mask=mask), mask=mask)
    return out


@nki.jit
def elementwise_sigmoid_wide_masked_kernel(x, y):
    p, f = x.shape; tile_f = 16384
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    nl.store(out[pi, fi], value=nl.sigmoid(nl.load(x[pi, fi], mask=mask), mask=mask), mask=mask)
    return out


@nki.jit
def masked_log_reduction_kernel(x, y):
    """Masked log/where/arithmetic/reduction compiler control."""
    p, f = x.shape
    out = nl.ndarray((p, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    lhs = nl.load(x[pi, fi], dtype=nl.float32)
    rhs = nl.load(y[pi, fi], dtype=nl.float32)
    zero = nl.zeros(rhs.shape, dtype=nl.float32, buffer=nl.sbuf)
    safe_log = nl.where(nl.greater(rhs, 0.0), nl.log(rhs), zero)
    term = nl.multiply(rhs, nl.subtract(safe_log, lhs))
    nl.store(out[pi, nl.arange(1)[None, :]], value=nl.sum(term, axis=1, keepdims=True))
    return out


@nki.jit
def softmax_reduction_kernel(x, y):
    """Two-reduction transcendental normalization control."""
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value = nl.load(x[pi, fi], dtype=nl.float32)
    maximum = nl.max(value, axis=1, keepdims=True)
    exponent = nl.exp(nl.subtract(value, maximum))
    total = nl.sum(exponent, axis=1, keepdims=True)
    normalized = nl.divide(exponent, total)
    nl.store(out[pi, fi], value=nl.add(normalized, 0.0, dtype=x.dtype))
    return out


@nki.jit
def elementwise_multiply2_kernel(x, y):
    """Two dependent multiplies: the rmsnorm epilogue's minimal grammar."""
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    nl.store(out[pi, fi], value=nl.multiply(nl.multiply(a, b), b, dtype=x.dtype))
    return out


@nki.jit
def broadcast_multiply2_kernel(x, weight):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a = nl.load(x[pi, fi], dtype=nl.float32)
    w = nl.load(weight[nl.arange(1)[:, None], fi], dtype=nl.float32)
    wb = nl.broadcast_to(w, shape=(p, f))
    nl.store(out[pi, fi], value=nl.multiply(nl.multiply(a, 1.0), wb, dtype=x.dtype))
    return out


@nki.jit
def broadcast_affine_kernel(x, weight, bias):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a = nl.load(x[pi, fi], dtype=nl.float32)
    wi = nl.arange(1)[:, None]
    w = nl.broadcast_to(nl.load(weight[wi, fi], dtype=nl.float32), shape=(p, f))
    b = nl.broadcast_to(nl.load(bias[wi, fi], dtype=nl.float32), shape=(p, f))
    value = nl.add(nl.multiply(nl.multiply(nl.subtract(a, 0.0), 1.0), w), b, dtype=x.dtype)
    nl.store(out[pi, fi], value=value)
    return out


@nki.jit
def two_pass_reduce_affine_kernel(x, weight, bias):
    """Generic reduce/statistics pass followed by reload+broadcast affine."""
    p, f = x.shape; tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    if f > tile_f:
        pi = nl.arange(p)[:, None]; sx = nl.zeros((p,1),dtype=nl.float32,buffer=nl.sbuf); sx2=nl.zeros((p,1),dtype=nl.float32,buffer=nl.sbuf)
        for block in nl.static_range((f+tile_f-1)//tile_f):
            fi=nl.arange(tile_f)[None,:];mask=(pi<p)&(fi<f-block*tile_f)
            a=nl.load(x[pi,block*tile_f+fi],mask=mask,dtype=nl.float32);zero=nl.zeros(a.shape,dtype=nl.float32,buffer=nl.sbuf);safe=nl.where(mask,a,zero)
            sx[...]=nl.add(sx,nl.sum(safe,axis=1,keepdims=True));sx2[...]=nl.add(sx2,nl.sum(nl.multiply(safe,safe),axis=1,keepdims=True))
        mean=nl.divide(sx,float(f));var=nl.subtract(nl.divide(sx2,float(f)),nl.multiply(mean,mean));val=nl.add(var,1e-5,dtype=nl.float32)
        half=nl.multiply(val,.5);root=nl.rsqrt(val);root=nl.multiply(root,nl.subtract(1.5,nl.multiply(half,nl.multiply(root,root))));root=nl.multiply(root,nl.subtract(1.5,nl.multiply(half,nl.multiply(root,root))))
        for block in nl.static_range((f+tile_f-1)//tile_f):
            fi=nl.arange(tile_f)[None,:];mask=(pi<p)&(fi<f-block*tile_f);wi=nl.arange(1)[:,None]
            a=nl.load(x[pi,block*tile_f+fi],mask=mask,dtype=nl.float32)
            w=nl.broadcast_to(nl.load(weight[wi,block*tile_f+fi],mask=(fi<f-block*tile_f),dtype=nl.float32),shape=(p,tile_f));b=nl.broadcast_to(nl.load(bias[wi,block*tile_f+fi],mask=(fi<f-block*tile_f),dtype=nl.float32),shape=(p,tile_f))
            nl.store(out[pi,block*tile_f+fi],value=nl.add(nl.multiply(nl.multiply(nl.subtract(a,mean),root),w),b,dtype=x.dtype),mask=mask)
        return out
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    zero = nl.zeros(a.shape, dtype=nl.float32, buffer=nl.sbuf); safe = nl.where(mask, a, zero)
    sx = nl.zeros((p, 1), dtype=nl.float32, buffer=nl.sbuf); sx[...] = nl.add(sx, nl.sum(safe, axis=1, keepdims=True))
    sx2 = nl.zeros((p, 1), dtype=nl.float32, buffer=nl.sbuf); sx2[...] = nl.add(sx2, nl.sum(nl.multiply(safe, safe), axis=1, keepdims=True))
    mean = nl.divide(sx, float(f)); var = nl.subtract(nl.divide(sx2, float(f)), nl.multiply(mean, mean))
    val = nl.add(var, 1e-5, dtype=nl.float32); half = nl.multiply(val, .5); root = nl.rsqrt(val)
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    a = nl.load(x[pi, fi], mask=mask, dtype=nl.float32); wi = nl.arange(1)[:, None]
    w = nl.broadcast_to(nl.load(weight[wi, fi], mask=(fi < f), dtype=nl.float32), shape=(p, tile_f))
    b = nl.broadcast_to(nl.load(bias[wi, fi], mask=(fi < f), dtype=nl.float32), shape=(p, tile_f))
    value = nl.add(nl.multiply(nl.multiply(nl.subtract(a, mean), root), w), b, dtype=x.dtype)
    nl.store(out[pi, fi], value=value, mask=mask); return out


@nki.jit
def two_pass_reduce_multiply_kernel(x, weight):
    """Generic one-reduction statistics pass followed by broadcast multiply."""
    p, f = x.shape; out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    if f > 2048:
        tile = 2048; pi = nl.arange(p)[:, None]; total = nl.zeros((p, 1), dtype=nl.float32, buffer=nl.sbuf)
        for block in nl.static_range((f + tile - 1) // tile):
            fi = nl.arange(tile)[None, :]; mask = (pi < p) & (fi < f - block * tile)
            a = nl.load(x[pi, block * tile + fi], mask=mask, dtype=nl.float32)
            zero = nl.zeros(a.shape, dtype=nl.float32, buffer=nl.sbuf)
            total[...] = nl.add(total, nl.sum(nl.where(mask, nl.multiply(a, a), zero), axis=1, keepdims=True))
        mean_sq = nl.divide(total, float(f)); val = nl.add(mean_sq, 1e-6, dtype=nl.float32)
        half = nl.multiply(val, .5); root = nl.rsqrt(val)
        root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
        root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
        for block in nl.static_range((f + tile - 1) // tile):
            fi = nl.arange(tile)[None, :]; mask = (pi < p) & (fi < f - block * tile)
            a = nl.load(x[pi, block * tile + fi], mask=mask, dtype=nl.float32)
            w = nl.broadcast_to(nl.load(weight[nl.arange(1)[:, None], block * tile + fi], mask=(fi < f - block * tile), dtype=nl.float32), shape=(p, tile))
            nl.store(out[pi, block * tile + fi], value=nl.multiply(nl.multiply(a, root), w, dtype=x.dtype), mask=mask)
        return out
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    mean_sq = nl.divide(nl.sum(nl.multiply(a, a), axis=1, keepdims=True), float(f))
    val = nl.add(mean_sq, 1e-6, dtype=nl.float32); half = nl.multiply(val, .5); root = nl.rsqrt(val)
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    a = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    w = nl.broadcast_to(nl.load(weight[nl.arange(1)[:, None], fi], mask=(fi < f), dtype=nl.float32), shape=(p, f))
    nl.store(out[pi, fi], value=nl.multiply(nl.multiply(a, root), w, dtype=x.dtype), mask=mask); return out


@nki.jit
def reduce_broadcast_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    reduced = nl.sum(nl.multiply(a, b), axis=1, keepdims=True)
    nl.store(out[pi, fi], value=nl.multiply(a, reduced, dtype=x.dtype))
    return out


@nki.jit
def two_reductions_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    first = nl.sum(a, axis=1, keepdims=True)
    second = nl.sum(nl.multiply(a, a), axis=1, keepdims=True)
    value = nl.add(nl.multiply(a, first), nl.multiply(b, second), dtype=x.dtype)
    nl.store(out[pi, fi], value=value)
    return out


@nki.jit
def rsqrt_newton_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a = nl.load(x[pi, fi], dtype=nl.float32)
    reduced = nl.sum(nl.multiply(a, a), axis=1, keepdims=True)
    val = nl.add(nl.divide(reduced, float(f)), 1e-6, dtype=nl.float32)
    half, root = nl.multiply(val, 0.5), nl.rsqrt(val)
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    nl.store(out[pi, fi], value=nl.multiply(a, root, dtype=x.dtype))
    return out


@nki.jit
def two_reductions_rsqrt_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi], dtype=nl.float32), nl.load(y[pi, fi], dtype=nl.float32)
    mean = nl.divide(nl.sum(a, axis=1, keepdims=True), float(f))
    mean_sq = nl.divide(nl.sum(nl.multiply(a, a), axis=1, keepdims=True), float(f))
    val = nl.add(nl.subtract(mean_sq, nl.multiply(mean, mean)), 1e-5)
    half, root = nl.multiply(val, 0.5), nl.rsqrt(val)
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    root = nl.multiply(root, nl.subtract(1.5, nl.multiply(half, nl.multiply(root, root))))
    nl.store(out[pi, fi], value=nl.multiply(nl.subtract(a, mean), root, dtype=x.dtype))
    return out


@nki.jit
def elementwise_mixed_kernel(x, y):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    a, b = nl.load(x[pi, fi]), nl.load(y[pi, fi])
    value = nl.add(nl.multiply(nl.multiply(nl.subtract(a, b), b), a), b, dtype=x.dtype)
    nl.store(out[pi, fi], value=value)
    return out


@nki.jit
def elementwise_mixed_masked_kernel(x, y):
    p, f = x.shape; tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    a, b = nl.load(x[pi, fi], mask=mask), nl.load(y[pi, fi], mask=mask)
    value = nl.add(nl.multiply(nl.multiply(nl.subtract(a, b), b), a), b, dtype=x.dtype)
    nl.store(out[pi, fi], value=value, mask=mask); return out


@nki.jit
def two_reductions_rsqrt_masked_kernel(x, y):
    p, f = x.shape; tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]; mask = (pi < p) & (fi < f)
    a = nl.load(x[pi, fi], mask=mask, dtype=nl.float32); zero=nl.zeros(a.shape,dtype=nl.float32,buffer=nl.sbuf)
    safe=nl.where(mask,a,zero)
    sum_x=nl.zeros((p,1),dtype=nl.float32,buffer=nl.sbuf);sum_x[...]=nl.add(sum_x,nl.sum(safe,axis=1,keepdims=True))
    sum_x2=nl.zeros((p,1),dtype=nl.float32,buffer=nl.sbuf);sum_x2[...]=nl.add(sum_x2,nl.sum(nl.multiply(safe,safe),axis=1,keepdims=True))
    mean=nl.divide(sum_x,float(f));mean_sq=nl.divide(sum_x2,float(f))
    val=nl.add(nl.subtract(mean_sq,nl.multiply(mean,mean)),1e-5);half,root=nl.multiply(val,.5),nl.rsqrt(val)
    root=nl.multiply(root,nl.subtract(1.5,nl.multiply(half,nl.multiply(root,root))))
    root=nl.multiply(root,nl.subtract(1.5,nl.multiply(half,nl.multiply(root,root))))
    nl.store(out[pi,fi],value=nl.multiply(nl.subtract(safe,mean),root,dtype=x.dtype),mask=mask);return out


@nki.jit
def mask_tail_kernel(x, y):
    p, f = x.shape
    tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(tile_f)[None, :]
    mask = (pi < p) & (fi < f)
    a, b = nl.load(x[pi, fi], mask=mask), nl.load(y[pi, fi], mask=mask)
    zero = nl.zeros(a.shape, dtype=x.dtype, buffer=nl.sbuf)
    nl.store(out[pi, fi], value=nl.where(mask, nl.add(a, b), zero), mask=mask)
    return out


def _padded_partition_indices(x):
    """PMAX physical tile with a source-visible logical-row mask."""
    logical_p, f = x.shape
    pi = nl.arange(nl.tile_size.pmax)[:, None]
    fi = nl.arange(f)[None, :]
    return logical_p, f, pi, fi, pi < logical_p


@nki.jit
def padded_add_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.add(nl.load(x[pi, fi], mask=mask), 0.25)
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_multiply_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.multiply(nl.load(x[pi, fi], mask=mask), 1.125)
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_sigmoid_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.sigmoid(nl.load(x[pi, fi], mask=mask))
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_mixed_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    a = nl.load(x[pi, fi], mask=mask)
    b = nl.load(y[pi, fi], mask=mask)
    value = nl.add(nl.multiply(nl.subtract(a, b), b), a)
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_reduce_affine_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    reduced = nl.sum(value, axis=1, keepdims=True)
    value = nl.add(value, nl.multiply(reduced, 0.001))
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_reduce_transcendental_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    value = nl.exp(nl.multiply(value, 0.125))
    reduced = nl.sum(value, axis=1, keepdims=True)
    value = nl.divide(value, nl.add(reduced, 1.0))
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_reduce_pair_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    first = nl.sum(value, axis=1, keepdims=True)
    second = nl.sum(nl.multiply(value, value), axis=1, keepdims=True)
    value = nl.add(value, nl.multiply(nl.add(first, second), 0.0001))
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_reduce_rsqrt_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    reduced = nl.sum(nl.multiply(value, value), axis=1, keepdims=True)
    scale = nl.rsqrt(nl.add(reduced, 1.0))
    nl.store(out[pi, fi], value=nl.multiply(value, scale), mask=mask)
    return out


@nki.jit
def padded_maximum_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.maximum(nl.load(x[pi, fi], mask=mask), 0.0)
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


@nki.jit
def padded_reduce_maximum_kernel(x, y):
    logical_p, _, pi, fi, mask = _padded_partition_indices(x)
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    value = nl.load(x[pi, fi], mask=mask, dtype=nl.float32)
    reduced = nl.sum(value, axis=1, keepdims=True)
    value = nl.maximum(nl.subtract(value, reduced), 0.0)
    nl.store(out[pi, fi], value=value, mask=mask)
    return out


def padded_randomdag_factory(schedule_id: int):
    """Target-independent random DAG in a PMAX-padded physical context."""
    schedule = random_dag_schedule(schedule_id)

    @nki.jit
    def kernel(x, y):
        logical_p, _, pi, fi, mask = _padded_partition_indices(x)
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        a = nl.add(nl.load(x[pi, fi], mask=mask), 2.0)
        b = nl.multiply(nl.load(y[pi, fi], mask=mask), 1.001)
        for action in schedule:
            if action == "a_add": a = nl.add(a, .1)
            elif action == "a_multiply": a = nl.multiply(a, 1.001)
            elif action == "a_exp": a = nl.exp(nl.multiply(a, .1))
            elif action == "b_subtract": b = nl.subtract(b, .01)
            elif action == "b_maximum": b = nl.maximum(b, 0.0)
            elif action == "b_rsqrt": b = nl.rsqrt(b)
            elif action == "a_reduce":
                a = nl.add(a, nl.multiply(nl.sum(a, axis=1, keepdims=True), .0001))
            elif action == "b_reduce":
                b = nl.add(b, nl.multiply(nl.sum(b, axis=1, keepdims=True), .0001))
            elif action == "cross_add": a = nl.add(a, b)
            else: b = nl.multiply(a, b)
        nl.store(out[pi, fi], value=nl.add(a, b), mask=mask)
        return out

    return kernel


@nki.jit
def multi_block_kernel(x, y):
    p, f = x.shape
    tile_f = 2048
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    pi = nl.arange(p)[:, None]
    for block in nl.static_range((f + tile_f - 1) // tile_f):
        fi = nl.arange(tile_f)[None, :]
        mask = fi < (f - block * tile_f)
        a = nl.load(x[pi, block * tile_f + fi], mask=mask)
        b = nl.load(y[pi, block * tile_f + fi], mask=mask)
        nl.store(out[pi, block * tile_f + fi], value=nl.add(a, b), mask=mask)
    return out


@nki.jit
def elementwise_one_bf16_kernel(x, y, chain):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value = nl.load(x[pi, fi], dtype=nl.bfloat16)
    for index in nl.static_range(chain):
        value = nl.add(value, float(index + 1), dtype=nl.bfloat16)
    nl.store(out[pi, fi], value=value)
    return out


@nki.jit
def elementwise_two_bf16_kernel(x, y, chain):
    p, f = x.shape
    out = nl.ndarray(x.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
    pi, fi = nl.arange(p)[:, None], nl.arange(f)[None, :]
    value = nl.load(x[pi, fi], dtype=nl.bfloat16)
    rhs = nl.load(y[pi, fi], dtype=nl.bfloat16)
    for _ in nl.static_range(chain):
        value = nl.add(value, rhs, dtype=nl.bfloat16)
    nl.store(out[pi, fi], value=value)
    return out


KERNELS = {
    "elementwise_one": elementwise_one_kernel,
    "elementwise_two": elementwise_two_kernel,
    "elementwise_maximum": elementwise_maximum_kernel,
    "elementwise_multiply": elementwise_multiply_kernel,
    "elementwise_sigmoid": elementwise_sigmoid_kernel,
    "primitive_divide": primitive_divide_kernel,
    "primitive_add": primitive_add_kernel,
    "primitive_subtract": primitive_subtract_kernel,
    "primitive_multiply": primitive_multiply_kernel,
    "primitive_where_bundle": primitive_where_bundle_kernel,
    "sequence_add_multiply": sequence_add_multiply_kernel,
    "sequence_multiply_add": sequence_multiply_add_kernel,
    "sequence_subtract_multiply_add": sequence_subtract_multiply_add_kernel,
    "sequence_multiply_subtract_multiply": sequence_multiply_subtract_multiply_kernel,
    "sequence_exp_multiply_add": sequence_exp_multiply_add_kernel,
    "sequence_log_add_multiply": sequence_log_add_multiply_kernel,
    "sequence_reduce_add_multiply": sequence_reduce_add_multiply_kernel,
    "sequence_reduce_divide_multiply": sequence_reduce_divide_multiply_kernel,
    "sequence_two_reduce_add": sequence_two_reduce_add_kernel,
    "sequence_two_reduce_multiply": sequence_two_reduce_multiply_kernel,
    "sequence_rsqrt_multiply_add": sequence_rsqrt_multiply_add_kernel,
    "sequence_add_multiply_wide_memory": sequence_add_multiply_wide_memory_kernel,
    "sequence_subtract_multiply_add_wide_memory": sequence_subtract_multiply_add_wide_memory_kernel,
    "sequence_exp_multiply_add_wide_memory": sequence_exp_multiply_add_wide_memory_kernel,
    "sequence_reduce_add_multiply_wide_memory": sequence_reduce_add_multiply_wide_memory_kernel,
    "sequence_two_reduce_multiply_wide_memory": sequence_two_reduce_multiply_wide_memory_kernel,
    "sequence_rsqrt_multiply_add_wide_memory": sequence_rsqrt_multiply_add_wide_memory_kernel,
    "sequence_multiply_add_wide_memory": sequence_multiply_add_wide_memory_kernel,
    "sequence_multiply_subtract_multiply_wide_memory": sequence_multiply_subtract_multiply_wide_memory_kernel,
    "sequence_log_add_multiply_wide_memory": sequence_log_add_multiply_wide_memory_kernel,
    "sequence_reduce_divide_multiply_wide_memory": sequence_reduce_divide_multiply_wide_memory_kernel,
    "sequence_two_reduce_add_wide_memory": sequence_two_reduce_add_wide_memory_kernel,
    "sequence_perm2k_ams": sequence_perm2k_ams_kernel,
    "sequence_perm2k_asm": sequence_perm2k_asm_kernel,
    "sequence_perm2k_mas": sequence_perm2k_mas_kernel,
    "sequence_perm2k_msa": sequence_perm2k_msa_kernel,
    "sequence_perm2k_sam": sequence_perm2k_sam_kernel,
    "sequence_perm2k_sma": sequence_perm2k_sma_kernel,
    "sequence_perm2k_eam": sequence_perm2k_eam_kernel,
    "sequence_perm2k_aem": sequence_perm2k_aem_kernel,
    "sequence_perm2k_mea": sequence_perm2k_mea_kernel,
    "sequence_perm2k_ram": sequence_perm2k_ram_kernel,
    "sequence_perm2k_arm": sequence_perm2k_arm_kernel,
    "sequence_perm2k_mra": sequence_perm2k_mra_kernel,
    **{f"sequence_perm2k_long{i:02d}": kernel for i, kernel in enumerate((
        sequence_perm2k_long00_kernel, sequence_perm2k_long01_kernel,
        sequence_perm2k_long02_kernel, sequence_perm2k_long03_kernel,
        sequence_perm2k_long04_kernel, sequence_perm2k_long05_kernel,
        sequence_perm2k_long06_kernel, sequence_perm2k_long07_kernel,
        sequence_perm2k_long08_kernel, sequence_perm2k_long09_kernel,
        sequence_perm2k_long10_kernel, sequence_perm2k_long11_kernel,
    ))},
    "sequence_deep2k_add": sequence_deep2k_add_kernel,
    "sequence_deep2k_multiply": sequence_deep2k_multiply_kernel,
    "sequence_deep2k_add_multiply": sequence_deep2k_add_multiply_kernel,
    **{f"sequence_deepmixed2k_{i:02d}": kernel for i, kernel in enumerate((
        sequence_deepmixed2k_00_kernel, sequence_deepmixed2k_01_kernel,
        sequence_deepmixed2k_02_kernel, sequence_deepmixed2k_03_kernel,
        sequence_deepmixed2k_04_kernel, sequence_deepmixed2k_05_kernel,
        sequence_deepmixed2k_06_kernel, sequence_deepmixed2k_07_kernel,
        sequence_deepmixed2k_08_kernel, sequence_deepmixed2k_09_kernel,
        sequence_deepmixed2k_10_kernel, sequence_deepmixed2k_11_kernel,
        sequence_deepmixed2k_12_kernel, sequence_deepmixed2k_13_kernel,
        sequence_deepmixed2k_14_kernel, sequence_deepmixed2k_15_kernel,
    ))},
    "primitive_exp": primitive_exp_kernel,
    "primitive_log": primitive_log_kernel,
    "primitive_rsqrt": primitive_rsqrt_kernel,
    "primitive_reduce_sum": primitive_reduce_sum_kernel,
    "elementwise_maximum_masked": elementwise_maximum_masked_kernel,
    "elementwise_multiply_masked": elementwise_multiply_masked_kernel,
    "elementwise_sigmoid_masked": elementwise_sigmoid_masked_kernel,
    "elementwise_maximum_wide_masked": elementwise_maximum_wide_masked_kernel,
    "elementwise_multiply_wide_masked": elementwise_multiply_wide_masked_kernel,
    "elementwise_sigmoid_wide_masked": elementwise_sigmoid_wide_masked_kernel,
    "masked_log_reduction": masked_log_reduction_kernel,
    "softmax_reduction": softmax_reduction_kernel,
    "elementwise_multiply2": elementwise_multiply2_kernel,
    "broadcast_multiply2": broadcast_multiply2_kernel,
    "broadcast_affine": broadcast_affine_kernel,
    "two_pass_reduce_affine": two_pass_reduce_affine_kernel,
    "two_pass_reduce_multiply": two_pass_reduce_multiply_kernel,
    "reduce_broadcast": reduce_broadcast_kernel,
    "two_reductions": two_reductions_kernel,
    "rsqrt_newton": rsqrt_newton_kernel,
    "two_reductions_rsqrt": two_reductions_rsqrt_kernel,
    "elementwise_mixed": elementwise_mixed_kernel,
    "elementwise_mixed_masked": elementwise_mixed_masked_kernel,
    "two_reductions_rsqrt_masked": two_reductions_rsqrt_masked_kernel,
    "mask_tail": mask_tail_kernel,
    "padded_add": padded_add_kernel,
    "padded_multiply": padded_multiply_kernel,
    "padded_sigmoid": padded_sigmoid_kernel,
    "padded_mixed": padded_mixed_kernel,
    "padded_reduce_affine": padded_reduce_affine_kernel,
    "padded_reduce_transcendental": padded_reduce_transcendental_kernel,
    "padded_reduce_pair": padded_reduce_pair_kernel,
    "padded_reduce_rsqrt": padded_reduce_rsqrt_kernel,
    "padded_maximum": padded_maximum_kernel,
    "padded_reduce_maximum": padded_reduce_maximum_kernel,
}


def region_control_factory(*, kind: str, p: int, f: int, chain: int = 1,
                           dtype_name: str = "float32", tile_f: int = 2048):
    if f > tile_f and kind not in {"two_pass_reduce_affine", "two_pass_reduce_multiply"} and not kind.startswith("padded_"):
        return multi_block_kernel, [(p, f), (p, f)], []
    extras = [chain] if kind in {"elementwise_one", "elementwise_two"} else []
    if kind == "sequence_randommixed2k":
        return sequence_randommixed2k_factory(chain), [(p, 2048), (p, 2048)], [f]
    if kind == "sequence_randomsemantic2k":
        return sequence_randomsemantic2k_factory(chain), [(p, 2048), (p, 2048)], [f]
    if kind == "sequence_randomdag2k":
        return sequence_randomdag2k_factory(chain), [(p, 2048), (p, 2048)], [f]
    if kind == "padded_randomdag":
        return padded_randomdag_factory(chain), [(p, f), (p, f)], []
    if kind == "sequence_factorialdag2k":
        return sequence_factorialdag2k_factory(chain), [(p, 2048), (p, 2048)], [f]
    if kind == "sequence_factorialdagaudit2k":
        return sequence_factorialdag2k_factory(chain, audit=True), [(p, 2048), (p, 2048)], [f]
    if kind == "sequence_factorialdaginterleave2k":
        return sequence_factorialdaginterleave2k_factory(chain), [(p, 2048), (p, 2048)], [f]
    if kind.startswith("sequence_deep2k_"):
        return KERNELS[kind], [(p, 2048), (p, 2048)], [f, chain]
    if kind.startswith("sequence_deepmixed2k_"):
        return KERNELS[kind], [(p, 2048), (p, 2048)], [f]
    if kind.startswith("sequence_perm2k_"):
        return KERNELS[kind], [(p, 2048), (p, 2048)], [f]
    if kind in {"broadcast_multiply2", "broadcast_affine", "two_pass_reduce_affine", "two_pass_reduce_multiply"}:
        shapes = [(p, f), (1, f)] + ([(1, f)] if kind in {"broadcast_affine", "two_pass_reduce_affine"} else [])
    else:
        shapes = [(p, f), (p, f)]
    if dtype_name == "bfloat16" and kind in {"elementwise_one", "elementwise_two"}:
        selected = elementwise_one_bf16_kernel if kind == "elementwise_one" else elementwise_two_bf16_kernel
    else:
        selected = KERNELS[kind]
    return selected, shapes, extras
