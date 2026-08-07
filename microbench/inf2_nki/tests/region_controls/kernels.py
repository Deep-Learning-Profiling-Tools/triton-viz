"""Minimal top-level nl.* controls for compositional Level-A calibration."""
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
}


def region_control_factory(*, kind: str, p: int, f: int, chain: int = 1,
                           dtype_name: str = "float32", tile_f: int = 2048):
    if f > tile_f and kind not in {"two_pass_reduce_affine", "two_pass_reduce_multiply"}:
        return multi_block_kernel, [(p, f), (p, f)], []
    extras = [chain] if kind in {"elementwise_one", "elementwise_two"} else []
    if kind in {"broadcast_multiply2", "broadcast_affine", "two_pass_reduce_affine", "two_pass_reduce_multiply"}:
        shapes = [(p, f), (1, f)] + ([(1, f)] if kind in {"broadcast_affine", "two_pass_reduce_affine"} else [])
    else:
        shapes = [(p, f), (p, f)]
    if dtype_name == "bfloat16" and kind in {"elementwise_one", "elementwise_two"}:
        selected = elementwise_one_bf16_kernel if kind == "elementwise_one" else elementwise_two_bf16_kernel
    else:
        selected = KERNELS[kind]
    return selected, shapes, extras
