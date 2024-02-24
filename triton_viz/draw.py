from chalk import *
from colour import Color
from triton_viz.data import *
from .interpreter import record_builder
import sys
import numpy as np
import triton
import planar
sys.setrecursionlimit(15000000)
planar.EPSILON = 0.000000000000000000

def box(d: Diagram, x, y):
    h, w = d.get_envelope().height, d.get_envelope().width
    m = max(h, w)
    d = rectangle(1.1 * m, 1.1 * m).fill_color(Color("white")).center_xy() + d.center_xy()
    d = d.scale(1 / m)
    # if d.get_envelope().height * y > d.get_envelope().width * x:
    #     d = d.scale(1 / d.get_envelope().height)
    # else:
    #     d = d.scale(1 / d.get_envelope().width)
    return d

def reshape(d, max_ratio):
    h, w = d.get_envelope().height, d.get_envelope().width
    if h / w > max_ratio:
        d = d.scale_y(max_ratio / (h / w))
    else:
        d = d.scale_x(max_ratio / (w / h))
    return d

def draw(output: str):
    for record in record_builder.launches:
        diagram = draw_launch(record)
    diagram.render(output, 500)

def draw_launch(launch: Launch):

    tensor_table = {}
    for t in launch.tensors:
        tensor_table[t.ptr[0]] = t

    def draw(x):
        if isinstance(x, Tensor):
            return draw_tensor(x)
        if isinstance(x, Grid):
            return draw_grid(x)
        if isinstance(x, Store):
            return draw_store(x)
        if isinstance(x, Load):
            return draw_load(x, tensor_table)
        if isinstance(x, BinaryOps):
            return draw_binary_op(x)
    def draw_record(x):
        print(x)
        y = draw(x).center_xy()
        if isinstance(x, Load):
            return box(y, 2, 1)
        else:
            return y

    d = vcat([draw_record(r) for r in launch.records])
    return d


def base_tensor(shape : Tuple):
    d = empty()
    if len(shape) == 2:
        d = rectangle(*shape)
    elif len(shape) == 1:
        #if shape[0] <= 64:
        d = rectangle(shape[0], 1)
        #else:
        #    d = hcat([rectangle(64, 1) for _ in range(shape[0] // 64)])
    else:
        assert "bad shape", shape
    return d.align_tl()

def delinearize(shape: Tuple, x: Tensor, dtype):
    x = x.copy() // (dtype.element_ty.primitive_bitwidth // 8)
    vals = []
    for s in shape[1:] + (10000,):
        vals.append((x % s).ravel())
        x = x // s
    if len(shape) == 1:
        vals = [np.zeros(len(vals[0]))] + vals
    return vals

def cover(shape: Tuple, dtype, load: Tensor, color):

    y, x = delinearize(shape, load, dtype)
    r = rectangle(1, 1).line_width(0).fill_color(color).align_tl()
    return concat([
        r.translate(x1, y1)
        for x1, y1 in zip(x.tolist(), y.tolist())])

def mask(shape: Tuple, mask: Tensor, color):
    if len(mask.shape) == 1:
        mask = mask.reshape(-1, 1)
    r = rectangle(1, 1).line_width(0).fill_color(color).align_tl()
    x, y = mask.nonzero()
    return concat([r.translate(x, y) 
                   for x, y in zip(x.tolist(), y.tolist())]) 

def draw_tensor(x: Tensor):
    print("tensor", x)
    return empty()

def draw_grid(x: Grid):
    print("grid", x)
    return empty()

def draw_store(x: Store):
    print("store", x)
    return empty()

def draw_load(x: Load, tensor_table):
    tensor: Tensor = tensor_table[x.ptr[0]]
    inp = base_tensor(tensor.shape) + cover(tensor.shape, tensor.dtype, x.offsets, Color("red"))
    inp = reshape(inp, 5/1)
    out = x.offsets.shape
    out = base_tensor(out) + mask(out, x.masks, Color("blue"))
    out = reshape(out, 5/1)
    return hcat([box(inp, 1, 1), 
                 box(out, 1, 1)], 0.1)

def draw_binary_op(x: BinaryOps):
    print("bin op", x)
    return empty()


