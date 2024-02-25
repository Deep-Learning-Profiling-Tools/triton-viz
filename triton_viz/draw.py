from chalk import *
from colour import Color
from triton_viz.data import *
from .interpreter import record_builder
import sys
import numpy as np
import triton
import planar
import math
import chalk
sys.setrecursionlimit(15000000)
planar.EPSILON = 0.000000000000000000
chalk.set_svg_draw_height(500)
def box(d: Diagram, x, y, width=0):
    h, w = d.get_envelope().height, d.get_envelope().width
    m = max(h, w)
    d = rectangle(0.2 + m, 0.2 + m).line_width(width).fill_color(Color("white")).center_xy() + d.center_xy()
    d = d.scale_x(x / m).scale_y(y / m)
    # if d.get_envelope().height * y > d.get_envelope().width * x:
    #     d = d.scale(1 / d.get_envelope().height)
    # else:
    #     d = d.scale(1 / d.get_envelope().width)
    return d

def reshape(d, max_ratio):
    h, w = d.get_envelope().height, d.get_envelope().width
    
    if h / w > max_ratio:
        d = d.scale_y(math.log(h+1, 2) / h).scale_x(math.log(w+1, 2)/ w)
    elif w / h > max_ratio:
        d = d.scale_y(math.log(h+1, 2) / h).scale_x(math.log(w+1, 2) / w)
    return d

def draw(output: str):
    for record in record_builder.launches[-1:]:
        diagram : Diagram = draw_launch(record)
    diagram.render(output, 2500)
    diagram.render_svg(output + ".svg", 2500, draw_height=400)

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
            return draw_store(x, tensor_table)
        if isinstance(x, Load):
            return draw_load(x, tensor_table)
        if isinstance(x, BinaryOps):
            return draw_binary_op(x)
        if isinstance(x, MakeRange):
            return draw_make_range(x)
        if isinstance(x, Reduce):
            return draw_reduce(x)
        if isinstance(x, Dot):
            return draw_dot(x)
    def draw_record(x):
        print(x)
        y = draw(x)
        if y is None:
            return empty()
        
        return y.center_xy() #box(y.center_xy(), 3, 1, 0.1)
        

    d = vcat([draw_record(r) for r in launch.records], 0.2)
    return rectangle(1.1 * d.get_envelope().width, 1.1 * d.get_envelope().height).fill_color(Color("white")).center_xy() + d.center_xy()


def base_tensor(shape : Tuple, color = Color("green")):
    d = empty()
    if len(shape) == 2:
        d = rectangle(shape[1], shape[0])
    elif len(shape) == 1:
        #if shape[0] <= 64:
        shape = (shape[0], 1)
        d = rectangle(*shape)
        #else:
        #    d = hcat([rectangle(64, 1) for _ in range(shape[0] // 64)])
    else:
        assert "bad shape", shape
    d = d.align_tl()
    return d.fill_color(color)

def add_whiskers(d, shape: Tuple):
    w, h = d.get_envelope().width, d.get_envelope().height
    if len(shape) == 1:
        shape = (1, shape[0])
    step0, step1 = 1, 1
    delta0, delta1 = 1, 1
    if shape[0] > 128:
        step0 = 16
        delta0 = 4
    if shape[1] > 128:
        step1 = 16
        delta1 = 4
    whisker = rectangle(0.01 * delta1, 0.5).fill_color(Color("black")).line_width(0)
    d = vcat([concat([whisker.translate(w * ((i + 0.5)/ shape[1]), 0) 
                      for i in range(0, shape[1], step1)]),  d], 0.1)
    whisker = rectangle(0.5, 0.01*delta0).fill_color(Color("black")).line_width(0)
    d = hcat([concat([whisker.translate(0.0, h * ((i + 0.5)/ shape[0])) 
                      for i in range(0, shape[0], step0)]).align_b().translate(0, -0.5 * (h / shape[0])),  d.align_b()], 0.1)
    return d.align_l()

def delinearize(shape: Tuple, x: Tensor, dtype, mask):
    if len(shape) == 1:
        shape = (1, shape[0])
    x = x.copy() // (dtype.element_ty.primitive_bitwidth // 8)
    vals = []
    for s in shape[1:] + (10000,):
        vals.append(((x % s) * mask - (1 - mask)).ravel())
        x = x // s
    if len(shape) == 1:
        vals = [np.zeros(len(vals[0]))] + vals
    return vals

def cover(shape: Tuple, dtype, load: Tensor, mask, color):
    x, y = delinearize(shape, load, dtype, mask)
    r = rectangle(1, 1).line_width(0).fill_color(color).align_tl()
    d = empty()
    for x1, y1 in zip(x.tolist(), y.tolist()):
        if x1 == -1 and y1 == -1: continue
        d = d + r.translate(x1, y1)
    return d


def mask(shape: Tuple, mask: Tensor, color):
    if len(mask.shape) == 1:
        mask = mask.reshape(1, -1)
    r = rectangle(1, 1).line_width(0).fill_color(color).align_tl()
    y, x = mask.nonzero()
    d = empty()
    for x, y in zip(x.tolist(), y.tolist()):
        d = d + r.translate(x, y) 
    return d

def draw_tensor(x: Tensor):
    return None

def draw_grid(x: Grid):
    return text("Program: " + ", ".join(map(str, x.idx)), 0.3).fill_color(Color("black")).line_width(0) + rectangle(3,1).line_width(0)

def draw_make_range(x: MakeRange):
    return None

def pair_draw(x, y, command, shape=True):
    if shape:
        x = reshape(x, 3/1)
        y = reshape(y, 3/1)
    return hcat([box(x, 3, 2.5), 
                 box(y, 3, 2.5)], 1).center_xy() + text(command, 0.2).fill_color(Color("black")).line_width(0).translate(0, -1)

def draw_reduce(x: Reduce):
    print("reduce", x)
    inp = reshape(base_tensor(x.input_shape), 3/1)
    inp = add_whiskers(inp, x.input_shape)
    if x.index == 0 and len(x.input_shape) == 2:
        inp = hcat([rectangle(0.1, inp.get_envelope().height).align_t().line_width(0).fill_color(Color("black")), 
                    inp], 0.3)
    else:
        inp = vcat([rectangle(inp.get_envelope().width, 0.1).align_l().line_width(0).fill_color(Color("black")), 
                    inp], 0.3)

        
    out = reshape(base_tensor(x.output_shape), 3/1)
    out = add_whiskers(out, x.output_shape)
    return pair_draw(inp, out, x.op, shape=False)

def draw_load(x: Load, tensor_table):
    inp, out = store_load(x, tensor_table)
    out = add_whiskers(reshape(out, 3/1), x.offsets.shape)
    return pair_draw(inp, out, "load", shape=False)

def draw_store(x: Store, tensor_table):
    inp, out = store_load(x, tensor_table)
    out = add_whiskers(reshape(out, 3/1), x.offsets.shape)
    return pair_draw(out, inp, "store", shape=False)

def store_load(x, tensor_table):
    tensor: Tensor = tensor_table[x.ptr[0]]
    inp = base_tensor(tensor.shape, Color("blue")) + cover(tensor.shape, tensor.dtype, x.offsets, x.masks, Color("red"))
    inp = reshape(inp, 3/1)
    out = x.offsets.shape
    out = base_tensor(out, Color("blue")) #+ mask(out, x.masks, Color("green"))
    return inp, out 

def draw_binary_op(x: BinaryOps):
    if x.input_shape == (1,):
        return None
    inp = reshape(base_tensor(x.input_shape), 3/1)
    inp = add_whiskers(inp, x.input_shape)
    out = reshape(base_tensor(x.output_shape), 3/1)
    out = add_whiskers(out, x.output_shape)
    return pair_draw(out, inp, x.op, shape=False).center_xy() 
        
def draw_dot(x: BinaryOps):
    if x.input_shape == (1,):
        return None
    inp = reshape(base_tensor(x.input_shape[0]), 3/1)
    inp = add_whiskers(inp, x.input_shape[0])
    inp2 = reshape(base_tensor(x.input_shape[0]), 3/1)
    inp2 = add_whiskers(inp, x.input_shape[0])
    out = reshape(base_tensor(x.output_shape), 3/1)
    out = add_whiskers(out, x.output_shape)
    return hcat([box(inp, 1, 2.5/3), box(inp2, 1, 2.5/3),
                 box(out, 3, 2.5)], 1).center_xy() + text("dot", 0.2).fill_color(Color("black")).line_width(0).translate(0, -1)




