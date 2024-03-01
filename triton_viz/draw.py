from colour import Color
from triton_viz.data import (
    Launch,
    Tensor,
    Grid,
    Store,
    Load,
    BinaryOp,
    MakeRange,
    Reduce,
    Dot,
)
from .interpreter import record_builder
import numpy as np
import numpy.typing as npt
import planar
import math
import chalk
from typing import Tuple, Union, Optional, List, Dict
from chalk import Diagram, rectangle, text, hcat, vcat, empty, Path, Trail, P2, V2

planar.EPSILON = 0.0
chalk.set_svg_draw_height(500)

BG = Color("white")
DEFAULT = Color("grey")
BLACK = Color("black")
ACTIVE = Color("red")

MRATIO = 1 / 3


# Generic render helpers
def box(d: Diagram, width: float, height: float) -> Diagram:
    "Put diagram in a box of shape height, width"
    h, w = d.get_envelope().height, d.get_envelope().width
    m = max(h, w)
    d = (
        rectangle(0.2 + m, 0.2 + m).line_width(0).fill_color(BG).center_xy()
        + d.center_xy()
    )
    return d.scale_x(width / m).scale_y(height / m)


def reshape(d: Diagram) -> Diagram:
    "Use log-scale if ratio is too sharp"
    h, w = d.get_envelope().height, d.get_envelope().width
    if h / w > MRATIO:
        d = d.scale_y(math.log(h + 1, 2) / h).scale_x(math.log(w + 1, 2) / w)
    elif w / h > MRATIO:
        d = d.scale_y(math.log(h + 1, 2) / h).scale_x(math.log(w + 1, 2) / w)
    return d


def draw(output: str, size: int = 2500):
    "Draw the last record to a file (PNG)."
    for record in record_builder.launches[-1:]:
        diagram: Diagram = draw_launch(record)
    diagram.render(output, 2500)


def draw_launch(launch: Launch) -> Diagram:
    "Render Launch to a diagram."

    # Map tensor pointers to a table.
    tensor_table = {}
    for t in launch.tensors:
        tensor_table[t.ptr] = t

    def draw(x):
        "Dispatch"
        if isinstance(x, Tensor):
            return draw_tensor(x)
        if isinstance(x, Grid):
            return draw_grid(x)
        if isinstance(x, Store):
            return draw_store(x, tensor_table)
        if isinstance(x, Load):
            return draw_load(x, tensor_table)
        if isinstance(x, BinaryOp):
            # Not sure if we should draw binops
            return None  # draw_binary_op(x)
        if isinstance(x, MakeRange):
            return draw_make_range(x)
        if isinstance(x, Reduce):
            return draw_reduce(x)
        if isinstance(x, Dot):
            return draw_dot(x)

    def draw_record(x):
        "Render one record"
        y = draw(x)
        if y is None:
            return empty()

        return y.center_xy()

    d = vcat([draw_record(r) for r in launch.records], 0.2)
    env = d.get_envelope()
    return (
        rectangle(1.1 * env.width, 1.1 * env.height).fill_color(BG).center_xy()
        + d.center_xy()
    )


def base_tensor(shape: Tuple, color=DEFAULT) -> Diagram:
    "Draw a 2d base tensor"
    d = empty()
    if len(shape) == 2:
        d = rectangle(shape[1], shape[0])
    elif len(shape) == 1:
        shape = (shape[0], 1)
        d = rectangle(*shape)
    else:
        assert "bad shape", shape
    d = d.align_tl()
    return d.fill_color(color)


def add_whiskers(d: Diagram, shape: Tuple) -> Diagram:
    "Add whiskers to a 2d tensor for rows and columns"
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
    whisker = Trail.rectangle(0.01 * delta1, 0.5)
    top_whiskers = Path(
        [
            whisker.at(P2(w * ((i + 0.5) / shape[1]), 0))
            for i in range(0, shape[1], step1)
        ]
    )
    tw = (
        top_whiskers.stroke()
        .line_width(0)
        .fill_color(BLACK)
        .with_envelope(rectangle(w, 0.5).align_l())
    )
    d = vcat([tw, d], 0.1)
    whisker = Trail.rectangle(0.5, 0.01 * delta0)
    left_whiskers = Path(
        [
            whisker.at(P2(0.0, h * ((i + 0.5) / shape[0])))
            for i in range(0, shape[0], step0)
        ]
    )
    lw = (
        left_whiskers.stroke()
        .line_width(0)
        .fill_color(BLACK)
        .align_b()
        .translate(0, -0.5 * (h / shape[0]))
    )
    lw = lw.with_envelope(rectangle(0.5, h).align_t())
    d = hcat([lw, d.align_b()], 0.1)
    return d.center_xy()


def delinearize(shape: Tuple, x: npt.NDArray, dtype, mask) -> List[npt.NDArray]:
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


trail = Trail.from_offsets([V2(0, 1), V2(1, 0), V2(0, -1), V2(-1, 0)], closed=True)


def cover(
    d: Diagram, shape: Tuple, dtype, load: Tensor, mask: npt.NDArray, color: Color
) -> Diagram:
    "Draw the values from load on top of the loading tensor"
    x, y = delinearize(shape, load, dtype, mask)
    path = []
    for x1, y1 in zip(x.tolist(), y.tolist()):
        if x1 == -1 and y1 == -1:
            continue
        path.append(trail.at(P2(x1, y1)))
    return d + Path(loc_trails=path).stroke().fill_color(color).line_width(
        0
    ).with_envelope(d)


def mask(d: Diagram, shape: Tuple, mask: npt.NDArray, color: Color) -> Diagram:
    "Color the values from mask on top of the loaded tensor"
    if len(mask.shape) == 1:
        mask = mask.reshape(1, -1)
    y, x = mask.nonzero()
    path = []
    for x, y in zip(x.tolist(), y.tolist()):
        path.append(trail.at(P2(x, y)))
    return d + Path(loc_trails=path).stroke().fill_color(color).line_width(
        0
    ).with_envelope(d)


def pair_draw(x: Diagram, y: Diagram, command: str) -> Diagram:
    "Draw two diagrams next to each other with a command in the middle."
    return hcat([box(x, 3, 2.5), box(y, 3, 2.5)], 1).center_xy() + text(
        command, 0.2
    ).fill_color(BLACK).line_width(0).translate(0, -1)


# Individual renderers


def draw_tensor(x: Tensor) -> Optional[Diagram]:
    return None


def draw_grid(x: Grid) -> Optional[Diagram]:
    return text("Program: " + ", ".join(map(str, x.idx)), 0.5).fill_color(
        BLACK
    ).line_width(0) + rectangle(3, 1).line_width(0)


def draw_make_range(x: MakeRange) -> Optional[Diagram]:
    return None


def draw_reduce(x: Reduce) -> Optional[Diagram]:
    inp = reshape(base_tensor(x.input_shape))
    inp = add_whiskers(inp, x.input_shape)
    if x.index == 0 and len(x.input_shape) == 2:
        inp = hcat(
            [
                rectangle(0.1, inp.get_envelope().height)
                .align_t()
                .line_width(0)
                .fill_color(BLACK),
                inp,
            ],
            0.3,
        )
    else:
        inp = vcat(
            [
                rectangle(inp.get_envelope().width, 0.1)
                .align_l()
                .line_width(0)
                .fill_color(BLACK),
                inp,
            ],
            0.3,
        )

    out = reshape(base_tensor(x.output_shape))
    out = add_whiskers(out, x.output_shape)
    return pair_draw(inp, out, x.op)


def draw_load(x, tensor_table) -> Optional[Diagram]:
    inp, out = store_load(x, tensor_table)
    out = add_whiskers(reshape(out), x.offsets.shape)
    return pair_draw(inp, out, "load")


def draw_store(x, tensor_table) -> Optional[Diagram]:
    inp, out = store_load(x, tensor_table)
    out = add_whiskers(reshape(out), x.offsets.shape)
    return pair_draw(out, inp, "store")


def store_load(
    x: Union[Store, Load], tensor_table: Dict[int, Tensor]
) -> Tuple[Diagram, Diagram]:
    tensor: Tensor = tensor_table[x.ptr]
    inp = base_tensor(tensor.shape, DEFAULT)
    inp = cover(inp, tensor.shape, tensor.dtype, x.offsets, x.masks, ACTIVE)
    inp = reshape(inp)
    out = base_tensor(x.offsets.shape, DEFAULT)
    out = mask(out, x.offsets.shape, x.masks, ACTIVE)
    return inp, out


def draw_binary_op(x: BinaryOp) -> Optional[Diagram]:
    if x.input_shape == (1,):
        return None
    inp = reshape(base_tensor(x.input_shape, color=ACTIVE))
    inp = add_whiskers(inp, x.input_shape)
    out = reshape(base_tensor(x.output_shape, color=ACTIVE))
    out = add_whiskers(out, x.output_shape)
    return pair_draw(out, inp, x.op).center_xy()


def draw_dot(x: BinaryOp) -> Optional[Diagram]:
    if x.input_shape == (1,):
        return None
    inp = reshape(base_tensor(x.input_shape[0], color=ACTIVE))
    inp = add_whiskers(inp, x.input_shape[0])
    inp2 = reshape(base_tensor(x.input_shape[0], color=ACTIVE))
    inp2 = add_whiskers(inp2, x.input_shape[0])
    out = reshape(base_tensor(x.output_shape, color=ACTIVE))
    out = add_whiskers(out, x.output_shape)
    return hcat(
        [box(inp, 1, 2.5 / 3), box(inp2, 1, 2.5 / 3), box(out, 3, 2.5)], 1
    ).center_xy() + text("dot", 0.2).fill_color(BLACK).line_width(0).translate(0, -1)
