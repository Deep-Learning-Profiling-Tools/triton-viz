from colour import Color
from triton_viz.data import (
    Tensor,
    Grid,
    Store,
    Load,
    Op,
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
from chalk import Diagram, rectangle, text, hcat, vcat, empty, Path, Trail, V2, concat
from dataclasses import dataclass
from numpy.typing import ArrayLike
import sys

sys.setrecursionlimit(100000)


planar.EPSILON = 0.0
chalk.set_svg_draw_height(500)
BG = Color("white")
WHITE = Color("white")
DEFAULT = Color("grey")
BLACK = Color("black")
GREY = Color("grey")
palette = [
    "#f29f05",
    "#f25c05",
    "#d6568c",
    "#4d8584",
    "#a62f03",
    "#400d01",
    "#274001",
    "#828a00",
]
ACTIVE = [Color(p) for p in palette]

MRATIO = 1 / 3


# Generic render helpers


def box(d: Diagram, width: float, height: float, outer=0.2) -> Diagram:
    "Put diagram in a box of shape height, width"
    h, w = d.get_envelope().height, d.get_envelope().width
    m = max(h, w)
    back = rectangle(outer + m, outer + m).line_width(0).fill_color(BG).center_xy()
    d = (back + d.center_xy()).with_envelope(back)
    return d.scale_x(width / m).scale_y(height / m)


def reshape(d: Diagram) -> Diagram:
    "Use log-scale if ratio is too sharp"
    h, w = d.get_envelope().height, d.get_envelope().width
    if (h / w > MRATIO) or (w / h > MRATIO):
        d = d.scale_y(math.log(h + 1, 2) / h).scale_x(math.log(w + 1, 2) / w)
    return d


def collect_grid():
    for launch in record_builder.launches[-1:]:
        records, tensor_table, failures = collect_launch(launch)
    return records, tensor_table, failures


def collect_launch(launch):
    tensor_table = {}
    for i, t in enumerate(launch.tensors):
        tensor_table[t.ptr] = (t, i)
    failures = {}
    all_grids = {}
    last_grid = None
    program_records = []
    for r in launch.records:
        if isinstance(r, Grid):
            if last_grid is not None:
                all_grids[last_grid.idx] = program_records
                program_records = []
            last_grid = r
        program_records.append(r)
        if (
            isinstance(r, (Store, Load))
            and (r.invalid_access_masks & r.original_masks).any()
        ):
            failures[last_grid.idx] = True
    all_grids[last_grid.idx] = program_records
    return all_grids, tensor_table, failures


def draw_record(program_record, tensor_table, output):
    return draw_launch(program_record, tensor_table, output)


def draw_launch(program_records, tensor_table, base) -> Diagram:
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
        if isinstance(x, Op):
            return draw_op(x)
        if isinstance(x, MakeRange):
            return draw_make_range(x)
        if isinstance(x, Reduce):
            return draw_reduce(x)
        if isinstance(x, Dot):
            return None  # draw_dot(x)

    def draw_record(x):
        "Render one record"
        y = draw(x)
        if y is None:
            return empty()

        return (chalk.vstrut(0.2) / y).center_xy()

    records = []
    for r in program_records:
        dr = draw_record(r)
        # env = dr.get_envelope()
        # dr = dr.center_xy().with_envelope(rectangle(env.width, env.height).center_xy())
        records.append(dr)

    dr = vcat(records)
    dr = dr.center_xy()
    env = dr.get_envelope()
    dr = rectangle(env.width + 1, env.height + 1).fill_color(BG).center_xy() + dr
    dr.render(base, 2500)
    return env.width, env.height


def delinearize(shape: Tuple, x: npt.NDArray, dtype, mask) -> List[npt.NDArray]:
    if len(shape) == 1:
        shape = (1, 1, shape[0])
    x = x.copy() // (dtype.element_ty.primitive_bitwidth // 8)
    vals = []
    for s in list(reversed(shape[1:])) + [10000]:
        vals.append(((x % s) * mask - (1 - mask)).ravel())
        x = x // s
    return vals


trail = Trail.from_offsets([V2(0, 1), V2(1, 0), V2(0, -1), V2(-1, 0)], closed=True)


def cover(
    shape: Tuple, dtype, load: Tensor, mask: npt.NDArray, color: Color
) -> Diagram:
    shape = make_3d(shape)
    "Draw the values from load on top of the loading tensor"
    x, y, z = delinearize(shape, load, dtype, mask)
    return draw_tensor_3d(shape, z, y, x, color)


def pair_draw(x: Diagram, y: Diagram, command: str) -> Diagram:
    "Draw two diagrams next to each other with a command in the middle."
    return hcat([box(x, 3, 2.5), box(y, 3, 2.5)], 1).center_xy() + text(
        command, 0.2
    ).fill_color(BLACK).line_width(0).translate(0, -1)


# Individual renderers


def draw_tensor(x: Tensor) -> Optional[Diagram]:
    return None


def draw_grid(x: Grid) -> Optional[Diagram]:
    return None


def draw_make_range(x: MakeRange) -> Optional[Diagram]:
    return None


def draw_reduce(x: Reduce) -> Optional[Diagram]:
    color = ACTIVE[0]
    inp = draw_tensor_3d(make_3d(x.input_shape), None, None, None, color)
    if x.index == 0 and len(x.input_shape) == 2:
        inp = hcat(
            [
                rectangle(0.1, inp.get_envelope().height)
                .align_t()
                .line_width(0)
                .fill_color(BLACK),
                inp,
            ],
            0.5,
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
            0.5,
        )
    out = draw_tensor_3d(x.output_shape, None, None, None, color)
    return pair_draw(reshape(inp), reshape(out), x.op)


def draw_load(x, tensor_table) -> Optional[Diagram]:
    inp, out = store_load(x, tensor_table)
    out = reshape(out)
    return pair_draw(inp, out, "load")


def draw_store(x, tensor_table) -> Optional[Diagram]:
    inp, out = store_load(x, tensor_table)
    out = reshape(out)
    return pair_draw(out, inp, "store")


def make_3d(shape):
    "Make a 3d shape"
    if len(shape) == 1:
        return (1, 1, shape[0])
    if len(shape) == 2:
        return (1, shape[0], shape[1])
    return shape


def store_load(
    x: Union[Store, Load], tensor_table: Dict[int, Tuple[Tensor, int]]
) -> Tuple[Diagram, Diagram]:
    tensor, tensor_id = tensor_table[x.ptr]
    # inp = base_tensor(tensor.shape, DEFAULT)
    color = ACTIVE[tensor_id]
    invalid = x.invalid_access_masks.any()
    if invalid:
        color = Color("red")
    inp = cover(tensor.shape, tensor.dtype, x.original_offsets, x.original_masks, color)
    inp = reshape(inp)
    s = make_3d(x.original_offsets.shape)
    a, b, c = x.original_masks.reshape(*s).nonzero()
    out = draw_tensor_3d(s, a, b, c, color)
    return inp, out


def draw_op(x: Op) -> Optional[Diagram]:
    return None


def draw_dot(x: Dot) -> Optional[Diagram]:
    if x.input_shape == (1,):
        return None
    inp = draw_tensor_3d(x.input_shape[0], None, None, None)
    # inp = reshape(base_tensor(x.input_shape[0], color=ACTIVE))
    # inp = add_whiskers(inp, x.input_shape[0])
    inp2 = draw_tensor_3d(x.input_shape[1], None, None, None)
    # inp2 = reshape(base_tensor(x.input_shape[0], color=ACTIVE))
    # inp2 = add_whiskers(inp2, x.input_shape[0])
    out = draw_tensor_3d(x.output_shape, None, None, None)
    # out = reshape(base_tensor(x.output_shape, color=ACTIVE))
    # out = add_whiskers(out, x.output_shape)
    return hcat(
        [box(inp, 1.5, 2), box(inp2, 1.5, 2), box(out, 1.5, 2)], 1
    ).center_xy() + text("dot", 0.2).fill_color(BLACK).line_width(0).translate(0, -1)


# For 3d


def lookAt(eye: ArrayLike, center: ArrayLike, up: ArrayLike):
    "Python version of the haskell lookAt function in linear.projections"
    f = (center - eye) / np.linalg.norm(center - eye)
    s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    u = np.cross(s, f)
    return np.array([[*s, 0], [*u, 0], [*-f, 0], [0, 0, 0, 1]])


def scale3(x, y, z):
    return np.array([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])


@dataclass
class D3:
    x: float
    y: float
    z: float

    def to_np(self):
        return np.array([self.x, self.y, self.z])


V3 = D3


def homogeneous(trails: List[List[D3]]):
    "Convert list of directions to a np.array of homogeneous coordinates"
    return np.array([[[*o.to_np(), 1] for o in offsets] for offsets in trails])


def cube():
    "3 faces of a cube drawn as offsets from the origin."
    return homogeneous(
        [
            [D3(*v) for v in offset]
            for offset in [
                [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
                [(1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1)],
                [(0, 0, 1), (0, 1, 0), (0, 0, -1), (0, -1, 0)],
            ]
        ]
    )


def to_trail(trail: ArrayLike, locations: ArrayLike):
    return [
        (
            Path(
                [
                    Trail.from_offsets([V2(*v[:2]) for v in trail])
                    .close()
                    .at(V2(*loc[:2]))
                ]
            ),
            loc[2],
        )
        for loc in locations
    ]


def project(projection, shape3, positions):
    p = homogeneous([positions for _ in range(shape3.shape[0])])
    locations = p @ projection.T
    trails = shape3 @ projection.T
    return [out for t, loc in zip(trails, locations) for out in to_trail(t, loc)]


def draw_tensor_3d(shape, a, b, c, color=WHITE):
    shape = make_3d(shape)

    # Big Cube
    s = scale3(*shape)
    big_cube = cube() @ s.T
    back = scale3(0, shape[1], shape[2])
    back_cube = cube() @ back.T

    # Isometric projection of tensor
    projection = lookAt(
        V3(-1.0, -0.3, -0.15).to_np(),
        V3(0, 0, 0).to_np(),
        V3(0, 1, 0).to_np(),
    )
    outer = project(projection, big_cube, [V3(0, 0, 0)])
    outer2 = project(projection, back_cube, [V3(shape[0], 0, 0)])
    d = (
        concat([p.stroke().fill_color(GREY).fill_opacity(0.1) for p, _ in outer2])
        .line_width(0.005)
        .line_color(GREY)
    )
    d += (
        concat([p.stroke().fill_color(GREY).fill_opacity(0.05) for p, _ in outer])
        .line_width(0.01)
        .line_color(BLACK)
    )
    if a is not None:
        out = group(a, b, c)
        d2 = [
            (b, loc)
            for i in range(len(out))
            for b, loc in make_cube(projection, out[i][0], out[i][1], color)
        ]
        d2.sort(key=lambda x: x[1], reverse=True)
        d2 = concat([b.with_envelope(empty()) for b, _ in d2])
        d = d2.with_envelope(d) + d
    return d


def lines(s):
    "Draw lines to mimic a cube of cubes"
    bs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    return [
        homogeneous(
            [
                [D3(*p) for _ in range(s[i]) for p in [a / s[i], b, -b]]
                for b in bs
                if not np.all(a == b)
            ]
        )
        for i, a in enumerate(bs)
    ]


def make_cube(projection, start, end, color):
    "Draws a cube from start position to end position."
    start = np.array(start).astype(int)
    end = np.array(end).astype(int)
    s2 = end - start + 1
    s = scale3(*s2)
    small_cube = cube() @ s.T
    loc = [
        project(projection, l2 @ s.T, [V3(*start)])
        for l2 in lines(s2)
        if l2.shape[1] > 0
    ]
    outer2 = project(projection, small_cube, [V3(*start)])
    ls = loc
    box = [
        (p.stroke().fill_color(color).fill_opacity(0.4).line_width(0), loc)
        for p, loc in outer2
    ]
    line = [
        (p.stroke().line_width(0.001).line_color(BLACK), l_)
        for loc in ls
        for p, l_ in loc
    ]
    return [(b, loc) for b, loc in box + line]


def group(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    "Groups together cubes into bigger cubes"
    x = list(zip(zip(x, y, z), zip(x, y, z)))
    x = [(a, b) for a, b in x if not (a[0] == -1 and a[1] == -1 and a[2] == -1)]

    start = x

    def remove_dups(ls):
        "Remove duplicates"
        out = []
        for y in ls:
            if not out or y != out[-1]:
                out.append(y)
        return out

    for j in range(2, -1, -1):
        x = remove_dups(start)
        start = []
        while True:
            if len(x) <= 1:
                break
            _, _, rest = x[0], x[1], x[2:]
            m = 0
            for k in range(2):
                a = x[0][k]
                b = x[1][k]
                if (
                    (k == 0 or a[j % 3] == b[j % 3] - 1)
                    and a[(j + 1) % 3] == b[(j + 1) % 3]
                    and a[(j + 2) % 3] == b[(j + 2) % 3]
                ):
                    m += 1
            if m == 2:
                x = [[x[0][0], x[1][1]]] + rest
            else:
                start.append(x[0])
                x = [x[1]] + rest
        start += x
    return start
