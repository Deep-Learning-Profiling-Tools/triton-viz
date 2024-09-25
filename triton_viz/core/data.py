from dataclasses import dataclass, field
from typing import ClassVar
import traceback
import numpy.typing as npt
import numpy as np
import torch

TRITON_FRAMES = [
    "triton/runtime",
    "triton/language",
    "triton_viz/core/client.py",
    "triton_viz/core/trace.py",
]


@dataclass
class Op:
    name: ClassVar[str] = "op"
    call_path: list[traceback.StackSummary] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.call_path = traceback.extract_stack()[:-2]
        clean_call_path = []
        for frame in self.call_path:
            if not any(
                triton_frame in frame.filename for triton_frame in TRITON_FRAMES
            ):
                clean_call_path.append(frame)
        self.call_path = clean_call_path


@dataclass
class ProgramId(Op):
    name: ClassVar[str] = "program_id"


@dataclass
class RawStore(Op):
    name: ClassVar[str] = "raw_store"


@dataclass
class Store(Op):
    name: ClassVar[str] = "store"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]


@dataclass
class RawLoad(Op):
    name: ClassVar[str] = "raw_load"


@dataclass
class Load(Op):
    name: ClassVar[str] = "load"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]


@dataclass
class UnaryOp(Op):
    name: ClassVar[str] = "unary_op"


@dataclass
class BinaryOp(Op):
    name: ClassVar[str] = "binary_op"
    op: str
    input_shape: tuple
    output_shape: tuple


@dataclass
class TernaryOp(Op):
    name: ClassVar[str] = "ternary_op"


@dataclass
class Dot(Op):
    name: ClassVar[str] = "dot"
    input_shape: tuple
    other_shape: tuple
    output_shape: tuple
    input_data: list[list[float]]
    other_data: list[list[float]]
    intermediate_results: dict[tuple[int, int], float] = field(
        default_factory=dict
    )  # Only storing the result now

    def update_intermediate(self, row: int, col: int, result: float):
        # Store only the result as a float
        self.intermediate_results[(row, col)] = result


@dataclass
class MakeRange(Op):
    name: ClassVar[str] = "make_range"
    start: int
    end: int


@dataclass
class AddPtr(Op):
    name: ClassVar[str] = "addptr"


@dataclass
class ExpandDims(Op):
    name: ClassVar[str] = "expand_dims"
    input_shape: tuple
    index: int
    output_shape: tuple


@dataclass
class Broadcast(Op):
    name: ClassVar[str] = "broadcast"


@dataclass
class Reduce(Op):
    name: ClassVar[str] = "reduce"
    input_shape: tuple
    index: int
    keep_dims: bool
    output_shape: tuple


@dataclass
class ReduceMin(Reduce):
    name: ClassVar[str] = "reduce_min"
    reduce_type: ClassVar[str] = "min"


@dataclass
class ReduceMax(Reduce):
    name: ClassVar[str] = "reduce_max"
    reduce_type: ClassVar[str] = "max"


@dataclass
class ReduceSum(Reduce):
    name: ClassVar[str] = "reduce_sum"
    reduce_type: ClassVar[str] = "sum"


@dataclass
class Splat(Op):
    # Broadcasts a scalar to a tensor
    name: ClassVar[str] = "splat"


@dataclass
class MakeBlockPointer(Op):
    name: ClassVar[str] = "make_block_ptr"


@dataclass
class TensorPointerLoad(Op):
    name: ClassVar[str] = "tensor_pointer_load"


@dataclass
class TensorPointerStore(Op):
    name: ClassVar[str] = "tensor_pointer_store"


@dataclass
class Idiv(Op):
    name: ClassVar[str] = "idiv"


@dataclass
class Rsqrt(Op):
    name: ClassVar[str] = "rsqrt"


@dataclass
class CastImpl(Op):
    name: ClassVar[str] = "cast_impl"


@dataclass
class Tensor:
    ptr: int
    dtype: str
    stride: tuple
    shape: tuple
    element_size: int
    data: torch.Tensor


@dataclass
class Grid:
    idx: tuple


@dataclass
class Launch:
    grid: tuple | None = None
    tensors: list[Tensor] = field(default_factory=list)
    records: list = field(default_factory=list)
