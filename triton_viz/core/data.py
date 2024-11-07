from dataclasses import dataclass, field
from typing import List, Tuple, ClassVar, Optional, Any
import traceback
import numpy.typing as npt
import numpy as np

TRITON_FRAMES = [
    "triton/runtime",
    "triton/language",
    "triton_viz/core/client.py",
    "triton_viz/core/trace.py",
]


@dataclass
class Op:
    name: ClassVar[str] = "op"
    call_path: List[traceback.StackSummary] = field(init=False, default_factory=list)

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
class Store(Op):
    name: ClassVar[str] = "store"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]


@dataclass
class Load(Op):
    name: ClassVar[str] = "load"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]


@dataclass
class BinaryOp(Op):
    name: ClassVar[str] = "binary_op"
    op: str
    input_shape: Tuple
    output_shape: Tuple


@dataclass
class MakeRange(Op):
    name: ClassVar[str] = "make_range"
    start: int
    end: int


@dataclass
class ExpandDims(Op):
    name: ClassVar[str] = "expand_dims"
    input_shape: Tuple
    index: int
    output_shape: Tuple


@dataclass
class Dot(Op):
    name: ClassVar[str] = "dot"
    input_shape: Tuple
    other_shape: Tuple
    output_shape: Tuple


@dataclass
class Reduce(Op):
    name: ClassVar[str] = "reduce"
    input_shape: Tuple
    index: int
    keep_dims: bool
    output_shape: Tuple


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
class Tensor:
    ptr: int
    dtype: str
    stride: Tuple
    shape: Tuple
    element_size: int


@dataclass
class Grid:
    idx: Tuple


@dataclass
class Launch:
    grid: Optional[Tuple] = None
    tensors: List[Tensor] = field(default_factory=list)
    records: List[Any] = field(default_factory=list)