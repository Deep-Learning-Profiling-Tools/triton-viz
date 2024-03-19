from dataclasses import dataclass, field
from typing import List, Tuple, Any
import traceback
import numpy.typing as npt
import numpy as np


@dataclass
class Op:
    call_path: List[traceback.StackSummary] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.call_path = traceback.extract_stack()[:-2]
        clean_call_path = []
        triton_frames = [
            "triton/runtime",
            "triton/language",
            "triton_viz/interpreter.py",
            "triton_viz/trace.py",
        ]
        for frame in self.call_path:
            if not any(
                triton_frame in frame.filename for triton_frame in triton_frames
            ):
                clean_call_path.append(frame)
        self.call_path = clean_call_path


@dataclass
class Store(Op):
    ptr: int
    shape: Tuple
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]
    invalid_access_masks: npt.NDArray[np.bool_]
    original_offsets: npt.NDArray[np.int_]
    original_mask: npt.NDArray[np.bool_]


@dataclass
class Load(Op):
    ptr: int
    shape: Tuple
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]
    invalid_access_masks: npt.NDArray[np.bool_]
    original_offsets: npt.NDArray[np.int_]
    original_mask: npt.NDArray[np.bool_]


@dataclass
class BinaryOp(Op):
    op: str
    input_shape: Tuple
    output_shape: Tuple


@dataclass
class MakeRange(Op):
    start: int
    end: int


@dataclass
class ExpandDims(Op):
    input_shape: Tuple
    index: int
    output_shape: Tuple


@dataclass
class Dot(Op):
    input_shape: Tuple
    other_shape: Tuple
    output_shape: Tuple


@dataclass
class Reduce(Op):
    input_shape: Tuple
    index: int
    op: Any
    keep_dims: bool
    output_shape: Tuple


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
    grid: Tuple
    tensors: List[Tensor]
    records: List
