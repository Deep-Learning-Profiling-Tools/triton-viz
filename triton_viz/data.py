from dataclasses import dataclass
from typing import List, Tuple, Any


@dataclass
class StoreRecord:
    ptr: int
    shape: Tuple
    offsets: List[Tuple]
    masks: List[bool]


@dataclass
class LoadRecord:
    ptr: int
    shape: Tuple
    offsets: List[Tuple]
    masks: List[bool]


@dataclass
class BinaryOps:
    op: str
    shape: Tuple


@dataclass
class MakeRange:
    start: int
    end: int


@dataclass
class ExpandDims:
    input_shape: Tuple
    index: int


@dataclass
class DotRecord:
    input_shape: Tuple
    other_shape: Tuple


@dataclass
class Reduce:
    input_shape: Tuple
    index: int
    op: Any
    keep_dims: bool


@dataclass
class TensorRecord:
    ptr: int
    shape: Tuple
    stride: Tuple
    dtype: str


@dataclass
class GridRecord:
    idx: Tuple


@dataclass
class LaunchRecord:
    grid: Tuple
    tensors: List[TensorRecord]
    records: List
