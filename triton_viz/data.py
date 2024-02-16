from dataclasses import dataclass
from typing import List, Tuple, Any


@dataclass
class Store:
    ptr: int
    shape: Tuple
    offsets: List[Tuple]
    masks: List[bool]


@dataclass
class Load:
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
class Tensor:
    ptr: int
    dtype: str
    stride: Tuple
    shape: Tuple


@dataclass
class Grid:
    idx: Tuple


@dataclass
class Launch:
    grid: Tuple
    tensors: List[Tensor]
    records: List
