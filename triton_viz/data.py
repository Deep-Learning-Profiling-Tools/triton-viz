from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np


@dataclass
class Store:
    ptr: int
    shape: Tuple
    offsets: np.array
    masks: np.array


@dataclass
class Load:
    ptr: int
    shape: Tuple
    offsets: np.array
    masks: np.array


@dataclass
class BinaryOps:
    op: str
    input_shape: Tuple
    output_shape: Tuple


@dataclass
class MakeRange:
    start: int
    end: int


@dataclass
class ExpandDims:
    input_shape: Tuple
    index: int
    output_shape: Tuple


@dataclass
class Dot:
    input_shape: Tuple
    other_shape: Tuple
    output_shape: Tuple


@dataclass
class Reduce:
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


@dataclass
class Grid:
    idx: Tuple


@dataclass
class Launch:
    grid: Tuple
    tensors: List[Tensor]
    records: List
