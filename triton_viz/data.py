from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np


@dataclass
class Store:
    ptr: np.Array
    shape: Tuple
    offsets: np.Array
    masks: np.Array


@dataclass
class Load:
    ptr: np.Array
    shape: Tuple
    offsets: np.Array
    masks: np.Array


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
    ptr: np.Array
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
