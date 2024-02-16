from dataclasses import dataclass
from typing import List, Tuple


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
    offsets: List[tuple]
    masks: List[bool]


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
