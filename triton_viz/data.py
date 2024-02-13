from dataclasses import dataclass
from typing import List, Tuple


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
    offsets: List[tuple]
    masks: List[bool]


@dataclass
class TensorRecord:
    ptr: int
    shape: List
    stride: List
    dtype: str


@dataclass
class GridRecord:
    idx: Tuple


@dataclass
class LaunchRecord:
    grid: Tuple
    tensors: List[TensorRecord]
    records: List
