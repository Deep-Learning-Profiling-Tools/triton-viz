from ...core.data import Store, Load
import numpy as np
from numpy.typing import NDArray
from typing import Type, Union, List
from dataclasses import dataclass
import torch
import z3


@dataclass
class TracebackInfo:
    filename: str
    lineno: int
    func_name: str
    line_of_code: str

@dataclass
class OutOfBoundsRecord:
    op_type: Type[Union[Store, Load]]
    tensor: torch.Tensor
    user_code_tracebacks: List[TracebackInfo]

@dataclass
class OutOfBoundsRecordBruteForce(OutOfBoundsRecord):
    offsets: NDArray[np.int_]
    masks: NDArray[np.bool_]
    valid_access_masks: NDArray[np.bool_]
    invalid_access_masks: NDArray[np.bool_]
    corrected_offsets: NDArray[np.int_]

@dataclass
class OutOfBoundsRecordZ3(OutOfBoundsRecord):
    constraints: List[z3.z3.BoolRef]
    violation_index: int
