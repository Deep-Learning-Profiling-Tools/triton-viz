from ...core.data import Store, Load
import numpy as np
from numpy.typing import NDArray
from typing import Type, Union
from dataclasses import dataclass
import torch


@dataclass
class OutOfBoundsRecord:
    op_type: Type[Union[Store, Load]]
    tensor: torch.Tensor
    offsets: NDArray[np.int_]
    masks: NDArray[np.bool_]
    valid_access_masks: NDArray[np.bool_]
    invalid_access_masks: NDArray[np.bool_]
    corrected_offsets: NDArray[np.int_]
