from ...core.data import Store, Load
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Union, Any, Optional
import torch
import z3


@dataclass
class TracebackInfo:
    """User code location associated with an op.

    Attributes:
        filename: Source file path.
        lineno: Line number in the file.
        func_name: Function name at the frame.
        line_of_code: Source line text.
    """

    filename: str
    lineno: int
    func_name: str
    line_of_code: str


@dataclass
class OutOfBoundsRecord:
    """Base out-of-bounds record for a load/store op.

    Attributes:
        op_type: Operation type for the access.
        tensor: Tensor involved in the access.
        user_code_tracebacks: User code frames for context.
    """

    op_type: type[Union[Store, Load]]
    tensor: torch.Tensor
    user_code_tracebacks: list[TracebackInfo]


@dataclass
class OutOfBoundsRecordBruteForce(OutOfBoundsRecord):
    """Out-of-bounds record with explicit mask diagnostics.

    Attributes:
        offsets: Offsets used for the access.
        masks: Mask values applied to the access.
        valid_access_masks: Mask for in-bounds lanes.
        invalid_access_masks: Mask for out-of-bounds lanes.
        corrected_offsets: Offsets corrected to in-bounds values.
    """

    offsets: NDArray[np.int_]
    masks: NDArray[np.bool_]
    valid_access_masks: NDArray[np.bool_]
    invalid_access_masks: NDArray[np.bool_]
    corrected_offsets: NDArray[np.int_]


@dataclass
class OutOfBoundsRecordZ3(OutOfBoundsRecord):
    """
    Attributes:
        constraints (Optional[z3.z3.BoolRef]):
            A conjunction of Z3 constraint expressions defining valid ranges
            for memory access. Any address falling outside these ranges is considered invalid.

        violation_address (int):
            The exact address where an invalid memory access was detected.
            For example:
            Invalid access detected at index: 200

            A few simplified constraints might look like:
              And(x >= 50, x <= 60, x != 55)
              And(x >= 70, x <= 80)
              ...

            In this scenario, 200 is an out-of-bounds address because it
            falls outside the valid ranges described by these constraints.

        symbolic_expr (Optional[Any]):
            The symbolic expression tree that led to the OOB access, if available.
    """

    constraints: Optional[z3.z3.BoolRef]
    violation_address: int
    symbolic_expr: "Any" = None  # Optional symbolic expression tree
    tensor_name: Optional[str] = None
