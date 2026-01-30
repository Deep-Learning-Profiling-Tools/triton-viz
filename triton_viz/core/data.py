from dataclasses import dataclass, field
from typing import ClassVar, Optional
import traceback
import numpy.typing as npt
import numpy as np
import torch

TRITON_FRAMES = [
    "triton/runtime",
    "triton/language",
    "triton_viz/core/client.py",
    "triton_viz/core/trace.py",
]


@dataclass
class Op:
    """Base record for a traced operation.

    Attributes:
        call_path: Filtered stack frames pointing to user code.
    """

    name: ClassVar[str] = "op"
    call_path: list[traceback.FrameSummary] = field(init=False, default_factory=list)

    def __post_init__(self):
        full_stack = traceback.extract_stack()[:-2]
        # keep original for fallback
        self.call_path = full_stack
        clean_call_path = []
        for frame in full_stack:
            if not any(
                triton_frame in frame.filename for triton_frame in TRITON_FRAMES
            ):
                clean_call_path.append(frame)
        # if filtering removed all frames, fallback to last meaningful frame(s)
        if clean_call_path:
            self.call_path = clean_call_path
        else:
            for frame in reversed(full_stack):
                if not str(frame.filename).startswith("<"):
                    self.call_path = [frame]
                    break


@dataclass
class ProgramId(Op):
    """Record a program id query."""

    name: ClassVar[str] = "program_id"


@dataclass
class Allocate(Op):
    """Record a tensor allocation.

    Attributes:
        ptr: Pointer address for the allocation.
    """

    name: ClassVar[str] = "allocate"
    ptr: int


@dataclass
class RawStore(Op):
    """Record a raw store op before masking."""

    name: ClassVar[str] = "raw_store"


@dataclass
class Store(Op):
    """Record a masked store op.

    Attributes:
        ptr: Base pointer address for the store.
        offsets: Element offsets applied to the base pointer.
        masks: Boolean mask indicating active lanes.
        mem_src: Source memory label.
        mem_dst: Destination memory label.
        backend: Backend name for the op.
        bytes: Payload size in bytes.
        time_idx: Logical ordering index for visualization.
    """

    name: ClassVar[str] = "store"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]
    mem_src: str = "SBUF"
    mem_dst: str = "HBM"
    backend: str = "nki"
    bytes: int = 0
    time_idx: int = 0


@dataclass
class RawLoad(Op):
    """Record a raw load op before masking."""

    name: ClassVar[str] = "raw_load"


@dataclass
class Load(Op):
    """Record a masked load op.

    Attributes:
        ptr: Base pointer address for the load.
        offsets: Element offsets applied to the base pointer.
        masks: Boolean mask indicating active lanes.
        mem_src: Source memory label.
        mem_dst: Destination memory label.
        backend: Backend name for the op.
        bytes: Payload size in bytes.
        time_idx: Logical ordering index for visualization.
    """

    name: ClassVar[str] = "load"
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]
    # buffer: str
    mem_src: str = "HBM"
    mem_dst: str = "SBUF"
    backend: str = "nki"
    bytes: int = 0
    time_idx: int = 0


@dataclass
class UnaryOp(Op):
    """Record a unary op."""

    name: ClassVar[str] = "unary_op"


@dataclass
class BinaryOp(Op):
    """Record a binary op with input/output shapes.

    Attributes:
        op: Operator name or token.
        input_shape: Shape of the left operand.
        output_shape: Shape of the output tensor.
    """

    name: ClassVar[str] = "binary_op"
    op: str
    input_shape: tuple
    output_shape: tuple


@dataclass
class TernaryOp(Op):
    """Record a ternary op."""

    name: ClassVar[str] = "ternary_op"


@dataclass
class Dot(Op):
    """Record a dot/matmul op with optional intermediate results.

    Attributes:
        input_shape: Shape of the left operand.
        other_shape: Shape of the right operand.
        output_shape: Shape of the output tensor.
        input_data: Raw data for the left operand.
        other_data: Raw data for the right operand.
        intermediate_results: Per-(row,col) partial results.
    """

    name: ClassVar[str] = "dot"
    input_shape: tuple
    other_shape: tuple
    output_shape: tuple
    input_data: list[list[float]]
    other_data: list[list[float]]
    intermediate_results: dict[tuple[int, int], float] = field(
        default_factory=dict
    )  # Only storing the result now

    def update_intermediate(self, row: int, col: int, result: float):
        # Store only the result as a float
        self.intermediate_results[(row, col)] = result


@dataclass
class Flip(Op):
    """Record a flip op with optional data payloads.

    Attributes:
        input_shape: Shape of the input tensor.
        output_shape: Shape of the output tensor.
        dim: Dimension to flip.
        input_data: Optional input values for visualization.
        output_data: Optional output values for visualization.
    """

    name: ClassVar[str] = "flip"
    input_shape: tuple
    output_shape: tuple
    dim: int
    # Optional payloads to help frontend render actual values when available
    input_data: list | None = None
    output_data: list | None = None


@dataclass
class MakeRange(Op):
    """Record a range creation op.

    Attributes:
        start: Range start.
        end: Range end.
    """

    name: ClassVar[str] = "make_range"
    start: int
    end: int


@dataclass
class AddPtr(Op):
    """Record a pointer arithmetic op."""

    name: ClassVar[str] = "addptr"


@dataclass
class ExpandDims(Op):
    """Record a dimension expansion op.

    Attributes:
        input_shape: Shape before expansion.
        index: Inserted dimension index.
        output_shape: Shape after expansion.
    """

    name: ClassVar[str] = "expand_dims"
    input_shape: tuple
    index: int
    output_shape: tuple


@dataclass
class Broadcast(Op):
    """Record a broadcast op."""

    name: ClassVar[str] = "broadcast"


@dataclass
class Reduce(Op):
    """Record a reduction op.

    Attributes:
        input_shape: Shape before reduction.
        index: Reduced axis index.
        keep_dims: Whether reduced dims are kept.
        output_shape: Shape after reduction.
    """

    name: ClassVar[str] = "reduce"
    input_shape: tuple
    index: int
    keep_dims: bool
    output_shape: tuple


@dataclass
class ReduceMin(Reduce):
    """Record a min reduction op."""

    name: ClassVar[str] = "reduce_min"
    reduce_type: ClassVar[str] = "min"


@dataclass
class ReduceMax(Reduce):
    """Record a max reduction op."""

    name: ClassVar[str] = "reduce_max"
    reduce_type: ClassVar[str] = "max"


@dataclass
class ReduceSum(Reduce):
    """Record a sum reduction op."""

    name: ClassVar[str] = "reduce_sum"
    reduce_type: ClassVar[str] = "sum"


@dataclass
class Splat(Op):
    """Record a scalar-to-tensor broadcast op."""

    # Broadcasts a scalar to a tensor
    name: ClassVar[str] = "splat"


@dataclass
class MakeBlockPointer(Op):
    """Record a block pointer creation op."""

    name: ClassVar[str] = "make_block_ptr"


@dataclass
class TensorPointerLoad(Op):
    """Record a tensor-pointer load op."""

    name: ClassVar[str] = "tensor_pointer_load"


@dataclass
class TensorPointerStore(Op):
    """Record a tensor-pointer store op."""

    name: ClassVar[str] = "tensor_pointer_store"


@dataclass
class Idiv(Op):
    """Record an integer division op."""

    name: ClassVar[str] = "idiv"


@dataclass
class Rsqrt(Op):
    """Record a reciprocal square root op."""

    name: ClassVar[str] = "rsqrt"


@dataclass
class CastImpl(Op):
    """Record a cast implementation op."""

    name: ClassVar[str] = "cast_impl"


@dataclass
class Reshape(Op):
    """Record a reshape op."""

    name: ClassVar[str] = "reshape"


@dataclass
class Join(Op):
    """Record a join/concat op."""

    name: ClassVar[str] = "join"


@dataclass
class Fabs(Op):
    """Record an absolute value op."""

    name: ClassVar[str] = "fabs"


@dataclass
class Ashr(Op):
    """Record an arithmetic shift right op."""

    name: ClassVar[str] = "ashr"


@dataclass
class Advance(Op):
    """Record a pointer advance op."""

    name: ClassVar[str] = "advance"


@dataclass
class FpToFp(Op):
    """Record a floating-point cast op."""

    name: ClassVar[str] = "fp_to_fp"


@dataclass
class Umulhi(Op):
    """Record a high-word multiply op."""

    name: ClassVar[str] = "umulhi"


@dataclass
class Trans(Op):
    """Record a transpose op."""

    name: ClassVar[str] = "trans"


@dataclass
class CumSum(Op):
    """Record a cumulative sum op."""

    name: ClassVar[str] = "cumsum"


@dataclass
class Bitcast(Op):
    """Record a bitcast op."""

    name: ClassVar[str] = "bitcast"


@dataclass
class AtomicCas(Op):
    """Record an atomic compare-and-swap op."""

    name: ClassVar[str] = "atomic_cas"


@dataclass
class AtomicRMW(Op):
    """Record an atomic read-modify-write op."""

    name: ClassVar[str] = "atomic_rmw"


@dataclass
class Tensor:
    """Tensor metadata captured for visualization.

    Attributes:
        ptr: Base pointer address for the tensor.
        dtype: Dtype string representation.
        stride: Tensor strides.
        shape: Tensor shape.
        element_size: Size in bytes for one element.
        data: Backing tensor object.
    """

    ptr: int
    dtype: str
    stride: tuple
    shape: tuple
    element_size: int
    data: torch.Tensor


@dataclass
class Grid:
    """Record the current grid index."""

    idx: tuple


@dataclass
class Launch:
    """Aggregate records for a single kernel launch.

    Attributes:
        grid: Grid dimensions used for the launch.
        tensors: Tensors seen during argument processing.
        records: Collected op records for this launch.
    """

    grid: Optional[tuple] = None
    tensors: set[Tensor] = field(default_factory=set)
    records: list = field(default_factory=list)
