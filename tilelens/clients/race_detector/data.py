from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from ...core.data import Op


@dataclass
class AccessEventRecord:
    op_type: type[Op]
    access_mode: Literal["read", "write"]
    tensor: torch.Tensor | None = None
    tensor_name: str | None = None
    symbolic_expr: Any = None
    addr_expr: Any = None
    premises: tuple[Any, ...] = field(default_factory=tuple)
    local_constraints: tuple[Any, ...] = field(default_factory=tuple)
    source_location: tuple[str, int, str] | None = None
    grid_idx: tuple[int, ...] | None = None
