from dataclasses import dataclass


@dataclass
class LoadStoreBytes:
    type: str  # "load" or "store"
    total_bytes_true: int
    total_bytes_attempted: int


@dataclass
class OpTypeCounts:
    type_counts: dict[str, int]
