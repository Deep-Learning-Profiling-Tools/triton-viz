from dataclasses import dataclass


@dataclass
class LoadStoreBytes:
    """Aggregated byte counts for masked load/store ops.

    Attributes:
        type: Operation type label (\"load\" or \"store\").
        total_bytes_true: Bytes for lanes where mask is True.
        total_bytes_attempted: Bytes for all lanes attempted.
    """

    type: str  # "load" or "store"
    total_bytes_true: int
    total_bytes_attempted: int


@dataclass
class OpTypeCounts:
    """Counts of op types observed during a launch.

    Attributes:
        type_counts: Mapping of op type names to counts.
    """

    type_counts: dict[str, int]
