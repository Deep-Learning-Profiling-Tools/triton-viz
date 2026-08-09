"""Operator-agnostic feature schema shared by NKI lowering and cost models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AccessPattern:
    src_space: str
    dst_space: str
    bytes: int
    partition_count: int
    free_stride_items: int
    partition_stride_bytes: int
    active_access_count: int
    access_span_bytes: int
    density: float
    item_bytes: int

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> AccessPattern | None:
        if event.get("op") not in {"load", "store", "transfer"}:
            return None
        active = int(event.get("active_access_count") or event.get("active_lanes") or 0)
        item_bytes = max(1, int(event.get("item_bytes") or 1))
        span = int(event.get("access_span_bytes") or event.get("bytes") or 0)
        density = event.get("access_density")
        if density is None:
            density = min(1.0, active * item_bytes / span) if span else 0.0
        op = str(event.get("op"))
        default_src = "hbm" if op == "load" else "sbuf"
        default_dst = "hbm" if op == "store" else "sbuf"
        return cls(
            src_space=str(event.get("mem_src") or default_src).lower(),
            dst_space=str(event.get("mem_dst") or default_dst).lower(),
            bytes=max(0, int(event.get("bytes") or 0)),
            partition_count=max(1, int(event.get("partition_count") or 1)),
            free_stride_items=max(0, int(event.get("free_stride_items") or 0)),
            partition_stride_bytes=max(
                0, int(event.get("partition_stride_bytes") or 0)
            ),
            active_access_count=max(0, active),
            access_span_bytes=max(0, span),
            density=float(density),
            item_bytes=item_bytes,
        )

    @property
    def layout_family(self) -> str:
        if self.active_access_count == 0:
            return "empty"
        if self.free_stride_items in (0, 1) and self.density >= 0.999:
            return "contiguous"
        return "strided"


@dataclass(frozen=True)
class ComputeRegion:
    dtype: str
    partition_count: int
    logical_free_dim: int
    op_histogram: tuple[tuple[str, int], ...]
    reduction_count: int
    broadcast_edge_count: int
    has_mask_or_tail: bool

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> ComputeRegion | None:
        region = event.get("region_ir")
        if not region:
            return None
        return cls(
            dtype=str(region.get("dtype") or event.get("output_dtype") or ""),
            partition_count=max(1, int(region.get("partition_count") or 1)),
            logical_free_dim=max(
                1,
                int(
                    region.get("logical_free_dim")
                    or region.get("free_dim")
                    or 1
                ),
            ),
            op_histogram=tuple(
                sorted(
                    (str(name), int(count))
                    for name, count in (region.get("op_histogram") or {}).items()
                )
            ),
            reduction_count=max(0, int(region.get("reduction_count") or 0)),
            broadcast_edge_count=max(
                0, int(region.get("broadcast_edge_count") or 0)
            ),
            has_mask_or_tail=bool(region.get("has_mask_or_tail")),
        )
