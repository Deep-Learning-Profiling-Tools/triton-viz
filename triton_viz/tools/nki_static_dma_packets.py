"""Timing-free packet features for the Inf2 Static DMA calibration."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PACKET_COLUMNS = [
    "engine_idx", "transfer_bytes", "read_bytes", "write_bytes", "queue_type",
]


def _size(row: dict) -> int:
    return max(*(int(row.get(key) or 0) for key in PACKET_COLUMNS[1:4]))


def packet_fingerprint(rows: list[dict]) -> str:
    counts = Counter(
        (
            str(row.get("queue_type") or "unknown"),
            int(row.get("engine_idx") or 0),
            int(row.get("read_bytes") or 0),
            int(row.get("write_bytes") or 0),
            int(row.get("transfer_bytes") or 0),
        )
        for row in rows
        if row.get("queue_type") != "software_dynamic"
    )
    return json.dumps(
        [[*key, count] for key, count in sorted(counts.items())],
        separators=(",", ":"),
    )


def packet_features(rows: list[dict]) -> list[float]:
    dynamic = [row for row in rows if row.get("queue_type") == "software_dynamic"]
    static = [row for row in rows if row.get("queue_type") != "software_dynamic"]
    by_engine: dict[int, list[dict]] = defaultdict(list)
    bins: Counter[int] = Counter()
    queues: Counter[str] = Counter()
    sizes = []
    for row in static:
        size = _size(row)
        sizes.append(size)
        by_engine[int(row.get("engine_idx") or 0)].append(row)
        bins[min(20, max(0, int(math.log2(max(1, size)))))] += 1
        queues[str(row.get("queue_type") or "unknown")] += 1
    engine_counts = sorted((len(value) for value in by_engine.values()), reverse=True)
    engine_bytes = sorted(
        (sum(_size(row) for row in value) for value in by_engine.values()),
        reverse=True,
    )
    vector: list[float] = [
        len(static), sum(sizes), max(sizes, default=0), len(by_engine),
        statistics.fmean(sizes) if sizes else 0.0,
        statistics.pstdev(sizes) if len(sizes) > 1 else 0.0,
        *(queues[name] for name in ("input", "output", "instruction", "unknown")),
        *(bins[index] for index in range(21)),
        *((engine_counts + [0] * 16)[:16]),
        *((engine_bytes + [0] * 16)[:16]),
    ]
    dynamic_sizes = [_size(row) for row in dynamic]
    dynamic_bins = Counter(
        min(20, max(0, int(math.log2(max(1, size))))) for size in dynamic_sizes
    )
    vector += [len(dynamic), sum(dynamic_sizes), max(dynamic_sizes, default=0)]
    vector += [dynamic_bins[index] for index in range(21)]
    assert len(vector) == 87
    return [float(value) for value in vector]


@dataclass(frozen=True)
class StaticDmaPacketCalibration:
    exact_ns: dict[str, float]
    vectors: tuple[tuple[float, ...], ...]
    targets_ns: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "StaticDmaPacketCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            exact_ns={str(k): float(v) for k, v in data["stable_exact_ns"].items()},
            vectors=tuple(tuple(map(float, row)) for row in data["vectors"]),
            targets_ns=tuple(map(float, data["targets_ns"])),
            means=tuple(map(float, data["feature_means"])),
            scales=tuple(map(float, data["feature_scales"])),
        )

    def predict_ns(self, rows: list[dict]) -> tuple[float, str]:
        fingerprint = packet_fingerprint(rows)
        if fingerprint in self.exact_ns:
            return self.exact_ns[fingerprint], "stable_exact"
        vector = packet_features(rows)
        best = min(
            range(len(self.vectors)),
            key=lambda index: sum(
                abs((value - train) / scale)
                for value, train, scale in zip(
                    vector, self.vectors[index], self.scales
                )
            ),
        )
        return self.targets_ns[best], "knn1_fallback"
