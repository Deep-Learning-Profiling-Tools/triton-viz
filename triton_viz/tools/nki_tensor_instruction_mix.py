"""Timing-free Tensor instruction-mix features and frozen KNN calibration."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


INSTRUCTION_COLUMNS = ["engine", "opcode", "tensor_instruction_type", "operands"]


def tensor_mix_features(rows: list[dict]) -> list[float]:
    tensor = [row for row in rows if str(row.get("engine") or "").lower() == "tensor"]
    types = Counter(str(row.get("tensor_instruction_type") or "") for row in tensor)
    opcodes = Counter(str(row.get("opcode") or "") for row in tensor)
    geometry = []
    for row in tensor:
        if row.get("opcode") not in {"MATMUL", "LDWEIGHTS"}:
            continue
        operands = str(row.get("operands") or "")
        geometry += [int(value) for value in re.findall(r"\[(\d+),1,1\]", operands)]
        geometry += [
            int(value)
            for pair in re.findall(r"\b(\d+)\*(\d+)\b", operands)
            for value in pair
        ]
    return [
        float(types["REGULAR"]), float(types["TRANSPOSE"]),
        float(opcodes["MATMUL"]), float(opcodes["LDWEIGHTS"]),
        float(len(tensor)),
        statistics.fmean(geometry) if geometry else 0.0,
        float(max(geometry, default=0)), float(sum(geometry)),
    ]


@dataclass(frozen=True)
class TensorInstructionMixCalibration:
    vectors: tuple[tuple[float, ...], ...]
    targets_ns: tuple[float, ...]
    scales: tuple[float, ...]
    neighbors: int

    @classmethod
    def from_json(cls, path: str | Path) -> "TensorInstructionMixCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vectors=tuple(tuple(map(float, row)) for row in data["vectors"]),
            targets_ns=tuple(map(float, data["targets_ns"])),
            scales=tuple(map(float, data["feature_scales"])),
            neighbors=int(data["neighbors"]),
        )

    def predict_ns(self, rows: list[dict]) -> tuple[float, str]:
        vector = tensor_mix_features(rows)
        if vector[1] <= 0 or vector[0] <= 0:
            return 0.0, "not_mixed"
        domain_max = [max(column) for column in zip(*self.vectors)]
        if any(value > maximum for value, maximum in zip(vector[:5], domain_max[:5])):
            return 0.0, "instruction_count_ood"
        distances = sorted(
            (
                sum(abs((value - train) / scale) for value, train, scale in zip(vector, row, self.scales)),
                target,
            )
            for row, target in zip(self.vectors, self.targets_ns)
        )[: self.neighbors]
        if distances[0][0] == 0:
            exact = [target for distance, target in distances if distance == 0]
            return statistics.fmean(exact), "exact_feature"
        weights = [1.0 / distance for distance, _target in distances]
        predicted = sum(weight * target for weight, (_distance, target) in zip(weights, distances)) / sum(weights)
        return predicted, f"knn{self.neighbors}_mixed"
