from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.traceback_utils import read_source_segment


@dataclass
class LLMOpRecord:
    """LLM-facing op record with compact metadata and debugging context."""

    uuid: str
    op_type: str
    grid_idx: str
    op_index: int | None = None
    time_idx: int | None = None
    overall_key: str | None = None
    input_shape: list[int] = field(default_factory=list)
    other_shape: list[int] = field(default_factory=list)
    output_shape: list[int] = field(default_factory=list)
    global_shape: list[int] = field(default_factory=list)
    slice_shape: list[int] = field(default_factory=list)
    mem_src: str | None = None
    mem_dst: str | None = None
    bytes: int | None = None
    tracebacks: list[dict[str, Any]] = field(default_factory=list)
    code: dict[str, Any] | None = None


@dataclass
class LLMRecordsSnapshot:
    """Serializable snapshot consumed by the frontend LLM chat widget."""

    total_ops: int
    grids: list[str]
    records: list[LLMOpRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ops": self.total_ops,
            "grids": self.grids,
            "records": [asdict(r) for r in self.records],
        }


class LLMRecordStore:
    """
    Builds and caches LLM-focused records from visualizer payloads.

    This stays separate from core tracer records to avoid affecting existing
    tracing and frontend rendering contracts.
    """

    def __init__(self, code_context: int = 8):
        self._snapshot = LLMRecordsSnapshot(total_ops=0, grids=[], records=[])
        self._code_context = max(1, int(code_context))

    def update(
        self,
        visualization_data: dict[str, list[dict[str, Any]]] | None,
        raw_tensor_data: dict[str, dict[str, Any]] | None,
    ) -> None:
        viz = visualization_data or {}
        raw = raw_tensor_data or {}
        records: list[LLMOpRecord] = []
        grids = sorted(viz.keys())

        for grid_idx in grids:
            for op in viz.get(grid_idx, []) or []:
                uuid = str(op.get("uuid") or "")
                payload = raw.get(uuid, {}) if uuid else {}
                tracebacks = list(payload.get("tracebacks") or [])
                code = self._code_from_tracebacks(tracebacks)
                record = LLMOpRecord(
                    uuid=uuid,
                    op_type=str(op.get("type") or "Unknown"),
                    grid_idx=grid_idx,
                    op_index=self._safe_int(op.get("op_index")),
                    time_idx=self._safe_int(op.get("time_idx")),
                    overall_key=self._safe_str(op.get("overall_key")),
                    input_shape=self._to_int_list(op.get("input_shape")),
                    other_shape=self._to_int_list(op.get("other_shape")),
                    output_shape=self._to_int_list(op.get("output_shape")),
                    global_shape=self._to_int_list(op.get("global_shape")),
                    slice_shape=self._to_int_list(op.get("slice_shape")),
                    mem_src=self._safe_str(op.get("mem_src")),
                    mem_dst=self._safe_str(op.get("mem_dst")),
                    bytes=self._safe_int(op.get("bytes")),
                    tracebacks=tracebacks,
                    code=code,
                )
                records.append(record)

        self._snapshot = LLMRecordsSnapshot(
            total_ops=len(records), grids=grids, records=records
        )

    def get_snapshot(self) -> dict[str, Any]:
        return self._snapshot.to_dict()

    def get_record(self, uuid: str) -> dict[str, Any] | None:
        for record in self._snapshot.records:
            if record.uuid == uuid:
                return asdict(record)
        return None

    def _code_from_tracebacks(
        self, tracebacks: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not tracebacks:
            return None
        frame = tracebacks[-1] or {}
        filename = frame.get("filename")
        lineno = self._safe_int(frame.get("lineno"))
        if not filename or lineno is None:
            return None

        cwd = Path.cwd().resolve()
        try:
            resolved = Path(filename).resolve()
            # Reject path-boundary tricks (e.g. cwd /foo vs /foo-secret) vs naive startswith.
            resolved.relative_to(cwd)
        except (ValueError, OSError):
            return None
        return read_source_segment(str(resolved), lineno, self._code_context)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text if text else None

    @staticmethod
    def _to_int_list(value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)):
            return []
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
