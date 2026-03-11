from dataclasses import fields, is_dataclass
from io import BytesIO
from pathlib import Path
import json
import zipfile
from typing import Any

import numpy as np
import torch

from ..clients.profiler.data import LoadStoreBytes, OpTypeCounts
from ..clients.sanitizer.data import (
    OutOfBoundsRecord,
    OutOfBoundsRecordBruteForce,
    OutOfBoundsRecordZ3,
)
from ..utils.traceback_utils import TracebackInfo
from .data import (
    AddPtr,
    Advance,
    Allocate,
    Ashr,
    AtomicCas,
    AtomicRMW,
    BinaryOp,
    Bitcast,
    Broadcast,
    CastImpl,
    CumSum,
    Dot,
    ExpandDims,
    Fabs,
    Flip,
    FpToFp,
    Grid,
    Idiv,
    Join,
    Launch,
    Load,
    MakeBlockPointer,
    MakeRange,
    Op,
    ProgramId,
    RawLoad,
    RawStore,
    Reduce,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    Reshape,
    Rsqrt,
    Splat,
    Store,
    TensorPointerLoad,
    TensorPointerStore,
    TensorSnapshot,
    TernaryOp,
    Trans,
    Umulhi,
    UnaryOp,
)


_TRACE_FORMAT_VERSION = 1
_ARRAY_PREFIX = "arr_"
_MANIFEST_NAME = "manifest.json"
_TENSORS_NAME = "tensors.npz"
_TRACE_CLASSES = {
    f"{cls.__module__}:{cls.__qualname__}": cls
    for cls in (
        Op,
        ProgramId,
        Allocate,
        RawStore,
        Store,
        RawLoad,
        Load,
        UnaryOp,
        BinaryOp,
        TernaryOp,
        Dot,
        Flip,
        MakeRange,
        AddPtr,
        ExpandDims,
        Broadcast,
        Reduce,
        ReduceMin,
        ReduceMax,
        ReduceSum,
        Splat,
        MakeBlockPointer,
        TensorPointerLoad,
        TensorPointerStore,
        Idiv,
        Rsqrt,
        CastImpl,
        Reshape,
        Join,
        Fabs,
        Ashr,
        Advance,
        FpToFp,
        Umulhi,
        Trans,
        CumSum,
        Bitcast,
        AtomicCas,
        AtomicRMW,
        TensorSnapshot,
        Grid,
        Launch,
        TracebackInfo,
        LoadStoreBytes,
        OpTypeCounts,
        OutOfBoundsRecord,
        OutOfBoundsRecordBruteForce,
        OutOfBoundsRecordZ3,
    )
}
# _TRACE_CLASSES rebuilds serialized dataclass instances like Load(...), Dot(...),
# Launch(...), etc. _TRACE_OP_TYPES is narrower: it restores class-valued fields
# such as OutOfBoundsRecord.op_type = Load, where the payload stores the class
# object itself rather than an instance of that class.
_TRACE_OP_TYPES = {
    "Load": Load,
    "Store": Store,
}


def _store_array(arrays: dict[str, np.ndarray], value) -> dict[str, str]:
    """Store an array payload in the NPZ sidecar and return its manifest ref."""
    key = f"{_ARRAY_PREFIX}{len(arrays)}"
    arrays[key] = np.asarray(value)
    return {"kind": "array", "key": key}


def _encode_trace_value(value, arrays: dict[str, np.ndarray]):
    """Convert a trace object graph into JSON-safe manifest data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, TensorSnapshot):
        return {
            "kind": "dataclass",
            "type": f"{TensorSnapshot.__module__}:{TensorSnapshot.__qualname__}",
            "fields": {
                "ptr": value.ptr,
                "dtype": value.dtype,
                "_stride": _encode_trace_value(value.stride(), arrays),
                "shape": _encode_trace_value(value.shape, arrays),
                "_element_size": value.element_size(),
                "data": _store_array(arrays, value.data.detach().cpu().numpy()),
                "device": value.device,
                "_contiguous": value.is_contiguous(),
            },
        }
    if isinstance(value, type):
        name = value.__name__
        if name not in _TRACE_OP_TYPES:
            raise TypeError(f"Unsupported trace class reference: {value}")
        return {"kind": "type", "name": name}
    if hasattr(value, "data_ptr") and hasattr(value, "detach"):
        return _encode_trace_value(TensorSnapshot.from_tensor(value), arrays)
    if isinstance(value, np.ndarray):
        return _store_array(arrays, value)
    if isinstance(value, list):
        return [_encode_trace_value(item, arrays) for item in value]
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_trace_value(item, arrays) for item in value],
        }
    if isinstance(value, set):
        return {
            "kind": "set",
            "items": [_encode_trace_value(item, arrays) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                {
                    "key": _encode_trace_value(key, arrays),
                    "value": _encode_trace_value(item, arrays),
                }
                for key, item in value.items()
            ],
        }
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        cls_key = f"{cls.__module__}:{cls.__qualname__}"
        if cls_key not in _TRACE_CLASSES:
            raise TypeError(f"Unsupported trace dataclass: {cls_key}")
        payload_fields: dict[str, Any] = {}
        for field in fields(value):
            field_value = getattr(value, field.name)
            if isinstance(value, OutOfBoundsRecordZ3) and field.name in {
                "constraints",
                "symbolic_expr",
            }:
                field_value = None if field_value is None else str(field_value)
            payload_fields[field.name] = _encode_trace_value(field_value, arrays)
        payload = {
            "kind": "dataclass",
            "type": cls_key,
            "fields": payload_fields,
        }
        return payload
    raise TypeError(f"Unsupported trace value: {type(value)}")


def _tensor_from_array(array: np.ndarray) -> torch.Tensor:
    """Rebuild a standalone CPU tensor from an NPZ-loaded array."""
    return torch.from_numpy(np.asarray(array).copy())


def _build_dataclass(cls: type, kwargs: dict):
    """Instantiate a dataclass without calling generated __init__/__post_init__."""
    obj: Any = object.__new__(cls)
    setter = object.__setattr__
    for field in fields(cls):
        if field.name in kwargs:
            setter(obj, field.name, kwargs[field.name])
    return obj


def _decode_trace_value(value, arrays: dict[str, np.ndarray]):
    """Reconstruct trace objects from manifest data and NPZ payloads."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_decode_trace_value(item, arrays) for item in value]
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported trace payload: {type(value)}")

    kind = value.get("kind")
    if kind == "array":
        return np.asarray(arrays[value["key"]]).copy()
    if kind == "path":
        return Path(value["value"])
    if kind == "tuple":
        return tuple(_decode_trace_value(item, arrays) for item in value["items"])
    if kind == "set":
        return {_decode_trace_value(item, arrays) for item in value["items"]}
    if kind == "dict":
        return {
            _decode_trace_value(item["key"], arrays): _decode_trace_value(
                item["value"], arrays
            )
            for item in value["items"]
        }
    if kind == "type":
        return _TRACE_OP_TYPES[value["name"]]
    if kind == "dataclass":
        cls = _TRACE_CLASSES.get(value["type"])
        if cls is None:
            raise TypeError(f"Unsupported trace dataclass: {value['type']}")
        kwargs = {
            name: _decode_trace_value(field_value, arrays)
            for name, field_value in value["fields"].items()
        }
        if cls is TensorSnapshot:
            kwargs["data"] = _tensor_from_array(kwargs["data"])
        return _build_dataclass(cls, kwargs)
    raise TypeError(f"Unsupported trace payload kind: {kind}")


def save(path: str | Path) -> Path:
    """Save the current trace launches to a zip archive."""
    from .trace import launches

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "format": "triton_viz_trace",
        "version": _TRACE_FORMAT_VERSION,
        "launches": _encode_trace_value(launches, arrays),
    }
    tensor_bytes = BytesIO()
    np.savez_compressed(tensor_bytes, **arrays)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_MANIFEST_NAME, json.dumps(manifest).encode("utf-8"))
        archive.writestr(_TENSORS_NAME, tensor_bytes.getvalue())
    return path


def load(path: str | Path, append: bool = False) -> list[Launch]:
    """Load trace launches from a zip archive into the global trace state."""
    from .trace import clear, launches

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(_MANIFEST_NAME))
        if manifest.get("format") != "triton_viz_trace":
            raise TypeError(f"{path} is not a triton-viz trace archive")
        if manifest.get("version") != _TRACE_FORMAT_VERSION:
            raise TypeError(
                f"Unsupported trace format version: {manifest.get('version')}"
            )
        with np.load(BytesIO(archive.read(_TENSORS_NAME)), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    loaded = _decode_trace_value(manifest["launches"], arrays)
    if not isinstance(loaded, list) or any(
        not isinstance(launch, Launch) for launch in loaded
    ):
        raise TypeError(f"{path} does not contain triton-viz trace launches")
    if not append:
        clear()
    launches.extend(loaded)
    return launches
