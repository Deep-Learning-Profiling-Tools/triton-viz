import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import tilelens
from tilelens.clients.sanitizer.data import OutOfBoundsRecordZ3
from tilelens.clients.sanitizer.report import print_oob_record
from tilelens.core.data import Grid, Launch, Load
from tilelens.core.trace import launches
from tilelens.utils.traceback_utils import TracebackInfo
from tilelens.visualizer.draw import collect_grid


def _legacy_class_names(value):
    if isinstance(value, list):
        return [_legacy_class_names(item) for item in value]
    if isinstance(value, dict):
        result = {key: _legacy_class_names(item) for key, item in value.items()}
        if result.get("kind") == "dataclass":
            result["type"] = result["type"].replace("tilelens.", "triton_viz.", 1)
        return result
    return value


def _roundtrip(path: Path, launch: Launch, legacy: bool) -> None:
    launches[:] = [launch]
    tilelens.save(path)
    if legacy:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["format"] == "tilelens_trace"
            manifest["format"] = "triton_viz_trace"
            manifest["launches"] = _legacy_class_names(manifest["launches"])
            tensors = archive.read("tensors.npz")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("tensors.npz", tensors)
    tilelens.load(path)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
    ],
)  # TODO: support bf16/other dtypes with ml_dtypes
@pytest.mark.parametrize("legacy", [False, True])
def test_trace_save_load_roundtrip_supports_visualizer(tmp_path: Path, dtype, legacy):
    tensor = torch.arange(4, dtype=dtype)
    ptr = tensor.data_ptr()
    _roundtrip(
        tmp_path / "trace.tvz",
        Launch(
            grid=(1, 1, 1),
            tensors={tensor},
            records=[
                Grid(idx=(0, 0, 0)),
                Load(
                    ptr=ptr,
                    offsets=np.array([0, 4], dtype=np.int64),
                    masks=np.array([True, False]),
                ),
            ],
        ),
        legacy,
    )

    records, tensor_table, failures = collect_grid()

    assert failures == {}
    assert (0, 0, 0) in records
    assert ptr in tensor_table
    saved_tensor, _ = tensor_table[ptr]
    assert saved_tensor.ptr == ptr
    assert tuple(saved_tensor.shape) == (4,)
    assert saved_tensor.data.dtype == dtype
    assert saved_tensor.data.float().tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("legacy", [False, True])
def test_trace_save_load_roundtrip_supports_sanitizer_report(
    tmp_path: Path, capsys, legacy
):
    tensor = torch.arange(4, dtype=torch.float32)
    ptr = tensor.data_ptr()
    _roundtrip(
        tmp_path / "sanitizer.tvz",
        Launch(
            records=[
                OutOfBoundsRecordZ3(
                    op_type=Load,
                    tensor=tensor,
                    user_code_tracebacks=[
                        TracebackInfo(
                            filename="kernel.py",
                            lineno=7,
                            func_name="kernel",
                            line_of_code="x = tl.load(ptr)",
                        )
                    ],
                    violation_address=ptr + tensor.numel() * tensor.element_size(),
                    constraints=None,
                )
            ]
        ),
        legacy,
    )

    print_oob_record(launches[-1].records[0])
    out = capsys.readouterr().out

    assert "Out-Of-Bounds Access Detected" in out
    assert f"Tensor base memory address: {ptr}" in out


def test_visualizer_cli_forwards_trace_file_and_flags():
    from tilelens.visualizer_cli import main

    trace_file = Path("trace.tvz")
    with patch("tilelens.load") as mock_load, patch("tilelens.launch") as mock_launch:
        main([str(trace_file), "--port", "9000", "--no-block"])

    mock_load.assert_called_once_with(trace_file)
    mock_launch.assert_called_once_with(share=False, port=9000, block=False)
