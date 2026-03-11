from pathlib import Path

import numpy as np
import torch

import triton_viz
from triton_viz.clients.sanitizer.data import OutOfBoundsRecordZ3
from triton_viz.clients.sanitizer.report import print_oob_record
from triton_viz.core.data import Grid, Launch, Load
from triton_viz.core.trace import launches
from triton_viz.utils.traceback_utils import TracebackInfo
from triton_viz.visualizer.draw import collect_grid


def test_trace_save_load_roundtrip_supports_visualizer(tmp_path: Path):
    triton_viz.clear()
    tensor = torch.arange(4, dtype=torch.float32)
    ptr = tensor.data_ptr()
    launches.append(
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
        )
    )

    path = tmp_path / "trace.tvz"
    triton_viz.save(path)
    triton_viz.clear()
    triton_viz.load(path)

    records, tensor_table, failures = collect_grid()

    assert failures == {}
    assert (0, 0, 0) in records
    assert len(records[(0, 0, 0)]) == 1
    assert ptr in tensor_table
    saved_tensor, _ = tensor_table[ptr]
    assert saved_tensor.ptr == ptr
    assert tuple(saved_tensor.shape) == (4,)
    assert saved_tensor.data.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_trace_save_load_roundtrip_supports_sanitizer_report(tmp_path: Path, capsys):
    triton_viz.clear()
    tensor = torch.arange(4, dtype=torch.float32)
    ptr = tensor.data_ptr()
    launches.append(
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
        )
    )

    path = tmp_path / "sanitizer.tvz"
    triton_viz.save(path)
    triton_viz.clear()
    triton_viz.load(path)

    print_oob_record(launches[-1].records[0])
    out = capsys.readouterr().out

    assert "Out-Of-Bounds Access Detected" in out
    assert f"Tensor base memory address: {ptr}" in out
