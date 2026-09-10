"""Pins for the harness's L1 pre-gate (``evaluation.harness._enum_track``):
only a reader-recognized await (``assumes_termination``) refuses before
executing; the reader's ``spin-shape`` refusal kind alone is not a gate
(it also covers plain carried-value ``scf.while`` iteration).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.harness import _enum_track  # noqa: E402


@triton.jit
def _copy_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))


def _spec(make_args):
    return SimpleNamespace(
        kernel_fn=_copy_kernel,
        constexprs={"BLOCK": 4},
        grid=(2,),
        make_args=make_args,
    )


def test_recognized_await_refuses_without_executing():
    def _explode(seed):
        raise AssertionError("the rung must not materialize the launch")

    row = _enum_track(
        _spec(_explode), 0, {"assumes_termination": True, "reason": "x: y"}
    )
    assert row["status"] == "unsupported"
    assert row["reason"].startswith("spin-shape:")
    assert row["instances"] == 0


def test_spin_shape_refusal_kind_alone_is_not_a_gate():
    def _args(seed):
        return (torch.zeros(8), torch.zeros(8))

    static = {
        "assumes_termination": False,
        "reason": "spin-shape: line 73: scf.while carries values (iter args or results)",
        "parse_unsupported": ["spin-shape: line 73: scf.while carries values"],
    }
    row = _enum_track(_spec(_args), 0, static)
    assert row["status"] == "ok"
    assert row["instances"] == 2
