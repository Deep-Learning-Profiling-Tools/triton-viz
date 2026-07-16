"""One-time GPU launch capture for the tilebench_cutile corpus —
TileBench's cuTile (cuda.tile) twin implementations, the first
non-Triton corpus.

Same TILEBENCH_ROOT checkout pin and harness-driven shape as the Triton
twin capture (evaluation/tilebench_capture): each case runs the suite's
own ``core.engine.run_benchmark_suite(op)`` with ``case_indices=[0]``
and ``report_benchmark`` stubbed out. The recorder patches
``cuda.tile.launch`` — records (kernel, grid, args) and then calls the
real launch, so the engine's verification still checks the output the
recorded launch produced.

Unlike the Triton corpora there is NO rebuild-by-import at sweep time:
cuTile compilation is a pure-Python AST pipeline, so each record
carries its CuTile IR TEXT compiled AT CAPTURE from the live args
(``compile_tile(..., return_final_ir=True)`` — no GPU or tileiras
needed to consume it later). Arg descriptors carry python param names
(the IR flattens an array param ``a`` into ``a_0``/``a_1``/... slots),
dtypes/shapes/strides, alias groups by storage pointer, and the
ct.Constant values that were baked into the IR.

Usage (GPU machine):
    uv run python -m evaluation.tilebench_cutile_capture           # all
    uv run python -m evaluation.tilebench_cutile_capture --one <case> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluation.tilebench_capture import OPS, TILEBENCH_ROOT, UPSTREAM, tilebench_commit

SPECS_PATH = Path(__file__).parent / "kernels" / "tilebench_cutile_specs.json"
PER_CASE_TIMEOUT_S = 600
# The bitonic-network operators launch one SPECIALIZATION per host-loop
# (stage, stride) step — stride is a ct.Constant, so each step is a
# distinct compiled kernel (385 raw records for 58 kernels). The stored
# payload keeps the first N per (case, kernel) and RECORDS the drop
# count (no silent caps): dropped steps differ only in baked constants,
# and each record embeds its full IR text, so an uncapped payload blows
# past the repo's large-file limit.
MAX_SPECIALIZATIONS = 2


def trim_specializations(payload: dict) -> dict:
    dropped_total = 0
    for case_entry in payload.get("cases", {}).values():
        per_kernel: dict[str, int] = {}
        kept: dict[str, dict] = {}
        dropped: list[str] = []
        for slot, rec in sorted(case_entry["kernels"].items()):
            n = per_kernel.get(rec["kernel"], 0)
            per_kernel[rec["kernel"]] = n + 1
            if n >= MAX_SPECIALIZATIONS:
                dropped.append(slot)
                continue
            kept[slot] = rec
        case_entry["kernels"] = kept
        case_entry["specializations_dropped"] = len(dropped)
        dropped_total += len(dropped)
    payload["specialization_cap"] = MAX_SPECIALIZATIONS
    payload["specializations_dropped_total"] = dropped_total
    return payload


def _describe_args(
    kernel: Any, args: tuple
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Per-arg descriptors named by the kernel's python params, the
    ct.Constant values (baked into the IR at compile time), and the
    tensor alias groups (name -> group id) for the shared fingerprint."""
    import torch

    af = kernel._annotated_function
    names = list(af.pysig.parameters)
    anns = af.parameter_annotations
    if len(names) != len(args) or len(anns) != len(args):
        raise RuntimeError(
            f"arg arity mismatch for {kernel._pyfunc.__name__}: "
            f"{len(names)} params, {len(anns)} annotations, {len(args)} args"
        )
    storage_groups: dict[int, int] = {}
    described: list[dict] = []
    constexprs: dict[str, int] = {}
    aliases: dict[str, int] = {}
    for i, (name, ann, val) in enumerate(zip(names, anns, args)):
        if getattr(ann, "constant", False):
            constexprs[name] = int(val)
            described.append({"kind": "constant", "name": name, "value": int(val)})
        elif isinstance(val, torch.Tensor):
            sp = val.untyped_storage().data_ptr()
            alias = storage_groups.setdefault(sp, i)
            aliases[name] = alias
            described.append(
                {
                    "kind": "tensor",
                    "name": name,
                    "dtype": str(val.dtype),
                    "shape": list(val.shape),
                    "strides": list(val.stride()),
                    "contiguous": bool(val.is_contiguous()),
                    "numel": int(val.numel()),
                    "elem_size": int(val.element_size()),
                    "alias": alias,
                }
            )
        elif isinstance(val, (bool, int, float)):
            described.append(
                {
                    "kind": "scalar",
                    "name": name,
                    "py_type": type(val).__name__,
                    "value": val if isinstance(val, (bool, int)) else float(val),
                }
            )
        else:
            raise RuntimeError(
                f"unsupported cuTile launch arg {name}={type(val).__name__}"
            )
    return described, constexprs, aliases


def _compile_ir(kernel: Any, args: tuple) -> str:
    """The final CuTile IR text for this kernel at these args — the
    artifact the static reader consumes. Compiled for the REAL device
    capability so dtype gates match what actually launched."""
    import torch
    from cuda.tile import compilation
    from cuda.tile._bytecode.version import BytecodeVersion
    from cuda.tile._compile import compile_tile

    cap = torch.cuda.get_device_capability()
    cc = compilation.CallingConvention.cutile_python_v2
    if callable(cc):
        cc = cc()
    sig = compilation.KernelSignature.from_kernel_args(kernel, args, cc)
    res = compile_tile(
        kernel._annotated_function,
        [sig],
        sm_arch=f"sm_{cap[0]}{cap[1]}",
        bytecode_version=BytecodeVersion.V_13_3,
        return_final_ir=True,
        return_bytecode=False,
        return_cubin=False,
    )
    return "\n".join(blk.to_string() for blk in res.final_ir)


class _CtLaunchRecorder:
    """Patches cuda.tile.launch: record first launch per specialization
    key, then run the real launch (the engine's correctness check then
    validates the very launch we recorded)."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.errors: list[str] = []
        self._orig: Any = None

    def __enter__(self) -> "_CtLaunchRecorder":
        import cuda.tile as ct

        self._orig = ct.launch

        def recording_launch(stream, grid, kernel, kernel_args, /):
            try:
                self._record(grid, kernel, tuple(kernel_args))
            except Exception as exc:  # noqa: BLE001 — capture must not alter runs
                self.errors.append(f"{type(exc).__name__}: {exc}")
            return self._orig(stream, grid, kernel, kernel_args)

        ct.launch = recording_launch
        return self

    def __exit__(self, *exc: Any) -> None:
        import cuda.tile as ct

        ct.launch = self._orig

    def _record(self, grid: Any, kernel: Any, args: tuple) -> None:
        fn = kernel._pyfunc
        described, constexprs, aliases = _describe_args(kernel, args)
        grid_t = [int(g) for g in tuple(grid)]
        key = json.dumps(
            [fn.__module__, fn.__name__, constexprs, grid_t]
            + [[d.get("dtype"), d.get("shape"), d.get("value")] for d in described],
            sort_keys=True,
        )
        if key in self.records:
            return
        self.records[key] = {
            "module": fn.__module__,
            "kernel": fn.__name__,
            "grid": grid_t,
            "args": described,
            "constexprs": constexprs,
            "aliases": aliases,
            "ir": _compile_ir(kernel, args),
        }


def _run_one(op: str) -> dict:
    import sys

    root = str(TILEBENCH_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    from core import engine as tb_engine

    tb_engine.report_benchmark = lambda *a, **k: {"mean": float("nan")}
    error = None
    with _CtLaunchRecorder() as rec:
        try:
            tb_engine.run_benchmark_suite(
                op, benchmark_overrides={"case_indices": [0], "autotune": False}
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    if rec.errors and not error:
        error = "; ".join(rec.errors[:3])
    kernels = {f"{r['kernel']}#{i}": r for i, r in enumerate(rec.records.values())}
    return {"case": op, "family": op, "error": error, "kernels": kernels}


def main() -> None:
    from evaluation.capture_common import run_case_capture

    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        out = args.out.resolve()  # _run_one chdirs into the checkout
        out.write_text(json.dumps(_run_one(args.one), indent=1))
        return

    commit = tilebench_commit()
    run_case_capture(
        "evaluation.tilebench_cutile_capture",
        {op: (op, False, None) for op in OPS},
        SPECS_PATH,
        payload_meta={
            "upstream": UPSTREAM,
            "tilebench_cutile": commit,
            "upstream_commit": commit,
            "tilebench_root": str(TILEBENCH_ROOT),
        },
        per_case_timeout_s=PER_CASE_TIMEOUT_S,
    )
    payload = trim_specializations(json.loads(SPECS_PATH.read_text()))
    SPECS_PATH.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    )
    print(
        f"trimmed to <= {MAX_SPECIALIZATIONS} specializations per kernel "
        f"({payload['specializations_dropped_total']} dropped, recorded)"
    )


if __name__ == "__main__":
    main()
