"""One-time GPU capture for the tritonracebench_cutile corpus — the
cuda.tile twin implementations of the TritonRaceBench litmus rows.

Unlike the tilebench_cutile capture there is no external checkout to
pin: the kernels live in evaluation/kernels/tritonracebench_cutile.py
(this repo), and each row's arguments come from the SAME make_args
recipe (same seed, same dtypes, same contents) as its Triton twin, so
the ground-truth label carries over by construction.

Per row: build the args on the GPU (identical tensor objects stay
identical, so the aliased in-place row records one storage group),
compile the final CuTile IR (``compile_tile(..., return_final_ir=True)``
for the real device capability), LAUNCH ONCE as a smoke check (every
spin in the corpus terminates: producers and consumers are co-resident
at these grid sizes), and record IR text + arg descriptors. Rebuild
from the JSON needs neither cuda-tile nor a GPU.

Usage (GPU machine):
    uv run python -m evaluation.tritonracebench_cutile_capture          # all rows
    uv run python -m evaluation.tritonracebench_cutile_capture --one trb001_pid_stride_no
    uv run python -m evaluation.tritonracebench_cutile_capture --no-launch  # compile only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPECS_PATH = Path(__file__).parent / "kernels" / "tritonracebench_cutile_specs.json"
SEED = 0


def _describe_args(
    kernel: Any, args: tuple
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Per-arg descriptors named by the kernel's python params, the
    ct.Constant values (baked into the IR at compile time), and the
    tensor alias groups (name -> group id) for the shared fingerprint.
    (Same shape as evaluation/tilebench_cutile_capture._describe_args,
    copied so this capture stays independent of the TileBench checkout.)"""
    import torch

    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

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
            desc = {
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
            # PRE-LAUNCH element values of a small integer tensor, under the
            # Triton track's own rule (CompiledRaceDetector.pre_warmup): the
            # static track's rf-init source for atomic observations (a spin
            # cannot read its exit value from an unknown initial state).
            # Described BEFORE the smoke launch mutates the storage.
            init = CompiledRaceDetector._capture_init_values(
                val, bool(val.is_contiguous())
            )
            if init is not None:
                desc["init_values"] = list(init)
            described.append(desc)
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
    """The final CuTile IR text for this kernel at these args."""
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


def _cuda_args(raw: tuple) -> tuple:
    """Move tensors to the GPU, mapping IDENTICAL tensor objects to one
    cuda copy so aliased rows stay aliased."""
    import torch

    memo: dict[int, Any] = {}
    out = []
    for a in raw:
        if isinstance(a, torch.Tensor):
            if id(a) not in memo:
                memo[id(a)] = a.cuda()
            out.append(memo[id(a)])
        else:
            out.append(a)
    return tuple(out)


def capture_one(name: str, row: dict, do_launch: bool) -> dict:
    import torch

    import cuda.tile as ct

    kernel = row["kernel"]
    args = _cuda_args(tuple(row["make_args"](SEED)) + tuple(row["consts"]))
    ir = _compile_ir(kernel, args)
    described, constexprs, aliases = _describe_args(kernel, args)
    launch = "skipped"
    if do_launch:
        grid3 = tuple(row["grid"]) + (1,) * (3 - len(row["grid"]))
        stream = torch.cuda.current_stream()
        t0 = time.perf_counter()
        ct.launch(stream, grid3, kernel, args)
        torch.cuda.synchronize()
        launch = f"ok ({time.perf_counter() - t0:.3f}s)"
    return {
        "module": kernel._pyfunc.__module__,
        "kernel": kernel._pyfunc.__name__,
        "grid": [int(g) for g in row["grid"]],
        "args": described,
        "constexprs": constexprs,
        "aliases": aliases,
        "launch": launch,
        "ir": ir,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help="capture a single row by name")
    ap.add_argument("--out", help="output JSON (default: the corpus specs path)")
    ap.add_argument(
        "--no-launch",
        action="store_true",
        help="compile and record without launching (no smoke check)",
    )
    ns = ap.parse_args()

    import torch

    import cuda.tile as ct
    from evaluation.kernels.tritonracebench_cutile import ROWS

    names = [ns.one] if ns.one else sorted(ROWS)
    rows_out: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for name in names:
        try:
            rows_out[name] = capture_one(name, ROWS[name], not ns.no_launch)
            print(f"{name}: {rows_out[name]['launch']}")
        except Exception as e:  # noqa: BLE001 — record and continue
            failures[name] = f"{type(e).__name__}: {e}"
            print(f"{name}: FAIL {failures[name]}")

    cap = torch.cuda.get_device_capability()
    payload = {
        "meta": {
            "tritonracebench_cutile_captured_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "tritonracebench_cutile_cuda_tile": getattr(ct, "__version__", None)
            or Path(ct.__file__).parent.joinpath("VERSION").read_text().strip(),
            "tritonracebench_cutile_torch": torch.__version__,
            "tritonracebench_cutile_device": torch.cuda.get_device_name(),
            "tritonracebench_cutile_sm": f"sm_{cap[0]}{cap[1]}",
            "tritonracebench_cutile_seed": SEED,
            "tritonracebench_cutile_rows": len(rows_out),
            "tritonracebench_cutile_capture_failures": failures,
        },
        "rows": rows_out,
    }
    out = Path(ns.out) if ns.out else SPECS_PATH
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(rows_out)} rows, {len(failures)} failures)")


if __name__ == "__main__":
    main()
