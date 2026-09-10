"""Focused conformance diagnostics, never a formal timing publication.

Use one fresh subprocess per case/track. Static probes consume a saved TTIR
file and disable replay so the IR decision can be diagnosed independently.
Input hashes bind the actual reconstructed tensors passed to the frontend.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--track", choices=("static", "dynamic", "enum"), required=True)
    parser.add_argument("--dynamic-budget", type=float, default=60.0)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--stack-delay", type=float)
    parser.add_argument("--ttir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args()
    if ns.stack_delay is not None:
        import faulthandler

        faulthandler.dump_traceback_later(ns.stack_delay)

    import torch
    from evaluation.kernels import load
    from evaluation.harness import _dynamic_track, _launch_binding, _static_result
    from triton_viz.clients.common.ttir_reader import parse_ttir
    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
    from triton_viz.clients.race_detector.ladder import LadderLevel
    from triton_viz.core.config import config as cfg

    cfg.race_detector_fence_order = True
    cfg.enable_race_detector = True
    cfg.num_sms = 1
    spec = next(s for s in load(ns.corpus).specs if s.name == ns.name)
    args = spec.make_args(0)
    bound = _launch_binding(spec, args)
    source_configuration = ns.name
    if ns.head_dim is not None:
        old_d = int(bound["D"])
        new_d = ns.head_dim
        if new_d < 2 or new_d > old_d or new_d & (new_d - 1):
            parser.error("head dimension must be a smaller power of two")
        for name, value in bound.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim == 4
                and value.shape[-1] == old_d
            ):
                bound[name] = value[..., :new_d]
        bound["D"] = new_d
        bound["LOG2_D"] = new_d.bit_length() - 1
        spec = replace(
            spec,
            constexprs={
                **spec.constexprs,
                "D": new_d,
                "LOG2_D": new_d.bit_length() - 1,
            },
        )
        source_configuration += f"__diagnostic_D{new_d}"
    if ns.sequence_length is not None:
        old_s = int(bound["S"])
        new_s = ns.sequence_length
        chunks = int(bound["num_chunks"])
        if new_s < chunks or new_s > old_s or new_s % chunks:
            parser.error("reduced sequence must cover all chunks evenly")
        for name, value in bound.items():
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                axes = [axis for axis, size in enumerate(value.shape) if size == old_s]
                if len(axes) == 1:
                    slices = [slice(None)] * value.ndim
                    slices[axes[0]] = slice(0, new_s)
                    bound[name] = value[tuple(slices)]
        bound["S"] = new_s
        bound["chunk_size"] = new_s // chunks
        source_configuration += f"__diagnostic_S{new_s}"
    if ns.head_dim is not None or ns.sequence_length is not None:
        arg_names = [
            name for name in spec.kernel_fn.arg_names if name not in spec.constexprs
        ]
        args = tuple(bound[name] for name in arg_names)
    inputs = {}
    for name, value in bound.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            inputs[name] = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "logical_bytes_sha256": sha256(tensor.numpy().tobytes()),
            }
        else:
            inputs[name] = {"scalar": repr(value)}
    source = inspect.getsource(spec.kernel_fn.fn)
    result = {
        "kind": "conformance-diagnostic-not-timing",
        "name": source_configuration,
        "original_configuration": ns.name,
        "sequence_length_override": ns.sequence_length,
        "head_dim_override": ns.head_dim,
        "corpus": ns.corpus,
        "track": ns.track,
        "ladder_level": "L2",
        "grid": list(spec.grid),
        "seed": 0,
        "inputs": inputs,
        "kernel_source_sha256": sha256(source.encode()),
    }
    if ns.track == "static":
        if ns.ttir is None:
            parser.error("--ttir is required for static probes")
        ttir = ns.ttir.read_text()
        result["ttir_sha256"] = sha256(ttir.encode())
        result["ttir_path"] = str(ns.ttir.resolve())
        graph = parse_ttir(ttir, multipath=True)
        result["dependencies"] = [
            {
                "access": i,
                "kind": a.kind,
                "line": a.line_no,
                "source": None
                if a.loc is None
                else [a.loc.file, a.loc.line, a.loc.col],
                "deps": list(getattr(a, "deps", ())),
            }
            for i, a in enumerate(graph.accesses)
        ]
        det = CompiledRaceDetector(
            confirm_races=False, differential_check=False, ladder_level=LadderLevel.L2
        )
        det.pre_warmup_callback(spec.kernel_fn, grid=spec.grid, **bound)
        det.post_warmup_callback(spec.kernel_fn, SimpleNamespace(asm={"ttir": ttir}))
        det.finalize()
        result["result"] = _static_result(det, 0.0, None)
        result["result"].pop("time_s", None)
    elif ns.track == "dynamic":
        import evaluation.harness as harness

        harness.DYNAMIC_TIMEOUT_S = ns.dynamic_budget
        result["diagnostic_dynamic_budget_s"] = ns.dynamic_budget
        result["result"] = _dynamic_track(
            replace(spec, make_args=lambda seed: args), 0, LadderLevel.L2
        )
    else:
        from triton_viz.clients.race_detector.concrete_enum import enumerate_launch

        outcome = enumerate_launch(spec.kernel_fn, (), bound, spec.grid)
        result["result"] = {
            "status": outcome.status,
            "reason": outcome.reason,
            "n_reports": len(outcome.reports),
            "n_instances": outcome.n_instances,
            "grid": None if outcome.grid is None else list(outcome.grid),
        }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"name": ns.name, "track": ns.track, "result": result["result"]}))


if __name__ == "__main__":
    main()
