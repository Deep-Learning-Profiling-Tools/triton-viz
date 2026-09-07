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
    parser.add_argument("--track", choices=("static", "dynamic"), required=True)
    parser.add_argument("--ttir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args()

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
        "name": ns.name,
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
    else:
        result["result"] = _dynamic_track(
            replace(spec, make_args=lambda seed: args), 0, LadderLevel.L2
        )
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"name": ns.name, "track": ns.track, "result": result["result"]}))


if __name__ == "__main__":
    main()
