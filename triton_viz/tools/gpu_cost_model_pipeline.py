"""GPU pilot: collect audited controls/holdouts, fit, then evaluate frozen costs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from triton_viz.performance.calibration import fit_controls, stable_digest
from triton_viz.performance.gpu import FEATURES, expand, predict, source_configuration


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _identity(device):
    import torch
    from triton_viz.performance.gpu_measure import snapshot

    props = torch.cuda.get_device_properties(device)
    package = Path(__file__).parents[1]
    files = sorted(package.rglob("*.py"))
    return {
        "backend": "gpu-source-service-v1",
        "gpu": props.name,
        "uuid": snapshot(device)["uuid"],
        "driver": snapshot(device)["driver"],
        "sm_count": props.multi_processor_count,
        "capability": [props.major, props.minor],
        "cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "numpy", "nvidia-ml-py")
        },
        "source_digest": stable_digest(
            {
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            }
        ),
        "num_warps": 4,
        "num_stages": 2,
        "metric": "cuda_graph_steady_cache_kernel_us",
    }


def collect(root, *, device=0, role="all", resume=False, allow_idle_graphics=False):
    import torch
    import triton_viz
    from triton_viz.performance.gpu_controls import cases, prepare, check_output
    from triton_viz.performance.gpu_measure import assert_available, snapshot, measure
    from triton_viz.performance.triton_observe import observe

    baseline = snapshot(device)
    allowed_graphics = (
        tuple(p["pid"] for p in baseline["graphics_processes"])
        if allow_idle_graphics
        else ()
    )
    for _ in range(3):
        assert_available(snapshot(device), allowed_graphics=allowed_graphics)
        time.sleep(0.2)
    torch.cuda.set_device(device)
    identity = _identity(device)
    identity["allow_idle_graphics"] = allow_idle_graphics
    fingerprint = stable_digest(identity)
    manifest = {
        "identity": identity,
        "fingerprint": fingerprint,
        "splits": {r: cases(r) for r in ("control", "holdout")},
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and _read(manifest_path) != manifest:
        raise ValueError(
            "Existing run has different compiler/hardware/source/splits; use a new root"
        )
    _write(manifest_path, manifest)
    for selected_role in ("control", "holdout") if role == "all" else (role,):
        for case in cases(selected_role):
            path = (
                root
                / ("controls" if selected_role == "control" else "holdouts")
                / (case["id"] + ".json")
            )
            if resume and path.exists():
                old = _read(path)
                if not old["contaminated"] and old["fingerprint"] == fingerprint:
                    continue
            assert_available(
                snapshot(device), own_pid=os.getpid(), allowed_graphics=allowed_graphics
            )
            kernel, grid, args, out = prepare(case, "cpu")
            source = observe(kernel, grid, *args, num_warps=4, num_stages=2)
            check_output(case, out)
            work = expand(source, sm_count=identity["sm_count"])
            if work["ood_reasons"]:
                raise ValueError(f"Unsupported control source: {work['ood_reasons']}")
            triton_viz.clear()
            kernel, grid, args, out = prepare(case, f"cuda:{device}")

            def launch():
                kernel[grid](*args, num_warps=4, num_stages=2)

            for attempt in range(3):
                measured = measure(
                    launch, device=device, allowed_graphics=allowed_graphics
                )
                _write(
                    root
                    / "measurement_attempts"
                    / selected_role
                    / case["id"]
                    / f"{time.time_ns()}.json",
                    measured,
                )
                if not measured["contaminated"]:
                    break
                print(
                    f"Rejected batch {case['id']} attempt {attempt + 1}: {measured['rejection_reasons']}",
                    flush=True,
                )
                # Recheck foreign processes before any repeat. Never retry into
                # somebody else's workload or retain only fast timing samples.
                assert_available(
                    snapshot(device),
                    own_pid=os.getpid(),
                    allowed_graphics=allowed_graphics,
                )
            check_output(case, out)
            row = {
                "role": selected_role,
                "case": case,
                "cv_group": case["cv_group"],
                "fingerprint": fingerprint,
                "features": work["features"],
                "source_configuration": source_configuration(work),
                "source": source,
                **measured,
            }
            _write(path, row)
            print(
                f"{selected_role} {case['id']}: {row['latency_us']:.4f} us, contaminated={row['contaminated']}",
                flush=True,
            )
            if measured["contaminated"]:
                raise ValueError(
                    f"Rejected measurement; resume after interference settles: {measured['rejection_reasons']}"
                )


def fit(root):
    manifest = _read(root / "manifest.json")
    # Deliberately enumerate ONLY declared control paths. Never scan the run
    # recursively: holdout data must remain inaccessible to fitting.
    rows = [
        _read(root / "controls" / (case["id"] + ".json"))
        for case in manifest["splits"]["control"]
    ]
    calibration = fit_controls(rows, FEATURES, fingerprint=manifest["fingerprint"])
    calibration["source_configurations"] = sorted(
        {json.dumps(row["source_configuration"]) for row in rows}
    )
    calibration["source_configurations"] = [
        json.loads(item) for item in calibration["source_configurations"]
    ]
    _write(root / "calibration" / "candidate.json", calibration)
    _write(
        root / "calibration" / "fit_status.json",
        {
            "passed": calibration["cv"]["passed"],
            "candidate_digest": stable_digest(calibration),
        },
    )
    print(json.dumps(calibration["cv"], indent=2), flush=True)
    if not calibration["cv"]["passed"]:
        raise ValueError(
            "CV gate failed; candidate is diagnostic only and was not promoted"
        )
    calibration["digest"] = stable_digest(calibration)
    _write(root / "calibration" / "frozen.json", calibration)


def evaluate(root):
    manifest = _read(root / "manifest.json")
    calibration = _read(root / "calibration" / "frozen.json")
    digest = calibration.pop("digest")
    if digest != stable_digest(calibration):
        raise ValueError("Frozen calibration digest mismatch")
    status = _read(root / "calibration" / "fit_status.json")
    if not status["passed"] or status["candidate_digest"] != digest:
        raise ValueError("Frozen calibration is stale relative to the latest fit")
    results = []
    for case in manifest["splits"]["holdout"]:
        row = _read(root / "holdouts" / (case["id"] + ".json"))
        if (
            row["role"] != "holdout"
            or row["fingerprint"] != manifest["fingerprint"]
            or row["contaminated"]
        ):
            raise ValueError(f"Invalid holdout measurement: {case['id']}")
        result = predict(
            row["source"],
            calibration,
            fingerprint=manifest["fingerprint"],
            sm_count=manifest["identity"]["sm_count"],
            strict=False,
        )
        # OOD results remain in the all-case metric, explicitly annotated.
        result.update(
            case=case,
            measured_us=row["latency_us"],
            error_pct=100 * (result["latency_us"] / row["latency_us"] - 1),
        )
        results.append(result)
    report = {
        "backend": "gpu-source-service-v1",
        "experimental": True,
        "calibration_digest": digest,
        "case_count": len(results),
        "mape_pct": sum(abs(r["error_pct"]) for r in results) / len(results),
        "ood_count": sum(bool(r["ood_reasons"]) for r in results),
        "metric": manifest["identity"]["metric"],
        "cases": results,
    }
    _write(root / "evaluation" / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "fit", "evaluate"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Physical nvidia-smi device index; CUDA visibility must be unset",
    )
    parser.add_argument("--role", choices=("all", "control", "holdout"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-idle-graphics",
        action="store_true",
        help="Permit existing graphics PIDs after three idle checks; results are labeled shared-desktop",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "root": str(args.root),
                    "role": args.role,
                    "device": args.device,
                }
            )
        )
        return 0
    if args.stage == "collect":
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValueError(
                "Unset CUDA_VISIBLE_DEVICES and select the physical GPU with --device"
            )
        collect(
            args.root,
            device=args.device,
            role=args.role,
            resume=args.resume,
            allow_idle_graphics=args.allow_idle_graphics,
        )
    elif args.stage == "fit":
        fit(args.root)
    else:
        evaluate(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
