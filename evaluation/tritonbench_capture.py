"""One-time launch capture for the vendored TritonBench_G_v1 corpus.

Each vendored file is a standalone operator whose test block executes AT
IMPORT TIME on CUDA, so this step needs a GPU machine — it runs every
file in a subprocess with a ``JITFunction.run`` hook that records, per
(file, kernel), the FIRST real launch: the full name→value binding split
into runtime args and constexprs, tensor descriptors (shape / dtype /
init class / contiguity / alias group), exact scalars, and the resolved
grid. The result is ``kernels/tritonbench_g_specs.json``; the corpus
module rebuilds CPU launches from it on any machine (no GPU, no test
blocks — only each file's pre-separator kernel section is executed).

Reconstruction is by-descriptor, not by-value: float tensors are seeded
randn/rand/zeros, int tensors randint over the OBSERVED value range (so
index tensors stay in-bounds). Aliased pointer args (in-place ops) are
rebuilt from one tensor and the spec is marked ``aliased``.
Non-contiguous tensors are recorded and the file is SKIPPED with a
reason — stride scalars captured from a strided layout would misdescribe
a contiguous rebuild.

Usage (GPU machine):
    uv run python -m evaluation.tritonbench_capture            # all files
    uv run python -m evaluation.tritonbench_capture --one <path> --out <json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent / "kernels" / "tritonbench_g_v1"
SPECS_PATH = Path(__file__).parent / "kernels" / "tritonbench_g_specs.json"
SEPARATOR = "#" * 100  # files use a ~146-char run; prefix match is enough
PER_FILE_TIMEOUT_S = 300

_TORCH_DTYPES = {
    "torch.float32": "*fp32",
    "torch.float16": "*fp16",
    "torch.bfloat16": "*bf16",
    "torch.float64": "*fp64",
    "torch.int64": "*i64",
    "torch.int32": "*i32",
    "torch.int16": "*i16",
    "torch.int8": "*i8",
    "torch.uint8": "*u8",
    "torch.bool": "*i1",
}


def _capture_one(path: Path) -> dict:
    import torch
    import triton
    from triton.runtime.jit import JITFunction

    captured: dict[str, dict] = {}
    skipped_kernels: dict[str, str] = {}

    def describe_tensor(t: "torch.Tensor") -> dict:
        d = {
            "kind": "tensor",
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "contiguous": bool(t.is_contiguous()),
        }
        if t.numel() == 0:
            d["init"] = "zeros"
        elif t.dtype.is_floating_point:
            d["init"] = "zeros" if bool((t == 0).all()) else "randn"
        elif t.dtype == torch.bool:
            d["init"] = "randbool"
        else:
            lo = int(t.min().item()) if t.numel() else 0
            hi = int(t.max().item()) if t.numel() else 0
            d["init"] = "randint"
            d["low"], d["high"] = lo, hi + 1
        return d

    def describe(v):
        if isinstance(v, torch.Tensor):
            return describe_tensor(v)
        if isinstance(v, bool):
            return {"kind": "scalar", "sig": "i1", "value": v}
        if isinstance(v, int):
            sig = "i64" if abs(v) >= 2**31 else "i32"
            return {"kind": "scalar", "sig": sig, "value": v}
        if isinstance(v, float):
            return {"kind": "scalar", "sig": "fp32", "value": v}
        if v is None:
            return {"kind": "none"}
        return {"kind": "unsupported", "type": type(v).__name__}

    real_run = JITFunction.run

    def hooked_run(self, *args, **kwargs):
        try:
            _record(self, args, dict(kwargs))
        except Exception as exc:  # noqa: BLE001 — capture must not break the run
            skipped_kernels.setdefault(self.__name__, f"capture error: {exc}")
        return real_run(self, *args, **kwargs)

    def _record(fn, args, kwargs) -> None:
        if fn.__name__ in captured or fn.__name__ in skipped_kernels:
            return
        if kwargs.get("warmup"):
            return
        grid = kwargs.pop("grid", None)
        if grid is None:
            return
        launch_opts = {
            "num_warps",
            "num_stages",
            "num_ctas",
            "enable_fp_fusion",
            "extern_libs",
            "stream",
            "device",
            "device_type",
            "debug",
            "maxnreg",
            "warmup",
            "launch_cooperative_grid",
            "launch_pdl",
        }
        meta = dict(zip(fn.arg_names, args))
        for k, v in kwargs.items():
            if k not in launch_opts:
                meta[k] = v
        params = {p.name: p for p in fn.params}
        for n in fn.arg_names:
            if n not in meta and params[n].has_default:
                meta[n] = params[n].default
        unbound = [n for n in fn.arg_names if n not in meta]
        if unbound:
            skipped_kernels[fn.__name__] = f"unbound params {unbound}"
            return
        g = grid(meta) if callable(grid) else grid
        g = tuple(int(x) for x in (g if isinstance(g, (tuple, list)) else (g,)))

        # alias groups over tensor args (in-place ops pass one tensor twice)
        ptrs: dict[int, str] = {}
        aliases: dict[str, str] = {}
        runtime_args = []
        constexprs = {}
        for name in fn.arg_names:
            v = meta[name]
            if params[name].is_constexpr:
                cv = getattr(v, "value", v)
                if not isinstance(cv, (int, float, bool, str, type(None))):
                    skipped_kernels[
                        fn.__name__
                    ] = f"non-literal constexpr {name}={type(cv).__name__}"
                    return
                constexprs[name] = cv
                continue
            d = describe(v)
            if d["kind"] == "unsupported":
                skipped_kernels[fn.__name__] = f"arg {name}: {d['type']}"
                return
            if d["kind"] == "tensor":
                if not d["contiguous"]:
                    skipped_kernels[fn.__name__] = f"non-contiguous arg {name}"
                    return
                p = v.data_ptr()
                if p in ptrs:
                    aliases[name] = ptrs[p]
                else:
                    ptrs[p] = name
            d["name"] = name
            runtime_args.append(d)

        captured[fn.__name__] = {
            "kernel": fn.__name__,
            "args": runtime_args,
            "constexprs": constexprs,
            "grid": list(g),
            "aliases": aliases,
        }

    src = path.read_text()
    JITFunction.run = hooked_run
    error = None
    try:
        exec(  # noqa: S102 — trusted vendored corpus
            compile(src, str(path), "exec"), {"__name__": f"tb_{path.stem}"}
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        JITFunction.run = real_run

    return {
        "file": path.name,
        "error": error,
        "kernels": captured,
        "skipped_kernels": skipped_kernels,
        "triton": triton.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one:
        result = _capture_one(args.one)
        args.out.write_text(json.dumps(result, indent=1))
        return

    files = sorted(VENDOR_DIR.glob("*.py"))
    merged: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for i, f in enumerate(files, 1):
        out = Path(f"/tmp/tb_capture_{f.stem}.json")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.tritonbench_capture",
                    "--one",
                    str(f),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                timeout=PER_FILE_TIMEOUT_S,
                cwd=Path(__file__).parent.parent,
            )
            if proc.returncode != 0:
                failures[f.name] = (proc.stderr or "").strip()[-300:]
                print(f"[{i}/{len(files)}] {f.name}: CRASH")
                continue
            result = json.loads(out.read_text())
        except subprocess.TimeoutExpired:
            failures[f.name] = f"timeout after {PER_FILE_TIMEOUT_S}s"
            print(f"[{i}/{len(files)}] {f.name}: TIMEOUT")
            continue
        finally:
            out.unlink(missing_ok=True)
        if result["error"] and not result["kernels"]:
            failures[f.name] = result["error"][:300]
            print(f"[{i}/{len(files)}] {f.name}: ERROR ({result['error'][:80]})")
            continue
        merged[f.name] = result
        n = len(result["kernels"])
        note = (
            f" (+error after capture: {result['error'][:60]})"
            if result["error"]
            else ""
        )
        print(f"[{i}/{len(files)}] {f.name}: {n} kernel(s){note}")

    payload = {
        "upstream": "https://github.com/thunlp/TritonBench data/TritonBench_G_v1",
        "upstream_commit": "603e28a5050e8c268f6883a69709d477a272d49a",
        "files": merged,
        "capture_failures": failures,
    }
    SPECS_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    total = sum(len(r["kernels"]) for r in merged.values())
    print(
        f"\ncaptured {total} launches from {len(merged)}/{len(files)} files "
        f"({len(failures)} failures) -> {SPECS_PATH}"
    )


if __name__ == "__main__":
    main()
