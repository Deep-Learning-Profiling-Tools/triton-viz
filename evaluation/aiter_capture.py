"""Capture the aiter Triton-op corpus from aiter's own triton tests.

NVIDIA-side capture over a plain ROCm/aiter checkout (AITER_ROOT, the
tilebench local-checkout pattern; commit-pinned, see
``kernels/_aiter_loader.py``). Each case is one file of
``op_tests/triton_tests/test_*.py``, run in its own subprocess under
pytest with (a) the package stubs (aiter's real inits require ROCm),
(b) the AMD-launch-kwarg strip shim (waves_per_eu etc.; the NVIDIA
backend rejects them), and (c) the shared LaunchRecorder hooked on
JITFunction.run. Whatever the tests actually launch on this GPU is
recorded (first launch per kernel per case, cross-case full-record
dedup by ``fingerprint``); AMD-gated or otherwise failing test params
simply do not launch and thus select themselves out, and a case whose
every test fails is recorded under ``capture_failures``.

Known selection effects on NVIDIA (2026-08-27 survey, TODO.md rq2 in
the paper repo): gemm-family wrappers look up per-arch config tables
(``configs/<cc>/``) that ship only for gfx architectures, so those
tests fail before launching; iris-comms and the two ROCm-only utils
modules cannot import; gluon kernels resolve only if triton's gluon
accepts them. The capture records reality: only what launched lands in
the specs.

Usage:
  python -m evaluation.aiter_capture                 # all cases
  python -m evaluation.aiter_capture --one test_softmax --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.capture_common import (
    SIG_FOR_DTYPE,
    LaunchRecorder,
    run_case_capture,
    write_case_result,
)
from evaluation.kernels._aiter_loader import (
    AITER_ROOT,
    aiter_commit,
    install_amd_kwarg_shim,
    install_stubs,
)

SPECS_PATH = Path(__file__).parent / "kernels" / "aiter_ops_specs.json"
TESTS_DIR = AITER_ROOT / "op_tests" / "triton_tests"


def _cases() -> dict[str, Path]:
    """All test files, RECURSIVELY: the suite nests most tests in family
    subdirectories (attention/, gemm/, moe/, ...). Case names join the
    relative path with '__' (they become temp-file prefixes)."""
    out = {}
    for p in sorted(TESTS_DIR.rglob("test_*.py")):
        rel = p.relative_to(TESTS_DIR)
        case = "__".join(rel.with_suffix("").parts)
        out[case] = p
    return out


def _capture_one(case: str, out: Path) -> None:
    import pytest
    import torch  # noqa: F401 — fail early if torch is broken
    import triton

    install_stubs()
    install_amd_kwarg_shim()
    test_file = _cases()[case]
    recorder = LaunchRecorder(key=lambda fn: f"{fn.fn.__module__}.{fn.__name__}")
    error = None
    with recorder.hooked():
        try:
            rc = pytest.main(
                [str(test_file), "-q", "--no-header", "-p", "no:cacheprovider"]
            )
            if rc not in (0, 1):  # 1 = some tests failed; still useful
                error = f"pytest exit code {rc}"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    # Keep only kernels that resolve inside the checkout's namespace;
    # anything else (runtime codegen, third-party jit) cannot rebuild.
    # Also drop records with unrebuildable tensor dtypes: the recorder
    # fires BEFORE the real run, so a launch the NVIDIA backend then
    # rejected (AMD fp8 fnuz flavors) still left a record.
    kept, skipped = {}, dict(recorder.skipped)
    for slot, rec in recorder.captured.items():
        mod = rec.get("module") or ""
        if not mod.startswith("aiter.ops.triton"):
            skipped[slot] = f"outside aiter.ops.triton (module {mod!r})"
            continue
        bad = sorted(
            {
                d["dtype"]
                for d in rec["args"]
                if d["kind"] == "tensor" and d["dtype"] not in SIG_FOR_DTYPE
            }
        )
        if bad:
            skipped[slot] = f"unrebuildable tensor dtype(s) {bad}"
            continue
        kept[slot] = rec

    write_case_result(
        {
            "case": case,
            "family": case.removeprefix("test_"),
            "error": error,
            "kernels": kept,
            "skipped_kernels": skipped,
            "triton": triton.__version__,
            "_values": recorder.values,  # the int/bool snapshots, beside the JSON
        },
        out,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--one", metavar="CASE")
    ap.add_argument("--out", type=Path)
    ns = ap.parse_args()
    if ns.one:
        _capture_one(ns.one, ns.out)
        return 0
    commit = aiter_commit()
    run_case_capture(
        runner_module="evaluation.aiter_capture",
        cases=_cases(),
        specs_path=SPECS_PATH,
        payload_meta={
            "upstream": "https://github.com/ROCm/aiter",
            "aiter": commit,
            "upstream_commit": commit,
        },
        per_case_timeout_s=600,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
