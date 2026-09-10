"""Loader stubs for an aiter checkout on the NVIDIA side.

aiter (ROCm/aiter) is not pip-installable without ROCm, and its package
``__init__`` chain hard-requires a ROCm runtime — but its Triton kernel
modules themselves are import-clean (2026-08-27 census on the b0d56a0
checkout: 275 of 313 leaf modules under ``aiter/ops/triton/`` import on
NVIDIA + triton 3.6 once the package inits are bypassed). Stubbing the
three package levels with the real ``__path__`` and a skipped
``__init__`` lets the leaf modules resolve from a plain checkout, the
tilebench local-checkout pattern. Shared by the capture driver
(``evaluation.aiter_capture``) and the corpus
(``evaluation.kernels.aiter_ops``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

AITER_ROOT = Path(
    os.environ.get("AITER_ROOT", str(Path.home() / "workspace" / "aiter-survey"))
)

# Launch kwargs of the AMD triton backend that aiter's op wrappers pass
# unconditionally; the NVIDIA backend rejects them with a KeyError, so
# the capture side pops them before the real run. Rebuilt corpus rows
# never carry them (the recorder keeps declared parameters only).
AMD_LAUNCH_KWARGS = (
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
    "instruction_sched_variant",
)


def aiter_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(AITER_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _make_dtypes_stub() -> types.ModuleType:
    """Synthetic ``aiter.dtypes``: the real one chains into the ROCm jit
    machinery (chip_info, C++ enum-header parsing), so the small surface
    the Triton ops and tests actually use (fp8/fp16/bf16/fp32,
    d_dtypes, str2tuple, fp8_e8m0, fp4x2) is synthesized instead; fp8
    is the OCP flavor, matching aiter's own non-gfx942 default."""
    import torch

    m = types.ModuleType("aiter.dtypes")
    values = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "fp8_e8m0": getattr(torch, "float8_e8m0fnu", torch.uint8),
        "fp4x2": getattr(torch, "float4_e2m1fn_x2", torch.uint8),
        "i8": torch.int8,
        "u8": torch.uint8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
    }
    for n, v in values.items():
        setattr(m, n, v)
    setattr(  # noqa: B010
        m,
        "d_dtypes",
        {
            n: values[n]
            for n in ("fp32", "fp16", "bf16", "fp8", "i8", "u8", "i16", "i32", "i64")
        },
    )

    def str2tuple(v: str):
        parts = [int(p.strip()) for p in v.strip("()").split(",") if p.strip()]
        if "," not in v and len(parts) == 1:
            return parts[0]
        return tuple(parts)

    setattr(m, "str2tuple", str2tuple)  # noqa: B010
    return m


def install_stubs() -> None:
    """sys.modules package stubs for aiter / aiter.ops / aiter.ops.triton
    / aiter.utility, plus the synthetic aiter.dtypes.

    Idempotent; raises ImportError when the checkout is missing so corpus
    loading fails loudly (the registry's fail-loud convention).
    """
    if not AITER_ROOT.is_dir():
        raise ImportError(
            f"the aiter_ops corpus needs an aiter checkout at {AITER_ROOT} "
            "(or set AITER_ROOT): git clone https://github.com/ROCm/aiter"
        )
    for name, sub in (
        ("aiter", "aiter"),
        ("aiter.ops", "aiter/ops"),
        ("aiter.ops.triton", "aiter/ops/triton"),
        ("aiter.utility", "aiter/utility"),
    ):
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        m.__path__ = [str(AITER_ROOT / sub)]
        m.__package__ = name
        sys.modules[name] = m
    if "aiter.dtypes" not in sys.modules:
        dt = _make_dtypes_stub()
        sys.modules["aiter.dtypes"] = dt
        # both `from aiter import dtypes` and `from aiter.utility import
        # dtypes` resolve through the parent attribute
        setattr(sys.modules["aiter"], "dtypes", dt)  # noqa: B010
        sys.modules["aiter.utility.dtypes"] = dt
        setattr(sys.modules["aiter.utility"], "dtypes", dt)  # noqa: B010
    if "aiter.jit.utils.chip_info" not in sys.modules:
        # The real chip_info lives on aiter's jit-internal import-path
        # hack (bare `import build_targets`) and shells out to ROCm
        # tooling; the three functions the Triton ops and tests use are
        # synthesized: an unknown gfx string routes every AMD-arch
        # check to its default/skip branch, and the CU count is the
        # CUDA device's SM count. Parent stubs carry no __path__ so
        # nothing else under aiter.jit resolves by accident.
        for name in ("aiter.jit", "aiter.jit.utils"):
            if name not in sys.modules:
                pm = types.ModuleType(name)
                pm.__path__ = []
                pm.__package__ = name
                sys.modules[name] = pm
        ci = types.ModuleType("aiter.jit.utils.chip_info")

        def _gfx() -> str:
            return "gfx000"

        def _cu_num() -> int:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).multi_processor_count
            return 64

        setattr(ci, "get_gfx", _gfx)  # noqa: B010
        setattr(ci, "get_gfx_runtime", _gfx)  # noqa: B010
        setattr(ci, "get_cu_num", _cu_num)  # noqa: B010
        sys.modules["aiter.jit.utils.chip_info"] = ci
        setattr(sys.modules["aiter.jit.utils"], "chip_info", ci)  # noqa: B010
    if "aiter.jit.core" not in sys.modules:
        # @compile_ops decorates aiter's HIP C++ ops; import-time it only
        # needs to exist. The stub keeps imports alive and turns any CALL
        # into a visible NotImplementedError (that test simply fails and
        # selects itself out of the capture).
        core = types.ModuleType("aiter.jit.core")

        def compile_ops(*_a, **_k):
            def deco(fn):
                def hip_op_stub(*args, **kwargs):
                    raise NotImplementedError(
                        "aiter HIP op unavailable under the NVIDIA stub loader"
                    )

                hip_op_stub.__name__ = getattr(fn, "__name__", "aiter_hip_op")
                return hip_op_stub

            return deco

        def get_module(*_a, **_k):
            raise NotImplementedError(
                "aiter HIP module unavailable under the NVIDIA stub loader"
            )

        for n, v in (
            ("compile_ops", compile_ops),
            ("get_module", get_module),
            ("is_experimental_enabled", lambda *a, **k: False),
            ("AITER_CSRC_DIR", str(AITER_ROOT / "csrc")),
            ("AITER_CONFIGS", str(AITER_ROOT / "aiter" / "configs")),
        ):
            setattr(core, n, v)
        sys.modules["aiter.jit.core"] = core
        setattr(sys.modules["aiter.jit"], "core", core)  # noqa: B010
    if "aiter.jit.utils.torch_guard" not in sys.modules:
        # torch_compile_guard wraps ops for torch.compile custom-op
        # registration; for capture the identity decorator suffices.
        tg = types.ModuleType("aiter.jit.utils.torch_guard")

        def torch_compile_guard(*_a, **_k):
            def deco(fn):
                return fn

            return deco

        setattr(tg, "torch_compile_guard", torch_compile_guard)  # noqa: B010
        sys.modules["aiter.jit.utils.torch_guard"] = tg
        setattr(sys.modules["aiter.jit.utils"], "torch_guard", tg)  # noqa: B010
    if not hasattr(sys.modules["aiter"], "logger"):
        import logging

        setattr(sys.modules["aiter"], "logger", logging.getLogger("aiter"))  # noqa: B010
    _install_compat_finder()


class _AliasLoader:
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        import importlib

        return importlib.import_module(self.target)

    def exec_module(self, module) -> None:
        pass


class _CompatFinder:
    """Mirror of the real aiter.ops.triton.__init__ backward-compat
    module redirects (old flat names -> reorganized nested paths),
    which the package stubs skip; the map is parsed from the real
    __init__ so it tracks the checkout."""

    PREFIX = "aiter.ops.triton."

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.PREFIX):
            return None
        new = self.mapping.get(fullname[len(self.PREFIX) :])
        if new is None:
            return None
        import importlib.util

        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(self.PREFIX + new)
        )


def _install_compat_finder() -> None:
    if any(isinstance(f, _CompatFinder) for f in sys.meta_path):
        return
    import ast
    import re

    src = (AITER_ROOT / "aiter" / "ops" / "triton" / "__init__.py").read_text()
    m = re.search(r"_BACKWARD_COMPAT_MAP\s*=\s*(\{.*?\n\})", src, re.S)
    mapping = ast.literal_eval(m.group(1)) if m else {}
    sys.meta_path.append(_CompatFinder(mapping))


def install_amd_kwarg_shim() -> None:
    """Capture-side only: strip AMD-only launch kwargs before the real run.

    Must be installed BEFORE the LaunchRecorder hooks JITFunction.run so
    the recorder's ``real_run`` is the stripped one (the recorder itself
    tolerates the extra kwargs: they are not declared parameters).
    """
    import triton

    jit_cls = triton.runtime.jit.JITFunction
    if getattr(jit_cls, "_aiter_amd_kwarg_shim", False):
        return
    orig = jit_cls.run

    def run(self, *args, **kwargs):
        for k in AMD_LAUNCH_KWARGS:
            kwargs.pop(k, None)
        return orig(self, *args, **kwargs)

    jit_cls.run = run
    jit_cls._aiter_amd_kwarg_shim = True
