"""Compile corpus cases to TTIR in a clean subprocess.

    python _ttir_capture.py OUT.json CASE [CASE ...]

The TTIR is what IR mode reads (D25): host-compiled by
``tilelens.core.host_compile.HostCompiler`` for the default target
``GPUTarget("cuda", 89, 32)`` (D26), through the ``ttir`` stage, with no
device involved; ``source`` is ``"host"``. The suite thereby validates the
host compile path together with the reader on every Triton it runs on, CPU
only included. The JIT's own compile of the same launch, for the same
target, is recorded apart (``jit_ttir``, ``jit_hash``, ``jit_target``; the
host's hash is ``hash``) for the explicit host-vs-JIT check
(test_host_ttir_is_the_jit_ttir): ``jit_fn.warmup(...)`` with a stand-in
for Triton's active driver that reports a cuda:89 device, so it needs no GPU
either. Launches are built on CPU tensors, so a record does not depend on
the machine. Each record also holds the launch the static evaluator needs:
the resolved 3-D grid and every integer argument by name.

A subprocess, because the test process may have imported Triton under
``TRITON_INTERPRET=1`` (tests/unit/test_multithreading.py sets it during
collection), where nothing compiles for real. The child imports tilelens
from this checkout (``PYTHONPATH``), and compiles into a Triton cache of
this checkout's own (:func:`cache_dir`): Triton keys a compiled kernel by
its source text and first line, not its file, so a cache shared by two
checkouts of the repository hands the second one TTIR whose ``loc()``
entries name the first checkout's files.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "_corpus.py")
REPO = os.path.dirname(os.path.dirname(HERE))


def cache_dir() -> str:
    """The capture child's Triton cache: a subdirectory, named after this
    checkout's path, of the cache Triton would use (``TRITON_CACHE_DIR``,
    else ``$TRITON_HOME/.triton/cache``), so warm runs stay warm."""
    root = os.environ.get("TRITON_CACHE_DIR") or os.path.join(
        os.environ.get("TRITON_HOME") or os.path.expanduser("~"), ".triton", "cache"
    )
    tag = hashlib.sha1(os.path.realpath(HERE).encode()).hexdigest()[:12]
    return os.path.join(root, f"tilelens-conformance-{tag}")


def capture(
    names: Sequence[str], *, timeout: float = 1800.0
) -> dict[str, dict[str, Any]]:
    """case name -> {"ttir", "hash", "source", "jit_ttir", "jit_hash",
    "jit_target", "grid", "params", "file"} or {"error"}."""
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    env["TRITON_CACHE_DIR"] = cache_dir()
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO, *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ttir.json")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), out, *names],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(
                f"TTIR capture child failed ({proc.returncode}):\n{proc.stderr[-4000:]}"
            )
        with open(out, encoding="utf-8") as f:
            return json.load(f)


# ─────────────────────────── the child ───────────────────────────


def load_corpus():
    import importlib.util

    name = "tilelens_conformance_corpus"
    spec = importlib.util.spec_from_file_location(name, CORPUS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def launch_record(kernel, grid, args, kwargs) -> dict[str, Any]:
    """The launch as the static evaluator reads it: integer arguments by
    Python parameter name (constexprs included; the reader has them folded)."""
    import inspect

    import torch

    fn = kernel.fn
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    params: dict[str, int] = {}
    tensors: dict[str, Any] = {}
    for name, v in bound.arguments.items():
        if isinstance(v, torch.Tensor):
            tensors[name] = {
                "shape": list(v.shape),
                "stride": list(v.stride()),
                "dtype": str(v.dtype),
            }
        elif isinstance(v, (bool, int)):
            params[name] = int(v)
    grid3 = [int(g) for g in grid] + [1] * (3 - len(grid))
    return {
        "grid": grid3,
        "params": params,
        "tensors": tensors,
        "file": os.path.realpath(fn.__code__.co_filename),
    }


class StandInDriver:
    """What JITFunction.run asks Triton's active driver for, as on a machine
    whose device 0 is a GPU of ``target``: nothing is launched or loaded on
    a warmup, so the JIT compiles without a GPU."""

    def __init__(self, target) -> None:
        self.target = target

    def get_current_device(self) -> int:
        return 0

    def get_current_stream(self, device=None) -> int:
        return 0

    def get_current_target(self):
        return self.target


def jit_compile(kernel, grid, args, kwargs) -> tuple[str, str, list]:
    """The JIT's own compile of the launch for the default target (the
    stand-in driver's): its TTIR, hash and target."""
    from triton.runtime.driver import driver

    from tilelens.core.host_compile import default_ir_target

    previous = driver._active
    driver.set_active(StandInDriver(default_ir_target()))
    try:
        compiled = kernel.warmup(*args, grid=grid, **kwargs)
    finally:
        driver._active = previous
    target = compiled.metadata.target
    return (
        compiled.asm["ttir"],
        compiled.hash,
        [target.backend, target.arch, target.warp_size],
    )


def host_compile(kernel, args, kwargs) -> tuple[str, str]:
    """The TTIR IR mode reads, and its hash: the launch host-compiled for
    the default target, as the core compiles it (D25, D26)."""
    from tilelens.core.host_compile import HostCompiler, default_ir_target

    compiled = HostCompiler().compile(
        kernel, tuple(args), kwargs, target=default_ir_target(), stages={"ttir"}
    )
    return compiled.asm["ttir"], compiled.hash


def main(argv: list[str]) -> int:
    out_path, names = argv[0], argv[1:]
    from triton.runtime.jit import JITFunction

    corpus = load_corpus()
    cases = corpus.by_name()
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        t0 = time.monotonic()
        try:
            c = cases[name]
            if not isinstance(c.kernel, JITFunction):
                raise TypeError(
                    f"{name}: kernel is {type(c.kernel).__name__}, not a JITFunction"
                )
            grid, args, kwargs = c.build("cpu")
            rec = launch_record(c.kernel, grid, args, kwargs)
            ttir, digest = host_compile(c.kernel, args, kwargs)
            rec.update(ttir=ttir, hash=digest, source="host")
            ttir, digest, target = jit_compile(c.kernel, grid, args, kwargs)
            rec.update(jit_ttir=ttir, jit_hash=digest, jit_target=target)
        except Exception as e:  # noqa: BLE001 - reported per case
            rec = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-3000:],
            }
        rec["seconds"] = round(time.monotonic() - t0, 3)
        results[name] = rec
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
