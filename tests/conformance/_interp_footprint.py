"""The footprint Triton's interpreter actually touches, per access site.

    python _interp_footprint.py OUT.json CASE [CASE ...]

Runs each corpus case in a subprocess under ``TRITON_INTERPRET=1`` on CPU
tensors (``CUDA_VISIBLE_DEVICES=""``): every program and loop iteration
executes with the interpreter's fixed-width numpy integers. The
interpreter builder's loads, stores and atomics are instrumented (the
approach of ir_mode_audit/probes_phase3/soundness/oracle.py): each active
lane's address is attributed to the tensor argument whose storage holds it
and recorded as a ``(pid_0, pid_1, pid_2, element offset)`` point, the
offset relative to that argument's ``data_ptr()`` and the program the
builder's ``grid_idx``, keyed ``(argument, kind, source line)`` like the
static side, the line being the innermost frame in a corpus source file.
Each site also records the element width (bits) of the pointers it
accessed.

Lanes outside every argument's storage (wild) are masked off, or for a
CAS redirected to scratch, so the interpreter never touches unmapped
memory; they are counted and make the case an error. Nothing here imports
tilelens: the interpreter is the independent oracle.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from types import FrameType
from typing import Any, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_TIMEOUT_S = int(os.environ.get("TILELENS_CONFORMANCE_CASE_TIMEOUT", "300"))


def run_interpreter(
    names: Sequence[str], *, timeout: float = 3600.0
) -> dict[str, dict[str, Any]]:
    """case name -> {"sites": [{"arg", "kind", "line", "file", "bits", "points"}],
    "errors": [...], "wild": n}; a point is [pid_0, pid_1, pid_2, element offset]."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "interp.json")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), out, *names],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(
                f"interpreter child failed ({proc.returncode}):\n{proc.stderr[-4000:]}"
            )
        with open(out, encoding="utf-8") as f:
            return json.load(f)


# ─────────────────────────── the child ───────────────────────────


class _State:
    def __init__(self) -> None:
        self.files: frozenset[str] = frozenset()
        self.reset()

    def reset(self) -> None:
        # (arg, kind, line, file) -> (pid_0, pid_1, pid_2, element offset) points
        self.sites: dict[tuple[str, str, int, str], set[tuple[int, int, int, int]]] = {}
        # (arg, kind, line, file) -> element widths (bits) of the accessed pointers
        self.bits: dict[tuple[str, str, int, str], set[int]] = {}
        self.errors: list[str] = []
        self.wild = 0
        # (name, data_ptr, element size, storage lo, storage hi) per tensor argument
        self.regions: list[tuple[str, int, int, int, int]] = []

    def error(self, msg: str) -> None:
        if msg not in self.errors:
            self.errors.append(msg)


STATE = _State()
_REALPATH: dict[str, str] = {}
_POSITIONS: dict[Any, list] = {}  # code object -> its co_positions()


def _site_line() -> tuple[int, str] | None:
    """(line, file) of the innermost frame in a corpus source file. The
    call it is in must sit on one line: Triton locates a wrapped call's op
    at another of its lines than the frame reports."""
    f: FrameType | None = sys._getframe(2)
    while f is not None:
        code = f.f_code
        path = _REALPATH.get(code.co_filename)
        if path is None:
            path = _REALPATH[code.co_filename] = os.path.realpath(code.co_filename)
        if path in STATE.files:
            positions = _POSITIONS.get(code)
            if positions is None:
                positions = _POSITIONS[code] = list(code.co_positions())
            start, end = positions[f.f_lasti // 2][:2]
            if start != end:
                STATE.error(
                    f"a memory-op call spans lines {start}-{end} of {path}: "
                    "keep it on one line"
                )
            return f.f_lineno, path
        f = f.f_back
    return None


def _bind(bound: dict[str, Any]) -> None:
    import torch

    STATE.regions = []
    seen: dict[int, str] = {}
    for name, v in bound.items():
        if not isinstance(v, torch.Tensor):
            continue
        storage = v.untyped_storage()
        lo = storage.data_ptr()
        if lo in seen:
            STATE.error(
                f"arguments {seen[lo]!r} and {name!r} share a storage: attribution is ambiguous"
            )
        seen[lo] = name
        STATE.regions.append(
            (name, v.data_ptr(), v.element_size(), lo, lo + storage.nbytes())
        )


def _record(kind: str, ptrs, mask, grid_idx):
    """Record the active lanes of one memory op run by program ``grid_idx``;
    return the lane mask with wild lanes off."""
    import numpy as np

    p = np.asarray(ptrs.data).astype(np.uint64)
    if mask is None:
        m = np.ones(p.shape, dtype=bool)
    else:
        # a TensorHandle, or a bare array (materialized block pointers)
        raw = mask if isinstance(mask, np.ndarray) else mask.data
        m = np.broadcast_to(np.asarray(raw).astype(bool), p.shape).copy()
    bits = int(ptrs.get_element_ty().primitive_bitwidth)
    width = max(1, bits // 8)
    flat_p, flat_m = p.ravel(), m.ravel()
    owner = np.full(flat_p.shape, -1, dtype=np.int64)
    for i, (_, _, _, lo, hi) in enumerate(STATE.regions):
        owner[
            (flat_p >= np.uint64(lo)) & (flat_p + np.uint64(width) <= np.uint64(hi))
        ] = i
    site = _site_line()
    if site is None and flat_m.any():
        STATE.error(f"{kind}: no frame in a corpus file")
    wild = flat_m & (owner < 0)
    if wild.any():
        STATE.wild += int(wild.sum())
        STATE.error(
            f"{kind} at line {site[0] if site else '?'}: {int(wild.sum())} lanes outside every argument"
        )
    if site is not None:
        line, path = site
        pid = tuple(int(g) for g in grid_idx)
        for i, (name, base, elem, _, _) in enumerate(STATE.regions):
            sel = flat_m & (owner == i)
            if not sel.any():
                continue
            rel = flat_p[sel].astype(np.int64) - np.int64(base)
            if (rel % elem).any():
                STATE.error(
                    f"{kind} at line {line}: an address misaligned to {name!r}'s elements"
                )
            key = (name, kind, line, path)
            STATE.sites.setdefault(key, set()).update(
                (*pid, off) for off in (rel // elem).tolist()
            )
            STATE.bits.setdefault(key, set()).add(bits)
    return (flat_m & (owner >= 0)).reshape(p.shape)


def _install() -> None:
    import numpy as np
    from triton.runtime import interpreter as I

    B, TH = I.InterpreterBuilder, I.TensorHandle
    scratch = np.zeros(64, dtype=np.uint64)

    orig_init = I.GridExecutor._init_args_hst

    def _init_args_hst(self, args_dev, kwargs):
        import inspect

        args_hst, kwargs_hst = orig_init(self, args_dev, kwargs)
        _bind(inspect.getcallargs(self.fn, *args_hst, **kwargs_hst))
        return args_hst, kwargs_hst

    I.GridExecutor._init_args_hst = _init_args_hst

    # NumPy >= 2.4 refuses int() of a 1-element 1-D array, which the
    # interpreter's tensor.__index__ does for every loop bound.
    orig_patch_tensor = I._patch_lang_tensor

    def _patch_lang_tensor(tensor, scope):
        orig_patch_tensor(tensor, scope)
        scope.set_attr(
            tensor,
            "__index__",
            lambda self: int(np.asarray(self.handle.data).reshape(-1)[0]),
        )

    I._patch_lang_tensor = _patch_lang_tensor

    def _mask(m, mask):
        if isinstance(mask, np.ndarray):
            return m
        return TH(m, mask.dtype) if mask is not None else TH(m, I.tl.int1)

    orig_load = B.create_masked_load

    def create_masked_load(self, ptrs, mask, *rest, **kw):
        return orig_load(
            self,
            ptrs,
            _mask(_record("load", ptrs, mask, self.grid_idx), mask),
            *rest,
            **kw,
        )

    orig_store = B.create_masked_store

    def create_masked_store(self, ptrs, value, mask, *rest, **kw):
        return orig_store(
            self,
            ptrs,
            value,
            _mask(_record("store", ptrs, mask, self.grid_idx), mask),
            *rest,
            **kw,
        )

    orig_rmw = B.create_atomic_rmw

    def create_atomic_rmw(self, rmw_op, ptr, val, mask, *rest, **kw):
        return orig_rmw(
            self,
            rmw_op,
            ptr,
            val,
            _mask(_record("atomic_rmw", ptr, mask, self.grid_idx), mask),
            *rest,
            **kw,
        )

    orig_cas = B.create_atomic_cas

    def create_atomic_cas(self, ptr, cmp, val, *rest, **kw):
        m = _record("atomic_cas", ptr, None, self.grid_idx)
        if not m.all():
            data = np.asarray(ptr.data).astype(np.uint64).copy()
            data[~m] = np.uint64(scratch.ctypes.data)
            ptr = TH(data, ptr.dtype)
        return orig_cas(self, ptr, cmp, val, *rest, **kw)

    B.create_masked_load = create_masked_load
    B.create_masked_store = create_masked_store
    B.create_atomic_rmw = create_atomic_rmw
    B.create_atomic_cas = create_atomic_cas
    # Block-pointer and descriptor loads / stores materialize their pointers
    # and go through create_masked_load / create_masked_store (Triton 3.6).


def _alarm(signum, frame):
    raise TimeoutError(f"case exceeded {CASE_TIMEOUT_S}s")


def main(argv: list[str]) -> int:
    out_path, names = argv[0], argv[1:]
    assert os.environ.get("TRITON_INTERPRET") == "1", "run through run_interpreter()"
    sys.path.insert(0, HERE)
    from _ttir_capture import load_corpus  # noqa: E402 - the child's own import path

    _install()
    corpus = load_corpus()
    STATE.files = frozenset(
        {os.path.realpath(corpus.__file__), os.path.realpath(corpus.READER_KERNELS)}
    )
    cases = corpus.by_name()
    signal.signal(signal.SIGALRM, _alarm)
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        STATE.reset()
        t0 = time.monotonic()
        signal.alarm(CASE_TIMEOUT_S)
        try:
            c = cases[name]
            grid, args, kwargs = c.build("cpu")
            c.kernel[grid](*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - reported per case
            if isinstance(e, KeyboardInterrupt):
                raise
            STATE.error(f"{type(e).__name__}: {str(e)[:1000]}")
            STATE.errors.append(traceback.format_exc()[-2000:])
        finally:
            signal.alarm(0)
        results[name] = {
            "sites": [
                {
                    "arg": a,
                    "kind": k,
                    "line": line,
                    "file": path,
                    "bits": sorted(STATE.bits[(a, k, line, path)]),
                    "points": sorted(points),
                }
                for (a, k, line, path), points in STATE.sites.items()
            ],
            "errors": list(STATE.errors),
            "wild": STATE.wild,
            "seconds": round(time.monotonic() - t0, 3),
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
