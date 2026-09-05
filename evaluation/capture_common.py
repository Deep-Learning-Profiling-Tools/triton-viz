"""Shared launch-capture machinery + descriptor rebuild for real-code corpora.

Capture side (GPU machine): ``LaunchRecorder`` hooks ``JITFunction.run``
and records, per kernel (dedup key configurable), the FIRST real launch:
the full name→value binding split into runtime args and constexprs,
tensor descriptors (shape / dtype / init class / contiguity / alias
group), exact scalars, and the resolved grid.

EVERY integer and bool tensor carries a VALUE SNAPSHOT (the exact
values): by-range ``randint`` rebuilds fabricate invalid inputs for
value-coupled tensors — non-monotone ``cu_seqlens``, repeated entries
in permutation/index tables, masks that no longer keep stores disjoint,
a snapshotted prefix sum next to a randomly rebuilt mask — which is
exactly the TritonBench interp-disagreement class and, at ladder level
L1, the capture-artifact class of the change-surface run (Hao,
2026-09-04: snapshot all int/bool, rebuild only floats). Small
snapshots (up to VALUE_SNAPSHOT_CAP elements) stay INLINE in the specs
JSON as before; larger ones live in a content-addressed sidecar
(``<specs stem>_values.npz`` next to the specs file, one compressed
array per SHA-256 of the raw bytes) and the descriptor carries
``values_ref`` (the hash). Float tensors stay by-descriptor (their
values only reach addresses through comparisons, and seeded randn keeps
them generic).

Rebuild side (any machine, CPU-only): ``make_tensor`` / ``make_args_fn``
reconstruct launch args from the descriptors, values-exact when a
snapshot is present. A ``values_ref`` whose sidecar or entry is missing
is a HARD error (``MissingValueSnapshot``), never a silent fall-back to
a random rebuild: that fall-back is the artifact this design removes.
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# int/bool snapshots up to this many elements are stored INLINE in the
# specs JSON; larger ones go to the content-addressed sidecar (every
# int/bool tensor is snapshotted either way)
VALUE_SNAPSHOT_CAP = 8192
VALUES_SIDECAR_SUFFIX = "_values.npz"


class MissingValueSnapshot(RuntimeError):
    """A descriptor references a value snapshot the sidecar does not
    provide. Fail-loud by design: rebuilding such a tensor at random
    would silently reintroduce the capture-artifact class."""


def _sha256_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class ValueStore:
    """Content-addressed store of value snapshots: SHA-256 of the raw
    bytes -> flat numpy array. Capture side: ``put`` returns the key the
    descriptor records. Rebuild side: ``get`` verifies the hash on read.
    Persisted as one compressed ``.npz`` (``save``/``load``); ``beside``
    names the sidecar that belongs to a specs file (loaded lazily, so a
    corpus whose descriptors need no sidecar never touches the disk)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._arrays: dict[str, np.ndarray] = {}
        self._loaded = path is None

    @classmethod
    def beside(cls, specs_path: Path) -> "ValueStore":
        specs_path = Path(specs_path)
        stem = specs_path.name
        if stem.endswith("_specs.json"):
            stem = stem[: -len("_specs.json")]
        else:
            stem = specs_path.stem
        return cls(specs_path.parent / f"{stem}{VALUES_SIDECAR_SUFFIX}")

    # ── capture side ──
    def put(self, arr: np.ndarray) -> str:
        flat = np.ascontiguousarray(arr).reshape(-1)
        key = _sha256_bytes(flat)
        self._arrays.setdefault(key, flat)
        return key

    def merge(self, other: "ValueStore") -> None:
        other._ensure_loaded()
        for k, v in other._arrays.items():
            self._arrays.setdefault(k, v)

    def save(self, path: Path | None = None) -> Path | None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("ValueStore.save needs a path")
        self._ensure_loaded()
        if not self._arrays:
            target.unlink(missing_ok=True)  # nothing referenced: no sidecar
            return None
        np.savez_compressed(target, **self._arrays)
        return target

    # ── rebuild side ──
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path is None or not self.path.exists():
            return
        with np.load(self.path) as z:
            for k in z.files:
                self._arrays[k] = np.asarray(z[k])

    def __contains__(self, key: str) -> bool:
        self._ensure_loaded()
        return key in self._arrays

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._arrays)

    def get(self, key: str, *, what: str = "") -> np.ndarray:
        self._ensure_loaded()
        arr = self._arrays.get(key)
        where = str(self.path) if self.path is not None else "<in-memory store>"
        if arr is None:
            raise MissingValueSnapshot(
                f"value snapshot {key[:12]}... {what}is not in {where}: the sidecar "
                "was not produced by the capture that wrote these specs (re-run "
                "the capture driver on the GPU machine) or is missing from this "
                "checkout; rebuilding the tensor at random is refused"
            )
        if _sha256_bytes(arr) != key:
            raise MissingValueSnapshot(
                f"value snapshot {key[:12]}... {what}in {where} fails its hash check"
            )
        return arr


# launch-config kwargs that are not kernel parameters
LAUNCH_OPTS = {
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

SIG_FOR_DTYPE = {
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
    # triton's own canonicalisation for the fp8 families (torchao quant
    # kernels take fp8 tensors as args); e8m0 has NO triton mapping as
    # of 3.6 — a kernel launched with an e8m0 arg fails upstream too
    "torch.float8_e4m3fn": "*fp8e4nv",
    "torch.float8_e5m2": "*fp8e5",
}
TORCH_DTYPE = {name: getattr(torch, name.split(".", 1)[1]) for name in SIG_FOR_DTYPE}


# ── capture side ─────────────────────────────────────────────────


def describe_tensor(t: torch.Tensor, store: "ValueStore | None" = None) -> dict:
    d = {
        "kind": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "contiguous": bool(t.is_contiguous()),
    }
    if not d["contiguous"]:
        # column-major and transposed-view args (torchao's blockwise fp8
        # quant family) rebuild via empty_strided + copy_
        d["strides"] = list(t.stride())
    if t.numel() == 0:
        d["init"] = "zeros"
    elif t.dtype.is_floating_point:
        # fp8 tensors don't implement eager comparison — widen first
        z = t.float() if t.dtype.itemsize == 1 else t
        d["init"] = "zeros" if bool((z == 0).all()) else "randn"
    elif t.dtype == torch.bool:
        d["init"] = "randbool"
        _snapshot_values(d, t, store)
    else:
        lo = int(t.min().item())
        hi = int(t.max().item())
        d["init"] = "randint"
        d["low"], d["high"] = lo, hi + 1
        _snapshot_values(d, t, store)
    return d


def _snapshot_values(d: dict, t: torch.Tensor, store: "ValueStore | None") -> None:
    """Every int/bool tensor is snapshotted: inline up to the cap,
    otherwise into the content-addressed sidecar. Without a store a
    large tensor keeps the by-descriptor rebuild and says so
    (``values_dropped``), so a capture run without a sidecar cannot
    pass for a full snapshot."""
    flat = t.detach().flatten().cpu()
    if t.numel() <= VALUE_SNAPSHOT_CAP:
        d["values"] = [int(x) for x in flat.tolist()]
        return
    if store is None:
        d["values_dropped"] = True
        return
    arr = flat.numpy() if t.dtype != torch.bool else flat.numpy().astype(np.bool_)
    d["values_ref"] = store.put(arr)


def describe(v: Any, store: "ValueStore | None" = None) -> dict:
    if isinstance(v, torch.Tensor):
        return describe_tensor(v, store)
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


_UNSUPPORTED_CONSTEXPR = object()


def encode_constexpr(cv: Any) -> Any:
    """JSON-able encoding of a constexpr value; dtype OBJECTS (torchao
    quant kernels take tl.float8e4nv / torch.float8_e4m3fn as constexpr
    params) round-trip through tagged dicts, decoded by the corpus
    builder. Returns _UNSUPPORTED_CONSTEXPR for anything else."""
    if isinstance(cv, (int, float, bool, str, type(None))):
        return cv
    import triton.language as tl

    if isinstance(cv, tl.core.dtype):
        return {"__tl_dtype__": str(cv)}
    if isinstance(cv, torch.dtype):
        return {"__torch_dtype__": str(cv)}
    return _UNSUPPORTED_CONSTEXPR


class LaunchRecorder:
    """Records the first real launch per dedup key while hooked.

    ``key(fn)`` names the capture slot (default: the kernel's plain
    name, right for one-file-per-subprocess corpora); records land in
    ``captured[key]``, rejects in ``skipped[key]`` with a reason. A
    capture error never breaks the hooked run.
    """

    def __init__(
        self,
        key: Callable[[Any], str] | None = None,
        values: "ValueStore | None" = None,
    ):
        self.captured: dict[str, dict] = {}
        self.skipped: dict[str, str] = {}
        self._key = key or (lambda fn: fn.__name__)
        # the value snapshots of every int/bool tensor above the inline
        # cap; the capture driver persists it as the specs' sidecar
        self.values = values if values is not None else ValueStore()

    @contextmanager
    def hooked(self):
        from triton.runtime.jit import JITFunction

        real_run = JITFunction.run
        recorder = self

        def hooked_run(self, *args, **kwargs):
            try:
                recorder._record(self, args, dict(kwargs))
            except Exception as exc:  # noqa: BLE001 — capture must not break the run
                recorder.skipped.setdefault(
                    recorder._key(self), f"capture error: {exc}"
                )
            return real_run(self, *args, **kwargs)

        JITFunction.run = hooked_run
        try:
            yield self
        finally:
            JITFunction.run = real_run

    def _record(self, fn, args, kwargs) -> None:
        slot = self._key(fn)
        if slot in self.captured or slot in self.skipped:
            return
        if kwargs.get("warmup"):
            return
        grid = kwargs.pop("grid", None)
        if grid is None:
            return
        meta = dict(zip(fn.arg_names, args))
        declared = set(fn.arg_names)
        for k, v in kwargs.items():
            # a kwarg naming a DECLARED parameter is a kernel arg even when
            # it collides with a launch option (fla's fused_recurrent kda /
            # gdn2 kernels declare `num_stages: tl.constexpr` and feed it
            # to tl.range) — triton's own binder resolves it the same way
            if k not in LAUNCH_OPTS or k in declared:
                meta[k] = v
        params = {p.name: p for p in fn.params}
        for n in fn.arg_names:
            if n not in meta and params[n].has_default:
                meta[n] = params[n].default
        unbound = [n for n in fn.arg_names if n not in meta]
        if unbound:
            self.skipped[slot] = f"unbound params {unbound}"
            return
        g = grid(meta) if callable(grid) else grid
        g = tuple(int(x) for x in (g if isinstance(g, (tuple, list)) else (g,)))

        # alias groups over tensor args (in-place ops pass one tensor twice);
        # value = (first arg name, layout) so later same-ptr args can verify
        # they are the SAME view before joining the alias group
        ptrs: dict[int, tuple[str, tuple[Any, Any, Any]]] = {}
        aliases: dict[str, str] = {}
        runtime_args = []
        constexprs = {}
        for name in fn.arg_names:
            v = meta[name]
            if params[name].is_constexpr:
                cv = getattr(v, "value", v)
                enc = encode_constexpr(cv)
                if enc is _UNSUPPORTED_CONSTEXPR:
                    self.skipped[
                        slot
                    ] = f"non-literal constexpr {name}={type(cv).__name__}"
                    return
                constexprs[name] = enc
                continue
            d = describe(v, self.values)
            if d["kind"] == "unsupported":
                self.skipped[slot] = f"arg {name}: {d['type']}"
                return
            if d["kind"] == "tensor":
                p = v.data_ptr()
                layout = (d["shape"], d["dtype"], d.get("strides"))
                if p in ptrs:
                    first_name, first_layout = ptrs[p]
                    if layout != first_layout:
                        # two DIFFERENT views of one buffer can't rebuild
                        # from independent tensors (the alias map hands the
                        # source tensor to the alias verbatim)
                        self.skipped[slot] = (
                            f"args {first_name}/{name} are distinct views "
                            "of one buffer"
                        )
                        return
                    aliases[name] = first_name
                else:
                    ptrs[p] = (name, layout)
            d["name"] = name
            runtime_args.append(d)

        self.captured[slot] = {
            "kernel": fn.__name__,
            "module": getattr(fn.fn, "__module__", None),
            "args": runtime_args,
            "constexprs": constexprs,
            "grid": list(g),
            "aliases": aliases,
        }


# ── rebuild side ─────────────────────────────────────────────────


def make_tensor(
    desc: dict, gen: torch.Generator, store: "ValueStore | None" = None
) -> torch.Tensor:
    shape = tuple(desc["shape"])
    dtype = TORCH_DTYPE[desc["dtype"]]
    t = _make_contiguous(desc, shape, dtype, gen, store)
    strides = desc.get("strides")
    if strides is not None:
        out = torch.empty_strided(shape, tuple(strides), dtype=dtype)
        if 0 in strides:
            # broadcast-expanded arg (torchao bsr passes beta*input
            # expanded): copy_ refuses overlapping writes — write the
            # de-overlapped slice, the zero strides replicate it
            sel = tuple(slice(0, 1) if s == 0 else slice(None) for s in strides)
            out[sel].copy_(t[sel])
        else:
            out.copy_(t)
        return out
    return t


def _make_contiguous(
    desc: dict,
    shape: tuple,
    dtype: torch.dtype,
    gen: torch.Generator,
    store: "ValueStore | None" = None,
) -> torch.Tensor:
    if "values" in desc:  # exact snapshot beats any by-descriptor init
        return torch.tensor(desc["values"], dtype=dtype).reshape(shape)
    if "values_ref" in desc:
        what = f"for {desc.get('name', '?')} {tuple(shape)} {desc['dtype']} "
        if store is None:
            raise MissingValueSnapshot(
                f"value snapshot {what}is referenced but no sidecar store was "
                "given to the rebuild"
            )
        arr = store.get(desc["values_ref"], what=what)
        return torch.from_numpy(np.array(arr)).to(dtype).reshape(shape)
    if desc["init"] == "zeros":
        return torch.zeros(shape, dtype=dtype)
    if desc["init"] == "randn":
        return torch.randn(shape, generator=gen).to(dtype)
    if desc["init"] == "randbool":
        return torch.rand(shape, generator=gen) > 0.5
    if desc["init"] == "randint":
        lo, hi = desc["low"], max(desc["high"], desc["low"] + 1)
        return torch.randint(lo, hi, shape, generator=gen, dtype=dtype)
    raise ValueError(f"unknown init {desc['init']!r}")


def make_args_fn(
    arg_descs: list[dict],
    aliases: dict[str, str],
    store: "ValueStore | None" = None,
):
    """None-valued args are NOT emitted — they live in ``constexprs``
    (triton specializes them away) and the harness launches all-kwargs,
    so declaration slots never shift. ``store`` resolves ``values_ref``
    snapshots (``ValueStore.beside(specs_path)``)."""

    def make_args(seed: int) -> tuple:
        gen = torch.Generator().manual_seed(seed)
        by_name: dict[str, Any] = {}
        out: list[Any] = []
        for d in arg_descs:
            if d["kind"] == "none":
                continue  # constexpr-None; the harness binds it by name
            if d["kind"] == "scalar":
                v: Any = d["value"]
            else:  # tensor
                src = aliases.get(d["name"])
                v = by_name[src] if src is not None else make_tensor(d, gen, store)
            by_name[d["name"]] = v
            out.append(v)
        return tuple(out)

    return make_args


# ── case-driven capture drivers (fla_capture, flagattn_capture) ──
# A driver contributes CASES = {name: (family, bwd, run)} where
# run(torch, device, dtype) calls one public op and returns its output
# tensors; everything else — per-case subprocess isolation, first-launch
# recording, full-record dedup, compact specs writing — is shared here.


def capture_one_case(
    cases: dict,
    case_name: str,
    dtype_name: str,
    module_prefix: str | None = None,
) -> dict:
    import triton

    family, bwd, run = cases[case_name]
    torch.manual_seed(0)
    recorder = LaunchRecorder(key=lambda fn: f"{fn.fn.__module__}.{fn.__name__}")
    error = None
    with recorder.hooked():
        try:
            outs = [
                o
                for o in run(torch, "cuda", getattr(torch, dtype_name))
                if isinstance(o, torch.Tensor)
            ]
            if bwd:
                grads = [o.float().sum() for o in outs if o.grad_fn is not None]
                if grads:
                    sum(grads).backward()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    captured = recorder.captured
    skipped = recorder.skipped
    if module_prefix is not None:
        # runtime-CODEGEN kernels (FlagGems pointwise_dynamic writes
        # generated modules under ~/.flaggems/code_cache with
        # process-dependent names) cannot be re-imported at rebuild time
        # — keep them out of the corpus, visibly
        kept = {}
        for slot, rec in captured.items():
            mod = rec.get("module") or ""
            if mod.startswith(module_prefix):
                kept[slot] = rec
            else:
                skipped[slot] = f"runtime-codegen kernel (module {mod!r})"
        captured = kept

    return {
        "case": case_name,
        "family": family,
        "error": error,
        "kernels": captured,
        "skipped_kernels": skipped,
        "triton": triton.__version__,
        # not JSON: the driver writes it beside the JSON (write_case_result)
        "_values": recorder.values,
    }


def write_case_result(result: dict, out: Path) -> None:
    """Write a per-case capture result: the JSON at ``out`` and the value
    snapshots at ``out`` + ``_values.npz`` (only when any exist)."""
    values = result.pop("_values", None)
    out.write_text(json.dumps(result, indent=1))
    if values is not None:
        values.save(values_sidecar_of(out))


def values_sidecar_of(json_path: Path) -> Path:
    json_path = Path(json_path)
    return json_path.with_name(json_path.name + VALUES_SIDECAR_SUFFIX)


def fingerprint(rec: dict) -> str:
    """The FULL rebuild-relevant record: two launches merge only when the
    corpus rows they would rebuild into are identical. Arg descriptors
    carry scalar values and int/bool snapshots, and aliases drive the
    spec's ``aliased`` flag — families share kernels but call them with
    different scalars (gsa's chunk_gla_bwd v-pass hardcodes scale=1 while
    gla passes K**-0.5), and shape-only fingerprints merged those."""
    return json.dumps(
        [
            rec["module"],
            rec["kernel"],
            rec["constexprs"],
            rec["grid"],
            rec["args"],
            rec["aliases"],
        ],
        sort_keys=True,
        default=str,
    )


def run_case_capture(
    runner_module: str,
    cases: dict,
    specs_path: Path,
    payload_meta: dict,
    per_case_timeout_s: int = 600,
) -> None:
    """Drive every case in its own subprocess (crash isolation) via
    ``python -m {runner_module} --one <case> --out <tmp>`` and merge the
    results into ``specs_path`` with cross-case full-record dedup."""
    merged: dict[str, dict] = {}
    failures: dict[str, str] = {}
    seen: dict[str, str] = {}  # specialization fingerprint -> first case
    values = ValueStore()  # the corpus sidecar, merged from the children
    for i, case in enumerate(sorted(cases), 1):
        # private per-run temp file: /tmp is shared and sticky, a fixed
        # path can collide with a concurrent sweep or another user's stale
        # file and merge records under the wrong run's provenance
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix=f"capture_{case}_")
        os.close(fd)
        out = Path(tmp)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", runner_module, "--one", case, "--out", str(out)],
                capture_output=True,
                text=True,
                timeout=per_case_timeout_s,
                cwd=Path(__file__).parent.parent,
            )
            if proc.returncode != 0:
                failures[case] = (proc.stderr or "").strip()[-300:]
                print(f"[{i}/{len(cases)}] {case}: CRASH")
                continue
            result = json.loads(out.read_text())
            child_values = values_sidecar_of(out)
            if child_values.exists():
                values.merge(ValueStore(child_values))
        except subprocess.TimeoutExpired:
            failures[case] = f"timeout after {per_case_timeout_s}s"
            print(f"[{i}/{len(cases)}] {case}: TIMEOUT")
            continue
        except (OSError, json.JSONDecodeError) as exc:
            failures[case] = f"capture output unreadable: {exc}"
            print(f"[{i}/{len(cases)}] {case}: UNREADABLE")
            continue
        finally:
            out.unlink(missing_ok=True)
            values_sidecar_of(out).unlink(missing_ok=True)
        if result["error"] and not result["kernels"]:
            failures[case] = result["error"][:300]
            print(f"[{i}/{len(cases)}] {case}: ERROR ({result['error'][:80]})")
            continue

        kept, dropped = {}, []
        for slot, rec in result["kernels"].items():
            fp = fingerprint(rec)
            if fp in seen:
                dropped.append(f"{rec['kernel']} (first: {seen[fp]})")
            else:
                seen[fp] = case
                kept[slot] = rec
        result["kernels"] = kept
        result["dedup_dropped"] = dropped
        merged[case] = result
        note = f", {len(dropped)} shared" if dropped else ""
        err = (
            f" (+error after capture: {result['error'][:60]})"
            if result["error"]
            else ""
        )
        print(f"[{i}/{len(cases)}] {case}: {len(kept)} kernel(s){note}{err}")

    payload = {**payload_meta, "cases": merged, "capture_failures": failures}
    # compact + sorted: value snapshots dominate the size (checked-in file)
    specs_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    )
    sidecar = prune_and_save_sidecar(
        values, payload, ValueStore.beside(specs_path).path
    )
    total = sum(len(r["kernels"]) for r in merged.values())
    print(
        f"\ncaptured {total} kernel specializations from "
        f"{len(merged)}/{len(cases)} cases ({len(failures)} failures) "
        f"-> {specs_path}"
        + (f" (+ {len(values)} value snapshots -> {sidecar})" if sidecar else "")
    )


def referenced_values(payload: Any) -> set[str]:
    """Every ``values_ref`` a specs payload mentions (any nesting)."""
    refs: set[str] = set()

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            ref = o.get("values_ref")
            if isinstance(ref, str):
                refs.add(ref)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(payload)
    return refs


def prune_and_save_sidecar(
    values: "ValueStore", payload: Any, sidecar_path: Path | None
) -> Path | None:
    """Keep only the snapshots the written specs reference (dedup dropped
    launches, skipped kernels) and persist them beside the specs; a
    reference without a snapshot is a capture bug and raises."""
    refs = referenced_values(payload)
    missing = [r for r in refs if r not in values]
    if missing:
        raise MissingValueSnapshot(
            f"{len(missing)} value snapshot(s) referenced by the specs were "
            f"not captured (first: {missing[0][:12]}...)"
        )
    pruned = ValueStore()
    for r in refs:
        pruned._arrays[r] = values.get(r)
    return pruned.save(sidecar_path)
