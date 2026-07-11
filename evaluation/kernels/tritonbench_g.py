"""TritonBench_G_v1 corpus: 184 real-world GitHub-crawled Triton operator
files (thunlp/TritonBench, Apache-2.0), vendored under
``tritonbench_g_v1/`` with the upstream commit pinned in its README and
in ``tritonbench_g_specs.json``.

Launches were captured ONCE on a CUDA machine by
``evaluation/tritonbench_capture.py`` (the files' test blocks execute at
import time on GPU); this module rebuilds them on ANY machine: it execs
only each file's pre-separator kernel section (never the test block) and
reconstructs CPU args from the captured descriptors — float tensors as
seeded randn/zeros, int tensors as randint over the OBSERVED value range
(index tensors stay in-bounds), aliased pointer args (in-place ops) from
one tensor with ``LaunchSpec.aliased=True``, scalars exactly.

Like the liger corpus, every row is labeled race-free (production code);
the point is the ladder distribution on real kernels, and "unsupported
dominating is itself the data".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from evaluation.spec import Corpus, LaunchSpec

VENDOR_DIR = Path(__file__).parent / "tritonbench_g_v1"
SPECS_PATH = Path(__file__).parent / "tritonbench_g_specs.json"
SEPARATOR_PREFIX = "#" * 100

_SIG_FOR_DTYPE = {
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
_TORCH_DTYPE = {name: getattr(torch, name.split(".", 1)[1]) for name in _SIG_FOR_DTYPE}


def _kernel_section(source: str) -> str:
    """Everything before the ``#####…`` separator — kernels and host
    wrappers, but never the import-time CUDA test block."""
    for i, line in enumerate(lines := source.splitlines()):
        if line.startswith(SEPARATOR_PREFIX):
            return "\n".join(lines[:i])
    return source


def _resolve_kernel(namespace: dict, name: str) -> Any:
    from triton.runtime.jit import JITFunction

    obj = namespace.get(name)
    # unwrap @triton.autotune / @triton.heuristics stacks to the JITFunction
    # (the wrappers proxy arg_names, so unwrap by TYPE, not by attribute)
    while obj is not None and not isinstance(obj, JITFunction):
        if not hasattr(obj, "fn"):
            return None
        obj = obj.fn
    return obj


def _make_tensor(desc: dict, gen: torch.Generator) -> torch.Tensor:
    shape = tuple(desc["shape"])
    dtype = _TORCH_DTYPE[desc["dtype"]]
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


def _make_args_fn(arg_descs: list[dict], aliases: dict[str, str]):
    """None-valued args are NOT emitted — they live in ``constexprs``
    (triton specializes them away) and the harness launches all-kwargs,
    so declaration slots never shift."""

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
                v = by_name[src] if src is not None else _make_tensor(d, gen)
            by_name[d["name"]] = v
            out.append(v)
        return tuple(out)

    return make_args


def _build() -> Corpus:
    corpus = Corpus("tritonbench_g")
    payload = json.loads(SPECS_PATH.read_text())
    corpus.provenance = {
        "tritonbench_upstream": payload["upstream"],
        "tritonbench_commit": payload["upstream_commit"],
    }
    section_cache: dict[str, dict] = {}

    for fname, entry in sorted(payload["files"].items()):
        stem = Path(fname).stem
        multi = len(entry["kernels"]) > 1
        for kname, spec in sorted(entry["kernels"].items()):
            if fname not in section_cache:
                ns: dict[str, Any] = {"__name__": f"tbk_{stem}"}
                try:
                    exec(  # noqa: S102 — trusted vendored corpus
                        compile(
                            _kernel_section((VENDOR_DIR / fname).read_text()),
                            str(VENDOR_DIR / fname),
                            "exec",
                        ),
                        ns,
                    )
                except Exception as exc:  # noqa: BLE001 — skip broken imports
                    ns = {"__error__": f"{type(exc).__name__}: {exc}"}
                section_cache[fname] = ns
            ns = section_cache[fname]
            if "__error__" in ns:
                continue
            kernel = _resolve_kernel(ns, kname)
            if kernel is None or not hasattr(kernel, "arg_names"):
                continue

            sig_by_name = {
                d["name"]: (
                    _SIG_FOR_DTYPE[d["dtype"]] if d["kind"] == "tensor" else d["sig"]
                )
                for d in spec["args"]
                if d["kind"] != "none"
            }
            # None-valued optional pointers (initial states, residuals,
            # optional masks) are constexpr-specialized away by triton.
            none_args = {d["name"] for d in spec["args"] if d["kind"] == "none"}
            constexprs = dict(spec["constexprs"])
            constexprs.update({n: None for n in none_args})
            signature: dict[str, str] = {}
            usable = True
            for arg_name in kernel.arg_names:
                if arg_name in constexprs:
                    signature[arg_name] = "constexpr"
                elif arg_name in sig_by_name:
                    signature[arg_name] = sig_by_name[arg_name]
                else:
                    usable = False  # unbound arg
                    break
            if not usable:
                continue

            corpus.add(
                LaunchSpec(
                    name=f"tb_{stem}__{kname}" if multi else f"tb_{stem}",
                    kernel_fn=kernel,
                    signature=signature,
                    constexprs=constexprs,
                    make_args=_make_args_fn(spec["args"], spec["aliases"]),
                    grid=tuple(spec["grid"]),
                    expected="race-free",
                    pattern="tritonbench_g",
                    params_note=f"captured launch from {fname}",
                    aliased=bool(spec["aliases"]),
                )
            )
    return corpus


CORPUS = _build()
