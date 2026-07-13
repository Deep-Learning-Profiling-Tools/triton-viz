"""Shared corpus builder for captured-launch corpora (fla, flagattn).

A specs JSON produced by a case-driven capture driver
(capture_common.run_case_capture) rebuilds into a Corpus on any
machine: each kernel is resolved by importing its recorded ``module``
and unwrapping the @triton.autotune/@triton.heuristics stack to the
JITFunction; args come from the captured descriptors with small
int/bool tensors value-exact.

Fail-loud invariants: the specs bind by module path + kernel name into
the INSTALLED package — on version drift kernels move/rename and rows
would vanish silently, so an installed-version mismatch and any
unresolved kernel are hard errors, never a shrunken corpus.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from evaluation.capture_common import SIG_FOR_DTYPE, make_args_fn
from evaluation.spec import Corpus, LaunchSpec


def _kernel_types() -> tuple:
    from triton.runtime.jit import JITFunction

    # under TRITON_INTERPRET=1 @triton.jit yields InterpretedFunction (not
    # a JITFunction subclass); the harness supports that mode, so must we
    try:
        from triton.runtime.interpreter import InterpretedFunction

        return (JITFunction, InterpretedFunction)
    except ImportError:  # pragma: no cover
        return (JITFunction,)


def _resolve_kernel(module_name: str, kernel_name: str) -> Any:
    """None on failure — the caller collects and raises loudly."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 — caller reports, with version context
        return None
    types = _kernel_types()

    def unwrap(obj: Any) -> Any:
        # unwrap @triton.autotune / @triton.heuristics stacks to the
        # JITFunction (the wrappers proxy arg_names, so unwrap by TYPE,
        # not by attribute); depth-capped so arbitrary .fn chains found
        # by the namespace scan below can't loop
        for _ in range(8):
            if obj is None or isinstance(obj, types):
                return obj
            if not hasattr(obj, "fn"):
                return None
            obj = obj.fn
        return None

    direct = unwrap(getattr(mod, kernel_name, None))
    if direct is not None:
        return direct
    # some packages publish a kernel under a DIFFERENT module-global name
    # (torchao.kernel.blockwise_quantization defines its kernels inside a
    # lazy-init closure and stores blockwise_fp8_gemm_kernel as
    # _blockwise_fp8_gemm_impl) or as a CLASS attribute (tritonbench's
    # softmax Operator carries @triton.jit kernels in its class body):
    # scan the namespace — one level into module-level classes — for a
    # def-name match, refusing ambiguity so a wrong kernel can never
    # resolve silently
    candidates = list(vars(mod).values())
    candidates += [
        v
        for cls in vars(mod).values()
        if isinstance(cls, type) and cls.__module__ == module_name
        for v in vars(cls).values()
    ]
    matches: dict[int, Any] = {}
    for value in candidates:
        k = unwrap(value)
        if k is not None and getattr(k.fn, "__name__", None) == kernel_name:
            matches[id(k)] = k
    if len(matches) == 1:
        return next(iter(matches.values()))
    return None


def _decode_constexpr(v: Any) -> Any:
    """Inverse of capture_common.encode_constexpr: tagged dicts back to
    dtype objects (triton binds dtype constexprs by object, not name)."""
    if isinstance(v, dict) and "__tl_dtype__" in v:
        import triton.language as tl

        return tl.core.dtype(v["__tl_dtype__"])
    if isinstance(v, dict) and "__torch_dtype__" in v:
        import torch

        return getattr(torch, v["__torch_dtype__"].removeprefix("torch."))
    return v


def build_captured_corpus(
    corpus_name: str,
    specs_path: Path,
    dist_name: str,
    version_field: str,
    install_hint: str,
) -> Corpus:
    """``version_field`` is the payload key carrying the captured package
    version (also the provenance key prefix); ``dist_name`` is the
    installed distribution to hard-check it against."""
    from importlib import metadata

    corpus = Corpus(corpus_name)
    payload = json.loads(specs_path.read_text())
    corpus.provenance = {
        f"{corpus_name}_upstream": payload["upstream"],
        f"{corpus_name}_captured_version": payload[version_field],
        f"{corpus_name}_upstream_commit": payload["upstream_commit"],
    }
    installed = metadata.version(dist_name)
    if installed != payload[version_field]:
        raise ImportError(
            f"{corpus_name} corpus was captured against {dist_name} "
            f"{payload[version_field]} but {installed} is installed; "
            f"{install_hint} (or re-run the capture driver on a GPU "
            f"machine and re-sweep)"
        )

    unresolved: list[str] = []
    used_names: set[str] = set()
    for case, entry in sorted(payload["cases"].items()):
        for _slot, spec in sorted(entry["kernels"].items()):
            kernel = _resolve_kernel(spec["module"], spec["kernel"])
            if kernel is None or not hasattr(kernel, "arg_names"):
                unresolved.append(f"{spec['module']}.{spec['kernel']}")
                continue

            sig_by_name = {
                d["name"]: (
                    SIG_FOR_DTYPE[d["dtype"]] if d["kind"] == "tensor" else d["sig"]
                )
                for d in spec["args"]
                if d["kind"] != "none"
            }
            # None-valued optional pointers (initial states, cu_seqlens on
            # dense launches) are constexpr-specialized away by triton.
            none_args = {d["name"] for d in spec["args"] if d["kind"] == "none"}
            constexprs = {
                k: _decode_constexpr(v) for k, v in spec["constexprs"].items()
            }
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
                unresolved.append(
                    f"{spec['module']}.{spec['kernel']} (unbound {arg_name})"
                )
                continue

            name = f"{corpus_name}_{case}__{spec['kernel']}"
            if name in used_names:
                # a bwd module re-defining its fwd twin under the same name
                name = f"{name}__{spec['module'].rsplit('.', 1)[-1]}"
            used_names.add(name)
            corpus.add(
                LaunchSpec(
                    name=name,
                    kernel_fn=kernel,
                    signature=signature,
                    constexprs=constexprs,
                    make_args=make_args_fn(spec["args"], spec["aliases"]),
                    grid=tuple(spec["grid"]),
                    expected="race-free",
                    pattern=f"{corpus_name}_{entry['family']}",
                    params_note=f"captured launch from case {case}",
                    aliased=bool(spec["aliases"]),
                )
            )
    if unresolved:
        raise RuntimeError(
            f"{corpus_name} corpus: {len(unresolved)} captured kernel(s) "
            f"failed to resolve against installed {dist_name} {installed}: "
            f"{unresolved[:10]}{'...' if len(unresolved) > 10 else ''}"
        )
    return corpus
