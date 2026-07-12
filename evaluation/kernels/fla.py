"""flash-linear-attention corpus: production linear-attention Triton
kernels (fla-org/flash-linear-attention) analyzed AS INSTALLED via the
``fla-core`` pip package (evaluation-only dependency, like liger;
``runner._fla_provenance()`` pins version + upstream commit).

Launches were captured ONCE on a CUDA machine by
``evaluation/fla_capture.py`` (public ``fla.ops`` entry points, forward
+ backward, dense + varlen, small fp32 shapes); this module rebuilds
them on ANY machine: each kernel is resolved by importing its recorded
``module`` and unwrapping the @triton.autotune/@triton.heuristics stack
to the JITFunction, args come from the captured descriptors
(capture_common.py) with small int/bool tensors value-exact
(cu_seqlens / chunk index tables stay coupled).

Every row is labeled race-free (production code); as with liger and
TritonBench the deliverable is the ladder distribution — and fla is the
block-pointer-heavy corpus (91 of 153 op files), so the
``block-pointer`` abstention bucket and the interp tier carry the load
the shared TTIR reader cannot yet.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

try:
    import fla  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the fla corpus needs fla-core: uv pip install fla-core==0.5.1"
    ) from e

from evaluation.capture_common import SIG_FOR_DTYPE, make_args_fn
from evaluation.spec import Corpus, LaunchSpec

SPECS_PATH = Path(__file__).parent / "fla_specs.json"


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
    obj = getattr(mod, kernel_name, None)
    # unwrap @triton.autotune / @triton.heuristics stacks to the JITFunction
    # (the wrappers proxy arg_names, so unwrap by TYPE, not by attribute)
    types = _kernel_types()
    while obj is not None and not isinstance(obj, types):
        if not hasattr(obj, "fn"):
            return None
        obj = obj.fn
    return obj


def _build() -> Corpus:
    from importlib import metadata

    corpus = Corpus("fla")
    payload = json.loads(SPECS_PATH.read_text())
    corpus.provenance = {
        "fla_upstream": payload["upstream"],
        "fla_captured_version": payload["fla_core"],
        "fla_upstream_commit": payload["upstream_commit"],
    }
    # the specs bind by module path + kernel name into the INSTALLED
    # package — on version drift kernels move/rename and rows would vanish
    # silently, so a mismatch is a hard error, not a shrunken corpus
    installed = metadata.version("fla-core")
    if installed != payload["fla_core"]:
        raise ImportError(
            f"fla corpus was captured against fla-core {payload['fla_core']} "
            f"but {installed} is installed; uv pip install "
            f"fla-core=={payload['fla_core']} (or re-run evaluation/fla_capture.py "
            f"on a GPU machine and re-sweep)"
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
                unresolved.append(
                    f"{spec['module']}.{spec['kernel']} (unbound {arg_name})"
                )
                continue

            name = f"fla_{case}__{spec['kernel']}"
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
                    pattern=f"fla_{entry['family']}",
                    params_note=f"captured launch from fla.ops case {case}",
                    aliased=bool(spec["aliases"]),
                )
            )
    if unresolved:
        raise RuntimeError(
            f"fla corpus: {len(unresolved)} captured kernel(s) failed to "
            f"resolve against installed fla-core {installed}: "
            f"{unresolved[:10]}{'...' if len(unresolved) > 10 else ''}"
        )
    return corpus


CORPUS = _build()
