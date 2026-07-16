"""TritonBench_G_v1 corpus: 184 real-world GitHub-crawled Triton operator
files (thunlp/TritonBench, Apache-2.0), vendored under
``tritonbench_g_v1/`` with the upstream commit pinned in its README and
in ``tritonbench_g_specs.json``.

Launches were captured ONCE on a CUDA machine by
``evaluation/tritonbench_capture.py`` (the files' test blocks execute at
import time on GPU); this module rebuilds them on ANY machine: it execs
only each file's pre-separator kernel section (never the test block) and
reconstructs CPU args from the captured descriptors (capture_common.py)
— float tensors as seeded randn/zeros, int tensors value-exact when the
capture carries a snapshot else randint over the OBSERVED value range,
aliased pointer args (in-place ops) from one tensor with
``LaunchSpec.aliased=True``, scalars exactly.

Like the liger corpus, every row is labeled race-free (production code);
the point is the ladder distribution on real kernels, and "unsupported
dominating is itself the data".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.capture_common import SIG_FOR_DTYPE, make_args_fn
from evaluation.spec import Corpus, LaunchSpec

VENDOR_DIR = Path(__file__).parent / "tritonbench_g_v1"
SPECS_PATH = Path(__file__).parent / "tritonbench_g_specs.json"
SEPARATOR_PREFIX = "#" * 100


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
                    SIG_FOR_DTYPE[d["dtype"]] if d["kind"] == "tensor" else d["sig"]
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
                    make_args=make_args_fn(spec["args"], spec["aliases"]),
                    grid=tuple(spec["grid"]),
                    expected="race-free",
                    pattern="tritonbench_g",
                    params_note=f"captured launch from {fname}",
                    aliased=bool(spec["aliases"]),
                )
            )
    return corpus


CORPUS = _build()
