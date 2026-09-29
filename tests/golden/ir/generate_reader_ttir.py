"""Regenerate the TTIR reader's regression goldens of the installed Triton release.

    python tests/golden/ir/generate_reader_ttir.py [NAME ...]

Host-compiles each kernel of ``reader_kernels.py`` (ASTSource +
``triton.compile`` for ``GPUTarget("cuda", 80, 32)``, no GPU needed) into
``reader_ttir/<name>.ttir`` under Triton 3.6 (the base release, whose
goldens every release's tests read) and into ``reader_ttir_<release>/``
under any other (that release's own printing, which shadows the base
golden of the same name under that release), using a throwaway Triton
cache. These goldens sit apart from ``ttir/``, whose every file the
walk-layer tests pin.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_RELEASE = "3.6"  # the release that printed reader_ttir/

_F32 = "*fp32"
# name -> (signature, constexprs)
SPECS: dict[str, tuple[dict[str, str], dict[str, Any]]] = {
    "p1_variant_delta": ({"x_ptr": _F32, "out_ptr": _F32, "n": "i32"}, {}),
    "p2_swap": ({"a_ptr": _F32, "b_ptr": _F32, "n": "i32"}, {}),
    "p3_call_guarded": ({"x_ptr": _F32, "n": "i32"}, {}),
    "p3_call_offset": ({"x_ptr": _F32, "n": "i32"}, {}),
    "p3_call_formals": ({"x_ptr": _F32, "n": "i32"}, {}),
    "p4_observed_direct": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "p4_observed_loop": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "p4_observed_delta": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "rv_trunci_alias": ({"x_ptr": "*i32"}, {}),
    "rv_i32_wrap": ({"x_ptr": "*i32", "S": "i32"}, {}),
    "unsigned_index": ({"x_ptr": _F32, "n": "i32"}, {}),
    "rv_inline_asm_store": ({"x_ptr": "*i32", "OFF": "constexpr"}, {"OFF": 4096}),
    "loop_two_step_advance": ({"x_ptr": _F32, "n": "i32", "s": "i32"}, {}),
    "loop_observed_advance": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "where_pointer": ({"x_ptr": _F32, "n": "i32"}, {}),
    "tile3d_shared_arange": ({"x_ptr": _F32, "N": "constexpr"}, {"N": 4}),
    "expand_iterarg_3d": (
        {"x_ptr": _F32, "out_ptr": _F32, "n": "i32", "N": "constexpr"},
        {"N": 4},
    ),
    "expand_iterarg_mask": (
        {"x_ptr": _F32, "out_ptr": _F32, "n": "i32", "M": "i32", "N": "constexpr"},
        {"N": 4},
    ),
    "int_iterarg_offset": ({"x_ptr": _F32, "n": "i32", "B": "constexpr"}, {"B": 8}),
    "iv_wrap": ({"x_ptr": _F32, "lo": "i32", "n": "i32"}, {}),
    "pure_asm_int_addr": ({"x_ptr": "*i32"}, {}),
    "observed_lanes": ({"cnt_ptr": "*i32", "x_ptr": _F32, "N": "constexpr"}, {"N": 4}),
}


def _kernels():
    spec = importlib.util.spec_from_file_location(
        "reader_kernels", os.path.join(HERE, "reader_kernels.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reader_kernels"] = module  # Triton reads the jit fn's module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    only = set(sys.argv[1:])
    kernels = _kernels()
    failed = 0
    release = ".".join(triton.__version__.split(".")[:2])
    out = os.path.join(
        HERE, "reader_ttir" if release == BASE_RELEASE else f"reader_ttir_{release}"
    )
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        for name, (sig, consts) in SPECS.items():
            if only and name not in only:
                continue
            src = ASTSource(fn=getattr(kernels, name), signature=sig, constexprs=consts)
            try:
                k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(
                    f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}",
                    file=sys.stderr,
                )
                continue
            path = os.path.join(out, f"{name}.ttir")
            with open(path, "w", encoding="utf-8") as f:
                f.write(k.asm["ttir"])
            print(f"[{name}] wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
