"""torchao corpus: pytorch/ao's hand-written Triton kernels (quantization
for fp8/int8/int4/blockwise formats, MoE-training scaling, split-k int
matmuls, BSR sparse ops) analyzed AS INSTALLED via a git-pinned
``USE_CPP=0`` pip install — the Triton kernels are pure Python, so the
C++ extension is skipped and the install has no torch-ABI coupling.
``runner._torchao_provenance()`` reads the exact commit from pip's
direct_url.json (the version string also embeds it: 0.18.0+git<sha>).

Launches were captured ONCE on a CUDA machine by
``evaluation/torchao_capture.py``; rebuild semantics and the fail-loud
invariants (version drift, unresolved kernels) live in
``_captured.build_captured_corpus``.

Every row is labeled race-free (production code). Race-relevant
surface: global-amax atomic_max folds in the fp8 scaling kernels
(float8nocompile, moe float8_rowwise) and split-k atomic accumulation
in the shared int matmul — the second atomics-heavy production corpus
after FlagGems.
"""

from __future__ import annotations

from pathlib import Path

try:
    import torchao  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the torchao corpus needs torchao: USE_CPP=0 uv pip install "
        '--no-build-isolation "torchao @ git+https://github.com/pytorch/'
        'ao@<captured commit>" (USE_CPP=0 skips the C++ extension; the '
        "corpus only exercises the pure-Python Triton kernels)"
    ) from e

import torchao.kernel.blockwise_quantization
import torchao.kernel.bsr_triton_ops

from evaluation.kernels._captured import build_captured_corpus


def _publish_closure_kernels(mod) -> None:
    """torchao/kernel's ``_lazy_init_triton()`` stores some kernels only
    inside a torch.library CustomOpDef whose impl function CLOSES OVER
    the Autotuner (blockwise_fp8_gemm_kernel) — dig those out of the
    closure cells and publish them under their def name so the shared
    resolver's getattr path finds them."""
    import types

    from triton.runtime.jit import JITFunction

    candidates = []
    for v in list(vars(mod).values()):
        if isinstance(v, types.FunctionType):
            candidates.append(v)
        init_fn = getattr(v, "_init_fn", None)  # CustomOpDef
        if isinstance(init_fn, types.FunctionType):
            candidates.append(init_fn)
    for fn in candidates:
        for cell in fn.__closure__ or ():
            try:
                wrapped = cell.cell_contents
            except ValueError:  # pragma: no cover — empty cell
                continue
            obj = wrapped
            for _ in range(8):
                if isinstance(obj, JITFunction):
                    if not hasattr(mod, obj.fn.__name__):
                        setattr(mod, obj.fn.__name__, wrapped)
                    break
                obj = getattr(obj, "fn", None)
                if obj is None:
                    break


# torchao/kernel builds its Triton kernels inside _lazy_init_triton()
# closures — importing the modules does NOT create them, so trigger the
# (idempotent, flag-guarded) init before kernel resolution, then surface
# the closure-held ones
torchao.kernel.blockwise_quantization._lazy_init_triton()
torchao.kernel.bsr_triton_ops._lazy_init_triton()
_publish_closure_kernels(torchao.kernel.blockwise_quantization)
_publish_closure_kernels(torchao.kernel.bsr_triton_ops)

SPECS_PATH = Path(__file__).parent / "torchao_specs.json"

CORPUS = build_captured_corpus(
    corpus_name="torchao",
    specs_path=SPECS_PATH,
    dist_name="torchao",
    version_field="torchao",
    install_hint=(
        "USE_CPP=0 uv pip install --no-build-isolation 'torchao @ "
        "git+https://github.com/pytorch/ao@<captured commit>'"
    ),
)
