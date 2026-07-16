"""flash-linear-attention corpus: production linear-attention Triton
kernels (fla-org/flash-linear-attention) analyzed AS INSTALLED via the
``fla-core`` pip package (evaluation-only dependency, like liger;
``runner._fla_provenance()`` pins version + upstream commit).

Launches were captured ONCE on a CUDA machine by
``evaluation/fla_capture.py`` (public ``fla.ops`` entry points, forward
+ backward, dense + varlen, small fp32 shapes); rebuild semantics and
the fail-loud invariants (version drift, unresolved kernels) live in
``_captured.build_captured_corpus``.

Every row is labeled race-free (production code); as with liger and
TritonBench the deliverable is the ladder distribution. NOTE the
dominant abstention is indirect-address (varlen cu_seqlens/chunk_indices
load chains) — NOT block pointers, which triton's make_ttir rewrites
away before the shared reader ever sees them.
"""

from __future__ import annotations

from pathlib import Path

try:
    import fla  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the fla corpus needs fla-core: uv pip install fla-core==0.5.1"
    ) from e

from evaluation.kernels._captured import build_captured_corpus

SPECS_PATH = Path(__file__).parent / "fla_specs.json"

CORPUS = build_captured_corpus(
    corpus_name="fla",
    specs_path=SPECS_PATH,
    dist_name="fla-core",
    version_field="fla_core",
    install_hint="uv pip install fla-core==<captured version>",
)
