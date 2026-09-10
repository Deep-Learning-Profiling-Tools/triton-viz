"""FlagAttention corpus: BAAI's production Triton attention kernels
(FlagOpen/FlagAttention — flash, piecewise, split-kv and paged
attention; 13 kernels, Apache-2.0) analyzed AS INSTALLED via a
git-pinned pip install (no PyPI release exists;
``runner._flagattn_provenance()`` reads the exact commit from pip's
direct_url.json).

Launches were captured ONCE on a CUDA machine by
``evaluation/flagattn_capture.py`` (public API, fp16, fwd + bwd, incl.
dropout/philox, GQA, non-divisible seqlens and the paged
partition+reduce pair); rebuild semantics and the fail-loud invariants
(version drift, unresolved kernels) live in
``_captured.build_captured_corpus``.

Every row is labeled race-free (production code). The corpus
complements fla: plain pointer arithmetic + masks, no autotune, no
atomics — flash/piecewise/split_kv aim at the static track's sweet
spot, while paged attention's block_tables/context_lens load chains
(vLLM-style indirect addressing) exercise the snapshot machinery on
the interp tier.
"""

from __future__ import annotations

from pathlib import Path

try:
    import flag_attn  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the flagattn corpus needs flag_attn: uv pip install "
        '"flag_attn @ git+https://github.com/FlagOpen/FlagAttention@<captured commit>"'
    ) from e

from evaluation.kernels._captured import build_captured_corpus

SPECS_PATH = Path(__file__).parent / "flagattn_specs.json"

CORPUS = build_captured_corpus(
    corpus_name="flagattn",
    specs_path=SPECS_PATH,
    dist_name="flag_attn",
    version_field="flag_attn",
    install_hint=(
        "uv pip install 'flag_attn @ git+https://github.com/FlagOpen/"
        "FlagAttention@<captured commit>'"
    ),
)
