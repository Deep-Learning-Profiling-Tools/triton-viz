"""FlagGems corpus: FlagOpen/flagos-ai's Triton ATen-operator library
analyzed AS INSTALLED via a git-pinned pip install (PyPI lags upstream;
``runner._flaggems_provenance()`` reads the exact commit from pip's
direct_url.json).

Launches were captured ONCE on a CUDA machine by
``evaluation/flaggems_capture.py``; rebuild semantics and the fail-loud
invariants (version drift, unresolved kernels) live in
``_captured.build_captured_corpus`` — libentry/libtuner wrappers expose
``.fn`` chains, so the shared type-descent unwrap applies unchanged.

Every row is labeled race-free (production code). This is the
race-relevant corpus: production ATOMIC scatter/index/histogram
kernels (the counting-axiom machinery's first at-scale field test),
``unique``'s cumsum-addressed stores, ``nonzero``'s loaded-prefix-sum
addressing, and ``mm_streamk``'s inter-CTA spinlock (atomic_xchg
arrive + atomic_cas spin — the await abstraction's first production
instance).
"""

from __future__ import annotations

from pathlib import Path

try:
    import flag_gems  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the flaggems corpus needs flag_gems: uv pip install --no-deps "
        '"flag_gems @ git+https://github.com/flagos-ai/FlagGems@<captured '
        'commit>" (plus sqlalchemy; --no-deps avoids its numpy==1.26.4 pin '
        "downgrading the env)"
    ) from e

from evaluation.kernels._captured import build_captured_corpus

SPECS_PATH = Path(__file__).parent / "flaggems_specs.json"

CORPUS = build_captured_corpus(
    corpus_name="flaggems",
    specs_path=SPECS_PATH,
    dist_name="flag_gems",
    version_field="flag_gems",
    install_hint=(
        "uv pip install --no-deps 'flag_gems @ git+https://github.com/"
        "flagos-ai/FlagGems@<captured commit>'"
    ),
)
