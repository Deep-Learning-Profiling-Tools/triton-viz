"""aiter Triton-op corpus (captured launches, NVIDIA side).

Coverage corpus over ROCm/aiter's Triton kernels, distinct from
``aiter_originals`` (which stays the two-row A1 case corpus by
design). Rows are captured from aiter's own ``op_tests/triton_tests``
files by ``evaluation.aiter_capture`` on this machine; kernels resolve
from a plain checkout at ``AITER_ROOT`` through the package stubs of
``_aiter_loader`` (aiter's real package inits require ROCm), pinned to
the captured commit, the tilebench local-checkout pattern.
"""

from pathlib import Path

from evaluation.kernels._aiter_loader import (
    AITER_ROOT,
    aiter_commit,
    install_stubs,
)
from evaluation.kernels._captured import build_captured_corpus

SPECS_PATH = Path(__file__).parent / "aiter_ops_specs.json"

install_stubs()

CORPUS = build_captured_corpus(
    corpus_name="aiter_ops",
    specs_path=SPECS_PATH,
    dist_name="aiter (local checkout)",
    version_field="aiter",
    install_hint=(
        f"git -C {AITER_ROOT} checkout <captured commit> "
        "(or set AITER_ROOT to a checkout at that commit)"
    ),
    installed_version=aiter_commit(),
)
