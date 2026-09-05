"""tilebench corpus: the Triton twin implementations from the group's
own TileBench multi-backend benchmark
(Deep-Learning-Profiling-Tools/Tilebench), captured by driving the
suite's ``core.engine`` harness (see evaluation/tilebench_capture).

TileBench has no packaging metadata, so unlike the pip-pinned corpora it
is analyzed as a LOCAL GIT CHECKOUT: ``TILEBENCH_ROOT`` goes on sys.path
for kernel resolution and the checkout HEAD commit is the version pin —
``tilebench_commit()`` refuses tracked-dirty trees, and the shared
drift guard (``installed_version=``) refuses a commit mismatch.

Every operator also ships a cuTile twin (impl_cutile.py) — this corpus
is the Triton-side baseline for the planned cuTile frontend, enabling
same-operator cross-DSL differential analysis later. Race-relevant
surface: destindex (duplicate-destination scatter, the
quantize_kv_copy family), streamk_matmul (atomic partial accumulation),
bitonic_sort/top_k_selection (in-place exchange networks), radix_sort
(data-dependent permutation scatter), histogramming (atomic scatter-add).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evaluation.kernels._captured import build_captured_corpus
from evaluation.tilebench_capture import TILEBENCH_ROOT, tilebench_commit

SPECS_PATH = Path(__file__).parent / "tilebench_specs.json"

if not TILEBENCH_ROOT.is_dir():
    raise ImportError(
        f"the tilebench corpus needs a TileBench checkout at "
        f"{TILEBENCH_ROOT} (or set TILEBENCH_ROOT): git clone "
        + json.loads(SPECS_PATH.read_text())["upstream"]
    )

# kernels resolve by module path (benchmarks.operators.<op>.impl_triton),
# which only imports with the checkout root on sys.path
_root = str(TILEBENCH_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

CORPUS = build_captured_corpus(
    corpus_name="tilebench",
    specs_path=SPECS_PATH,
    dist_name="tilebench (local checkout)",
    version_field="tilebench",
    install_hint=(
        f"git -C {TILEBENCH_ROOT} checkout <captured commit> "
        "(or set TILEBENCH_ROOT to a checkout at that commit)"
    ),
    installed_version=tilebench_commit(),
)
