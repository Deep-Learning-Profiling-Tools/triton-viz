"""tritonbench_meta corpus: meta-pytorch/tritonbench's OWN Triton
operator implementations (Meta's benchmark suite; distinct from
thunlp/TritonBench = tritonbench_g), analyzed AS INSTALLED via a
git-pinned pip install and captured by driving the suite's own
``BenchmarkOperator`` harness (see evaluation/tritonbench_meta_capture).

The dist version is a constant 0.0.1 — too weak for the shared version
drift guard — so this module ALSO hard-checks the installed
direct_url.json commit against the captured one.

Every row is labeled race-free (production benchmark code). Race-
relevant surface: streamk/partition-k matmul atomic accumulation, the
tutorial layer-norm backward dw/db lock (atomic spin), gdpa's atomic
sites, and the split-k decoding attention family.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

try:
    import tritonbench  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "the tritonbench_meta corpus needs tritonbench: uv pip install "
        '"tritonbench @ git+https://github.com/meta-pytorch/tritonbench@'
        '<captured commit>" (plus pynvml, transformers)'
    ) from e

from evaluation.kernels._captured import build_captured_corpus

SPECS_PATH = Path(__file__).parent / "tritonbench_meta_specs.json"


def _installed_commit() -> str | None:
    raw = metadata.distribution("tritonbench").read_text("direct_url.json")
    if not raw:
        return None
    return json.loads(raw).get("vcs_info", {}).get("commit_id")


_payload_commit = json.loads(SPECS_PATH.read_text())["upstream_commit"]
_commit = _installed_commit()
if _commit != _payload_commit:
    raise ImportError(
        f"tritonbench_meta corpus was captured at upstream commit "
        f"{_payload_commit} but the installed tritonbench is at "
        f"{_commit} (dist version 0.0.1 is constant, so the commit is "
        f"the real pin) — reinstall the captured commit or re-capture"
    )

CORPUS = build_captured_corpus(
    corpus_name="tritonbench_meta",
    specs_path=SPECS_PATH,
    dist_name="tritonbench",
    version_field="tritonbench_meta",
    install_hint=(
        "uv pip install 'tritonbench @ git+https://github.com/"
        "meta-pytorch/tritonbench@<captured commit>'"
    ),
)
