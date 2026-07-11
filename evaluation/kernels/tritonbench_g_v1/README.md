# Vendored: TritonBench_G_v1

Upstream: https://github.com/thunlp/TritonBench — `data/TritonBench_G_v1`
(184 standalone real-world Triton operator files, each: kernel(s) + host
wrapper + a `#####…`-separated test block that executes at import time on
CUDA).

- Upstream commit: `603e28a5050e8c268f6883a69709d477a272d49a`
- Retrieved: 2026-07-10
- License: Apache-2.0 (see LICENSE in this directory)
- Files are byte-identical to upstream (excluded from repo formatters);
  do not edit — regenerate from upstream instead.

Vendored (rather than a git submodule or download-on-demand) for artifact
self-containment: archived repo tarballs keep the corpus, evaluation runs
offline, and the exact sources are pinned. The launch specs consumed by
the harness are captured ONCE on a CUDA machine by
`evaluation/tritonbench_capture.py` (the test blocks need a GPU) into
`tritonbench_g_specs.json`; the corpus module then rebuilds CPU launches
from those specs on any machine, executing only each file's pre-separator
kernel section.
