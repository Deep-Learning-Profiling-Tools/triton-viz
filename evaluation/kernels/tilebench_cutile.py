"""tilebench_cutile corpus: TileBench's cuTile (cuda.tile) twin
implementations — the first non-Triton corpus, consumed through the
CuTile IR reader front-end.

Rows carry their CAPTURED CuTile IR text (compiled at launch capture;
see evaluation/tilebench_cutile_capture) plus arg descriptors, so
rebuild needs neither cuda-tile nor a GPU — only the same TileBench
checkout pin as the Triton twin corpus (the shared commit drift guard).

The bitonic-network operators (bitonic_sort, top_k_selection,
radix_sort) launch one SPECIALIZATION per (stage, stride) host-loop
step — stride is a ct.Constant, so each step is a distinct compiled
kernel. The capture payload is TRIMMED to the first
``MAX_SPECIALIZATIONS`` per (case, kernel) at store time (see
tilebench_cutile_capture.trim_specializations) with the drop count
recorded — no silent caps; the module keeps the same guard so a
re-captured untrimmed payload cannot silently balloon the corpus.

Every row has a same-operator Triton twin in the ``tilebench`` corpus —
the cross-DSL differential pairing is by case name.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.spec import Corpus, LaunchSpec
from evaluation.tilebench_capture import TILEBENCH_ROOT, tilebench_commit

SPECS_PATH = Path(__file__).parent / "tilebench_cutile_specs.json"
MAX_SPECIALIZATIONS = 2

_payload = json.loads(SPECS_PATH.read_text())
_commit = tilebench_commit()
if _commit != _payload["upstream_commit"]:
    raise ImportError(
        f"tilebench_cutile corpus was captured at TileBench commit "
        f"{_payload['upstream_commit']} but the checkout at "
        f"{TILEBENCH_ROOT} is at {_commit} — check out the captured "
        "commit or re-capture"
    )

CORPUS = Corpus("tilebench_cutile")

_kept = 0
_dropped = 0
for _case, _entry in sorted(_payload["cases"].items()):
    _per_kernel: dict[str, int] = {}
    for _slot, _rec in sorted(_entry["kernels"].items()):
        _n = _per_kernel.get(_rec["kernel"], 0)
        _per_kernel[_rec["kernel"]] = _n + 1
        if _n >= MAX_SPECIALIZATIONS:
            _dropped += 1
            continue
        _kept += 1
        _name = f"ctb_{_case}__{_rec['kernel']}"
        if _n:
            _name += f"__s{_n}"
        _aliases = _rec.get("aliases", {})
        CORPUS.add(
            LaunchSpec(
                name=_name,
                kernel_fn=None,
                signature={},
                constexprs=dict(_rec.get("constexprs", {})),
                make_args=lambda seed: (),
                grid=tuple(_rec["grid"]),
                expected="race-free",
                pattern="cutile-twin",
                params_note=f"cuTile twin of tilebench/{_case}",
                aliased=len(set(_aliases.values())) < len(_aliases),
                frontend="cutile",
                cutile={
                    "ir": _rec["ir"],
                    "args": _rec["args"],
                    "kernel": _rec["kernel"],
                    "module": _rec["module"],
                },
            )
        )

CORPUS.provenance = {
    "tilebench_cutile_upstream": _payload["upstream"],
    "tilebench_cutile_captured_version": _payload["tilebench_cutile"],
    "tilebench_cutile_upstream_commit": _payload["upstream_commit"],
    "tilebench_cutile_specializations_kept": _kept,
    "tilebench_cutile_specializations_dropped": _dropped
    + int(_payload.get("specializations_dropped_total", 0)),
}
