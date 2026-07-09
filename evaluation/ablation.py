"""RQ5 ablation study (plan S5 / paper sec:eval-baselines).

Three switches, each isolating one pillar of the encoding, run over the
LITMUS corpora (labels known, so a verdict flip is attributable):

  no-hb           solver ablations=("hb",): no happens-before at all.
                  Prediction: every ordering-based proof flips to races;
                  pure footprint-disjointness proofs survive.
  no-coherence    solver ablations=("coherence",): no per-location atomic
                  order (immediacy + the counting axiom go with it).
                  Prediction: single-winner / counting / mutex proofs flip;
                  plain release->acquire producer/consumer proofs SURVIVE
                  (their sw edge rides reads-from values, not order).
  no-load-values  dynamic detector ablations=("load-values",): one concrete
                  observation replaces the snapshot Select. Prediction:
                  value-dependent-mask verdicts flip on MIXED flag data
                  (demonstrated as a planted false proof).

Usage:  uv run python -m evaluation.ablation   (writes results/ABLATION.md)
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import torch

RESULTS_DIR = Path(__file__).parent / "results"
STATIC_CONFIGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline", ()),
    ("no-hb", ("hb",)),
    ("no-coherence", ("coherence",)),
)
LITMUS_CORPORA = ("golden_smoke", "rmw_sync", "await_sync")


def _static_status(spec, ttir: str, ablations: tuple[str, ...], seed: int) -> str:
    from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

    det = CompiledRaceDetector(
        confirm_races=False, differential_check=False, ablations=ablations
    )
    args = spec.make_args(seed)
    det.pre_warmup_callback(spec.kernel_fn, *args, grid=spec.grid, **spec.constexprs)
    det.post_warmup_callback(spec.kernel_fn, SimpleNamespace(asm={"ttir": ttir}))
    det.finalize()
    return det.last_global_status


def static_matrix(seed: int = 0) -> list[str]:
    from evaluation.harness import _host_compile_ttir
    from evaluation.kernels import load

    lines = [
        "## Static-track ablations (litmus corpora)",
        "",
        "| row | expected | " + " | ".join(n for n, _ in STATIC_CONFIGS) + " | flips |",
        "|---|---|" + "---|" * (len(STATIC_CONFIGS) + 1),
    ]
    flip_count = 0
    total = 0
    for corpus_name in LITMUS_CORPORA:
        corpus = load(corpus_name)
        for spec in corpus.specs:
            try:
                ttir = _host_compile_ttir(spec)
            except Exception as e:  # noqa: BLE001
                lines.append(
                    f"| {spec.name} | {spec.expected} | compile-error: "
                    f"{type(e).__name__} |" + " |" * len(STATIC_CONFIGS)
                )
                continue
            statuses = {
                name: _static_status(spec, ttir, abl, seed)
                for name, abl in STATIC_CONFIGS
            }
            base = statuses["baseline"]
            flips = [n for n, s in statuses.items() if n != "baseline" and s != base]
            total += 1
            flip_count += bool(flips)
            lines.append(
                f"| {spec.name} | {spec.expected} | "
                + " | ".join(statuses[n] for n, _ in STATIC_CONFIGS)
                + f" | {', '.join(flips) if flips else '-'} |"
            )
    lines += ["", f"rows with at least one flip: {flip_count}/{total}", ""]
    return lines


def load_value_ablation(seed: int = 0) -> list[str]:
    """The planted no-load-values false proof: dd_mask with MIXED flags
    (flags[0]=0, rest=1). Real semantics: lanes 1.. store into one shared
    range from every block — a race. The single-observation ablation reads
    flags[0]=0, collapses the mask to all-false, and proves the launch
    clean."""
    import triton_viz
    from evaluation.kernels.golden_smoke import dd_mask_kernel
    from triton_viz.clients import RaceDetector

    flags = torch.ones(64, dtype=torch.int32)
    flags[0] = 0

    def run(ablations: tuple[str, ...]) -> tuple[str, int]:
        triton_viz.clear()
        det = RaceDetector(ablations=ablations)
        g = torch.Generator().manual_seed(seed)
        args = (flags.clone(), torch.randn(256, generator=g), torch.zeros(64))
        traced = triton_viz.trace(det)(dd_mask_kernel)
        traced[(4,)](*args, BLOCK=64)
        return det.last_status, len(det.last_reports)

    base_status, base_n = run(())
    abl_status, abl_n = run(("load-values",))
    flipped = (base_status, base_n > 0) != (abl_status, abl_n > 0)
    return [
        "## Dynamic-track ablation: no-load-value-semantics",
        "",
        "dd_mask kernel, MIXED flags (flags[0]=0, rest=1), grid (4,):",
        "",
        f"- baseline (snapshot Select): status={base_status}, reports={base_n}",
        f"- no-load-values (single observation): status={abl_status}, "
        f"reports={abl_n}",
        "",
        (
            "**FLIP demonstrated** — the single-observation baseline erases a "
            "real value-gated race (the paper's predicted unsoundness)."
            if flipped and abl_n == 0
            else f"flip={'yes' if flipped else 'NO — investigate'}"
        ),
        "",
    ]


def main() -> None:
    t0 = time.perf_counter()
    lines = ["# RQ5 ablation study", ""]
    lines += static_matrix()
    lines += load_value_ablation()
    lines.append(f"(generated in {time.perf_counter() - t0:.1f}s)")
    out = "\n".join(lines)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ABLATION.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
