"""Compiled-mode race detector: static analysis of TritonGPU IR (TTGIR).

The dynamic mode (``SymbolicRaceDetector``) reasons about cross-CTA global
memory races from an interpreter-driven symbolic capture. Shared memory is
invisible at that level — it is introduced by TritonGPU compiler passes
(``ttg.local_alloc`` / ``ttg.async_copy_global_to_local`` / software
pipelining). This package analyzes the compiled TTGIR instead: it extracts
shared-memory access events plus the pipeline synchronization structure
(commit-group / async-wait counting, multibuffer rotation) and asks Z3
whether any pair of accesses can overlap unordered. UNSAT is a proof for
the specialization, over all inputs and grids.

See ``race_detector_compiled_mode_plan.md`` at the repository root for the
full design, scope and model boundary.
"""

from .client import CompiledRaceDetector
from .smt_encoder import AnalysisResult, CompiledRaceReport, analyze_ttgir

__all__ = [
    "AnalysisResult",
    "CompiledRaceDetector",
    "CompiledRaceReport",
    "analyze_ttgir",
]
