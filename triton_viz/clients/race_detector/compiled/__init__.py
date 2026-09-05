"""Compiled-mode race detector: static analysis of TritonGPU IR (TTGIR).

The dynamic mode (``SymbolicRaceDetector``) reasons about cross-CTA global
memory races from an interpreter-driven symbolic capture. Shared memory is
invisible at that level — it is introduced by TritonGPU compiler passes
(``ttg.local_alloc`` / ``ttg.async_copy_global_to_local`` / software
pipelining). This package analyzes the compiled TTGIR instead: it extracts
shared-memory access events plus the pipeline synchronization structure
(commit-group / async-wait counting, multibuffer rotation) and asks Z3, for
every (copy, load) pair on a slot, whether the load's guarding async_wait can
fail to cover the copy's commit group. UNSAT over all pairs proves the cp.async
pipeline carries no such wait-coverage violation for the specialization, over
all inputs/grids/trip counts — within the model boundary in ``hb.py`` (RAW
direction only, whole-tile slots, lockstep/Membar-barrier assumption). It is
that wait-coverage proof, not a full byte-level data-race proof; the per-report
``byte_offset`` is a representative witness byte, not part of the solved query.

See ``race_detector_static_hybrid_plan.md`` (Part II) at the repository root
for the full design, scope and model boundary.
"""

from .client import CompiledRaceDetector
from .smt_encoder import AnalysisResult, CompiledRaceReport, analyze_ttgir

__all__ = [
    "AnalysisResult",
    "CompiledRaceDetector",
    "CompiledRaceReport",
    "analyze_ttgir",
]
