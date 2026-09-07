# Frontend conformance repairs (2026-09-07)

This record follows the opposite-frontend review of the paper's immutable
`31c48f5` L0/L1/L2 publications. The repair branch starts from demo `f133ec8`.
The implementation commits are `234a8fe`, `52bf83d` and `edb37fb`; the normal
repository hooks pass. These are correctness diagnostics, not new
formal timing samples or a replacement pinned publication.

## Mechanisms and independent controls

1. **Mixed positional provenance.** A RoPE result includes both the original
   loaded tile and a permutation of that same tile: `x*cos + rotate(x)*sin`.
   Merging its per-source flags with AND erased the original direct path.
   Simultaneously evaluated elementwise operands now preserve an independently
   positional path with OR. A pure permutation remains non-positional.
   `arith.select` retains a branch-derived unconditional dependency only when
   both arms preserve it; the always-evaluated condition can supply its own
   positional dependency. Unrecognized select syntax retains no positional
   claim. Tests include both operand orders, permuted and inactive-arm
   negatives, a loaded condition, and actual solver decisions.

2. **Independent creation sites on one physical tile axis.** The interpreter
   previously allowed separate address and mask `tl.arange` calls to choose
   unrelated positions in one access. The symbolic solver could then invent
   an intra-instance WAW by keeping the address lane equal while changing
   the mask lane. Uniformly one-dimensional elementwise pointer/mask DAGs
   now carry access-local equality of coordinates (range value minus range
   start). This adds no global creation-site equality. Multidimensional
   intermediates, reshapes, reductions and transposes do not use this rule.
   Controls preserve a real duplicate-lane WAW, off-diagonal collisions from
   independent equal-size axes, and distinct nonzero range origins.

3. **Loop fence sequence.** Barriers received their sequence numbers during
   capture, while loop memory events received theirs only at loop flush.
   The resulting record order moved an intervening barrier before both
   accesses. Deferred accesses now reserve their sequence at first capture.
   Nested-loop controls distinguish an intervening fence from a fence before
   the store, and retain cross-instance conflicts despite local fences.
   This repairs ordering metadata without adding new synchronization.

4. **Tensor Boolean masks.** The installed interpreter treats a non-scalar
   tensor as a truthy object for Python `and`/`or`, whereas the installed
   compiler combines tensor operands elementwise. The frontend now rewrites
   Boolean expressions using the compiler's rules, evaluates each operand
   once, and preserves constexpr short circuit through deferred operands.
   Both symbolic capture and concrete enumeration use the rewrite. Mutation
   controls retain the active collision arm of a tensor `or`, and explicitly
   check concrete enumeration as well as symbolic detection.

The unchanged source kernels, input-dependent address terms and default
"clean" metadata are not used as accuracy oracles. The positive conclusions
below use source access geometry and the semantic controls above in addition
to the corrected frontend results.

## Exact real-case diagnostics

Raw diagnostic JSON and logs live in the isolated worktree at
`evaluation/results/frontend-conformance-20260907/`. Each successful probe
records the kernel-source SHA-256, seed, grid, scalar arguments, tensor
shape/stride/dtype and logical-byte hashes. Static probes consume the exact
saved TTIR selected by its full SHA-256 matching the published prefix.
They disable C2/C3 replay so the IR result is diagnosed independently.
Seven captured-value sidecars were copied into the isolated worktree and
verified byte-identical before execution. Source-side preservation of the
recorded inputs is not a universal certificate of every historical runtime
tensor byte or alias relation.

| Configuration | Original disagreement | Corrected diagnostic |
|---|---|---|
| aiter fused cosine cache | L1 enum proof, L2 same-instance WAR | `proved@T1+content`, zero reports |
| aiter fused reshape/cache | L1 enum proof, L2 same-instance WAR | `proved@T1+content`, zero reports |
| aiter cached THD GQA | L1 enum proof, L2 same-instance WAR | `proved@T1-launch+content`, zero selected reports |
| aiter cached THD GQA one-head | L1 enum proof, L2 same-instance WAR | `proved@T1+content`, zero reports |
| aiter THD forward | L1 enum proof, L2 same-instance WAR | `proved@T1+content`, zero reports |
| FlagGems embedding | Static proof, interpreter phantom WAW | Interpreter `ok`, zero reports, contents-snapshot premise retained |
| FlagGems weight norm first | Static proof, interpreter WAW across PIDs | Interpreter `ok`, zero reports |
| Recurrent-retention backward | Static proof, 24 interpreter same-PID WAWs | Interpreter `ok`, zero reports |
| TorchAO Hadamard QKV | Interpreter reports or budget-driven enum fallback | Full 32-instance enum proof; source-matched S4/D8 interpreter control changes three RAWs to zero |
| TorchAO Hadamard V | Interpreter reports or budget-driven enum fallback | Full 32-instance enum proof; source-matched S4/D8 interpreter control changes three RAWs to zero |
| GDN2 fused recurrent sentinel | Main L2 selects static proof before enum | Direct enum freshly exercises all eight captured instances and proves clean |

`graph-comparison.json` compares the five aiter graphs against the reader at
`f133ec8`: all non-dependency graph fields are identical and exactly ten
direct anchors are added. The general GQA kernel retains broader-grid
WAR/WAW evidence with PIDs `(32,0,0)` and `(0,1,0)`; its captured-launch proof
must not be described as an any-grid result. That comparison predates the
separate callsite-location repair `edb37fb`. `callsite-comparison.json`
verifies that this later fix changes only eight source-location fields in
the two fused-cache graphs, recovering the callee load sites at lines
552/581 and helper stores at 150/151. Direct and nested callsites resolve
the callee; unknown, cyclic and ambiguous fused locations remain unavailable.
Dependency and footprint fields do not change.

The embedding launch has 256 instances, one disjoint output row of 128
elements per instance, and indices affect only the weight reads. Weight
norm has two instances covering 32 rows each; its mask now restricts the
2048-column tile to the actual 128 columns. Retention backward has grid
`(1,1,8)`, `T=8` and `DK=DV=16`: the three output arrays use disjoint
instance/time slices and the separate range calls denote the same within-row
coordinate. These source facts explain why the removed reports were
fabricated by the frontend, without treating arbitrary real-kernel labels
as ground truth.

Two initial fused-cache attempts failed in diagnostic input hashing on
zero-dimensional floating tensors before invoking analysis. The serializer
now flattens logical elements before byte reinterpretation; their original
failed logs are retained alongside successful retries. No result is inferred
from either failed admission or an interpreter timeout.

The full Hadamard configurations retain `S=128`, `D=64`, six butterfly
stages, four chunks and grid `(2,4,4)`. Both interpreter probes time out at
the ordinary 60-second deadline and at the separate diagnostic 180-second
deadline (240-second outer cap). Neither timeout is credited as a proof.
The direct enum runs at the ordinary 60-second enum budget freshly cover
all 32 instances with zero reports. Their source, grid and input hashes
match the corresponding full interpreter probes.

Reducing only `S` to 4 and `chunk_size` to 1 still times out at 60 seconds
on both implementations. The final matched-source controls also reduce
`D` to 8 and `LOG2_D` to 3, retaining four chunks and all 32 instances.
Input/output tensors are sliced to those dimensions with their actual
strides preserved; temporary-buffer regions retain their disjoint original
strides. These are separate legal diagnostic inputs, not replacements for
the captured full inputs. On each kernel, baseline `31c48f5` completes and
reports three same-instance RAWs at the real Hadamard helper; the repaired
frontend completes and reports zero. All scalar parameters, shapes,
strides, dtypes and logical tensor-byte hashes match within each pair, and
the kernel-source hashes are identical. This closes the fence-capture
correctness repair while retaining full dynamic completion as a
scalability limitation. No performance comparison is claimed from these
diagnostics.

`SUMMARY.json` binds 23 completed worker receipts, including eight honest
timeouts, plus their logs and the two graph audits. Two additional
hash-serialization admission failures remain visible. It records four
before/after input-matched Hadamard pairs (two timing out at S4/D64, two
completing at S4/D8). The production source hashes are recorded separately
from the evolving diagnostic harness; the archived final probe is a
reproduction aid, not a claim that every earlier diagnostic used identical
serializer source bytes.

## Verification and rerun scope

The final combined local regression group passed 254 tests with three existing
skips, including the concrete-mask, nonzero-origin and callsite additions.
This final check also includes the cuTile-only upstream changes pulled from
demo `ddc4735`; the earlier diagnostic receipts retain their executed source
hashes. Normal hooks
run from the existing detector virtualenv and include Ruff, mypy, codespell,
whitespace and merge-conflict checks.

All three levels invoke the affected interpreter frontend, and the static
provenance repair is not L2-gated. The shared Boolean rewrite also affects
concrete enumeration and replay. Therefore a new adopted common-pin result
requires routine full L1 then L2 main rows, affected correctness certificates,
phase and budget studies, and input-sensitive experiments to be recomputed
at the integrated pin. The user's updated policy retains L0 only for
targeted regressions and selected-study controls, not a routine full-corpus
rerun. The old `31c48f5` receipts remain immutable. This diagnostic
record does not rewrite their verdict distributions or timing statistics.

The access-local one-dimensional rule does not claim to repair arbitrary
multidimensional lane reconstruction. Nested outer-load dependency anchors
that are unavailable until a later flush can still conservatively lose
ordering. Neither remaining boundary licenses an unconditional proof from
an incomplete capture.
