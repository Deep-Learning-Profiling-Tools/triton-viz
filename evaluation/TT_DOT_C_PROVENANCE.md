# Positional dependency of the `tt.dot` accumulator

2026-09-07. Production correction based on `f133ec8`, the demo branch after
cuTile token order. This changes the compiled reader at every analysis
level; it does not change the frozen `31c48f5` measurements.

## Change and justification

For the recognized three-operand custom assembly form, `tt.dot A, B, C`
computes the matrix product plus C. The contribution from C at `(i,j)` is
to result `(i,j)`. Preserve C's existing position-preserving load provenance
for that addition component. A/B remain non-positional. If the same source
appears in both A/B and C, its valid C path survives; a non-positional C
path stays non-positional. The general provenance merge is unchanged.

This implements the existing same-position D3 rule for a fused addition.
The installed FLA source writes the two expressions as `b_dq += tl.dot(...)`
and `b_dv += tl.dot(...)`; the captured compiler output fuses each addition
into C. A direct `tl.dot(..., acc=C)` receives the same operand-sensitive
interpretation. Neither the matrix multiplication as a whole nor A/B is
classified as elementwise. No hardware barrier or floating-point numerical
equivalence claim is introduced.

Recognition accepts exactly three SSA operands, optional named
`inputPrecision`, and the known `maxNumImpreciseAcc` integer attribute dict.
Unknown attributes, a fourth operand, generic quoted assembly, `dot_scaled`,
and different op names receive no new positional provenance. They retain
the reader's conservative value behavior. This is refusal of new ordering
evidence, not mandatory whole-kernel refusal when an unknown value is unused.
Dot values remain `DataDep`: unknown masks still widen and are flagged,
and unknown address arithmetic still refuses. Access offsets, masks,
activity predicates, sequence numbers, fences, scopes, budgets and the
shared solver's rules are unchanged.

The kernel's integer capture values remain snapshot premises. A shifted
store still has an overlapping cross-position pair, so a valid C dependency
cannot turn that pair into an ordered one. Permutation, reshape, broadcast,
unsupported C transformations, unsupported loop-carried bindings, and a
subsequent dot's A/B path cannot revive a discarded positional dependency.

Local semantic anchors: `triton/language/semantic.py` passes the accumulator
as `create_dot`'s third operand with the result shape/type;
`triton/runtime/interpreter.py:create_dot` adds that array elementwise.
The checked local Triton `TritonOps.td:TT_DotOp` custom assembly format
matches the accepted spelling. The latter tree was not established as the
installed compiler revision, so it is syntax evidence only. The two saved
installed-compiler TTIR modules are the actual regression inputs.

## Verification

The [39 focused regressions](../tests/unit/test_ttir_dot_dependency.py)
exercise both single-path and multipath readers, exact supported syntax,
A/B-only and shared A/C sources, unrelated source exclusion, transformed
and chained C, the unfused addition equivalent, unknown `dot_scaled` and
generic syntax, loop boundaries, inactive/partial masks, shifted stores,
unknown dot-derived masks/addresses, the production compiled client, and
the shared sanitizer consumer. Solver checks establish feasibility before
checking a race, so success is not a vacuous infeasible base.

The focused and relevant existing suites pass together: **96 passed**.

```bash
PYTHONPATH="$PWD" /home/hwu27/workspace/triton-viz/.venv/bin/python -m pytest \
  tests/unit/test_ttir_dot_dependency.py \
  tests/unit/test_ttir_reader_datadep_policy.py \
  tests/unit/test_compiled_race_detector_ttir.py \
  tests/unit/test_compiled_sanitizer_reader.py \
  tests/unit/test_t1_global_races.py -q
```

`git diff --check` passes. The existing pre-commit environment passes Ruff,
mypy, spelling, size, whitespace and the other applicable repository hooks.
Ruff formatting changed only the test and diagnostic runner layout. The
[executed runner](diagnostics/dot-c-20260907/executed-diagnose-dot-c.py.txt)
retains its recorded hash; its AST equals the formatted reproduction runner.
The measured production reader is unchanged by formatting.

## Exact FLA diagnostics

The [reproduction script](diagnose_dot_c.py) calls the existing `_static_track`
with the baseline parser and production parser on the same CPU objects.
It checks the captured integer values, seed-zero regenerated tensor bytes
against the archived earlier diagnostic, and unchanged bytes after each call.
The baseline uses the source at `f133ec8`; every other top-level parser
helper is checked unchanged before using it. The accepted reader source
hash is recorded in each output. No GPU kernel is launched and no new TTIR
is compiled. Separate invocations use an outer 180-second process limit
with a five-second kill grace. Both completed normally.

| Cached configuration | Baseline | Production | Only graph changes |
|---|---|---|---|
| Non-variable length | two WARs | `proved@T1+content`, zero reports | store 11 depends on load 9; store 12 on load 7 |
| Variable length | two WARs | `proved@T1+content`, zero reports | store 13 depends on load 11; store 14 on load 9 |

In both cases the stores are source lines 1439 (`dv`) and 1440 (`dq`).
All other access fields agree exactly. The proof scope is
**`this-params-any-grid` with content qualification**, with no termination
assumption. Single-path parsing retains the same `indirect-address` refusal.
Both baseline and production retain `differential check failed: unhandled
term Loaded`; this is **not independent footprint validation or a hardware
proof**. The variable-length L1 bounds abstention is a separate unresolved
condition, not an enum proof newly established here.

The exact TTIR hashes are unchanged from both the earlier diagnosis and
the final `31c48f5` cache:

- Non-variable length: `8d0882bb828b44105f2df07aa41ff6359033fade46d45e86b6c8381bb49e02f3`.
- Variable length: `7b67ea890ebfc18af38c2f220881dea6fdd44a173b5c455abacdeaa56b119e2d`.

Full outputs: [nonvarlen.json](diagnostics/dot-c-20260907/nonvarlen.json),
[varlen.json](diagnostics/dot-c-20260907/varlen.json).
[Verification bindings](diagnostics/dot-c-20260907/verification.json) include
output/source hashes and the frozen manifest's specification/sidecar hashes.
The diagnostic seconds are execution records only, not performance data.
Two preliminary non-varlen calls reached the same result but their receipt
assertion failed because `dataclasses.asdict(AccessEvent)` omits dynamically
attached `deps`. The serializer now records that field explicitly; no detector
change was made in response to those failed diagnostic assertions.

Reproduction on this installation, using the same command with `--varlen`
and a distinct output path for the second configuration:

```bash
env -u CUDA_VISIBLE_DEVICES -u TRITON_INTERPRET PYTHONPATH="$PWD" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --kill-after=5s 180s \
  /home/hwu27/workspace/triton-viz/.venv/bin/python -m evaluation.diagnose_dot_c \
  --baseline-ref f133ec8 \
  --cache-root /tmp/triton-viz-final-cache-31c48f5 \
  --prior-diagnostic-dir /home/hwu27/workspace/tile-race-paper/baselines/results/enum-fix-rerun-ff160f1 \
  --output /tmp/dot-c-reproduction-nonvarlen.json
```

## Affected rerun scope

Do not restrict the production change to these two FLA cases. The common
TTIR parser is used by L0/L1/L2 and the compiled sanitizer. Any supported
dot whose C has positional loaded provenance reaching a later memory
value/mask/compare can change the graph. A newly obtained static proof can
also suppress a fallback and change runtime even if the selected verdict
was already a proof. Dot-heavy kernels with a constant C need not change,
and cuTile's separate parser is outside this patch.

The minimal selective correctness population is that graph-derived set
across all three levels; this patch does not claim an exhaustive corpus
count. Dependent optimization pairs, especially the FLA groups previously
showing race/proof disagreement, need inclusion for attribution. Existing
timeout/censored samples remain missing timing evidence.

For paper adoption after the remaining fixes are integrated, create a new
common pin and rerun the declared full ladder plus affected study panels.
The completed `31c48f5` publication and all prior raw data keep their original
revision and labels. This correction itself does not decide the unrelated
aiter, TorchAO interpreter, strict-deadline, or cuTile ordering work.
