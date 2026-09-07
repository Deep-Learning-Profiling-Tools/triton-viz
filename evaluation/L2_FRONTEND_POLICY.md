# L2 frontend execution

The default L2 evaluation harness now stops after a static decision. A static
proof or report keeps its existing terminal, witnesses and qualifications.
Static abstention runs the interpreter; a remaining abstention runs concrete
enumeration under the existing budget. Static C2/C3 replay and checks are
unchanged. Requested mutation analysis still runs after a static proof.

This follows the existing composed classifier, which already selects static
decisions before consulting the interpreter. It changes which independent
frontend observations are collected. L0/L1 keep the historical protocol.
cuTile still has only its static frontend.

## Execution and comparison

The normal invocation needs no additional flag:

```sh
python -m evaluation.runner --corpus tritonracebench --ladder-level L2
```

To collect the independent interpreter result for every Triton configuration:

```sh
TRITON_VIZ_EVAL_ALL_FRONTENDS=1 python -m evaluation.runner \
  --corpus tritonracebench --ladder-level L2
```

The same environment switch applies to direct harness calls, reused workers
and pinned runs. Only unset, empty, `0` and `1` are accepted. The resolved
`frontend_policy` is `on-demand` for default L2 and `all` for the explicit
comparison or L0/L1. Historical records without this field mean `all`.

## Records and timing

A skipped interpreter has `status: not-run`, `reason: static-decided` and
`time_s: null`. It supplies no independent frontend outcome or timing sample.
Known skipped work contributes zero to a recorded-stage subtotal, while the
individual interpreter time remains unmeasured. Skipped observations must
not count as interpreter abstentions or establish static-only coverage.

Worker wall time measures the pipeline that actually ran, including startup,
imports, compilation and the selected analysis paths. Existing published
wall-time measurements remain attached to their original every-frontend
protocol; they cannot be relabeled as on-demand measurements. New median or
tail claims require a fresh run with the resolved policy recorded.

Pinned manifests freeze the execution policy and its environment override.
Resumption and publication reject a policy mismatch rather than mixing
comparison observations with on-demand observations in one dataset.

## Validation (2026-09-07)

The focused execution tests cover static proof and report short-circuiting,
proof qualifications, retained witnesses, mutation checks, interpreter
errors/timeouts, faithful-refutation scope guards, enumeration fallback,
L0/L1 comparison behavior, the explicit override, and cuTile dispatch.
All 31 cases and the 25 existing ladder-level tests pass.

A fresh-process L2 TritonRaceBench diagnostic at `9fee74c` runs all 64
configurations: all 63 labeled verdicts and all 29 designated access-pair
checks agree. The 57 static decisions skip the interpreter; seven
configurations use it; none needs enumeration. Every terminal matches the
archived `31c48f5` benchmark terminal. This is a correctness check; the
runner-header provenance change was integrated afterward.

The original FLA GDN2 recurrent forward configuration also returns its
`proved@T1` result with the interpreter explicitly not run. The diagnostic
uses a writable temporary Triton cache because the sandbox cannot write the
normal user cache. An earlier cache-blocked attempt is excluded from the
successful validation. Neither diagnostic replaces any paper timing.
