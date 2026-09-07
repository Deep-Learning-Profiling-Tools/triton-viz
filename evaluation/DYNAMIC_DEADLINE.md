# Dynamic-stage deadline and process accounting

The evaluation interpreter now runs in a fresh subprocess. Its parent accepts a result only after the child exits within the requested budget. If the deadline expires, the parent requests cancellation, allows 0.05 s of cleanup grace, kills a remaining child, and reaps it before returning a timeout. The reported `dynamic.time_s` is the measured parent READY/GO-to-reap interval, including actual cancellation and termination cost. It is never replaced by the requested budget.

The implementation is in `dynamic_subprocess.py` and the `_dynamic_track` / `_dynamic_track_local` split in `harness.py`. The validated candidate is `b10b8f8`, following the cancellation repair in `c812712` and the frame-lifetime correction in `66d0dba`. This is process containment with observed return slack, not a real-time scheduling guarantee for every host state. The existing evaluation rule still flags a returned dynamic time more than 0.5 s beyond its requested deadline.

## Cause and correction

The former watchdog repeatedly raised `TimeoutError` from SIGALRM. The four flagged context-attention observations in the frozen `31c48f5` budget study logged ignored exceptions from `z3.AstRef.__del__`. Python ignores exceptions from destructors, and an asynchronous exception can interrupt their remaining native-reference release. Ordinary interpreter fallback also catches `Exception` subclasses. Neither is a reliable cancellation mechanism.

The first repair made the signal request cancellation without raising an asynchronous exception. Only after expiry, a temporary Python trace raises a dedicated `BaseException` outside an active `__del__` stack. The existing context-interrupt thread releases native Z3 solver work; cancellation is not reinjected during unwinding. Prior tracing, timers and handlers are restored, the interrupter is stopped and joined, and trace bookkeeping does not retain interpreter frames or native expression graphs.

The fresh 60 s diagnostic then timestamped the signal at 60.009054 s, cancellation at 60.009206 s, and watchdog scope exit at 61.450825 s. The internal return was 61.450893 s. Thus approximately 1.442 s remained in actual post-cancellation unwinding even after ignored destructor exceptions were removed. Its wrapper took another approximately 1.54 s for the broader function envelope. The in-process repair alone did not satisfy the existing half-second return criterion. It was retained as cooperative cancellation inside the isolated child; forced process termination supplies containment when cleanup cannot finish promptly.

## Launch transport and setup

The parent creates the same fresh CPU arguments that the former dynamic stage requested. `torch.save` transports their full backing storages, including aliases, noncontiguous views, storage offsets and reinterpret wrappers. Logical tensor bytes and complete backing-storage bytes are hashed separately. The latter binds padding and gap bytes that pointer arithmetic could access. Runtime nonfinite scalar values are represented by their IEEE double bytes in the identity record; their actual transported values are unchanged. Unknown runtime argument types fail admission rather than receiving a weak string-only identity.

The `LaunchSpec.make_args` closure is removed from the callable payload. The child uses the transported arguments directly and does not execute another factory. `cloudpickle` transports the kernel and launch metadata. A custom JIT reducer reconstructs callable source and options without transferring compiled caches, runtime locks or device handles. Source, source locations, referenced JIT helpers and referenced module files are checked after deserialization. The parent also verifies the child's detector tree, harness and transport source hashes, Python version, dependency versions and module origins, and effective runtime configuration.

Serialization, child startup, imports, observer installation, tensor loading, identity validation and detector construction precede READY. They remain inside the full dynamic wrapper and whole-worker wall time. After READY the parent starts its clock and writes GO. The child begins its declared observers, executes the local dynamic stage, finishes its observers, writes its result and exits. A result message followed by a late exit is a timeout, even if the message contains a proof. Transport, source, configuration or observer failures are named harness errors; they cannot silently fall through to enumeration or retain a static proof as a successful complete row.

There is no fork of initialized CUDA, Torch or Z3 state. The child retains its parent's process group so the existing outer row-group kill also contains it. The per-row outer budget remains an additional boundary. Child setup has a separately named 60 s cap.

The optional dependency is declared under the `evaluation` extra:

```sh
uv sync --extra test --extra evaluation
```

For an already provisioned experiment environment, install only the added pinned dependency without updating the existing detector dependencies:

```sh
uv pip install --python .venv/bin/python cloudpickle==3.1.1
```

Development validation used version 3.1.1 under `/tmp/tilerace-deadline-deps`; that module origin is retained in the receipts. Its cached six-file source tree has SHA-256 `a6a75cbe76f314354415e448ddcbf9acae2ab9d25e479416f5595ccef69f3308` under the sorted relative-path-to-file-hash convention in `cloudpickle-source-receipt.json`. Existing frozen environment manifests were not modified. Cloudpickle is used only for private, locally created, short-lived transport between the same Python interpreter version, consistent with its [documented usage](https://github.com/cloudpipe/cloudpickle).

## Observer and memory contract

`harness.DYNAMIC_CHILD_HOOKS` is a tuple of JSON-compatible descriptors:

```python
{
    "name": "profile",
    "module": "dynamic_child_observer",
    "factory": "create_observer",
    "kwargs": {...},
}
```

The factory installs the selected optimization switches and profiler explicitly in the child. It returns an observer with `begin()`, `snapshot()` and `finish()`. `begin()` runs after GO. A monitor obtains copied snapshots every 0.1 s and writes them by atomic replacement; this channel cannot block the parent's deadline polling. `finish()` runs after the local dynamic call returns and the monitor is joined. The paper adapter owns its root span and closes it in `finish()` before reporting a complete profile.

`dynamic.execution.hooks[name]` records the payload, sample time and completeness. A forced kill retains the latest partial snapshot with unfinished spans and in-flight checks visible. Missing snapshots remain explicit unavailable entries. Snapshot errors remain explicit instrumentation errors. Completed counters in a partial snapshot are lower bounds, not a complete phase profile; they cannot supply completed-stage medians. Parent spawn/wait time is not added to child exclusive categories as though it were another detector phase.

The execution receipt separates:

- `startup_s`: parent function entry through READY/GO, including transport and setup.
- `parent_ready_to_reap_s` / `dynamic.time_s`: actual analysis-window, cancellation and child-reap time.
- `full_wall_s`: parent function entry through result handling and temporary-file cleanup, before final function return.
- `child_internal_time_s` and `deadline`: the child's narrower local clock and signal/cancellation/scope-exit observations when a complete child result survives.
- The child's high-water RSS at its reported sample, sampled child RSS and sampled simultaneous parent-plus-child RSS, and the parent's own high-water RSS.

Periodic RSS maxima are lower bounds at 0.1 s sampling. A last child sample before termination does not certify its through-exit peak. Summed parent/child RSS can count shared pages more than once and is not PSS. Parent-only `RUSAGE_SELF` omits the child's memory and must not be substituted for total analysis memory. These definitions deliberately remain distinct from the old single-process measurements.

## Validation at b10b8f8

Raw archive: `evaluation/results/dynamic-deadline-fix-20260907/`. `SUMMARY.json` binds 52 artifact files and has SHA-256 `34433d7ebbfc300644ca3347b63c42845e16e2ffea901e47cc770cc08831ee10`. The archive retains the unsuccessful preflight separately: sandbox GPU introspection omitted the corpus row before any detector analysis, and a startup-only RSS-sampling edge in the diagnostic wrapper was corrected before the admitted retry.

Every admitted context-attention run has exactly the same recorded named input bytes/layout and TTIR as frozen budget sample 0008. All six admitted runs return dynamic timeout with zero reports and premises, followed by `race@enum`. No new race claim or witness validation is inferred from these deadline diagnostics.

| Execution | Budget | Returned dynamic time | Full dynamic wrapper | Whole worker |
|---|---:|---:|---:|---:|
| In process, 66d0dba | 20 s | 20.403503 s | 21.918939 s | 49.610181 s |
| In process, 66d0dba | 60 s | 61.450893 s | 62.989390 s | 90.875501 s |
| Subprocess, b10b8f8 | 20 s | 20.201208 s | 25.906802 s | 53.637737 s |
| Subprocess, b10b8f8, repeat 0 | 60 s | 60.385488 s | 66.076685 s | 93.923392 s |
| Subprocess, b10b8f8, repeat 1 | 60 s | 60.398794 s | 66.162082 s | 93.722685 s |
| Subprocess, b10b8f8, repeat 2 | 60 s | 60.404858 s | 66.105165 s | 94.096033 s |

The first two rows use the old internal function clock; the last four use READY/GO through child reap. They are explicitly different protocols. All four subprocess observations satisfy the unchanged half-second return criterion, including all three 60 s repeats. Every child is killed and reaped before fallback enumeration. The full wrapper and worker columns retain the extra process/setup cost, so this table is not evidence of an overall speedup.

The successful `smoke_add_no` L2 control returns `proved@T0` with dynamic `ok`, a complete child profile containing nine solver checks, nonzero exclusive phase costs and no open spans. Twenty-two focused tests cover native interruption, finalizers, trace restoration, exact tensor transport, cache-free JIT reconstruction, a result published before late process exit, forced termination during native-owner cleanup, and plain transport errors failing the row closed. The relevant command is:

```sh
PYTHONPATH=/tmp/triton-viz-fix-dynamic-deadline:/tmp/tilerace-deadline-deps \
  .venv/bin/python -m pytest -q tests/unit/test_dynamic_subprocess.py \
  tests/unit/test_dynamic_watchdog.py tests/unit/test_diagnostic_export.py
```

All repository commit checks, including Ruff, formatting and mypy, passed for the candidate.

## Affected rerun scope

The detector's memory model and solver conclusions are unchanged by isolation, but startup, cleanup, process memory and dynamic timing boundaries change for every Triton row. Remaining-budget allocation to enumeration includes those real costs and can change which launch analyses finish. A new formal pin must therefore rerun the affected full-corpus and selected-study measurements, with successor ablation/budget/phase adapters explicitly installing their child observers. Archived study scripts and old receipts remain immutable.

Under the user policy recorded on 2026-09-07, routine full-corpus reruns run L1 and then L2; there is no new full L0 pass. Selected L0 controls and regressions remain where their experiments require them. Historical three-level data remain historical and must not be mixed with new subprocess timing as a paired comparison. This four-observation diagnostic validates the repaired return behavior; it does not replace a full formal rerun or recalibrate corpus-wide duration estimates.
