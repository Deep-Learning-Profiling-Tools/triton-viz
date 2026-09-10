# Session-owned dynamic preloading

The resumable evaluation driver supports `--dynamic-launcher preload`. It starts one clean CPU preloader per admitted serial session and forks a fresh dynamic analysis child for each launch. Every configuration still gets a fresh row worker. P17 avoids redundant input-identity copies; private mmap is not enabled.

This is checkout evaluation tooling, not a separately installed wheel entry point. The ordinary launcher remains the default (`--dynamic-launcher spawn`), and the debugging reused-row worker does not support a preloader socket. The chosen launcher and `dynamic-preload-normal-exit-v1` protocol are recorded in the frozen configuration and every result header. Existing spawn runs retain their original behavior.

## Prepare without executing

After committing the clean execution source and restoring its hash-checked corpus sidecars, `evaluation.pinned_run start --prepare-only --dynamic-launcher preload --ladder-level L2 --run-dir /absolute/artifact/results/new-run/L2` creates the manifest and durable ledger, then returns without launching a worker, broker, service or solver. `--prepare-only` and foreground execution are mutually exclusive. Preparation imports corpus declarations and verifies source/dependency/sidecar identities; it is not a numerical preflight. The separate `resume --run-dir ...` command starts the prepared run and requires an explicit execution decision.

Preparation fixes `TRITON_INTERPRET=0`, chooses dedicated Triton and Inductor cache paths beside the run directory, and records `FLAGGEMS_SOURCE_DIR` before fingerprinting. It locates FlagGems through installed distribution metadata, without importing that package to discover the default. Restoring a run uses the frozen environment. Every environment entry except the explicitly passed `PYTHONPATH` must match what the preloader inherited; source checks still validate actual module origins. No environment drift is silently ignored.

## Lifetime and acceptance

The controller owns a broker only while it holds the original host admission. The broker starts in a separate Python process, imports common CPU libraries and the harness, and verifies a single thread, no initialized CUDA state, no Z3 default context and no children before every fork. It does not preload corpora, detector instances, inputs or active solver objects. Each analysis child follows normal Python exit and the original READY/GO, input/kernel/source/config checks and watchdog.

The original row process group, cancellation rules and 200/320 second outer limits remain in effect. Dynamic analysis keeps its original 60 second budget; enumeration keeps the original remaining-budget policy. A timeout or operator cancellation first kills and waits for the row, then the independent broker closes its owner lease and reaps its analysis children. The controller verifies PID/start-time identities, genuine wait status, complete launch/reap files and the row's declared launch list before accepting a result. A result file alone is insufficient.

Within the service domain, only the exact live broker owned by this controller may remain between rows. Final publication and admission release require broker closure and zero remaining children. Graceful pause/resume creates a new broker per session. A hard-killed session without verifiable closure and cost receipts cannot silently produce a fully accounted publication. Failed or interrupted sessions with complete receipts retain their status and charged time; they are not rewritten as successful sessions.

The broker's three source files and the package initializer are included in dynamic source identity. Launch and reap records are atomically replaced and synced with their containing directory. Publication verifies their original hashes through each already-committed row's audit before binding them into the final receipt. Fault injection is disabled by default and requires an explicit diagnostic broker flag.

## Timing and validation scope

Native per-row `wall_s` keeps its existing boundary. Post-row broker verification, waiting for owner-death cleanup and durable audit writes are charged to measured session costs. Broker costs separately show startup, row intervals, controller gaps and shutdown; the enclosing controller interval also covers final broker receipt persistence. The observer's own checkpoint write and ordinary publication bookkeeping are outside that observer clock. Do not add a full session wall to its already-included row sum or treat preloading as free.

The prior artifact prototype had 18 lifecycle fault controls and targeted timing comparisons, including 66 observations of gated_delta_rule, RWKV7 and gdn2. This integration changes package paths, durability, ownership and publication binding; those prototype observations are not measurements of this integrated commit. A few seconds of isolated tail slowdown is accepted by the user, with final full-corpus median/P95 still to be assessed.

The integration's mocked unit tests prohibit real process, socket and kernel launch. They cover launcher isolation, environment/source checks, wait receipts, row identity/declaration binding, controller cleanup ordering, publication tamper rejection and prepare-only behavior. Real integrated lifecycle controls and any full L2 execution remain pending the user's explicit start order. No numerical result is adopted by committing this integration.
