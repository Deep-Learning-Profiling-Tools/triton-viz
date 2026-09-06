# Durable pinned reruns

`evaluation.pinned_run` saves every completed main and retry attempt in a
SQLite ledger before starting another attempt. The ledger uses DELETE journal
mode with EXTRA synchronization on persistent local storage. A run is one
immutable manifest across multiple driver sessions; JSONL files are exports,
not restart inputs. No completed result is selected again for a better verdict
or timing.

```sh
python -m evaluation.pinned_run start --ladder-level L2
python -m evaluation.pinned_run status --run-dir /absolute/path/to/run
python -m evaluation.pinned_run pause --run-dir /absolute/path/to/run
python -m evaluation.pinned_run pause --run-dir /absolute/path/to/run --now
python -m evaluation.pinned_run resume --run-dir /absolute/path/to/run
python -m evaluation.pinned_run verify --run-dir /absolute/path/to/run
```

The default pause lets the reserved current attempt finish and commits it.
Immediate pause cancels an unfinished attempt; resume repeats only that empty
slot. SIGSTOP is not a supported pause operation. Unexpected process loss can
repeat the current uncommitted attempt even when its solver had just finished.
Keep the entire run directory and its SQLite rollback journal, if present.
Do not place live checkpoints on a network filesystem or copy an active
database as a backup. The disk's durability guarantees still apply.

Formal execution uses an owned Linux user-service domain, with automatic
restart disabled. It must run on a host with an available user service manager.
The service log is `service.log` inside the run directory. A saved pause request
is not an acknowledgment: `status` must show no active domain before the machine
is considered released. An intentionally paused run resumes only on request.
The host lock and durable admitted-domain registry live in
`~/.local/state/triton-viz/pinned`; surviving domains are checked before another
driver can execute. `TRITON_VIZ_PINNED_STATE_DIR` overrides that location for
isolated rehearsals/tests only and is rejected for formal runs.

The manifest freezes the ordered corpus/spec roster, full execution commit,
source and installed package hashes, value sidecars, runtime/environment,
ladder level, budgets and counting rules. Keep the execution checkout,
environment and inputs unchanged until completion. Resume refuses a mismatch.
The definitive run is L2, all 1242 rows from 17 corpora, seed 0, jobs 1, main
budget 200 s and retry budget 320 s, fence order enabled with no environment
override. L1/L0 attribution requires `--purpose attribution` and its own run;
their main budgets are 200/180 s. Rehearsal can select smaller corpora and budgets:

```sh
python -m evaluation.pinned_run start --rehearsal --foreground \
  --corpora golden_smoke --ladder-level L2 --row-timeout 1 --retry-timeout 120
```

`--foreground` and `--no-load-guard` are rehearsal-only. Direct Python
`run_pinned(..., rehearsal=True)` returns a completed rehearsal dataset path;
a formal call returns the directory of its asynchronously dispatched service.
Ordinary `evaluation.runner.run_corpus` keeps its existing interface. Durable
main/retry scheduling is the responsibility of `pinned_run`.

Main and retry slots are separate. After every main row is saved, each row
whose terminal is timeout or full recorded wall reaches the main budget gets
one logical retry. An interrupted retry does not consume that slot. A completed
retry is never repeated. Existing merge rules are retained, including an
abstention replacing a timed-out main result. Overhead statistics use main raw
walls; verdict statistics use selected-attempt `pinned_wall_s`.

Each attempt still creates a fresh subprocess. Its wall timer starts before
temporary-output/process setup; the subprocess wait budget starts after process
creation. Checkpoint start/commit work is outside the row timer. This removes
direct accounting of checkpoint writes from the row wall, but does not prove
that cache/load effects cannot change later walls or budget-boundary verdicts.

Completed artifacts live in the run's `exports/` directory. `COMPLETE.json`
binds their hashes to the manifest after exact main/retry completeness checks.
The report and comparison loaders reject v1 files lacking a valid receipt.
There is no overwrite of historical top-level PINNED or per-corpus files, and
no automatic import of legacy aborted runs. Keep the run directory when moving
exports so their receipt remains verifiable. Repeated resume after COMPLETE
checks the ledger and files, reconciles final bookkeeping and executes no row.

## Checkpoint overhead measurement

`evaluation.checkpoint_overhead` writes labeled rehearsal evidence. Micro mode
replays real saved payloads through both durable transactions; paired mode
alternates no-ledger and ledger execution with the same fresh-process executor.
The tool saves the row selection, block order, source hashes, runtime, raw
samples and summary. See its module help for invocation. Paired runs wait for
the existing foreign-process/load guard before each block. Their common
diagnostic JSONL is saved identically in both variants; the comparison isolates
the incremental experiment ledger, not all possible disk activity.

Tests inject real process kills around SQLite transactions, row completion and
pause races; exercise main/retry resume and atomic export; and check identity,
duplicate ownership and corruption rejection. A successful rehearsal does not
re-pin or start the definitive paper experiment. The measured execution commit
and a new formal freeze must be recorded separately.
