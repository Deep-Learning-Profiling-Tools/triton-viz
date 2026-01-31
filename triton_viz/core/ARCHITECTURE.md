# Core Architecture

This folder owns the runtime plumbing that makes Triton-Viz work: kernel tracing, patching, shared data models, and configuration.

## File map (start here)
- `triton_viz/core/trace.py`: the entry point wrapper (`trace`) and tracing runners (`TritonTrace`, `NKITrace`).
- `triton_viz/core/client.py`: client interfaces plus `ClientManager` orchestration.
- `triton_viz/core/patch.py`: op/loop/lang patching and the interpreter execution loop.
- `triton_viz/core/data.py`: the data model for records produced by clients.
- `triton_viz/core/config.py`: runtime configuration flags.

## Terms defined here
- **launch**: a single traced kernel invocation; stored as a `Launch` record and appended to the global `launches` list.
- **record**: any `Op`-derived object emitted by clients during a launch (e.g. `Load`, `Store`, `Dot`).
- **client**: a data-collection module registered with `ClientManager` that receives callbacks during execution.
- **patch scope**: the period where ops/loops/lang are monkey-patched to route through callbacks.
- **backend**: a DSL execution target (`triton` or `nki`) used to select patch/adapter registries.
- **grid index**: the (x, y, z) program coordinate currently executing inside the interpreter.

## Subsystems used
- **Triton interpreter**: `GridExecutor` + `interpreter_builder` are patched to intercept operations.
- **DSL registry**: `triton_viz.dsls.OPERATION_REGISTRY` defines per-backend namespaces and adapters.
- **Threading**: optional multi-SM emulation uses a thread pool; locks guard shared client state.

## Main logic flows
- **Trace wrapper setup**:
  - `trace()` returns a `TritonTrace` or `NKITrace` wrapper.
  - For Triton, the wrapper keeps both the JIT function and an interpreted function.
  - Warmup is run with the JIT function to collect ASM if needed.
- **Run flow**:
  - `ClientManager.patch_run()` patches ops, loops, and language helpers.
  - Interpreter executes a grid; each program index triggers `grid_idx_callback`.
  - Each patched op calls client callbacks via standardized adapters.
  - `ClientManager.finalize()` aggregates records and attaches them to the `Launch`.
- **Patching flow**:
  - `patch_calls()` replaces `GridExecutor.__call__` and `JITFunction.__call__` for the duration.
  - `patch_op()` wraps each op with `PatchOp`, invoking adapter -> callbacks -> original.
  - `patch_for_loop()` instruments Python `for` loops in interpreter AST.

## Extension crash-course: add a new core record
1. Add a new `Op` subclass to `triton_viz/core/data.py` for the data you want to capture.
2. Update the DSL registry (see `triton_viz/dsls/`) to map a backend op to the new `Op` type.
3. Update clients (tracer/profiler/sanitizer) to emit the new record if needed.
4. If the visualizer should render it, add support under `triton_viz/visualizer/` and `frontend/ops/`.

## Required vs optional patterns
- **Required**: every op in `OPERATION_REGISTRY` must map to an `Op` type.
- **Optional**: clients can ignore op types they do not care about; `OpCallbacks()` is a no-op.

## Gotchas and invariants
- `patch_calls()` only patches at the outermost scope; nested calls do not re-patch.
- When `cfg.num_sms > 1`, client callbacks must be thread-safe or wrapped with `lock_fn()`.
- `TritonTrace.__call__` bypasses `run()` when invoked inside another kernel.
- `NKITrace` uses its own interpreter builder; it does not share Triton JIT behavior.
