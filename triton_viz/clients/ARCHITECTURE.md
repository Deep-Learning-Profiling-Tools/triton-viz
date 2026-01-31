# Clients Architecture

Clients are data-collection modules that attach to the core tracing pipeline. Each client decides which callbacks to implement and what records to emit.

## File map (start here)
- `triton_viz/clients/tracer/tracer.py`: collects op-level trace records for visualization.
- `triton_viz/clients/profiler/profiler.py`: collects performance and mask stats.
- `triton_viz/clients/sanitizer/sanitizer.py`: performs symbolic/bounds checks using Z3.
- `triton_viz/clients/utils.py`: shared helpers for stack inspection and analysis.

## Terms defined here
- **client**: a `Client` subclass with callback hooks invoked by `ClientManager`.
- **op**: A tensor operation (e.g. load/store/matmul). Ops are DSL-agnostic.
- **record**: a concrete `Op` instance or analysis record emitted by a client.
- **callback**: a hook like `register_op_callback()`, `pre_run_callback()`, or `grid_idx_callback()`.
- **op overrider**: an optional callback that replaces an operation (e.g. skip loads in profiler).
- **block sampling**: selecting a subset of grid indices for profiling.

## Subsystems used
- **Core patching**: callbacks are invoked by `PatchOp` in `core/patch.py`.
- **Op registry**: determines which op types can be hooked for each backend.
- **Thread-local state**: clients store per-thread data (grid index, sampling state).

## Main logic flows
- **Registration flow**:
  - `ClientManager.add_clients()` stores unique clients.
  - `register_op_callback()` allows the user to add extra code to run before/after an op is ran. Registration is called once per op type when patching.
  - `register_for_loop_callback()` allows adding extra code to run before/after/during for loop iterations. Registration is called once when patching.
- **Execution flow**:
  - `grid_callback()` fires once per launch to set up sampling.
  - `grid_idx_callback()` fires per program index.
  - `OpCallbacks.before_callback` runs before op execution; `after_callback` after.
  - `finalize()` returns the records for this launch.

## Extension crash-course: add a new client
1. Create `triton_viz/clients/<name>/` and implement a `Client` subclass.
2. Implement required methods: `pre_run_callback`, `post_run_callback`, `arg_callback`,
   `grid_callback`, `grid_idx_callback`, `register_op_callback`, `register_for_loop_callback`, `finalize`.
3. Use `lock_fn()` around any shared data mutation when `cfg.num_sms > 1`.
4. Export the client in `triton_viz/clients/__init__.py`.
5. If you want string-based selection (e.g. `trace(client='foo')`), add a case in
   `triton_viz/core/trace.py` `_normalize_client()`.

## Required vs optional patterns
- **Required**: `finalize()` must return a list (possibly empty) and should be idempotent.
- **Optional**: `op_overrider` can bypass expensive ops if you only need metadata.
- **Optional**: `pre_warmup_callback`/`post_warmup_callback` when ASM inspection is required.

## Gotchas and invariants
- `Tracer` uses a sorted tensor list to map raw pointers to tensors; keep that invariant if you copy the pattern.
- `Profiler` may skip ops to reduce overhead; do not assume real data is available.
- `Sanitizer` swaps implementations based on config, so construction can return a different class.
