# DSLs Architecture

This folder defines how each DSL backend (Triton, NKI, etc.) is patched and normalized so clients can work against a consistent API.

## File map (start here)
- `triton_viz/dsls/base.py`: `Frontend` descriptor, adapters, and `OPERATION_REGISTRY`.
- `triton_viz/dsls/triton.py`: Triton op namespaces and adapters.
- `triton_viz/dsls/nki.py`: NKI op namespaces and adapters (optional dependency).

## Terms defined here
- **frontend (python)**: a registry entry that describes patchable ops for a backend.
- **namespace**: a module/class where ops live (e.g. `triton.language`, `interpreter_builder`).
- **adapter**: a function that normalizes backend-specific op signatures into a common shape.
- **operation registry**: `OPERATION_REGISTRY`, mapping backend name -> `Frontend`.
- **builder**: the interpreter builder used when executing a backend.

## Subsystems used
- **Core patching**: `core/patch.py` reads `OPERATION_REGISTRY` for op patching.
- **Data model**: op types are `Op` subclasses in `core/data.py`.

## Main logic flows
- **Registry build**:
  - Each backend declares namespaces and op mappings.
  - `Frontend.from_namespaces()` captures original ops and fills missing adapters.
  - The resulting `Frontend` is stored in `OPERATION_REGISTRY`.
- **Execution time**:
  - `patch_op()` wraps each op and runs the adapter before calling client callbacks.
  - Clients receive consistent argument shapes even when backends differ.

## Extension crash-course: add a new op or backend
1. **New op in an existing backend**:
   - Add an `Op` subclass in `core/data.py`.
   - Add the op to the backend namespaces map (e.g. `TRITON_NAMESPACES`).
   - Add or reuse an adapter in `<backend>.py` if the signature needs normalization.
   - Update clients or visualizer rendering if needed.
2. **New backend**:
   - Create `triton_viz/dsls/<backend>.py` with namespaces, adapters, and a `Frontend`.
   - Register it in `OPERATION_REGISTRY`.
   - Add an import in `triton_viz/dsls/__init__.py` to populate the registry.
   - Implement a trace wrapper similar to `TritonTrace` or `NKITrace`.

## Required vs optional patterns
- **Required**: every op in `namespaces` must map to an `Op` type.
- **Required**: every op type must have an adapter; `passthrough_adapter` is acceptable.
- **Optional**: omit a backend entirely (e.g. NKI) if optional deps are missing.

## Gotchas and invariants
- `OPERATION_REGISTRY` must be populated at import time; missing imports mean no patching.
- If NKI is not installed, `NKI_NAMESPACES` stays empty; do not rely on it by default.
- Adapter output order defines what clients see; keep it stable.
