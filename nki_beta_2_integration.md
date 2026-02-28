# NKI Beta 2 integration plan (updated for ml_dtypes quantization)

## goal
Integrate `../nki-namespace` into `main` as small, reviewable PRs with minimal cross-PR conflicts.

## prototype baseline
Source branch: `../nki-namespace` (`nki-namespace`)

Commits on top of `main`:
1. `b149002` initial nki beta 2 interpreter
2. `aaf6e9b` quantize
3. `dacf01d` add almost all the ops
4. `590d058` fix type errors
5. `9873ccd` fix ty errors
6. `0692573` debloat
7. `b218865` tests
8. `9eaf4aa` examples
9. `8ecd095` use ml_dtypes quant

## staged PR split

### stage 1: dtypes foundation (current PR)
Scope:
- Add `triton_viz/utils/dtypes.py` with `STORAGE_DTYPES` backed by `ml_dtypes` for low-precision formats.
- Add unit tests that validate alias resolution and cast behavior for low-precision formats.

Files:
- `triton_viz/utils/dtypes.py`
- `tests/unit/test_dtypes.py`

Validation:
- `pytest tests/unit/test_dtypes.py`

---

### stage 2: beta2 interpreter scaffold (side-by-side)
Scope:
- Add `triton_viz/core/nki_beta2.py` without routing trace/patch to it yet.
- Add minimal beta2 tests/examples that exercise direct behavior.

---

### stage 3: runtime wiring switch
Scope:
- Route NKI tracing/patching to beta2 interpreter.
- Include small compatibility/typing adjustments (`trace.py`, `patch.py`, `client.py`).

---

### stage 4: op coverage expansion
Scope:
- Land broad `nl`/`nisa` operation support in `nki_beta2.py`.

---

### stage 5: hardening cleanup
Scope:
- Apply type-fix and debloat commits (`590d058`, `9873ccd`, `0692573`) as cleanup-only PRs.

---

### stage 6: test expansion
Scope:
- Land large beta2 test coverage and nkilib e2e tests (`b218865` + quantization-related updates from `8ecd095`).

---

### stage 7: beta2 examples
Scope:
- Land beta2 examples (`rmsnorm_beta2.py`, `rope_beta2.py`, `softmax_beta2.py`, `tiled_attention_beta2.py`, `nki2.py`).

## notes
- Keep each PR behavior-focused; avoid mixing routing, feature growth, and cleanup in one diff.
- Treat commit `8ecd095` as a logical replacement of the original quantization approach, not an additive step.
