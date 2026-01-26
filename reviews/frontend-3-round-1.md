# Review: frontend-3 (round 1)

BLOCKING
- `npm run build:frontend` fails with TypeScript errors, so the frontend does not compile from TS as required by acceptance. Fixes are needed before review can proceed.
  - `src/api.ts:31-52` uses `unknown` for `body`, which is not assignable to `BodyInit` in `fetch`. Type `RequestOptions.body` as `BodyInit | null` (or cast `nextBody` to `BodyInit | null`) so `requestJson` compiles.
  - `src/dimension_utils.ts:4-50` uses `options = {}` leading to property access on `{}`. Define typed option interfaces for `createVectorText` and `createCadDimension` (or type `options` as `Partial<...>`) so destructuring compiles.
  - `src/histogram.ts:85-90` assigns numbers to `HTMLInputElement.min/max/step` which are `string` in DOM types. Assign string values (e.g., `'4'`) or cast.
  - `src/histogram.ts:126-132` calls `createDisposer().listen` with 3 args, but the signature expects 4. Make the options arg optional in `src/utils/dispose.ts` or pass `undefined` explicitly.

NIT
- Consider tightening types for `OpRecord` fields used in `nki.ts` and `tensor_view.ts` once compile errors are fixed.

Tests
- `npm run build:frontend` (fails with errors above)
- `npm run test:frontend` (passes)

REQUEST-CHANGES
