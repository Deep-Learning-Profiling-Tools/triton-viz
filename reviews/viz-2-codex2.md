# Review: viz-2

Status: REQUEST-CHANGES

## Findings
- BLOCKING: `triton_viz/static/dimension_utils.js:16-75` uses `length`/`isFlipped` before initialization and has an extra closing brace, which makes the module throw/parse-fail. Define `length`/`isFlipped` before any logging and remove the stray `}`.
- BLOCKING: `triton_viz/static/dimension_utils.js:54-74` never defines `isFlipped` or uses `flipThreshold`, so the required inside/outside arrow flip logic is not implemented. Compute `isFlipped` from the measured distance and align arrow direction with the spec.
- BLOCKING: `triton_viz/static/dimension_utils.js:93-96` logs `axis`/`label`/`length`/`isFlipped` inside `createArrowhead`, which are out of scope and will throw at runtime. Remove the log or pass the values in.
