# Review: viz-3 (round 1)

BLOCKING:
- `triton_viz/static/load_utils.js`: `createVectorText` runs `sync()` before `strokeWidth`/`strokeColor` are set in `addLabel`. Troika text only applies style changes after `sync()`, so the stroke is currently ignored. Set stroke props before the initial `sync()` or call `vectorText.sync()` after setting them.
- `tasks/code-backlog/viz-3_vector-text-rendering.md`: task metadata is out of date (Review-Round is 0 and the plan differs from the review-backlog plan). Update the task file to match `tasks/review-backlog/viz-3_vector-text-rendering.md` and keep it in the correct backlog state before resubmitting.

NIT:
- `triton_viz/static/dimension_utils.js`: consider adding `depthWrite = false` when `depthTest` is disabled to preserve the previous overlay behavior for labels.

REQUEST-CHANGES
