# Review: viz-2

Status: REQUEST-CHANGES

## Findings
- BLOCKING: Arrow flip behavior is reversed vs the requirement. `triton_viz/static/dimension_utils.js:54-74` uses inward-pointing arrows for the default case and outward-pointing arrows for the flipped case. The requirement says default should be inside pointing outward, and flipped should be outside pointing inward. Swap arrow directions/placement (and update comments) so inside/outside behavior matches the spec.
- BLOCKING: Dimension lines/numbers are not color-distinct per tensor. In `triton_viz/static/load_utils.js:363-379`, both tensors share the same background-derived color, and `triton_viz/static/matmul.js:384-395` hardcodes white. The acceptance criteria call for dimension lines/numbers with distinct colors and matching shape legend color coding. Use the tensor color mapping for each axis label/dimension line.
- BLOCKING: Plan items remain unimplemented. The task plan in `tasks/review-backlog/viz-2_autocad-dimension-lines.md` includes tests, MANUAL.md updates, logging/observability, and CSS variable usage, but none appear in the diff. Either implement the planned items or update the plan to reflect the actual scope.

## Questions
- QUESTION: Should the legend be re-rendered on resize as mentioned in the plan? Currently `createShapeLegend` is only called during view initialization.
