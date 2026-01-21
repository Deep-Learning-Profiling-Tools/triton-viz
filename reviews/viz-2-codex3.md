# Review: viz-2

Status: REQUEST-CHANGES

## Findings
- BLOCKING: Outside-arrow placement does not match the spec. In `triton_viz/static/dimension_utils.js:58-84`, the flipped case keeps arrowhead tips at `d1`/`d2` (extension-line intersections) and only extends the dimension line outward. The requirement says flipped arrows should be outside and point inward toward the extension lines. Move the arrowhead tips to the outside points (e.g., `arrow1_outer`/`arrow2_outer`) and point them inward, leaving the dimension line between the extension lines.
