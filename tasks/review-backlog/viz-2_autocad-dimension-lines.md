Context:
Replace the existing dimension markers with CAD-style dimension lines:
    - Draw Extension Lines perpendicular to the dimension axis at both terminal points. Optional: Add a 2-unit offset so they don't touch the object.
    - Draw a Dimension Line parallel to the object.
    - Place Arrowheads at the intersection of the dimension and extension lines.
    - Conditional Logic: If the distance between extension lines is less than [X] units, flip the arrows to the 'Outside' (pointing inward toward the extension lines). Otherwise, keep them 'Inside' (pointing outward)."
Make the dimension lines/numbers different colors and display each visualize tensors' shape (with color coding) somewhere on the screen.

Review-Round: 1
Owner:
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/store.js, triton_viz/static/matmul.js, triton_viz/static/visualizer.css
Acceptance:
- Dimension markers are replaced with AutoCAD-style dimension lines in tensor views.
- Dimension markers contain all components: extension lines, parallel dimension lines, arrowheads, conditional arrow flipping
- Dimension lines/numbers use distinct colors and tensor shapes are displayed with matching color coding.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan: TBD (low-level plan)
