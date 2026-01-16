Context:
Replace the existing dimension markers with AutoCAD-style dimension lines (bars at the ends, arrows pointing to the bars, dimension size in the middle.
Make the dimension lines/numbers different colors and display each visualize tensors' shape (with color coding) somewhere on the screen.

Review-Round: 0
Owner:
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/store.js, triton_viz/static/matmul.js, triton_viz/static/visualizer.css
Acceptance:
- Dimension markers are replaced with AutoCAD-style dimension lines in tensor views.
- Dimension lines/numbers use distinct colors and tensor shapes are displayed with matching color coding.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan: TBD (low-level plan)
