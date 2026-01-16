Context:
Fix bug: When I run `examples/matmul_demo.py`, on the first load record, the matrix is split into four quadrants top-left (0, 0), top-right (0, 1), bottom-left (1, 0), and bottom-right (1, 1). For this record, program (x, y) loads quadrant (x, 0).
Expected: Since program (x, y) loads quadrant (x, 0) for record 0 so only quadrants (0, 0) and (1, 0) should be highlighted with rainbow colors (other quadrants remain gray) for this record.
Reality: When I press all program IDs on, all quadrants are highlighted with rainbow colors.

Review-Round: 0
Owner:
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/load_utils.js
Acceptance:
- For record 0 in `examples/matmul_demo.py`, all-programs mode highlights only quadrants (0,0) and (1,0).
- Quadrants (0,1) and (1,1) remain gray in all-programs mode for that record.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan: TBD (low-level plan)
