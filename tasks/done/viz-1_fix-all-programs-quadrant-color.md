Context:
Fix bug: When I run `examples/matmul_demo.py`, on the first load record, the matrix is split into four quadrants top-left (0, 0), top-right (0, 1), bottom-left (1, 0), and bottom-right (1, 1). For this record, program (x, y) loads quadrant (x, 0).
Expected: Since program (x, y) loads quadrant (x, 0) for record 0 so only quadrants (0, 0) and (1, 0) should be highlighted with rainbow colors (other quadrants remain gray) for this record.
Reality: When I press all program IDs on, all quadrants are highlighted with rainbow colors.

Review-Round: 1
Owner: ai
Deps: none
Touches: triton_viz/static/load.js, triton_viz/static/load_utils.js
Acceptance:
- For record 0 in `examples/matmul_demo.py`, all-programs mode highlights only quadrants (0,0) and (1,0).
- Quadrants (0,1) and (1,1) remain gray in all-programs mode for that record.
Notes:
- base branch is thaihoa/viz-fixes per user
Plan:
- **Plan: Overview**
  - Fix the "All Program IDs" mode in Load/Store visualizations so it only highlights tiles for the currently viewed record index across all programs, instead of aggregating all records.
- **Plan: Mental model**
  - The "All Program IDs" mode currently fetches all tiles for a given pointer (pointer-key) from the backend. The backend aggregates all Load/Store operations on that pointer across all time steps and all programs. We need to introduce a `time_idx` to identify specific operations and filter by it when requested by the frontend.
- **Plan: Interfaces / contracts**
  - Update `/api/load_overall` and `/api/store_overall` to accept an optional `time_idx` (int) in the JSON body.
  - Update the `tiles` returned by these APIs to include `time_idx`.
- **Plan: Files / functions / data structures**
  - `triton_viz/visualizer/draw.py`: `prepare_visualization_data` will track an incrementing `op_idx` for operations added to `visualization_data` and assign it to `time_idx`.
  - `triton_viz/visualizer/interface.py`: `get_load_overall` and `get_store_overall` will implement the filtering logic.
  - `triton_viz/static/load.js` & `triton_viz/static/store.js`: `fetchAllProgramTiles` will pass the current operation's `time_idx`.
- **Plan: Implementation steps**
  - Update `draw.py` to assign consistent `time_idx` to operations and their corresponding "overall" tiles.
  - Update `interface.py` to support filtering by `time_idx` in `load_overall` and `store_overall` endpoints.
  - Update frontend JS files (`load.js`, `store.js`) to pass `time_idx` when fetching tiles for the "All Program IDs" mode.
- **Plan: Logging / observability**
  - No new logging required, but ensuring `time_idx` is present in API payloads will help debugging.
- **Plan: Documentation**
  - No documentation updates needed as this is a bug fix in internal visualization logic.
- **Plan: Tests**
  - Since this is a UI bug, manual verification with `examples/matmu.py` is the primary way to test.
- **Plan: Risk analysis**
  - This change only affects the "All Program IDs" mode and should not impact other parts of the visualization.
  - Ensure that the "Overall" view (which aggregates everything) still works by making `time_idx` optional in the API.
