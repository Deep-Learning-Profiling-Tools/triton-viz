# Architecture

## Front-end Annotation Layer
The front-end utilizes Three.js for 3D tensor visualization and a custom annotation layer for dimension lines and legends.

### CAD Dimension Overlays
Dimension lines are rendered as CAD-style annotations:
- **Extension Lines**: Perpendicular lines from tensor boundaries.
- **Dimension Lines**: Parallel lines showing the span.
- **Arrowheads**: Located at intersections, with automatic inside/outside flipping logic.
- **Logic**: Implemented in `triton_viz/static/dimension_utils.js`.

### Shape Legend
A floating DOM overlay that lists all active tensors and their shapes, color-coded to match the 3D meshes.
