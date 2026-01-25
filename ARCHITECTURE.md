# Architecture

## Front-end Annotation Layer
The front-end utilizes Three.js for 3D tensor visualization and a custom annotation layer for dimension lines and legends.

### CAD Dimension Overlays
Dimension lines are rendered as CAD-style annotations:
- **Extension Lines**: Perpendicular lines from tensor boundaries.
- **Dimension Lines**: Parallel lines showing the span.
- **Arrowheads**: Located at intersections, with automatic inside/outside placement logic.
- **Text Rendering**: High-quality vector text rendering using Multi-channel Signed Distance Fields (MSDF) via `Troika-Three-Text`, ensuring readability at all zoom levels without pixelation.
- **Logic**: Implemented in `triton_viz/static/dimension_utils.js`.

### Shape Legend
A floating DOM overlay that lists all active tensors and their shapes, color-coded to match the 3D meshes.

### Active Program Workspace
Program ID sliders select the active program (X/Y/Z). The workspace renders op tabs, Tensor/Flow views, and code peek for that active program.
Logic lives in `triton_viz/static/op_workspace.js`, with wiring in `triton_viz/static/visualization.js`.
