# Triton Viz User Manual

## Navigation and Controls
- **Orbit Controls**: Use mouse left-click to rotate, right-click to pan, and scroll to zoom.
- **Keyboard Shortcuts**:
  - `W/A/S/D`: Pan camera.
  - `Arrow Keys`: Tilt/Rotate camera.
  - `O/P`: Zoom out/in.

## Visual Annotations

### CAD-style Dimension Lines
Tensors in the visualization are annotated with CAD-style dimension lines to indicate their shapes.
- **Extension Lines**: Perpendicular lines at the start and end of each dimension.
- **Dimension Lines**: Parallel lines showing the span of the dimension.
- **Arrowheads**: Located at the intersection of dimension and extension lines.
- **Auto-flipping Arrows**: If a dimension is too small to fit the arrows inside, they will automatically flip to the outside, pointing inward.

### Color Coding
Dimension lines and numbers are color-coded to match the tensor they belong to.
- **Global Tensor**: Usually rendered with its designated theme color.
- **Slice Tensor**: Rendered with a distinct color (e.g., Cyan or Magenta) to highlight the loaded/stored region.

### Shape Legend
A floating legend is displayed at the bottom-left of the screen, showing the name, shape, and color of each tensor currently being visualized. This legend updates automatically when window is resized or when switching between different operations.

## Overlays and Panels
- **Value Legend**: Shows the colormap and value range when colorization is enabled.
- **Side Menu**: Displays detailed information about the hovered element (coordinates, dimensions, value).
- **Code Panel**: Shows the source code context for the current operation.
