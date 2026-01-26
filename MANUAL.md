# Triton Viz User Manual

Refer to `GLOSSARY.md` for canonical UI terms and payload names.

## Navigation and Controls
- Mouse: left-drag to orbit, right-drag to pan, scroll to zoom.
- Keyboard:
  - W/A/S/D: pan camera.
  - Arrow keys: tilt/rotate camera.
  - O/P: zoom out/in.

## Program ID Controls
- Use the X/Y/Z sliders to set the Active Program.
- The Reset button returns all axes to their default values.
- Changing the Active Program refreshes Op Tabs and all views.

## Operation Controls
- Color by Value: enables heatmap coloring in the Tensor View.
- All Program IDs: shows per-program sampling across all program IDs.
- Value Histogram: opens the histogram overlay for the active tensor.

## Active Program Workspace
- Op Tabs switch between operations for the Active Program.
- Tensor View shows 3D tensor meshes with CAD-style dimension lines.
- Flow View appears for ops that provide a flow diagram.
- The Side Info Panel shows hover details (coordinates, shapes, values).

## Code Peek Panel
- The Code Peek Panel displays source context for the active op.
- Use the op controls to toggle or refresh code visibility.

## Legends and Overlays
- Shape Legend: lists visible tensors with their shapes and colors.
- Value Legend: shows the colormap range when Color by Value is enabled.
- Histogram Overlay: lets you pick a tensor source and bin count.

## Dev Overlay
- Enable the Dev Overlay with `?dev=1` in the URL or press Ctrl+Shift+D.
- Badges display the `data-component` values for major UI roots.
- The overlay is off by default and does not affect layout or clicks.

## Frontend Build
If you modify frontend TypeScript sources in `src/`, rebuild the static assets:

```sh
npm install
npm run build:frontend
```
