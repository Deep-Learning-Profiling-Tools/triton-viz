export const COLORMAPS = {
    Load: [[0.0, 1.0, 0.95], [0.0, 0.2, 1.0]],
    Store: [[1.0, 1.0, 0.0], [1.0, 0.25, 0.0]],
    A: [[0.0, 1.0, 0.95], [0.0, 0.2, 1.0]],
    B: [[1.0, 1.0, 0.0], [1.0, 0.25, 0.0]],
    C: [[0.2, 1.0, 0.2], [1.0, 0.0, 0.9]],
};

export function clamp01(value) { return Math.min(1, Math.max(0, value)); }
export function lerp(a, b, t) { return a + (b - a) * t; }
export function getColormap(label) { return COLORMAPS[label] || COLORMAPS.Load; }
