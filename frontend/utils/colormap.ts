/** Default hue assignments for known labels. */
export const HUES: Record<string, number> = {
    Load: 200,
    Store: 45,
    A: 200,
    B: 45,
    C: 140,
};
const DEFAULT_HUE = 200;

/**
 * Clamp a value to [0, 1].
 * @param value - Input value.
 * @returns The clamped value.
 */
export function clamp01(value: number): number { return Math.min(1, Math.max(0, value)); }
/**
 * Linearly interpolate between two values.
 * @param a - Start value.
 * @param b - End value.
 * @param t - Interpolation factor in [0, 1].
 * @returns The interpolated value.
 */
export function lerp(a: number, b: number, t: number): number { return a + (b - a) * t; }
/**
 * Resolve a hue for a label, falling back to the default.
 * @param label - Label to resolve.
 * @returns The hue in degrees.
 */
export function getHue(label: string): number { return HUES[label] ?? DEFAULT_HUE; }

/**
 * Convert HSL to RGB.
 * @param h - Hue in degrees.
 * @param s - Saturation in [0, 1].
 * @param l - Lightness in [0, 1].
 * @returns RGB channels in [0, 1].
 */
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r1 = 0, g1 = 0, b1 = 0;
    if (hp >= 0 && hp < 1) { r1 = c; g1 = x; }
    else if (hp >= 1 && hp < 2) { r1 = x; g1 = c; }
    else if (hp >= 2 && hp < 3) { g1 = c; b1 = x; }
    else if (hp >= 3 && hp < 4) { g1 = x; b1 = c; }
    else if (hp >= 4 && hp < 5) { r1 = x; b1 = c; }
    else { r1 = c; b1 = x; }
    const m = l - c / 2;
    return [r1 + m, g1 + m, b1 + m];
}
