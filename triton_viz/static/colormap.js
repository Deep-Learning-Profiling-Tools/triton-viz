export const HUES = {
    Load: 200,
    Store: 45,
    A: 200,
    B: 45,
    C: 140,
};
const DEFAULT_HUE = 200;
export function clamp01(value) { return Math.min(1, Math.max(0, value)); }
export function lerp(a, b, t) { return a + (b - a) * t; }
export function getHue(label) { return HUES[label] ?? DEFAULT_HUE; }
export function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r1 = 0, g1 = 0, b1 = 0;
    if (hp >= 0 && hp < 1) {
        r1 = c;
        g1 = x;
    }
    else if (hp >= 1 && hp < 2) {
        r1 = x;
        g1 = c;
    }
    else if (hp >= 2 && hp < 3) {
        g1 = c;
        b1 = x;
    }
    else if (hp >= 3 && hp < 4) {
        g1 = x;
        b1 = c;
    }
    else if (hp >= 4 && hp < 5) {
        r1 = x;
        b1 = c;
    }
    else {
        r1 = c;
        b1 = x;
    }
    const m = l - c / 2;
    return [r1 + m, g1 + m, b1 + m];
}
