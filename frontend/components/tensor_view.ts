import { createCadDimension, createShapeLegend } from '../utils/dimension_utils.js';
import { clamp01, getHue, hslToRgb } from '../utils/colormap.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';
import {
    setupScene,
    setupGeometries,
    createTensor,
    calculateTensorSize,
    setupCamera,
    fitCameraToBounds,
    setupEventListeners,
    cameraControls,
    addLabels,
    COLOR_EDGE,
    CUBE_SIZE,
    GAP,
    COLOR_HOVER,
    updateTensorHighlights,
} from '../utils/three_utils.js';
import { createHistogramOverlay } from './histogram.js';
import { enableDrag } from '../utils/ui_helpers.js';
import { getApiBase, postJson } from '../core/api.js';
import { getState } from '../core/state.js';
import { createDisposer } from '../utils/dispose.js';
import type { HistogramOverlay } from './histogram.js';
import type { OpRecord, ProgramCountsPayload, ProgramSubsetsPayload, TensorPayload } from '../types/types.js';

type Coord3 = [number, number, number];
type TensorCoords = Coord3[];
type ColorInput = any;
type ThreeColor = any;
type ThreeScene = any;
type ThreeCamera = any;
type ThreeRenderer = any;
type ThreeLineMaterial = any;
type ThreeRaycaster = any;
type ThreeVector2 = any;
type ThreeVector3 = any;
type ThreeLineSegments = any;
type ThreeOrbitControls = any;
type TensorConfig = {
    name: string;
    shape: number[];
    color: ColorInput;
    position?: Coord3;
    endpoint?: string;
};
type PayloadCache = {
    scaleMin: number;
    scaleMax: number;
    values?: TensorPayload['values'] | undefined;
    shape?: number[] | undefined;
    dims?: number | undefined;
    highlights?: TensorPayload['highlights'] | null | undefined;
};
type ProgramCountState = { map: Map<string, number>; maxCount: number };
type ProgramSubsetState = {
    subsetMap: Map<string, string>;
    subsets: Record<string, Coord3[]>;
    subsetCount: number;
    countMap: Map<string, number>;
    maxCount: number;
};
type TensorMesh = {
    count: number;
    userData: {
        shape: { width: number; height: number; depth: number };
        shape_raw?: number[];
        coords?: TensorCoords;
        tensorName?: string;
        color_base?: ThreeColor;
    };
    setColorAt: (index: number, color: ThreeColor) => void;
    getMatrixAt: (index: number, matrix: any) => void;
    localToWorld: (pos: any) => void;
    instanceColor?: { needsUpdate: boolean };
};
type TensorGroup = { userData: { mesh: TensorMesh; endpoint?: string }; position: any; children: any[] };
type VizState = {
    colorizeOn: boolean;
    payloads: Map<string, PayloadCache>;
    rafId: number | null;
    renderPending: boolean;
    lastHoverKey: string | null;
    activeHoverOutline: ThreeLineSegments;
    dotHoverKey: string | null;
    allProgramsOn: boolean;
    allProgramsMode: 'subset' | 'count';
    programCounts: ProgramCountState | null;
    programSubsets: ProgramSubsetState | null;
    programSubsetHues: Map<string, number> | null;
    programDataLoading: boolean;
    isDragging?: boolean;
};
export type ViewState = {
    camera?: { position?: number[]; quaternion?: number[] };
    target?: number[];
    colorizeOn?: boolean;
    allProgramsOn?: boolean;
    histogramVisible?: boolean;
    histogramSource?: string | null;
    histogramBins?: number | null;
};
type VizContext = {
    type: string;
    shapeKey: string;
    containerElement: HTMLElement;
    sideMenu: HTMLDivElement;
    histogramUI: HistogramOverlay;
    stage: HTMLDivElement;
    API_BASE: string;
    op: OpRecord;
    scene: ThreeScene;
    camera: ThreeCamera;
    renderer: ThreeRenderer;
    tensors: Map<string, TensorGroup>;
    orbitControls: ThreeOrbitControls;
    lineMaterial: ThreeLineMaterial | null;
    state: VizState;
    disposer: ReturnType<typeof createDisposer>;
    raycaster: ThreeRaycaster;
    mouse: ThreeVector2;
    legendContainer: HTMLElement | null;
    dimLineGroups: any[];
    highlightColor: ThreeColor;
    requestRender: () => void;
    applyBackgroundTheme: (hex: string) => void;
    destroyLegends: () => void;
    createLegends: (items: HTMLElement[]) => void;
    cleanup?: () => void;
    ensureProgramDataForHover?: () => void;
};

const VIZ_CACHE = new Map<HTMLElement, VizContext>();
const PROGRAM_COUNT_COLORS = [
    '#22c55e',
    '#facc15',
    '#f97316',
    '#ef4444',
    '#a855f7',
    '#3b82f6',
    '#14b8a6',
    '#f472b6',
];
const PROGRAM_COUNT_PALETTE = PROGRAM_COUNT_COLORS.map((c) => new THREE.Color(c));
const PROGRAM_SUBSET_LIMIT = 256;

// --- Top-Level Helpers ---

function coordsFromIndex(index: number, shape: { width: number; height: number }): Coord3 {
    const w = Math.max(1, shape.width), h = Math.max(1, shape.height);
    const z = Math.floor(index / (w * h)), rem = index % (w * h);
    const y = Math.floor(rem / w), x = rem % w;
    return [x, y, z];
}

function sampleValueFromCache(cache: PayloadCache, coords: Coord3): number {
    if (!cache || !cache.values) return 0;
    const [x, y, z] = coords;
    const dims = cache.dims || (Array.isArray(cache.values[0]) ? (Array.isArray(cache.values[0][0]) ? 3 : 2) : 1);
    if (dims >= 3) return Number((cache.values as number[][][] | undefined)?.[y]?.[x]?.[z] ?? 0);
    if (dims === 2) return Number((cache.values as number[][] | undefined)?.[y]?.[x] ?? 0);
    return Number((cache.values as number[] | undefined)?.[x] ?? 0);
}

function applyValueColormap(
    mesh: TensorMesh,
    cache: PayloadCache,
    paint: (color: ThreeColor, coords: Coord3, val: number, t: number) => void,
): void {
    if (!mesh || !cache) return;
    const min = cache.scaleMin, max = cache.scaleMax, denom = max - min || 1;
    const count = mesh.count, shape = mesh.userData.shape;
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
        const coords = coordsFromIndex(i, shape);
        const val = sampleValueFromCache(cache, coords);
        const t = clamp01((val - min) / denom);
        paint(c, coords, val, t);
        mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

function applyCoordColormap(
    mesh: TensorMesh,
    paint: (color: ThreeColor, coords: Coord3, index: number) => void,
): void {
    if (!mesh || !paint) return;
    const coordsList = mesh.userData.coords;
    if (!coordsList) return;
    const c = new THREE.Color();
    for (let i = 0; i < mesh.count; i += 1) {
        const coords = coordsList[i];
        if (!coords) continue;
        paint(c, coords, i);
        mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

function createSideMenu(container: HTMLElement): HTMLDivElement {
    const menu = document.createElement('div');
    Object.assign(menu.style, {
        position: 'absolute', right: '16px', bottom: '16px', width: '300px', padding: '12px',
        background: 'rgba(0,0,0,0.65)', color: '#fff', borderRadius: '8px',
        fontFamily: 'var(--font-sans)', fontSize: '12px', zIndex: 2500,
    });
    container.appendChild(menu);
    return menu;
}

function updateSideMenu(
    el: HTMLElement,
    name: string,
    coords: number[],
    val: number | string,
    shape: number[] | null,
    extraHtml = '',
): void {
    const shapeStr = Array.isArray(shape) ? `[${shape.join(', ')}]` : '(unknown)';
    const indexCoords = Array.isArray(coords) ? coords.slice().reverse() : [];
    const indexExpr = indexCoords.map((coord) => `[${coord}]`).join('');
    const valueLine = indexExpr ? `tensor${indexExpr} = ${val}` : `tensor = ${val}`;
    el.innerHTML = `<h3>${name} Tensor</h3><p>${valueLine}</p><p>Shape: ${shapeStr}</p>${extraHtml}`;
}

async function fetchTensorPayload(apiBase: string, uuid: string, endpoint = 'getLoadTensor'): Promise<PayloadCache | null> {
    try {
        const data = await postJson<TensorPayload>(`/api/${endpoint}`, { uuid }, { base: apiBase });
        if (!data) return null;
        return {
            scaleMin: data.min ?? 0, scaleMax: data.max ?? 0,
            values: data.values, shape: data.shape, dims: data.dims,
            highlights: data.highlights,
        };
    } catch (e) { return null; }
}

async function fetchProgramCounts(apiBase: string, op: OpRecord): Promise<ProgramCountsPayload | null> {
    if (!op || !op.overall_key || op.time_idx === undefined) return null;
    // fetch a sparse list of coords -> program count for this op
    try {
        return await postJson<ProgramCountsPayload>('/api/getLoadStoreAllPrograms', {
            type: op.type,
            overall_key: op.overall_key,
            time_idx: op.time_idx,
            op_index: op.op_index,
        }, { base: apiBase });
    } catch (e) { return null; }
}

function applyColorToMesh(mesh: TensorMesh, cache: PayloadCache, label: string): void {
    if (!mesh || !cache) return;
    const hue = getHue(label);
    applyValueColormap(mesh, cache, (color, _coords, _val, t) => {
        const [r, g, b] = hslToRgb(hue, 0.9, t);
        color.setRGB(r, g, b);
    });
}

function applyDimmedColormap(
    mesh: TensorMesh,
    cache: PayloadCache,
    label: string,
    isHighlighted: (coords: Coord3) => boolean,
): void {
    if (!mesh || !cache) return;
    const hue = getHue(label);
    applyValueColormap(mesh, cache, (color, coords, _val, t) => {
        if (isHighlighted(coords)) {
            const [r, g, b] = hslToRgb(hue, 0.9, t);
            color.setRGB(r, g, b);
        } else {
            color.setRGB(t, t, t);
        }
    });
}

function applyMonochromeColormap(mesh: TensorMesh, cache: PayloadCache): void {
    if (!mesh || !cache) return;
    applyValueColormap(mesh, cache, (color, _coords, _val, t) => {
        color.setRGB(t, t, t);
    });
}

function normalizeProgramCounts(payload: ProgramCountsPayload | null): ProgramCountState {
    // normalize the sparse payload into a map for fast lookups
    const map = new Map();
    let maxCount = 0;
    (payload?.counts || []).forEach((entry) => {
        if (!entry || entry.length < 4) return;
        const [x, y, z, count] = entry;
        const safeCount = Number(count) || 0;
        map.set(`${x},${y},${z}`, safeCount);
        if (safeCount > maxCount) maxCount = safeCount;
    });
    maxCount = Math.max(maxCount, Number(payload?.max_count) || 0);
    return { map, maxCount };
}

function normalizeProgramSubsets(payload: ProgramSubsetsPayload | null): ProgramSubsetState {
    // normalize subset payload into lookup maps
    const subsetMap = new Map<string, string>();
    const subsets = (payload?.subsets || {}) as Record<string, Coord3[]>;
    const subsetCount = Number(payload?.subset_count) || Object.keys(subsets).length;
    const countMap = new Map<string, number>();
    let maxCount = Number(payload?.max_count) || 0;
    (payload?.coords || []).forEach((entry) => {
        if (!entry || entry.length < 4) return;
        const [x, y, z, key] = entry;
        subsetMap.set(`${x},${y},${z}`, String(key));
    });
    (payload?.counts || []).forEach((entry) => {
        if (!entry || entry.length < 4) return;
        const [x, y, z, count] = entry;
        const safeCount = Number(count) || 0;
        countMap.set(`${x},${y},${z}`, safeCount);
        if (safeCount > maxCount) maxCount = safeCount;
    });
    return { subsetMap, subsets, subsetCount, countMap, maxCount };
}

function buildSubsetHues(subsets: Record<string, Coord3[]>): Map<string, number> {
    const keys = Object.keys(subsets || {}).sort();
    const hues = new Map<string, number>();
    keys.forEach((key) => {
        let hash = 5381;
        for (let i = 0; i < key.length; i += 1) {
            hash = ((hash << 5) + hash) ^ key.charCodeAt(i); // deterministic hash
        }
        const hue = ((hash >>> 0) % 360) / 360;
        hues.set(key, hue);
    });
    return hues;
}

function applyProgramCountColors(
    mesh: TensorMesh,
    counts: ProgramCountState,
    baseColor: ColorInput,
    palette?: any[],
): void {
    if (!mesh || !counts) return;
    const base = baseColor instanceof THREE.Color ? baseColor : new THREE.Color(baseColor);
    const countMap = counts.map;
    const colors = palette || PROGRAM_COUNT_PALETTE;
    applyCoordColormap(mesh, (color, coords) => {
        const count = coords ? countMap.get(`${coords[0]},${coords[1]},${coords[2]}`) || 0 : 0;
        if (count <= 0) {
            color.copy(base);
        } else {
            const idx = Math.min(count - 1, colors.length - 1);
            color.copy(colors[idx]);
        }
    });
}

function applyProgramCountHeatmap(
    mesh: TensorMesh,
    cache: PayloadCache | null | undefined,
    counts: ProgramCountState,
    palette?: any[],
    baseColor?: ColorInput,
): void {
    if (!mesh || !cache || !counts) return;
    const min = cache.scaleMin, max = cache.scaleMax, denom = max - min || 1;
    const countMap = counts.map;
    const colors = palette || PROGRAM_COUNT_PALETTE;
    const hsl = { h: 0, s: 0, l: 0.5 };
    applyCoordColormap(mesh, (color, coords) => {
        const count = coords ? countMap.get(`${coords[0]},${coords[1]},${coords[2]}`) || 0 : 0;
        if (count <= 0) {
            const val = coords ? sampleValueFromCache(cache, coords) : 0;
            const t = clamp01((val - min) / denom);
            color.setRGB(t, t, t);
            return;
        }
        const idx = Math.min(count - 1, colors.length - 1);
        colors[idx].getHSL(hsl);
        const val = coords ? sampleValueFromCache(cache, coords) : 0;
        const t = clamp01((val - min) / denom);
        color.setHSL(hsl.h, 0.7, t);
    });
}

function applyProgramSubsetColors(
    mesh: TensorMesh,
    subsetState: ProgramSubsetState,
    hues: Map<string, number> | null,
    baseColor: ColorInput,
): void {
    if (!mesh || !subsetState || !hues) return;
    const base = baseColor instanceof THREE.Color ? baseColor : new THREE.Color(baseColor);
    const subsetMap = subsetState.subsetMap;
    applyCoordColormap(mesh, (color, coords) => {
        const key = coords ? subsetMap.get(`${coords[0]},${coords[1]},${coords[2]}`) : null;
        if (key && hues.has(key)) {
            color.setHSL(hues.get(key), 0.6, 0.55);
        } else {
            color.copy(base);
        }
    });
}

function applyProgramSubsetHeatmap(
    mesh: TensorMesh,
    cache: PayloadCache | null | undefined,
    subsetState: ProgramSubsetState,
    hues: Map<string, number> | null,
    baseColor: ColorInput,
): void {
    if (!mesh || !cache || !subsetState || !hues) return;
    const min = cache.scaleMin, max = cache.scaleMax, denom = max - min || 1;
    const subsetMap = subsetState.subsetMap;
    applyCoordColormap(mesh, (color, coords) => {
        const key = coords ? subsetMap.get(`${coords[0]},${coords[1]},${coords[2]}`) : null;
        if (!key || !hues.has(key)) {
            const val = coords ? sampleValueFromCache(cache, coords) : 0;
            const t = clamp01((val - min) / denom);
            color.setRGB(t, t, t);
            return;
        }
        const val = coords ? sampleValueFromCache(cache, coords) : 0;
        const t = clamp01((val - min) / denom);
        color.setHSL(hues.get(key), 0.7, t);
    });
}
function getHighlightPredicate(
    highlights: TensorPayload['highlights'] | null | undefined,
): ((coords: Coord3) => boolean) | null {
    if (!highlights) return null;
    if (highlights.type === 'descriptor') {
        const { start, shape } = highlights;
        const sx = start?.[0] ?? 0;
        const sy = start?.[1] ?? 0;
        const sz = start?.[2] ?? 0;
        const dx = shape?.[0] ?? 0;
        const dy = shape?.[1] ?? 0;
        const dz = shape?.[2] ?? 0;
        if (dx <= 0 || dy <= 0 || dz <= 0) return null;
        return (coords) => {
            const [x, y, z] = coords;
            return x >= sx && x < sx + dx && y >= sy && y < sy + dy && z >= sz && z < sz + dz;
        };
    }
    if (Array.isArray(highlights.data) && highlights.data.length) {
        const set = new Set();
        highlights.data.forEach((c) => { set.add(`${c[0]},${c[1]},${c[2]}`); });
        return (coords) => set.has(`${coords[0]},${coords[1]},${coords[2]}`);
    }
    return null;
}

function applyColorizedMesh(ctx: VizContext, group: TensorGroup, name: string): void {
    const mesh = group.userData.mesh;
    const p = ctx.state.payloads.get(name);
    if (!p) return;
    const label = name === 'Global' ? ctx.type : name;
    const predicate = getHighlightPredicate(p.highlights);
    if (predicate) {
        applyDimmedColormap(mesh, p, label, predicate);
    } else {
        applyMonochromeColormap(mesh, p);
    }
}

function restoreTensorColors(ctx: VizContext): void {
    const { state, tensors, type } = ctx;
    const supportsAllPrograms = type === 'Load' || type === 'Store';
    tensors.forEach((group, name) => {
        const mesh = group.userData.mesh;
        const p = state.payloads.get(name);
        if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'subset' && state.programSubsets) {
            if (state.colorizeOn && p) {
                applyProgramSubsetHeatmap(mesh, p, state.programSubsets, state.programSubsetHues, mesh.userData.color_base);
            } else {
                applyProgramSubsetColors(mesh, state.programSubsets, state.programSubsetHues, mesh.userData.color_base);
            }
        } else if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'count' && state.programCounts) {
            if (state.colorizeOn && p) {
                applyProgramCountHeatmap(mesh, p, state.programCounts, PROGRAM_COUNT_PALETTE, mesh.userData.color_base);
            } else {
                applyProgramCountColors(mesh, state.programCounts, mesh.userData.color_base, PROGRAM_COUNT_PALETTE);
            }
        } else if (state.colorizeOn && p) {
            applyColorizedMesh(ctx, group, name);
        } else {
            updateTensorHighlights(group, p?.highlights, ctx.highlightColor, mesh.userData.color_base);
        }
    });
}

function applyDotHoverHighlight(ctx: VizContext, row: number, col: number): void {
    const { tensors, state } = ctx;
    const aGroup = tensors.get('A');
    const bGroup = tensors.get('B');
    if (!aGroup || !bGroup) return;
    const aCache = state.payloads.get('A');
    const bCache = state.payloads.get('B');
    if (!aCache || !bCache) return;
    applyDimmedColormap(aGroup.userData.mesh, aCache, 'A', (coords) => coords[1] === row);
    applyDimmedColormap(bGroup.userData.mesh, bCache, 'B', (coords) => coords[0] === col);
}

function applyDotHoverOutline(ctx: VizContext, row: number, col: number): void {
    const { tensors } = ctx;
    const aGroup = tensors.get('A');
    const bGroup = tensors.get('B');
    if (!aGroup || !bGroup) return;
    updateTensorHighlights(aGroup, null, ctx.highlightColor, aGroup.userData.mesh.userData.color_base, (x, y) => y === row);
    updateTensorHighlights(bGroup, null, ctx.highlightColor, bGroup.userData.mesh.userData.color_base, (x) => x === col);
}

function captureHistogramState(histogramUI: HistogramOverlay): {
    histogramVisible: boolean;
    histogramSource: string | null;
    histogramBins: number | null;
} {
    const overlay = histogramUI?.overlay;
    if (!overlay) {
        return { histogramVisible: false, histogramSource: null, histogramBins: null };
    }
    const select = overlay.querySelector('#histogram-source') as HTMLSelectElement | null;
    const bins = overlay.querySelector('#histogram-bins') as HTMLInputElement | null;
    return {
        histogramVisible: overlay.style.display === 'block',
        histogramSource: select ? select.value : null,
        histogramBins: bins ? Number(bins.value) : null,
    };
}

function applyHistogramState(
    histogramUI: HistogramOverlay,
    state?: { histogramVisible?: boolean; histogramSource?: string | null; histogramBins?: number | null },
): void {
    const overlay = histogramUI?.overlay;
    if (!overlay || !state) return;
    const select = overlay.querySelector('#histogram-source') as HTMLSelectElement | null;
    const bins = overlay.querySelector('#histogram-bins') as HTMLInputElement | null;
    if (select && state.histogramSource) {
        select.value = state.histogramSource;
    }
    if (bins && Number.isFinite(state.histogramBins)) {
        bins.value = String(state.histogramBins);
    }
    if (state.histogramVisible) {
        histogramUI.show?.();
    } else {
        histogramUI.hide?.();
    }
}

function createLegendItem(label: string, min: number, max: number): HTMLDivElement {
    const item = document.createElement('div');
    Object.assign(item.style, { display: 'grid', gap: '4px', fontFamily: 'monospace', fontSize: '12px' });
    const title = document.createElement('div');
    title.textContent = `${label} Value`;
    title.style.opacity = '0.9'; title.style.fontWeight = 'bold';
    const canvas = document.createElement('canvas');
    canvas.width = 220; canvas.height = 10;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        item.appendChild(title); item.appendChild(canvas);
        return item;
    }
    for (let x = 0; x < canvas.width; x++) {
        const t = clamp01(x / (canvas.width - 1));
        const v = Math.round(t * 255);
        ctx.fillStyle = `rgb(${v},${v},${v})`;
        ctx.fillRect(x, 0, 1, canvas.height);
    }
    const labels = document.createElement('div');
    labels.style.display = 'flex'; labels.style.justifyContent = 'space-between';
    labels.style.marginTop = '2px'; labels.innerHTML = `<span>${min.toFixed(3)}</span><span>${max.toFixed(3)}</span>`;
    item.appendChild(title); item.appendChild(canvas); item.appendChild(labels);
    return item;
}

function createProgramCountLegendItem(
    baseColor: ColorInput,
    maxCount: number,
    palette = PROGRAM_COUNT_PALETTE,
): HTMLDivElement {
    const item = document.createElement('div');
    Object.assign(item.style, { display: 'grid', gap: '6px', fontFamily: 'monospace', fontSize: '12px' });
    const title = document.createElement('div');
    title.textContent = 'Programs per element';
    title.style.opacity = '0.9';
    title.style.fontWeight = 'bold';
    item.appendChild(title);
    const rows = document.createElement('div');
    rows.style.display = 'grid';
    rows.style.gap = '4px';
    const base = baseColor instanceof THREE.Color ? baseColor : new THREE.Color(baseColor);
    const addRow = (label: string, color: ThreeColor): void => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '6px';
        const swatch = document.createElement('span');
        swatch.style.display = 'inline-block';
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.borderRadius = '3px';
        swatch.style.background = `#${color.getHexString()}`;
        row.appendChild(swatch);
        const text = document.createElement('span');
        text.textContent = label;
        row.appendChild(text);
        rows.appendChild(row);
    };
    addRow('none', base);
    const maxLegend = Math.min(maxCount, palette.length);
    for (let i = 1; i <= maxLegend; i += 1) {
        addRow(String(i), palette[i - 1]);
    }
    if (maxCount > palette.length && palette.length > 0) {
        addRow(`>=${palette.length}`, palette[palette.length - 1]);
    }
    item.appendChild(rows);
    return item;
}

function createProgramSubsetLegendItem(
    baseColor: ColorInput,
    subsets: Record<string, Coord3[]>,
    hues: Map<string, number> | null,
): HTMLDivElement {
    const item = document.createElement('div');
    Object.assign(item.style, { display: 'grid', gap: '6px', fontFamily: 'monospace', fontSize: '12px' });
    const title = document.createElement('div');
    title.textContent = 'Program ID subsets';
    title.style.opacity = '0.9';
    title.style.fontWeight = 'bold';
    item.appendChild(title);
    const rows = document.createElement('div');
    rows.style.display = 'grid';
    rows.style.gap = '4px';
    const base = baseColor instanceof THREE.Color ? baseColor : new THREE.Color(baseColor);
    const addRow = (label: string, color: ThreeColor): void => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '6px';
        const swatch = document.createElement('span');
        swatch.style.display = 'inline-block';
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.borderRadius = '3px';
        swatch.style.background = `#${color.getHexString()}`;
        row.appendChild(swatch);
        const text = document.createElement('span');
        text.textContent = label;
        row.appendChild(text);
        rows.appendChild(row);
    };
    addRow('none', base);
    Object.keys(subsets || {}).forEach((key) => {
        const pids = subsets[key] || [];
        const label = pids.length
            ? pids.map(([x, y, z]) => `(${x},${y},${z})`).join(' ')
            : '(empty)';
        const color = new THREE.Color();
        if (hues && hues.has(key)) {
            color.setHSL(hues.get(key), 0.6, 0.55);
        } else {
            color.copy(base);
        }
        addRow(label, color);
    });
    item.appendChild(rows);
    return item;
}
function addDimensionLines(scene: ThreeScene, tensorGroup: TensorGroup, dimColors: string[] = []): any[] {
    const mesh = tensorGroup?.userData?.mesh;
    if (!mesh) return [];
    const shape = mesh.userData.shape;
    const shapeRaw = mesh.userData.shape_raw || [];
    const bbox = new THREE.Box3().setFromObject(tensorGroup);
    const offsetBase = (CUBE_SIZE + GAP) * 1.5;
    const axisDefaults = { x: '#f87171', y: '#4ade80', z: '#60a5fa' };
    const getColor = (axis: 'x' | 'y' | 'z'): string => {
        if (shapeRaw.length === 1 && axis === 'x') return dimColors[0] || axisDefaults.x;
        if (shapeRaw.length === 2 && axis === 'y') return dimColors[0] || axisDefaults.y;
        if (shapeRaw.length === 2 && axis === 'x') return dimColors[1] || axisDefaults.x;
        if (shapeRaw.length >= 3 && axis === 'z') return dimColors[0] || axisDefaults.z;
        if (shapeRaw.length >= 3 && axis === 'y') return dimColors[1] || axisDefaults.y;
        if (shapeRaw.length >= 3 && axis === 'x') return dimColors[2] || axisDefaults.x;
        return axisDefaults[axis];
    };
    const groups: any[] = [];
    if (shapeRaw.length >= 2) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${shape.width}`, 'x', getColor('x'), { offset: offsetBase }));
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), `${shape.height}`, 'y', getColor('y'), { offset: offsetBase }));
    } else if (shapeRaw.length === 1) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${shape.width}`, 'x', getColor('x'), { offset: offsetBase }));
    }
    if (shapeRaw.length >= 3) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z), new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), `${shape.depth}`, 'z', getColor('z'), { offset: offsetBase }));
    }
    return groups;
}

// --- Interaction Handlers ---

function onMouseMove(event: MouseEvent, ctx: VizContext): void {
    const { renderer, camera, tensors, state, sideMenu, requestRender, raycaster, mouse, API_BASE, op } = ctx;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const meshes = Array.from(tensors.values()).map((t) => t.userData.mesh);
    const hits = raycaster.intersectObjects(meshes);

    if (hits.length > 0) {
        const hit = hits[0];
        const mesh = hit.object as TensorMesh;
        const instanceId = hit.instanceId ?? 0;
        const tensorName = mesh.userData.tensorName || '';
        const key = `${tensorName}_${instanceId}`;

        if (key !== state.lastHoverKey) {
            state.lastHoverKey = key;
        }
        const coords = (mesh.userData.coords?.[instanceId] ?? [0, 0, 0]) as Coord3;
        if (state.activeHoverOutline) {
            const matrix = new THREE.Matrix4();
            mesh.getMatrixAt(instanceId, matrix);
            const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scale = new THREE.Vector3();
            matrix.decompose(pos, quat, scale);
            mesh.localToWorld(pos);
            state.activeHoverOutline.position.copy(pos);
            state.activeHoverOutline.visible = true;
        }
        const cacheEntry = state.payloads.get(tensorName);
        const val = cacheEntry ? sampleValueFromCache(cacheEntry, coords) : 'Loading...';
        const currentShape = mesh.userData.shape_raw;

        let extraHtml = '';
        if (tensorName === 'Global') {
            if (!state.programSubsets && !state.programCounts) {
                ctx.ensureProgramDataForHover?.();
            }
            const key = `${coords[0]},${coords[1]},${coords[2]}`;
            if (state.programSubsets) {
                const subsetKey = state.programSubsets.subsetMap.get(key);
                const subset = subsetKey ? state.programSubsets.subsets?.[subsetKey] || [] : [];
                const label = subset.length
                    ? subset.map(([x, y, z]) => `(${x},${y},${z})`).join(' ')
                    : 'none';
                extraHtml = `<p>Programs: ${label}</p>`;
            } else if (state.programCounts) {
                const count = state.programCounts.map.get(key) || 0;
                extraHtml = `<p>Programs: ${count}</p>`;
            } else if (state.programDataLoading) {
                extraHtml = '<p>Programs: loading...</p>';
            }
        }
        let displayCoords: Coord3 | number[] = (ctx.type === 'Dot' && tensorName !== 'Global' && Array.isArray(coords))
            ? [coords[1], coords[0], coords[2]]
            : coords;
        if (Array.isArray(displayCoords) && Array.isArray(currentShape) && currentShape.length === 2) {
            displayCoords = displayCoords.slice(0, 2);
        }
        updateSideMenu(sideMenu, tensorName, displayCoords, val, currentShape || null, extraHtml);
        if (ctx.type === 'Dot' && tensorName === 'C') {
            const row = coords[1];
            const col = coords[0];
            const hoverKey = `${row},${col}`;
            if (state.dotHoverKey !== hoverKey) {
                state.dotHoverKey = hoverKey;
                if (state.colorizeOn) {
                    applyDotHoverHighlight(ctx, row, col);
                } else {
                    applyDotHoverOutline(ctx, row, col);
                }
            }
        } else if (state.dotHoverKey) {
            state.dotHoverKey = null;
            restoreTensorColors(ctx);
        }
        requestRender();
    } else {
        if (state.lastHoverKey !== null) {
            state.lastHoverKey = null;
            if (state.activeHoverOutline) state.activeHoverOutline.visible = false;
            sideMenu.innerHTML = '';
            if (state.dotHoverKey) {
                state.dotHoverKey = null;
                restoreTensorColors(ctx);
            }
            requestRender();
        }
    }
}

function onMouseUp(ctx: VizContext): void { ctx.state.isDragging = false; if (ctx.stage) ctx.stage.style.cursor = ''; }

// --- Main Exports ---

export function createTensorVisualization(
    containerElement: HTMLElement,
    op: OpRecord,
    options: {
        type?: string;
        colors?: Record<string, ColorInput>;
        tensorConfigs?: TensorConfig[];
        dimColors?: Record<string, string[]>;
        showDimLines?: boolean;
        viewState?: ViewState | null;
        hasHeatmap?: boolean;
        layoutBounds?: { width: number; height: number; depth?: number; center?: [number, number, number] };
        fitToTensors?: boolean;
        cameraPadding?: number;
    } = {},
): (() => void) | void {
    const { type = 'Load', colors = {}, tensorConfigs = [], dimColors = {}, showDimLines = true, viewState = null, layoutBounds = null, fitToTensors = true, cameraPadding = 1.15 } = options;
    const supportsAllPrograms = type === 'Load' || type === 'Store';
    const API_BASE = getApiBase();
    const initialToggles = getState().toggles;
    const configs = tensorConfigs.length > 0 ? tensorConfigs : [
        { name: 'Global', shape: op.global_shape || [], color: colors.GLOBAL || '#333', position: [0,0,0], endpoint: 'getLoadTensor' }
    ];

    let cache = VIZ_CACHE.get(containerElement);
    const shapeKey = JSON.stringify({ shapes: configs.map(c => c.shape), layoutBounds });
    const isSameContext = cache && cache.type === type && cache.shapeKey === shapeKey;

    if (!isSameContext) {
        if (cache && cache.cleanup) cache.cleanup();
        containerElement.innerHTML = '';
        containerElement.style.position = 'relative';
        const stage = document.createElement('div');
        stage.className = 'viz-stage';
        containerElement.appendChild(stage);
        const sideMenu = createSideMenu(containerElement);
        const histogramUI = createHistogramOverlay(containerElement, {
            title: `${type} Value Distribution`,
            apiBase: API_BASE,
            sources: configs.map(c => ({ value: c.name.toUpperCase(), label: `${c.name} Tensor` })),
            buildRequestBody: (s, b) => ({ uuid: op.uuid, source: s, bins: b }),
        });
        const { scene, camera, renderer } = setupScene(stage, 0x000000);
        const disposer = createDisposer();
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();
        const tensors = new Map<string, TensorGroup>();
        configs.forEach(cfg => {
            const group = createTensor(cfg.shape, null, cfg.color, cfg.name, cubeGeometry, edgesGeometry, lineMaterial) as TensorGroup;
            group.position.set(...(cfg.position || [0,0,0]));
            if (cfg.endpoint) {
                group.userData.endpoint = cfg.endpoint;
            } else if (group.userData.endpoint) {
                delete group.userData.endpoint;
            }
            scene.add(group);
            tensors.set(cfg.name, group);
        });
        if (layoutBounds) {
            const depth = layoutBounds.depth ?? CUBE_SIZE;
            const layoutBox = new THREE.Mesh(
                new THREE.BoxGeometry(layoutBounds.width, layoutBounds.height, depth),
                new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 }),
            );
            const center = layoutBounds.center ?? [0, 0, 0];
            layoutBox.position.set(center[0], center[1], center[2] ?? 0);
            scene.add(layoutBox);
        }
        const hoverOutline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05)), new THREE.LineBasicMaterial({ color: COLOR_HOVER }));
        hoverOutline.visible = false;
        scene.add(hoverOutline);
        const dimLineGroups: any[] = [];
        if (showDimLines) {
            tensors.forEach((group, name) => {
                dimLineGroups.push(...addDimensionLines(scene, group, dimColors[name]));
            });
        }
        let cameraCenter = new THREE.Vector3(0, 0, 0);
        let fitRadius = 0;
        const bounds = new THREE.Box3();
        let hasBounds = false;
        let tensorCenter: ThreeVector3 | null = null;
        if (fitToTensors) {
            tensors.forEach((group) => bounds.union(new THREE.Box3().setFromObject(group)));
            hasBounds = true;
            if (!bounds.isEmpty()) tensorCenter = bounds.getCenter(new THREE.Vector3());
        }
        if (showDimLines) {
            dimLineGroups.forEach((group) => bounds.union(new THREE.Box3().setFromObject(group)));
            bounds.expandByScalar(1.0);
            hasBounds = true;
        }
        if (layoutBounds) {
            const c = layoutBounds.center ?? [0, 0, 0];
            const center = new THREE.Vector3(c[0], c[1], c[2] ?? 0);
            const size = new THREE.Vector3(layoutBounds.width, layoutBounds.height, layoutBounds.depth ?? CUBE_SIZE);
            bounds.union(new THREE.Box3().setFromCenterAndSize(center, size));
            hasBounds = true;
        }
        if (hasBounds && !bounds.isEmpty()) {
            cameraCenter = fitCameraToBounds(camera, bounds, tensorCenter ?? undefined, cameraPadding).center;
            fitRadius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
        } else {
            const { center } = setupCamera(scene, camera);
            cameraCenter = center;
        }
        const orbitControls = new OrbitControls(camera, renderer.domElement);
        orbitControls.enableDamping = false;
        orbitControls.target.copy(cameraCenter);
        orbitControls.update();
        disposer.add(() => orbitControls.dispose());
        const syncClipPlanes = (): void => {
            if (!fitRadius) return;
            const dist = camera.position.distanceTo(orbitControls.target);
            camera.near = Math.max(0.05, dist - fitRadius * 2.5);
            camera.far = Math.max(camera.far, dist + fitRadius * 2.5);
            camera.updateProjectionMatrix();
        };
        const state: VizState = {
            colorizeOn: !!initialToggles.colorize,
            payloads: new Map(),
            rafId: null,
            renderPending: false,
            lastHoverKey: null,
            activeHoverOutline: hoverOutline,
            dotHoverKey: null,
            allProgramsOn: !!initialToggles.allPrograms,
            allProgramsMode: 'subset',
            programCounts: null,
            programSubsets: null,
            programSubsetHues: null,
            programDataLoading: false,
        };
        const highlightColor = (colors.HIGHLIGHT instanceof THREE.Color) ? colors.HIGHLIGHT : new THREE.Color(colors.HIGHLIGHT || 0x00b3ff);
        const ctx: VizContext = { type, shapeKey, containerElement, sideMenu, histogramUI, stage, API_BASE, op, scene, camera, renderer, tensors, orbitControls, lineMaterial, state, disposer, raycaster: new THREE.Raycaster(), mouse: new THREE.Vector2(), legendContainer: null, dimLineGroups, highlightColor, requestRender: () => {}, applyBackgroundTheme: () => {}, destroyLegends: () => {}, createLegends: () => {} };
        ctx.requestRender = () => {
            if (state.rafId !== null) { state.renderPending = true; return; }
            state.rafId = requestAnimationFrame(() => { state.rafId = null; orbitControls.update(); syncClipPlanes(); renderer.render(scene, camera); if (state.renderPending) { state.renderPending = false; ctx.requestRender(); } });
        };
        disposer.listen(orbitControls, 'change', ctx.requestRender);
        ctx.applyBackgroundTheme = (hex: string) => {
            const isLight = (hex || '').toLowerCase() === '#ffffff';
            const baseGlobalLight = new THREE.Color('#fefce8');
            tensors.forEach((group, name) => {
                const mesh = group.userData.mesh;
                const baseColor = isLight ? baseGlobalLight : mesh.userData.color_base;
                if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'subset' && state.programSubsets) {
                    if (state.colorizeOn) {
                        const payload = state.payloads.get(name);
                        if (payload) applyProgramSubsetHeatmap(mesh, payload, state.programSubsets, state.programSubsetHues, baseColor);
                    } else {
                        applyProgramSubsetColors(mesh, state.programSubsets, state.programSubsetHues, baseColor);
                    }
                } else if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'count' && state.programCounts) {
                    if (state.colorizeOn) {
                        const payload = state.payloads.get(name);
                        if (payload) applyProgramCountHeatmap(mesh, payload, state.programCounts, PROGRAM_COUNT_PALETTE, baseColor);
                    } else {
                        applyProgramCountColors(mesh, state.programCounts, baseColor, PROGRAM_COUNT_PALETTE);
                    }
                } else {
                    updateTensorHighlights(group, state.payloads.get(name)?.highlights, highlightColor, baseColor);
                }
                group.children.forEach(c => { if (c?.userData?.edges) c.userData.edges.visible = !isLight; });
            });
            if (!isLight && lineMaterial) { lineMaterial.color.set('#ffffff'); lineMaterial.opacity = 0.28; }
            ctx.requestRender();
        };
        ctx.destroyLegends = (): void => { if (ctx.legendContainer?.remove) ctx.legendContainer.remove(); ctx.legendContainer = null; };
        ctx.createLegends = (items: HTMLElement[]): void => {
            ctx.destroyLegends();
            if (!items.length) return;
            const shapeLegend = containerElement.querySelector('.viz-shape-legend');
            const wrapper = document.createElement('div');
            Object.assign(wrapper.style, { display: 'grid', gap: '12px', marginTop: '10px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.15)' });
            items.forEach(i => wrapper.appendChild(i));
            if (shapeLegend) {
                shapeLegend.appendChild(wrapper);
            } else {
                Object.assign(wrapper.style, { position: 'absolute', left: '24px', bottom: '20px', background: 'rgba(0,0,0,0.65)', color: '#fff', padding: '12px', borderRadius: '8px', zIndex: '2000', pointerEvents: 'auto', border: '1px solid rgba(255,255,255,0.1)' });
                containerElement.appendChild(wrapper);
            }
            ctx.legendContainer = wrapper;
        };
        const applyViewState = (nextState?: ViewState | null): void => {
            if (!nextState) return;
            const pos = nextState.camera?.position;
            const quat = nextState.camera?.quaternion;
            const target = nextState.target;
            if (pos && pos.length === 3) {
                camera.position.set(pos[0], pos[1], pos[2]);
            }
            if (quat && quat.length === 4) {
                camera.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
            }
            if (target && target.length === 3) {
                orbitControls.target.set(target[0], target[1], target[2]);
            }
            orbitControls.update();
            state.colorizeOn = !!nextState.colorizeOn;
            state.allProgramsOn = !!nextState.allProgramsOn;
            applyHistogramState(histogramUI, nextState);
            if (window.setOpControlState) {
                window.setOpControlState({
                    colorize: state.colorizeOn,
                    histogram: !!nextState.histogramVisible,
                    allPrograms: state.allProgramsOn,
                });
            }
        };
        const getViewState = (): {
            camera: { position: Coord3; quaternion: [number, number, number, number] };
            target: Coord3;
            colorizeOn: boolean;
            allProgramsOn: boolean;
            histogramVisible?: boolean;
            histogramSource?: string | null;
            histogramBins?: number | null;
        } => {
            return {
                camera: {
                    position: [camera.position.x, camera.position.y, camera.position.z],
                    quaternion: [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w],
                },
                target: [orbitControls.target.x, orbitControls.target.y, orbitControls.target.z],
                colorizeOn: state.colorizeOn,
                allProgramsOn: state.allProgramsOn,
                ...captureHistogramState(histogramUI),
            };
        };
        const cleanupListeners = setupEventListeners(
            stage,
            camera,
            renderer,
            (e) => onMouseMove(e, ctx),
            cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ')),
            ctx.requestRender,
        );
        disposer.add(cleanupListeners);
        disposer.add(() => renderer.dispose());
        const fallbackState = !viewState ? {
            colorizeOn: !!initialToggles.colorize,
            allProgramsOn: !!initialToggles.allPrograms,
            histogramVisible: !!initialToggles.histogram,
        } : null;
        applyViewState(viewState || fallbackState);
        containerElement.__vizGetState = getViewState;
        ctx.cleanup = () => {
            if (state.rafId) cancelAnimationFrame(state.rafId);
            ctx.disposer.dispose();
            ctx.destroyLegends();
            ctx.dimLineGroups.forEach((group) => scene.remove(group));
            ctx.dimLineGroups = [];
            if (stage.parentElement) stage.parentElement.removeChild(stage);
            if (sideMenu.parentElement) sideMenu.parentElement.removeChild(sideMenu);
            histogramUI.destroy?.();
            if (histogramUI.overlay?.parentElement) histogramUI.overlay.parentElement.removeChild(histogramUI.overlay);
            if (containerElement.__vizGetState) {
                delete containerElement.__vizGetState;
            }
            VIZ_CACHE.delete(containerElement);
        };
        cache = ctx;
        VIZ_CACHE.set(containerElement, cache);
    } else if (cache) {
        if (cache.stage.parentElement !== containerElement) {
            containerElement.innerHTML = '';
            containerElement.appendChild(cache.stage);
            containerElement.appendChild(cache.sideMenu);
            if (cache.histogramUI.overlay) containerElement.appendChild(cache.histogramUI.overlay);
            if (cache.legendContainer) containerElement.appendChild(cache.legendContainer);
        }
    }

    if (!cache) return;
    const vizCache = cache;
    const { state, tensors, sideMenu, requestRender, applyBackgroundTheme, createLegends, destroyLegends } = vizCache;
    state.payloads.clear();
    state.programCounts = null;
    state.programSubsets = null;
    state.programSubsetHues = null;
    sideMenu.innerHTML = '';
    const opUuid = op.uuid ?? null;
    window.current_op_uuid = opUuid;
    const getBaseCountColor = () => tensors.values().next().value?.userData?.mesh?.userData?.color_base || '#333333';
    const getValueLegendItems = (): HTMLElement[] => {
        const items: HTMLElement[] = [];
        if (!state.colorizeOn) return items;
        tensors.forEach((group, name) => {
            const p = state.payloads.get(name);
            if (p) {
                items.push(createLegendItem(name === 'Global' ? type : name, p.scaleMin, p.scaleMax));
            }
        });
        return items;
    };
    const renderProgramCounts = (): void => {
        // paint counts across all cubes and attach the legend
        const programCounts = state.programCounts;
        if (!programCounts) return;
        destroyLegends();
        tensors.forEach((group, name) => {
            const p = state.payloads.get(name);
            if (state.colorizeOn && p) {
                applyProgramCountHeatmap(group.userData.mesh, p, programCounts, PROGRAM_COUNT_PALETTE, group.userData.mesh.userData.color_base);
            } else {
                applyProgramCountColors(group.userData.mesh, programCounts, group.userData.mesh.userData.color_base, PROGRAM_COUNT_PALETTE);
            }
        });
        const items = getValueLegendItems();
        items.push(createProgramCountLegendItem(getBaseCountColor(), programCounts.maxCount, PROGRAM_COUNT_PALETTE));
        createLegends(items);
    };
    const renderProgramSubsets = (): void => {
        // paint subset colors across all cubes and attach the legend
        const programSubsets = state.programSubsets;
        if (!programSubsets) return;
        destroyLegends();
        tensors.forEach((group, name) => {
            const p = state.payloads.get(name);
            if (state.colorizeOn && p) {
                applyProgramSubsetHeatmap(group.userData.mesh, p, programSubsets, state.programSubsetHues, group.userData.mesh.userData.color_base);
            } else {
                applyProgramSubsetColors(group.userData.mesh, programSubsets, state.programSubsetHues, group.userData.mesh.userData.color_base);
            }
        });
        const items = getValueLegendItems();
        items.push(createProgramSubsetLegendItem(getBaseCountColor(), programSubsets.subsets, state.programSubsetHues));
        createLegends(items);
    };
    const applyProgramPayload = (payload: ProgramSubsetsPayload): void => {
        const subsets = normalizeProgramSubsets(payload);
        if (subsets.subsetCount > PROGRAM_SUBSET_LIMIT) {
            state.allProgramsMode = 'count';
            state.programCounts = normalizeProgramCounts(payload);
            state.programSubsets = null;
            state.programSubsetHues = null;
        } else {
            state.allProgramsMode = 'subset';
            state.programSubsets = subsets;
            state.programSubsetHues = buildSubsetHues(subsets.subsets);
            state.programCounts = null;
        }
    };
    const ensureProgramData = async (): Promise<boolean> => {
        // fetch subset data for the current op if needed
        if (!supportsAllPrograms || !state.allProgramsOn) return false;
        if (!state.programSubsets && !state.programCounts) {
            const payload = await fetchProgramCounts(API_BASE, op);
            if (!payload) {
                state.allProgramsOn = false;
                if (window.setOpControlState) window.setOpControlState({ allPrograms: false });
                return false;
            }
            applyProgramPayload(payload);
        }
        if (state.allProgramsMode === 'subset' && state.programSubsets) {
            renderProgramSubsets();
        } else if (state.programCounts) {
            renderProgramCounts();
        }
        requestRender();
        return true;
    };
    const ensureProgramDataForHover = (): void => {
        if (!supportsAllPrograms) return;
        if (state.programSubsets || state.programCounts || state.programDataLoading) return;
        state.programDataLoading = true;
        fetchProgramCounts(API_BASE, op).then((payload) => {
            if (payload) {
                applyProgramPayload(payload);
                if (state.allProgramsOn) {
                    if (state.allProgramsMode === 'subset' && state.programSubsets) {
                        renderProgramSubsets();
                    } else if (state.programCounts) {
                        renderProgramCounts();
                    }
                    requestRender();
                }
            }
        }).finally(() => {
            state.programDataLoading = false;
        });
    };
    vizCache.ensureProgramDataForHover = ensureProgramDataForHover;
    if (window.setOpControlHandlers) {
        window.setOpControlHandlers({
            toggleColorize: async () => {
                state.colorizeOn = !state.colorizeOn;
                if (supportsAllPrograms && state.allProgramsOn) {
                    if (!state.programSubsets && !state.programCounts) {
                        await ensureProgramData();
                        return state.colorizeOn;
                    }
                    if (state.allProgramsMode === 'subset' && state.programSubsets) {
                        renderProgramSubsets();
                    } else if (state.programCounts) {
                        renderProgramCounts();
                    }
                } else if (!state.colorizeOn) {
                    destroyLegends();
                    tensors.forEach((group, name) => updateTensorHighlights(group, state.payloads.get(name)?.highlights, vizCache.highlightColor, group.userData.mesh.userData.color_base));
                } else {
                    const items: HTMLElement[] = [];
                    tensors.forEach((group, name) => {
                        const p = state.payloads.get(name);
                        if (p) {
                            items.push(createLegendItem(name === 'Global' ? type : name, p.scaleMin, p.scaleMax));
                            applyColorizedMesh(vizCache, group, name);
                        }
                    });
                    createLegends(items);
                }
                requestRender();
                return state.colorizeOn;
            },
            toggleShowCode: () => window.__tritonVizCodeToggle?.(),
            toggleHistogram: (): boolean => {
                const isVisible = vizCache.histogramUI.overlay.style.display === 'block';
                if (isVisible) {
                    vizCache.histogramUI.hide?.();
                } else {
                    vizCache.histogramUI.show?.();
                }
                return !isVisible;
            },
            toggleAllPrograms: supportsAllPrograms ? async () => {
                state.allProgramsOn = !state.allProgramsOn;
                if (!state.allProgramsOn) {
                    state.programCounts = null;
                    state.programSubsets = null;
                    state.programSubsetHues = null;
                    state.allProgramsMode = 'subset';
                    destroyLegends();
                    restoreTensorColors(vizCache);
                    if (state.colorizeOn) {
                        const items = getValueLegendItems();
                        if (items.length) {
                            createLegends(items);
                        }
                    }
                    requestRender();
                    return state.allProgramsOn;
                }
                return ensureProgramData();
            } : null,
        });
    }

    const fetchers = opUuid ? Array.from(tensors.entries()).map(([name, group]) => {
        return fetchTensorPayload(API_BASE, opUuid, group.userData.endpoint || 'getLoadTensor').then(p => {
            if (p) {
                state.payloads.set(name, p);
                if (!supportsAllPrograms || !state.allProgramsOn) {
                    updateTensorHighlights(group, p.highlights, vizCache.highlightColor, group.userData.mesh.userData.color_base);
                }
            }
        });
    }) : [];

    Promise.all(fetchers).then(async () => {
        if (supportsAllPrograms && state.allProgramsOn) {
            await ensureProgramData();
        } else if (state.colorizeOn) {
            const items: HTMLElement[] = [];
            tensors.forEach((group, name) => {
                const p = state.payloads.get(name);
                if (p) {
                    items.push(createLegendItem(name === 'Global' ? type : name, p.scaleMin, p.scaleMax));
                    applyColorizedMesh(vizCache, group, name);
                }
            });
            createLegends(items);
        } else {
            tensors.forEach((group, name) => {
                const p = state.payloads.get(name);
                if (p) {
                    updateTensorHighlights(group, p.highlights, vizCache.highlightColor, group.userData.mesh.userData.color_base);
                }
            });
            if (state.dotHoverKey && type === 'Dot') {
                const parts = state.dotHoverKey.split(',');
                const row = Number(parts[0] ?? NaN);
                const col = Number(parts[1] ?? NaN);
                if (Number.isFinite(row) && Number.isFinite(col)) {
                    if (state.colorizeOn) {
                        applyDotHoverHighlight(vizCache, row, col);
                    } else {
                        applyDotHoverOutline(vizCache, row, col);
                    }
                }
            }
        }
        requestRender();
    });

    createShapeLegend(containerElement, Array.from(tensors.entries()).map(([name, group]) => {
        const entry: { name: string; color: string; shape?: number[]; dimColors?: string[] } = {
            name: name === 'Global' ? type : `Matrix ${name}`,
            color: '#' + group.userData.mesh.userData.color_base.getHexString(),
        };
        const shape = group.userData.mesh.userData.shape_raw;
        if (shape) entry.shape = shape;
        const dimColor = dimColors?.[name];
        if (dimColor) entry.dimColors = dimColor;
        return entry;
    }));

    applyBackgroundTheme('#000000');
    requestRender();
    return vizCache.cleanup;
}
