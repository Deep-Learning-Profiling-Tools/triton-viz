import { createCadDimension, createShapeLegend, defaultAxisColor } from '../utils/dimension_utils.js';
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
    COLOR_EDGE,
    CUBE_SIZE,
    GAP,
    COLOR_HOVER,
    updateTensorHighlights,
    canUseWebgl,
    renderWebglWarning,
    positionForTensorCoord,
    tensorBoundsSizeForShape,
} from '../utils/three_utils.js';
import { createHistogramOverlay } from './histogram.js';
import { enableDrag } from '../utils/ui_helpers.js';
import { getApiBase, postJson } from '../core/api.js';
import { getState } from '../core/state.js';
import { createDisposer } from '../utils/dispose.js';
import type { HistogramOverlay } from './histogram.js';
import type { OpRecord, ProgramCountsPayload, ProgramSubsetsPayload, TensorPayload } from '../types/types.js';

type Coord3 = [number, number, number];
type TensorCoord = number[];
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
    subsets: Record<string, number[][]>;
    subsetCount: number;
    countMap: Map<string, number>;
    maxCount: number;
};
type DescriptorHighlight = {
    start: number[];
    shape: number[];
    stride: number[];
};
type HiddenAxisGroup = {
    token: string;
    axes: number[];
    size: number;
    value: number;
    outlineAxis: number;
};
type TensorViewSpec = {
    axisShape: number[];
    axisLabels: string[];
    displaySlots: Array<number[] | null>;
    outlineSlots: Array<number[] | null>;
    displayToOutline: number[];
    visibleAxes: number[];
    hiddenAxes: number[];
    hiddenGroups: HiddenAxisGroup[];
    hiddenIndices: number[];
    visibleText: string;
    displayShape: number[];
    outlineShape: number[];
};
type TensorViewSnapshot = {
    visible: string;
    hiddenIndices: number[];
};
type TensorMesh = {
    count: number;
    userData: {
        shape: { width: number; height: number; depth: number };
        shape_raw?: number[];
        coords?: TensorCoords;
        coords_display?: TensorCoord[];
        coords_full?: TensorCoord[];
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
    editTensorViewOn: boolean;
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
    tensorViews: Map<string, TensorViewSpec>;
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
    tensorViews?: Record<string, TensorViewSnapshot>;
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
    configByName: Map<string, TensorConfig>;
    cubeGeometry: any;
    edgesGeometry: any;
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
    sliceOutlineGroups: any[];
    descriptorDimLineGroups: any[];
    showDimLines: boolean;
    supportsAllPrograms: boolean;
    highlightColor: ThreeColor;
    tensorViewControls: HTMLElement | null;
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
const SHOW_DESCRIPTOR_DIM_LINES = false;
const DESCRIPTOR_AXIS_COLORS = {
    x: '#f59e0b',
    y: '#d946ef',
    z: '#22d3ee',
};

// --- Top-Level Helpers ---

function sampleValueFromCache(cache: PayloadCache, coords: TensorCoord): number {
    if (!cache || !cache.values) return 0;
    let cursor: any = cache.values;
    for (let axis = 0; axis < coords.length; axis += 1) {
        if (!Array.isArray(cursor)) return Number(cursor ?? 0);
        const idx = coords[axis] ?? 0;
        cursor = cursor[idx];
    }
    return Number(cursor ?? 0);
}

function getMeshFullCoords(mesh: TensorMesh, index: number): TensorCoord {
    return (
        mesh.userData.coords_full?.[index]
        || mesh.userData.coords_display?.[index]
        || mesh.userData.coords?.[index]
        || [0, 0, 0]
    );
}

function getMeshDisplayCoords(mesh: TensorMesh, index: number): TensorCoord {
    return mesh.userData.coords_display?.[index] || getMeshFullCoords(mesh, index);
}

function getAxisLabel(axis: number): string {
    const letters = 'abcdefghijklmnopqrstuvwxyz';
    return letters[axis] || `d${axis}`;
}

function getAxisLabels(rank: number): string[] {
    return Array.from({ length: rank }, (_, axis) => getAxisLabel(axis));
}

function normalizeViewShape(shapeRaw: number[]): number[] {
    if (!Array.isArray(shapeRaw) || shapeRaw.length === 0) return [1];
    return shapeRaw.map((dim) => Math.max(1, Number(dim) || 1));
}

function flattenAxesIndex(
    axes: number[],
    values: number[],
    shapeRaw: number[],
): number {
    let linear = 0;
    axes.forEach((axis) => {
        const dim = Math.max(1, shapeRaw[axis] ?? 1);
        const v = Math.min(dim - 1, Math.max(0, Number(values[axis] ?? 0)));
        linear = (linear * dim) + v;
    });
    return linear;
}

function unflattenAxesIndex(
    linearIndex: number,
    axes: number[],
    shapeRaw: number[],
): number[] {
    const out = new Array(axes.length).fill(0);
    let remaining = Math.max(0, Math.round(Number(linearIndex) || 0));
    for (let i = axes.length - 1; i >= 0; i -= 1) {
        const axis = axes[i];
        if (axis === undefined) continue;
        const dim = Math.max(1, shapeRaw[axis] ?? 1);
        out[i] = remaining % dim;
        remaining = Math.floor(remaining / dim);
    }
    return out;
}

function productForAxes(
    axes: number[],
    shapeRaw: number[],
): number {
    return axes.reduce((acc, axis) => acc * Math.max(1, shapeRaw[axis] ?? 1), 1);
}

function buildDefaultTensorViewSpec(
    shapeRaw: number[],
    hiddenIndices: number[] = [],
): TensorViewSpec {
    const rank = shapeRaw.length;
    const axisShape = normalizeViewShape(shapeRaw);
    const axisLabels = getAxisLabels(rank).map((label) => label.toUpperCase());
    const nextHiddenIndices = Array.from({ length: rank }, (_, axis) => {
        const dim = Math.max(1, axisShape[axis] ?? 1);
        const prev = Number(hiddenIndices[axis] ?? 0);
        return Math.min(dim - 1, Math.max(0, Number.isFinite(prev) ? prev : 0));
    });
    const displaySlots = Array.from({ length: rank }, (_, axis) => [axis]);
    const outlineSlots = displaySlots.slice();
    return {
        axisShape,
        axisLabels,
        displaySlots,
        outlineSlots,
        displayToOutline: Array.from({ length: rank }, (_, axis) => axis),
        visibleAxes: Array.from({ length: rank }, (_, axis) => axis),
        hiddenAxes: [],
        hiddenGroups: [],
        hiddenIndices: nextHiddenIndices,
        visibleText: axisLabels.join(' '),
        displayShape: axisShape.slice(),
        outlineShape: axisShape.slice(),
    };
}

function buildTensorViewSpec(
    shapeRaw: number[],
    visibleText = '',
    hiddenIndices: number[] = [],
): TensorViewSpec {
    const rank = shapeRaw.length;
    if (rank <= 0) return buildDefaultTensorViewSpec(shapeRaw, hiddenIndices);
    const axisShape = normalizeViewShape(shapeRaw);
    const axisLabels = getAxisLabels(rank).map((label) => label.toUpperCase());
    const rawText = (visibleText || '').trim().replace(/\s+/g, ' ');
    if (!rawText) return buildDefaultTensorViewSpec(axisShape, hiddenIndices);
    const tokens = rawText.split(' ').filter(Boolean);
    if (tokens.length === 0) return buildDefaultTensorViewSpec(axisShape, hiddenIndices);

    const labelEntries = axisLabels.map((label, axis) => ({ axis, lower: label.toLowerCase() }));
    labelEntries.sort((a, b) => b.lower.length - a.lower.length);
    const nextHiddenIndices = Array.from({ length: rank }, (_, axis) => {
        const dim = Math.max(1, axisShape[axis] ?? 1);
        const prev = Number(hiddenIndices[axis] ?? 0);
        return Math.min(dim - 1, Math.max(0, Number.isFinite(prev) ? prev : 0));
    });
    const displaySlots: Array<number[] | null> = [];
    const outlineSlots: Array<number[] | null> = [];
    const displayToOutline: number[] = [];
    const visibleAxes: number[] = [];
    const hiddenAxes: number[] = [];
    const hiddenGroups: HiddenAxisGroup[] = [];
    const canonicalTokens: string[] = [];
    const outlineShape: number[] = [];
    const seenAxes = new Set<number>();

    for (const token of tokens) {
        const outlineAxis = outlineSlots.length;
        if (token === '1') {
            displaySlots.push(null);
            outlineSlots.push(null);
            displayToOutline.push(outlineAxis);
            canonicalTokens.push('1');
            outlineShape.push(1);
            continue;
        }
        const lettersOnly = token.replace(/[^a-zA-Z]/g, '');
        const hasUpper = /[A-Z]/.test(lettersOnly);
        const hasLower = /[a-z]/.test(lettersOnly);
        if (!lettersOnly || (hasUpper && hasLower)) {
            return buildDefaultTensorViewSpec(axisShape, hiddenIndices);
        }
        const tokenLower = token.toLowerCase();
        const parsedAxes: number[] = [];
        let cursor = 0;
        while (cursor < tokenLower.length) {
            const match = labelEntries.find((entry) => tokenLower.startsWith(entry.lower, cursor));
            if (!match) return buildDefaultTensorViewSpec(axisShape, hiddenIndices);
            cursor += match.lower.length;
            if (seenAxes.has(match.axis)) return buildDefaultTensorViewSpec(axisShape, hiddenIndices);
            seenAxes.add(match.axis);
            parsedAxes.push(match.axis);
        }
        if (parsedAxes.length === 0) return buildDefaultTensorViewSpec(axisShape, hiddenIndices);
        const label = parsedAxes.map((axis) => axisLabels[axis]).join('');
        const size = productForAxes(parsedAxes, axisShape);
        outlineSlots.push(parsedAxes);
        outlineShape.push(size);
        if (hasLower) {
            hiddenAxes.push(...parsedAxes);
            hiddenGroups.push({
                token: label.toLowerCase(),
                axes: parsedAxes,
                size,
                value: Math.min(size - 1, flattenAxesIndex(parsedAxes, nextHiddenIndices, axisShape)),
                outlineAxis,
            });
            canonicalTokens.push(label.toLowerCase());
        } else {
            displaySlots.push(parsedAxes);
            displayToOutline.push(outlineAxis);
            visibleAxes.push(...parsedAxes);
            canonicalTokens.push(label);
        }
    }

    if (seenAxes.size !== rank) {
        return buildDefaultTensorViewSpec(axisShape, hiddenIndices);
    }

    const displayShape = displaySlots.map((slot) => {
        if (slot === null) return 1;
        return productForAxes(slot, axisShape);
    });
    return {
        axisShape,
        axisLabels,
        displaySlots,
        outlineSlots,
        displayToOutline,
        visibleAxes,
        hiddenAxes,
        hiddenGroups,
        hiddenIndices: nextHiddenIndices,
        visibleText: canonicalTokens.join(' '),
        displayShape,
        outlineShape,
    };
}

function mapDisplayToFullCoords(displayCoord: TensorCoord, spec: TensorViewSpec): TensorCoord {
    const full = spec.hiddenIndices.slice();
    spec.displaySlots.forEach((axes, displayAxis) => {
        if (axes === null) return;
        const dim = Math.max(1, spec.displayShape[displayAxis] ?? 1);
        const raw = Number(displayCoord[displayAxis] ?? 0);
        const linear = Math.min(dim - 1, Math.max(0, Number.isFinite(raw) ? Math.round(raw) : 0));
        const coords = unflattenAxesIndex(linear, axes, spec.axisShape);
        axes.forEach((axis, axisIdx) => {
            full[axis] = coords[axisIdx] ?? 0;
        });
    });
    return full;
}

function mapDisplayToOutlineCoords(displayCoord: TensorCoord, spec: TensorViewSpec): TensorCoord {
    const outline = new Array(spec.outlineSlots.length).fill(0);
    spec.hiddenGroups.forEach((group) => {
        const max = Math.max(0, group.size - 1);
        outline[group.outlineAxis] = Math.min(max, Math.max(0, Number(group.value) || 0));
    });
    spec.displaySlots.forEach((axes, displayAxis) => {
        const outlineAxis = spec.displayToOutline[displayAxis];
        if (outlineAxis === undefined) return;
        if (axes === null) {
            outline[outlineAxis] = 0;
            return;
        }
        const dim = Math.max(1, spec.displayShape[displayAxis] ?? 1);
        const raw = Number(displayCoord[displayAxis] ?? 0);
        outline[outlineAxis] = Math.min(dim - 1, Math.max(0, Number.isFinite(raw) ? Math.round(raw) : 0));
    });
    return outline;
}

function isLayoutPreservingView(spec: TensorViewSpec, fullShape: number[]): boolean {
    if (spec.displaySlots.length !== fullShape.length) return false;
    for (let axis = 0; axis < fullShape.length; axis += 1) {
        const slot = spec.displaySlots[axis];
        if (!slot || slot.length !== 1 || slot[0] !== axis) return false;
    }
    return true;
}

function shouldUseFullLayoutPosition(spec: TensorViewSpec, _fullShape: number[]): boolean {
    return spec.hiddenAxes.length > 0;
}

function computeViewPlacementOffset(spec: TensorViewSpec, fullShape: number[]): Coord3 {
    if (shouldUseFullLayoutPosition(spec, fullShape)) return [0, 0, 0];
    if (!isLayoutPreservingView(spec, fullShape) || spec.hiddenAxes.length === 0) return [0, 0, 0];
    const displayOrigin = new Array(spec.displaySlots.length).fill(0);
    const fullOrigin = mapDisplayToFullCoords(displayOrigin, spec);
    const displayPos = positionForTensorCoord(displayOrigin, spec.displayShape);
    const fullPos = positionForTensorCoord(fullOrigin, fullShape);
    return [
        fullPos.x - displayPos.x,
        fullPos.y - displayPos.y,
        fullPos.z - displayPos.z,
    ];
}

function buildTensorViewPreview(spec: TensorViewSpec): string {
    const rank = spec.axisLabels.length;
    const hiddenSet = new Set(spec.hiddenAxes);
    const slices = Array.from({ length: rank }, (_, axis) => (
        hiddenSet.has(axis) ? String(spec.hiddenIndices[axis] ?? 0) : ':'
    ));
    let expr = `tensor[${slices.join(', ')}]`;
    const keptAxes = Array.from({ length: rank }, (_, axis) => axis).filter((axis) => !hiddenSet.has(axis));
    const desiredAxes = spec.displaySlots
        .filter((axes): axes is number[] => axes !== null)
        .flat();
    if (desiredAxes.length > 1) {
        const permute = desiredAxes.map((axis) => keptAxes.indexOf(axis));
        const isIdentity = permute.every((idx, i) => idx === i);
        if (!isIdentity) expr += `.permute(${permute.join(', ')})`;
    }
    if (spec.displaySlots.some((axes) => axes === null || axes.length !== 1)) {
        expr += `.reshape(${spec.displayShape.join(', ')})`;
    }
    return expr;
}

function arraysEqual(a: number[] = [], b: number[] = []): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function coordKey(coords: TensorCoord): string {
    return coords.join(',');
}

function applyValueColormap(
    mesh: TensorMesh,
    cache: PayloadCache,
    paint: (color: ThreeColor, coords: TensorCoord, val: number, t: number) => void,
): void {
    if (!mesh || !cache) return;
    const min = cache.scaleMin, max = cache.scaleMax, denom = max - min || 1;
    const count = mesh.count;
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
        const coords = getMeshFullCoords(mesh, i);
        const val = sampleValueFromCache(cache, coords);
        const t = clamp01((val - min) / denom);
        paint(c, coords, val, t);
        mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

function applyCoordColormap(
    mesh: TensorMesh,
    paint: (color: ThreeColor, coords: TensorCoord, index: number) => void,
): void {
    if (!mesh || !paint) return;
    const coordsList = mesh.userData.coords_full || mesh.userData.coords_display || mesh.userData.coords;
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
    const indexExpr = (Array.isArray(coords) ? coords : []).map((coord) => `[${coord}]`).join('');
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
    isHighlighted: (coords: TensorCoord) => boolean,
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
        if (!entry || entry.length < 2) return;
        const coord = entry.slice(0, -1).map((v) => Number(v));
        const count = entry[entry.length - 1];
        const safeCount = Number(count) || 0;
        map.set(coordKey(coord), safeCount);
        if (safeCount > maxCount) maxCount = safeCount;
    });
    maxCount = Math.max(maxCount, Number(payload?.max_count) || 0);
    return { map, maxCount };
}

function normalizeProgramSubsets(payload: ProgramSubsetsPayload | null): ProgramSubsetState {
    // normalize subset payload into lookup maps
    const subsetMap = new Map<string, string>();
    const subsets = (payload?.subsets || {}) as Record<string, number[][]>;
    const subsetCount = Number(payload?.subset_count) || Object.keys(subsets).length;
    const countMap = new Map<string, number>();
    let maxCount = Number(payload?.max_count) || 0;
    (payload?.coords || []).forEach((entry) => {
        if (!entry || entry.length < 2) return;
        const coord = entry.slice(0, -1).map((v) => Number(v));
        const key = entry[entry.length - 1];
        subsetMap.set(coordKey(coord), String(key));
    });
    (payload?.counts || []).forEach((entry) => {
        if (!entry || entry.length < 2) return;
        const coord = entry.slice(0, -1).map((v) => Number(v));
        const count = entry[entry.length - 1];
        const safeCount = Number(count) || 0;
        countMap.set(coordKey(coord), safeCount);
        if (safeCount > maxCount) maxCount = safeCount;
    });
    return { subsetMap, subsets, subsetCount, countMap, maxCount };
}

function buildSubsetHues(subsets: Record<string, number[][]>): Map<string, number> {
    const keys = Object.keys(subsets || {}).sort();
    const hues = new Map<string, number>();
    const used: number[] = [];
    const GOLDEN = 0.618033988749895;
    const minSep = 1 / Math.min(36, Math.max(8, keys.length * 2));
    keys.forEach((key) => {
        let hash = 2166136261;
        for (let i = 0; i < key.length; i += 1) {
            hash ^= key.charCodeAt(i);
            hash = Math.imul(hash, 16777619);
        }
        let hue = (hash >>> 0) / 4294967296;
        let tries = 0;
        const tooClose = (candidate: number): boolean => {
            return used.some((h) => {
                const d = Math.abs(candidate - h);
                return Math.min(d, 1 - d) < minSep;
            });
        };
        while (tooClose(hue) && tries < 128) { // avoid visible repeats
            hue = (hue + GOLDEN) % 1;
            tries += 1;
        }
        used.push(hue);
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
        const count = coords ? countMap.get(coordKey(coords)) || 0 : 0;
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
        const count = coords ? countMap.get(coordKey(coords)) || 0 : 0;
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
        const key = coords ? subsetMap.get(coordKey(coords)) : null;
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
        const key = coords ? subsetMap.get(coordKey(coords)) : null;
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

function parseDescriptorHighlight(
    highlights: TensorPayload['highlights'] | null | undefined,
): DescriptorHighlight | null {
    if (!highlights || highlights.type !== 'descriptor') return null;
    const rank = Math.max(
        highlights.start?.length || 0,
        highlights.shape?.length || 0,
        highlights.stride?.length || 0,
    );
    if (rank <= 0) return null;
    const start = Array.from({ length: rank }, (_, axis) => Number(highlights.start?.[axis] ?? 0));
    const shape = Array.from({ length: rank }, (_, axis) => Number(highlights.shape?.[axis] ?? 0));
    const stride = Array.from({ length: rank }, (_, axis) => Math.max(1, Math.abs(Number(highlights.stride?.[axis] ?? 1))));
    if (shape.some((dim) => dim <= 0)) return null;
    return {
        start,
        shape,
        stride,
    };
}

function descriptorContainsCoord(descriptor: DescriptorHighlight, coords: TensorCoord): boolean {
    const inAxis = (coord: number, axisStart: number, axisShape: number, axisStride: number): boolean => {
        if (axisShape <= 0) return false;
        const delta = coord - axisStart;
        if (delta < 0 || delta % axisStride !== 0) return false;
        return (delta / axisStride) < axisShape;
    };
    for (let axis = 0; axis < descriptor.shape.length; axis += 1) {
        if (
            !inAxis(
                coords[axis] ?? 0,
                descriptor.start[axis] ?? 0,
                descriptor.shape[axis] ?? 0,
                descriptor.stride[axis] ?? 1,
            )
        ) {
            return false;
        }
    }
    return true;
}

function projectDescriptorForView(descriptor: DescriptorHighlight, spec: TensorViewSpec): DescriptorHighlight | null {
    for (let i = 0; i < spec.hiddenAxes.length; i += 1) {
        const axis = spec.hiddenAxes[i];
        if (axis === undefined) continue;
        const idx = spec.hiddenIndices[axis] ?? 0;
        const start = descriptor.start[axis] ?? 0;
        const shape = descriptor.shape[axis] ?? 0;
        const stride = descriptor.stride[axis] ?? 1;
        if (shape <= 0) return null;
        const delta = idx - start;
        if (delta < 0 || delta % stride !== 0) return null;
        if ((delta / stride) >= shape) return null;
    }
    return {
        start: spec.displaySlots.map((axes) => {
            if (axes === null) return 0;
            if (axes.length === 1) {
                const axis = axes[0];
                return descriptor.start[axis ?? 0] ?? 0;
            }
            return flattenAxesIndex(axes, descriptor.start, spec.axisShape);
        }),
        shape: spec.displaySlots.map((axes) => {
            if (axes === null) return 1;
            if (axes.length === 1) {
                const axis = axes[0];
                return descriptor.shape[axis ?? 0] ?? 0;
            }
            return axes.reduce((acc, axis) => acc * Math.max(1, descriptor.shape[axis] ?? 1), 1);
        }),
        stride: spec.displaySlots.map((axes) => {
            if (axes === null) return 1;
            if (axes.length === 1) {
                const axis = axes[0];
                return descriptor.stride[axis ?? 0] ?? 1;
            }
            return 1;
        }),
    };
}

function reorderDescriptorForTensor<T>(values: T[], tensorRank: number, fallback: T): T[] {
    const out: T[] = [];
    for (let i = 0; i < tensorRank; i += 1) {
        out.push(values[i] ?? fallback);
    }
    return out;
}

function getHighlightPredicate(
    highlights: TensorPayload['highlights'] | null | undefined,
): ((coords: TensorCoord) => boolean) | null {
    if (!highlights) return null;
    const descriptor = parseDescriptorHighlight(highlights);
    if (descriptor) {
        return (coords) => descriptorContainsCoord(descriptor, coords);
    }
    if (Array.isArray(highlights.data) && highlights.data.length) {
        const set = new Set();
        highlights.data.forEach((c) => { set.add(c.join(',')); });
        return (coords) => set.has(coords.join(','));
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
    const { state, tensors } = ctx;
    const supportsAllPrograms = ctx.supportsAllPrograms;
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
    applyDimmedColormap(aGroup.userData.mesh, aCache, 'A', (coords) => coords[0] === row);
    applyDimmedColormap(bGroup.userData.mesh, bCache, 'B', (coords) => coords[1] === col);
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
        row.style.alignItems = 'flex-start';
        row.style.gap = '6px';
        row.style.whiteSpace = 'normal';
        const swatch = document.createElement('span');
        swatch.style.display = 'inline-block';
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.borderRadius = '3px';
        swatch.style.background = `#${color.getHexString()}`;
        swatch.style.flex = '0 0 12px';
        swatch.style.marginTop = '2px';
        row.appendChild(swatch);
        const text = document.createElement('span');
        text.textContent = label;
        text.style.whiteSpace = 'normal';
        text.style.overflowWrap = 'anywhere';
        text.style.lineHeight = '1.25';
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
    subsets: Record<string, number[][]>,
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
        row.style.alignItems = 'flex-start';
        row.style.gap = '6px';
        row.style.whiteSpace = 'normal';
        const swatch = document.createElement('span');
        swatch.style.display = 'inline-block';
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.borderRadius = '3px';
        swatch.style.background = `#${color.getHexString()}`;
        swatch.style.flex = '0 0 12px';
        swatch.style.marginTop = '2px';
        row.appendChild(swatch);
        const text = document.createElement('span');
        text.textContent = label;
        text.style.whiteSpace = 'normal';
        text.style.overflowWrap = 'anywhere';
        text.style.lineHeight = '1.25';
        row.appendChild(text);
        rows.appendChild(row);
    };
    addRow('none', base);
    Object.keys(subsets || {}).forEach((key) => {
        const pids = subsets[key] || [];
        const label = pids.length
            ? pids.map((pid) => `(${(pid || []).join(',')})`).join(' ')
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

function worldPositionForTensorCoord(
    tensorGroup: TensorGroup,
    coord: number[],
    shape: number[],
): ThreeVector3 {
    const localPos = positionForTensorCoord(coord, shape);
    const world = localPos.clone();
    world.add(tensorGroup.position);
    return world;
}

function dominantAxis(delta: ThreeVector3): 'x' | 'y' | 'z' {
    const ax = Math.abs(delta.x);
    const ay = Math.abs(delta.y);
    const az = Math.abs(delta.z);
    if (ay >= ax && ay >= az) return 'y';
    if (az >= ax && az >= ay) return 'z';
    return 'x';
}

function defaultDimColorForAxis(_rank: number, axis: number): string {
    return defaultAxisColor(axis);
}

function fallbackWorldDirection(rank: number, axis: number): ThreeVector3 {
    const mapped = (rank - 1 - axis) % 3;
    if (mapped === 0) return new THREE.Vector3(1, 0, 0);
    if (mapped === 1) return new THREE.Vector3(0, -1, 0);
    return new THREE.Vector3(0, 0, -1);
}

function axisWorldKey(rank: number, axis: number): number {
    return (rank - 1 - axis) % 3;
}

function axisWorldName(worldKey: number): 'x' | 'y' | 'z' {
    if (worldKey === 1) return 'y';
    if (worldKey === 2) return 'z';
    return 'x';
}

function colorForAxisFamily(worldKey: number, familyPos: number, familyCount: number): string {
    const p = Math.max(1, familyCount);
    const level = Math.min(p, Math.max(1, familyPos + 1));
    const t = level / p;
    const color = new THREE.Color();
    if (worldKey === 1) color.setRGB(0, t, 0);
    else if (worldKey === 2) color.setRGB(0, 0, t);
    else color.setRGB(t, 0, 0);
    return `#${color.getHexString()}`;
}

function defaultDimColorsForShape(rank: number): string[] {
    const familyAxes = new Map<number, number[]>();
    for (let axis = 0; axis < rank; axis += 1) {
        const key = axisWorldKey(rank, axis);
        const prev = familyAxes.get(key) || [];
        prev.push(axis);
        familyAxes.set(key, prev);
    }
    return Array.from({ length: rank }, (_, axis) => {
        const key = axisWorldKey(rank, axis);
        const family = familyAxes.get(key) || [];
        const familyPos = Math.max(0, family.indexOf(axis));
        return colorForAxisFamily(key, familyPos, family.length);
    });
}

function extensionDirectionFor(worldAxis: 'x' | 'y' | 'z', order: number, diagonal = false): ThreeVector3 {
    const baseByAxis = {
        x: [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)],
        y: [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1)],
        z: [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0)],
    };
    const [a, b] = baseByAxis[worldAxis];
    if (!a || !b) return new THREE.Vector3(0, 1, 0);
    if (!diagonal) {
        const orthogonal = [a, b, a.clone().negate(), b.clone().negate()];
        return orthogonal[order % orthogonal.length].clone();
    }
    const d1 = a.clone().add(b).normalize();
    const d2 = a.clone().sub(b).normalize();
    const diagonalDirs = [d1, d2, d1.clone().negate(), d2.clone().negate()];
    return diagonalDirs[order % diagonalDirs.length].clone();
}

function addAxisDimensionLines(
    scene: ThreeScene,
    tensorGroup: TensorGroup,
    axisSizes: number[],
    axisStarts: number[],
    axisStrides: number[],
    dimColors: string[] = [],
    options: {
        offsetBase?: number;
        opacity?: number;
        colorOverride?: string;
        labelPrefix?: string;
        shapeOverride?: number[];
    } = {},
): any[] {
    const mesh = tensorGroup?.userData?.mesh;
    if (!mesh) return [];
    const shapeRaw = normalizeViewShape(options.shapeOverride || mesh.userData.shape_raw || []);
    if (shapeRaw.length === 0) return [];
    const { offsetBase = (CUBE_SIZE + GAP) * 1.5, opacity, colorOverride, labelPrefix = '' } = options;
    const groups: any[] = [];
    const tensorCenter = new THREE.Box3().setFromObject(tensorGroup).getCenter(new THREE.Vector3());
    const entries: Array<{
        axis: number;
        axisWorld: 'x' | 'y' | 'z';
        extentStart: ThreeVector3;
        extentEnd: ThreeVector3;
        span: number;
        color: string;
        label: string;
    }> = [];
    const startCoord = axisStarts.map((v, axis) => {
        const dim = shapeRaw[axis] ?? 1;
        return Math.min(dim - 1, Math.max(0, Number(v) || 0));
    });
    const familyAxes = new Map<number, number[]>();
    for (let axis = 0; axis < axisSizes.length; axis += 1) {
        const key = axisWorldKey(shapeRaw.length, axis);
        const prev = familyAxes.get(key) || [];
        prev.push(axis);
        familyAxes.set(key, prev);
    }
    for (let axis = 0; axis < axisSizes.length; axis += 1) {
        const size = Math.max(1, Number(axisSizes[axis] ?? 1));
        const stride = Math.max(1, Number(axisStrides[axis] ?? 1));
        const start = startCoord.slice();
        const end = startCoord.slice();
        const targetWorldKey = axisWorldKey(shapeRaw.length, axis);
        const axisWorld = axisWorldName(targetWorldKey);
        if (targetWorldKey === 2) {
            const zAxes: number[] = [];
            const xAxes: number[] = [];
            const yAxes: number[] = [];
            for (let dimAxis = 0; dimAxis < shapeRaw.length; dimAxis += 1) {
                const key = axisWorldKey(shapeRaw.length, dimAxis);
                if (key === 2) zAxes.push(dimAxis);
                else if (key === 1) yAxes.push(dimAxis);
                else xAxes.push(dimAxis);
            }
            const setMin = (dimAxis: number): void => {
                start[dimAxis] = 0;
                end[dimAxis] = 0;
            };
            xAxes.forEach(setMin);
            yAxes.forEach(setMin);
            const zPos = zAxes.indexOf(axis);
            for (let i = 0; i < zAxes.length; i += 1) {
                const zAxis = zAxes[i];
                if (zAxis === undefined) continue;
                const zMax = Math.max(0, (shapeRaw[zAxis] ?? 1) - 1);
                const zSize = Math.max(1, Number(axisSizes[zAxis] ?? 1));
                const zStride = Math.max(1, Number(axisStrides[zAxis] ?? 1));
                const zEnd = Math.min(zMax, Math.max(0, (start[zAxis] ?? 0) + (zSize - 1) * zStride));
                end[zAxis] = i >= zPos ? zEnd : (start[zAxis] ?? 0);
            }
        } else {
            const max = shapeRaw[axis] ?? 1;
            end[axis] = Math.min(max - 1, Math.max(0, (start[axis] ?? 0) + (size - 1) * stride));
            for (let companion = axis + 1; companion < axisSizes.length; companion += 1) {
                if (axisWorldKey(shapeRaw.length, companion) !== targetWorldKey) continue;
                const companionSize = Math.max(1, Number(axisSizes[companion] ?? 1));
                const companionStride = Math.max(1, Number(axisStrides[companion] ?? 1));
                const companionMax = shapeRaw[companion] ?? 1;
                end[companion] = Math.min(
                    companionMax - 1,
                    Math.max(0, (start[companion] ?? 0) + (companionSize - 1) * companionStride),
                );
            }
        }
        const startPos = worldPositionForTensorCoord(tensorGroup, start, shapeRaw);
        const endPos = worldPositionForTensorCoord(tensorGroup, end, shapeRaw);
        const delta = new THREE.Vector3().subVectors(endPos, startPos);
        const axisDir = delta.lengthSq() > 1e-9
            ? delta.clone().normalize()
            : fallbackWorldDirection(shapeRaw.length, axis);
        const extentStart = startPos.clone().add(axisDir.clone().multiplyScalar(-CUBE_SIZE / 2));
        const extentEnd = endPos.clone().add(axisDir.clone().multiplyScalar(CUBE_SIZE / 2));
        const family = familyAxes.get(targetWorldKey) || [];
        const familyPos = Math.max(0, family.indexOf(axis));
        const familyColor = colorForAxisFamily(targetWorldKey, familyPos, family.length);
        const color = colorOverride || dimColors[axis] || familyColor || defaultDimColorForAxis(shapeRaw.length, axis);
        const labelCore = `${getAxisLabel(axis)}: ${size}`;
        entries.push({
            axis,
            axisWorld,
            extentStart,
            extentEnd,
            span: extentStart.distanceTo(extentEnd),
            color,
            label: `${labelPrefix}${labelCore}`,
        });
    }
    const groupByAxis: Record<'x' | 'y' | 'z', typeof entries> = { x: [], y: [], z: [] };
    entries.forEach((entry) => {
        groupByAxis[entry.axisWorld].push(entry);
    });
    const linearStep = (CUBE_SIZE + GAP) * 1.35;
    const xyCornerDirection = new THREE.Vector3();
    (['x', 'y', 'z'] as const).forEach((axisWorld) => {
        const axisEntries = groupByAxis[axisWorld];
        if (axisEntries.length === 0) return;
        axisEntries.sort((a, b) => a.axis - b.axis);
        axisEntries.forEach((entry, lineIdx) => {
            const reverseIdx = axisEntries.length - 1 - lineIdx;
            const useDiagonal = axisWorld === 'z';
            const directionOrder = useDiagonal ? 3 : 0; // z family anchored to top-left
            let extensionDirection = extensionDirectionFor(axisWorld, directionOrder, useDiagonal);
            const mid = entry.extentStart.clone().add(entry.extentEnd).multiplyScalar(0.5);
            const outward = mid.clone().sub(tensorCenter);
            if (useDiagonal) {
                extensionDirection = extensionDirectionFor(axisWorld, directionOrder, true);
            } else {
                if (outward.dot(extensionDirection) < 0) {
                    extensionDirection = extensionDirection.clone().negate();
                }
                if (axisWorld === 'x' || axisWorld === 'y') {
                    xyCornerDirection.add(extensionDirection);
                }
            }
            const edgeShift = extensionDirection.clone().multiplyScalar(CUBE_SIZE / 2);
            groups.push(createCadDimension(
                scene,
                entry.extentStart.clone().add(edgeShift),
                entry.extentEnd.clone().add(edgeShift),
                entry.label,
                axisWorld,
                entry.color,
                {
                    offset: offsetBase + reverseIdx * linearStep,
                    extensionOffset: 0,
                    extensionLength: (CUBE_SIZE + GAP) * 0.2,
                    textOffset: 0,
                    extensionDirection,
                    opacity: opacity ?? 0.9,
                },
            ));
        });
    });
    return groups;
}

function addDimensionLines(
    scene: ThreeScene,
    tensorGroup: TensorGroup,
    dimColors: string[] = [],
    spec: TensorViewSpec | null = null,
): any[] {
    const mesh = tensorGroup?.userData?.mesh;
    if (!mesh) return [];
    const shapeRaw = normalizeViewShape(spec?.outlineShape || mesh.userData.shape_raw || []);
    return addAxisDimensionLines(
        scene,
        tensorGroup,
        shapeRaw,
        new Array(shapeRaw.length).fill(0),
        new Array(shapeRaw.length).fill(1),
        dimColors,
        { shapeOverride: shapeRaw },
    );
}

function createSliceReferenceOutline(shapeRaw: number[], baseColor: ColorInput): any {
    const size = tensorBoundsSizeForShape(shapeRaw);
    const color = baseColor instanceof THREE.Color
        ? baseColor.clone().lerp(new THREE.Color('#ffffff'), 0.45)
        : new THREE.Color(baseColor || '#cbd5e1').lerp(new THREE.Color('#ffffff'), 0.45);
    const material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: 0.35,
        depthTest: true,
        depthWrite: false,
    });
    const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z)),
        material,
    );
    outline.frustumCulled = false;
    outline.renderOrder = 1500;
    return outline;
}

function buildSliceReferenceOutlines(
    scene: ThreeScene,
    tensors: Map<string, TensorGroup>,
    configByName: Map<string, TensorConfig>,
    tensorViews: Map<string, TensorViewSpec> | null = null,
): any[] {
    const groups: any[] = [];
    tensors.forEach((tensorGroup, name) => {
        const cfg = configByName.get(name);
        if (!cfg) return;
        const spec = tensorViews?.get(name) || null;
        const shape = spec?.outlineShape?.length
            ? normalizeViewShape(spec.outlineShape)
            : normalizeViewShape(cfg.shape || []);
        const outline = createSliceReferenceOutline(shape, tensorGroup.userData.mesh.userData.color_base || '#cbd5e1');
        const [bx = 0, by = 0, bz = 0] = cfg.position || [0, 0, 0];
        outline.position.set(bx, by, bz);
        scene.add(outline);
        groups.push(outline);
    });
    return groups;
}

function clearSliceReferenceOutlines(ctx: VizContext): void {
    ctx.sliceOutlineGroups.forEach((group) => {
        ctx.scene.remove(group);
        const geometry = group?.geometry;
        const material = group?.material;
        if (geometry?.dispose) geometry.dispose();
        if (Array.isArray(material)) material.forEach((m: any) => m?.dispose?.());
        else if (material?.dispose) material.dispose();
    });
    ctx.sliceOutlineGroups = [];
}

function getDescriptorSelectionBounds(mesh: TensorMesh, descriptor: DescriptorHighlight): { min: ThreeVector3; max: ThreeVector3 } {
    const tensorShape = mesh.userData.shape;
    if (!descriptor || descriptor.shape.length < 1 || descriptor.shape.length > 3) {
        const fallback = new THREE.Vector3(0, 0, 0);
        return { min: fallback.clone(), max: fallback };
    }
    const spacing = CUBE_SIZE + GAP;
    const centerX = (tensorShape.width - 1) * spacing / 2;
    const centerY = -((tensorShape.height - 1) * spacing / 2);
    const centerZ = -((tensorShape.depth - 1) * spacing / 2);
    const axisFromDisplay = (axis: 'x' | 'y' | 'z'): number => {
        if (descriptor.shape.length === 1) return 0;
        if (descriptor.shape.length === 2) return axis === 'x' ? 1 : 0;
        if (axis === 'x') return 2;
        if (axis === 'y') return 1;
        return 0;
    };
    const sx = descriptor.start[axisFromDisplay('x')] ?? 0;
    const sy = descriptor.start[axisFromDisplay('y')] ?? 0;
    const sz = descriptor.start[axisFromDisplay('z')] ?? 0;
    const dx = descriptor.shape[axisFromDisplay('x')] ?? 1;
    const dy = descriptor.shape[axisFromDisplay('y')] ?? 1;
    const dz = descriptor.shape[axisFromDisplay('z')] ?? 1;
    const tx = descriptor.stride[axisFromDisplay('x')] ?? 1;
    const ty = descriptor.stride[axisFromDisplay('y')] ?? 1;
    const tz = descriptor.stride[axisFromDisplay('z')] ?? 1;
    const ex = sx + (dx - 1) * tx;
    const ey = sy + (dy - 1) * ty;
    const ez = sz + (dz - 1) * tz;
    const toX = (x: number): number => x * spacing - centerX;
    const toY = (y: number): number => -y * spacing - centerY;
    const toZ = (z: number): number => -z * spacing - centerZ;
    const half = CUBE_SIZE / 2;
    const min = new THREE.Vector3(
        Math.min(toX(sx), toX(ex)) - half,
        Math.min(toY(sy), toY(ey)) - half,
        Math.min(toZ(sz), toZ(ez)) - half,
    );
    const max = new THREE.Vector3(
        Math.max(toX(sx), toX(ex)) + half,
        Math.max(toY(sy), toY(ey)) + half,
        Math.max(toZ(sz), toZ(ez)) + half,
    );
    return { min, max };
}

function addDescriptorDimensionLines(
    scene: ThreeScene,
    tensorGroup: TensorGroup,
    highlights: TensorPayload['highlights'] | null | undefined,
    spec: TensorViewSpec | null = null,
): any[] {
    const mesh = tensorGroup?.userData?.mesh;
    if (!mesh) return [];
    const descriptorRaw = parseDescriptorHighlight(highlights);
    const descriptor = descriptorRaw && spec ? projectDescriptorForView(descriptorRaw, spec) : descriptorRaw;
    if (!descriptor) return [];
    const shapeRaw = mesh.userData.shape_raw || [];
    const rank = Math.min(shapeRaw.length, descriptor.shape.length);
    if (rank <= 0) return [];
    return addAxisDimensionLines(
        scene,
        tensorGroup,
        descriptor.shape.slice(0, rank),
        descriptor.start.slice(0, rank),
        descriptor.stride.slice(0, rank),
        [],
        {
            offsetBase: (CUBE_SIZE + GAP) * 0.85,
            opacity: 0.95,
            colorOverride: '#00b3ff',
        },
    );
}

function clearDescriptorDimensionLines(ctx: VizContext): void {
    ctx.descriptorDimLineGroups.forEach((group) => ctx.scene.remove(group));
    ctx.descriptorDimLineGroups = [];
}

function refreshDescriptorDimensionLines(ctx: VizContext): void {
    clearDescriptorDimensionLines(ctx);
    if (!SHOW_DESCRIPTOR_DIM_LINES) return;
    if (!ctx.showDimLines || ctx.state.allProgramsOn) return;
    ctx.tensors.forEach((group, name) => {
        const p = ctx.state.payloads.get(name);
        if (!p) return;
        const spec = ctx.state.tensorViews.get(name) || null;
        ctx.descriptorDimLineGroups.push(
            ...addDescriptorDimensionLines(ctx.scene, group, p.highlights, spec),
        );
    });
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
        const coords3 = (mesh.userData.coords?.[instanceId] ?? [0, 0, 0]) as Coord3;
        const coordsFull = getMeshFullCoords(mesh, instanceId);
        const coordsDisplay = getMeshDisplayCoords(mesh, instanceId);
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
        const val = cacheEntry ? sampleValueFromCache(cacheEntry, coordsFull) : 'Loading...';
        const currentShape = mesh.userData.shape_raw;

        let extraHtml = '';
        if (tensorName === 'Global' && ctx.supportsAllPrograms) {
            if (!state.programSubsets && !state.programCounts) {
                ctx.ensureProgramDataForHover?.();
            }
            const key = coordKey(coordsFull);
            if (state.programSubsets) {
                const subsetKey = state.programSubsets.subsetMap.get(key);
                const subset = subsetKey ? state.programSubsets.subsets?.[subsetKey] || [] : [];
                const label = subset.length
                    ? subset.map((pid) => `(${(pid || []).join(',')})`).join(' ')
                    : 'none';
                extraHtml = `<p>Programs: ${label}</p>`;
            } else if (state.programCounts) {
                const count = state.programCounts.map.get(key) || 0;
                extraHtml = `<p>Programs: ${count}</p>`;
            } else if (state.programDataLoading) {
                extraHtml = '<p>Programs: loading...</p>';
            }
        }
        updateSideMenu(sideMenu, tensorName, coordsFull, val, currentShape || null, extraHtml);
        if (ctx.type === 'Dot' && tensorName === 'C') {
            const row = Number(coordsDisplay[0] ?? 0);
            const col = Number(coordsDisplay[1] ?? 0);
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
        layoutBounds?: { width: number; height: number; depth?: number; center?: [number, number, number] };
        fitToTensors?: boolean;
        cameraPadding?: number;
    } = {},
): (() => void) | void {
    const { type = 'Load', colors = {}, tensorConfigs = [], dimColors = {}, showDimLines = true, viewState = null, layoutBounds = null, fitToTensors = true, cameraPadding = 1.15 } = options;
    const API_BASE = getApiBase();
    const initialToggles = getState().toggles;
    const configs = tensorConfigs.length > 0 ? tensorConfigs : [
        { name: 'Global', shape: op.global_shape || [], color: colors.GLOBAL || '#333', position: [0,0,0], endpoint: 'getLoadTensor' }
    ];
    const supportsAllPrograms = type === 'Load' || type === 'Store';
    const configByNameMap = new Map<string, TensorConfig>(configs.map((cfg) => [cfg.name, cfg as TensorConfig]));

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
        if (!canUseWebgl()) {
            // show a visible message when WebGL is disabled and skip initialization.
            return renderWebglWarning(containerElement);
        }
        const sideMenu = createSideMenu(containerElement);
        const histogramUI = createHistogramOverlay(containerElement, {
            title: `${type} Value Distribution`,
            apiBase: API_BASE,
            sources: configs.map(c => ({ value: c.name.toUpperCase(), label: `${c.name} Tensor` })),
            buildRequestBody: (s, b) => ({ uuid: op.uuid, source: s, bins: b }),
        });
        let scene: ThreeScene;
        let camera: ThreeCamera;
        let renderer: ThreeRenderer;
        try {
            ({ scene, camera, renderer } = setupScene(stage, 0x2f343d));
        } catch (err) {
            // webgl can still fail even after a feature test.
            return renderWebglWarning(containerElement);
        }
        const disposer = createDisposer();
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();
        const initialTensorViews = new Map<string, TensorViewSpec>();
        configs.forEach((cfg) => {
            const snapshot = viewState?.tensorViews?.[cfg.name];
            initialTensorViews.set(
                cfg.name,
                buildTensorViewSpec(
                    normalizeViewShape(cfg.shape || []),
                    snapshot?.visible || '',
                    snapshot?.hiddenIndices || [],
                ),
            );
        });
        const tensors = new Map<string, TensorGroup>();
        configs.forEach(cfg => {
            const shape = normalizeViewShape(cfg.shape || []);
            const spec = initialTensorViews.get(cfg.name) || buildTensorViewSpec(shape);
            const useFullLayoutPosition = shouldUseFullLayoutPosition(spec, shape);
            const group = createTensor(
                spec.displayShape,
                null,
                cfg.color,
                cfg.name,
                cubeGeometry,
                edgesGeometry,
                lineMaterial,
                {
                    mapDisplayCoordToFull: (coord) => mapDisplayToFullCoords(coord, spec),
                    ...(useFullLayoutPosition ? {
                        mapDisplayCoordToPosition: (coord: TensorCoord) => mapDisplayToOutlineCoords(coord, spec),
                        positionShape: spec.outlineShape,
                    } : {}),
                },
            ) as TensorGroup;
            const [bx = 0, by = 0, bz = 0] = cfg.position || [0, 0, 0];
            const [ox, oy, oz] = computeViewPlacementOffset(spec, shape);
            group.position.set(bx + ox, by + oy, bz + oz);
            if (cfg.endpoint) {
                group.userData.endpoint = cfg.endpoint;
            } else if (group.userData.endpoint) {
                delete group.userData.endpoint;
            }
            scene.add(group);
            tensors.set(cfg.name, group);
        });
        const hoverOutline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05)), new THREE.LineBasicMaterial({ color: COLOR_HOVER }));
        hoverOutline.visible = false;
        scene.add(hoverOutline);
        const dimLineGroups: any[] = [];
        if (showDimLines) {
            tensors.forEach((group, name) => {
                dimLineGroups.push(
                    ...addDimensionLines(scene, group, dimColors[name], initialTensorViews.get(name) || null),
                );
            });
        }
        const sliceOutlineGroups = buildSliceReferenceOutlines(scene, tensors, configByNameMap, initialTensorViews);
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
        if (sliceOutlineGroups.length > 0) {
            sliceOutlineGroups.forEach((group) => bounds.union(new THREE.Box3().setFromObject(group)));
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
            editTensorViewOn: !!initialToggles.editTensorView,
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
            tensorViews: new Map(initialTensorViews),
        };
        const highlightColor = (colors.HIGHLIGHT instanceof THREE.Color) ? colors.HIGHLIGHT : new THREE.Color(colors.HIGHLIGHT || 0x00b3ff);
        const ctx: VizContext = {
            type,
            shapeKey,
            containerElement,
            sideMenu,
            histogramUI,
            stage,
            API_BASE,
            op,
            configByName: configByNameMap,
            cubeGeometry,
            edgesGeometry,
            scene,
            camera,
            renderer,
            tensors,
            orbitControls,
            lineMaterial,
            state,
            disposer,
            raycaster: new THREE.Raycaster(),
            mouse: new THREE.Vector2(),
            legendContainer: null,
            dimLineGroups,
            sliceOutlineGroups,
            descriptorDimLineGroups: [],
            showDimLines,
            supportsAllPrograms,
            highlightColor,
            tensorViewControls: null,
            requestRender: () => {},
            applyBackgroundTheme: () => {},
            destroyLegends: () => {},
            createLegends: () => {},
        };
        ctx.requestRender = () => {
            if (state.rafId !== null) { state.renderPending = true; return; }
            state.rafId = requestAnimationFrame(() => { state.rafId = null; orbitControls.update(); syncClipPlanes(); renderer.render(scene, camera); if (state.renderPending) { state.renderPending = false; ctx.requestRender(); } });
        };
        const handleTextSync = (): void => { ctx.requestRender(); };
        window.addEventListener('triton-viz-text-sync', handleTextSync);
        disposer.add(() => window.removeEventListener('triton-viz-text-sync', handleTextSync));
        let lastWidth = 0;
        let lastHeight = 0;
        const resizeRenderer = (): void => {
            const width = Math.max(1, Math.floor(stage.clientWidth));
            const height = Math.max(1, Math.floor(stage.clientHeight));
            if (width === lastWidth && height === lastHeight) return;
            lastWidth = width; lastHeight = height;
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            ctx.requestRender();
        };
        const resizeObserver = new ResizeObserver(() => resizeRenderer());
        resizeObserver.observe(stage);
        disposer.add(() => resizeObserver.disconnect()); // keep webgl canvas in sync with layout resizes
        resizeRenderer();
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
        const serializeTensorViews = (): Record<string, TensorViewSnapshot> => {
            const out: Record<string, TensorViewSnapshot> = {};
            state.tensorViews.forEach((spec, name) => {
                out[name] = {
                    visible: spec.visibleText,
                    hiddenIndices: spec.hiddenIndices.slice(),
                };
            });
            return out;
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
            state.allProgramsOn = supportsAllPrograms ? !!nextState.allProgramsOn : false;
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
            tensorViews: Record<string, TensorViewSnapshot>;
        } => {
            return {
                camera: {
                    position: [camera.position.x, camera.position.y, camera.position.z],
                    quaternion: [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w],
                },
                target: [orbitControls.target.x, orbitControls.target.y, orbitControls.target.z],
                colorizeOn: state.colorizeOn,
                allProgramsOn: state.allProgramsOn,
                tensorViews: serializeTensorViews(),
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
            clearSliceReferenceOutlines(ctx);
            clearDescriptorDimensionLines(ctx);
            if (stage.parentElement) stage.parentElement.removeChild(stage);
            if (sideMenu.parentElement) sideMenu.parentElement.removeChild(sideMenu);
            if (ctx.tensorViewControls?.parentElement) ctx.tensorViewControls.parentElement.removeChild(ctx.tensorViewControls);
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
    const {
        state,
        tensors,
        sideMenu,
        requestRender,
        applyBackgroundTheme,
        createLegends,
        destroyLegends,
        configByName,
        cubeGeometry,
        edgesGeometry,
        scene,
        stage,
    } = vizCache;
    const syncOpControlState = (): void => {
        if (!window.setOpControlState) return;
        window.setOpControlState({
            colorize: state.colorizeOn,
            histogram: vizCache.histogramUI.overlay.style.display === 'block',
            allPrograms: state.allProgramsOn,
            editTensorView: state.editTensorViewOn,
        });
    };
    state.payloads.clear();
    state.programCounts = null;
    state.programSubsets = null;
    state.programSubsetHues = null;
    clearDescriptorDimensionLines(vizCache);
    sideMenu.innerHTML = '';
    const opUuid = op.uuid ?? null;
    window.current_op_uuid = opUuid;
    const getBaseCountColor = () => tensors.values().next().value?.userData?.mesh?.userData?.color_base || '#333333';
    const renderShapeLegend = (): void => {
        createShapeLegend(containerElement, Array.from(tensors.entries()).map(([name, group]) => {
            const entry: {
                name: string;
                color: string;
                shape?: number[];
                dimColors?: string[];
                descriptor?: { shape: number[]; stride?: number[]; color: string; dimColors?: string[] };
            } = {
                name: name === 'Global' ? type : `Matrix ${name}`,
                color: '#' + group.userData.mesh.userData.color_base.getHexString(),
            };
            const shape = group.userData.mesh.userData.shape_raw;
            if (shape) {
                entry.shape = shape;
                entry.dimColors = defaultDimColorsForShape(shape.length);
            }
            const dimColor = dimColors?.[name];
            if (dimColor) entry.dimColors = dimColor;
            const spec = state.tensorViews.get(name) || null;
            const descriptorRaw = parseDescriptorHighlight(state.payloads.get(name)?.highlights);
            const descriptor = descriptorRaw && spec ? projectDescriptorForView(descriptorRaw, spec) : descriptorRaw;
            if (descriptor && shape && shape.length > 0) {
                const selectionColor = vizCache.highlightColor instanceof THREE.Color
                    ? `#${vizCache.highlightColor.getHexString()}`
                    : String(vizCache.highlightColor || '#00b3ff');
                const selectionDimColors = reorderDescriptorForTensor(
                    [DESCRIPTOR_AXIS_COLORS.x, DESCRIPTOR_AXIS_COLORS.y, DESCRIPTOR_AXIS_COLORS.z],
                    descriptor.shape.length,
                    selectionColor,
                );
                entry.descriptor = {
                    shape: descriptor.shape.slice(),
                    color: selectionColor,
                    dimColors: selectionDimColors,
                };
            }
            return entry;
        }));
    };
    const rebuildDimensionLines = (): void => {
        vizCache.dimLineGroups.forEach((group) => scene.remove(group));
        vizCache.dimLineGroups = [];
        if (!vizCache.showDimLines) return;
        tensors.forEach((group, name) => {
            vizCache.dimLineGroups.push(
                ...addDimensionLines(scene, group, dimColors[name], state.tensorViews.get(name) || null),
            );
        });
    };
    const rebuildSliceReferenceOutlines = (): void => {
        clearSliceReferenceOutlines(vizCache);
        vizCache.sliceOutlineGroups.push(
            ...buildSliceReferenceOutlines(scene, tensors, configByName, state.tensorViews),
        );
    };
    const rebuildTensorFromView = (name: string): void => {
        const cfg = configByName.get(name);
        const oldGroup = tensors.get(name);
        if (!cfg || !oldGroup) return;
        const shape = normalizeViewShape(cfg.shape || []);
        const spec = state.tensorViews.get(name) || buildTensorViewSpec(shape);
        const useFullLayoutPosition = shouldUseFullLayoutPosition(spec, shape);
        const nextGroup = createTensor(
            spec.displayShape,
            null,
            oldGroup.userData.mesh.userData.color_base || cfg.color,
            name,
            cubeGeometry,
            edgesGeometry,
            vizCache.lineMaterial,
            {
                mapDisplayCoordToFull: (coord) => mapDisplayToFullCoords(coord, spec),
                ...(useFullLayoutPosition ? {
                    mapDisplayCoordToPosition: (coord: TensorCoord) => mapDisplayToOutlineCoords(coord, spec),
                    positionShape: spec.outlineShape,
                } : {}),
            },
        ) as TensorGroup;
        const [bx = 0, by = 0, bz = 0] = cfg.position || [0, 0, 0];
        const [ox, oy, oz] = computeViewPlacementOffset(spec, shape);
        nextGroup.position.set(bx + ox, by + oy, bz + oz);
        const endpoint = oldGroup.userData.endpoint || cfg.endpoint;
        if (endpoint) nextGroup.userData.endpoint = endpoint;
        else if (nextGroup.userData.endpoint) delete nextGroup.userData.endpoint;
        scene.remove(oldGroup);
        const oldMaterial = (oldGroup.userData.mesh as any)?.material;
        if (oldMaterial && typeof oldMaterial.dispose === 'function') oldMaterial.dispose();
        scene.add(nextGroup);
        tensors.set(name, nextGroup);
    };
    const rebuildAllTensorsFromView = (): void => {
        Array.from(tensors.keys()).forEach((name) => rebuildTensorFromView(name));
        rebuildSliceReferenceOutlines();
        rebuildDimensionLines();
        renderShapeLegend();
        restoreTensorColors(vizCache);
        refreshDescriptorDimensionLines(vizCache);
        requestRender();
    };
    const ensureTensorViewsInitialized = (): void => {
        configByName.forEach((cfg, name) => {
            if (state.tensorViews.has(name)) return;
            state.tensorViews.set(name, buildTensorViewSpec(normalizeViewShape(cfg.shape || [])));
        });
    };
    const applyTensorViewState = (snapshots?: Record<string, TensorViewSnapshot>): void => {
        ensureTensorViewsInitialized();
        if (!snapshots) return;
        let changed = false;
        configByName.forEach((cfg, name) => {
            const snapshot = snapshots[name];
            if (!snapshot) return;
            const shape = normalizeViewShape(cfg.shape || []);
            const nextSpec = buildTensorViewSpec(shape, snapshot.visible, snapshot.hiddenIndices || []);
            const prevSpec = state.tensorViews.get(name);
            if (
                !prevSpec
                || prevSpec.visibleText !== nextSpec.visibleText
                || !arraysEqual(prevSpec.hiddenIndices, nextSpec.hiddenIndices)
            ) {
                state.tensorViews.set(name, nextSpec);
                changed = true;
            }
        });
        if (changed) rebuildAllTensorsFromView();
    };
    const renderTensorViewControls = (): void => {
        if (configByName.size === 0) {
            if (vizCache.tensorViewControls) {
                vizCache.tensorViewControls.remove();
                vizCache.tensorViewControls = null;
            }
            return;
        }
        if (!vizCache.tensorViewControls) {
            const panel = document.createElement('div');
            panel.className = 'viz-ndim-controls';
            stage.appendChild(panel);
            vizCache.tensorViewControls = panel;
        }
        const root = vizCache.tensorViewControls as HTMLElement;
        if (!state.editTensorViewOn) {
            root.style.display = 'none';
            return;
        }
        root.style.display = '';
        root.innerHTML = '';
        Array.from(configByName.keys()).sort().forEach((name) => {
            const cfg = configByName.get(name);
            if (!cfg) return;
            const shape = normalizeViewShape(cfg.shape || []);
            const spec = state.tensorViews.get(name) || buildTensorViewSpec(shape);
            state.tensorViews.set(name, spec);

            const section = document.createElement('div');
            section.className = 'viz-ndim-section';
            const title = document.createElement('div');
            title.className = 'viz-ndim-title';
            title.textContent = `${name} tensor`;
            section.appendChild(title);

            const visibleRow = document.createElement('div');
            visibleRow.className = 'viz-ndim-row';
            const visibleLabel = document.createElement('label');
            visibleLabel.textContent = 'Visible dimensions:';
            visibleLabel.className = 'viz-ndim-label';
            const visibleInput = document.createElement('input');
            visibleInput.type = 'text';
            visibleInput.value = spec.visibleText;
            visibleInput.className = 'viz-ndim-visible';
            visibleInput.spellcheck = false;
            const applyVisible = (): void => {
                const nextSpec = buildTensorViewSpec(shape, visibleInput.value, state.tensorViews.get(name)?.hiddenIndices || []);
                state.tensorViews.set(name, nextSpec);
                visibleInput.value = nextSpec.visibleText;
                renderTensorViewControls();
                rebuildAllTensorsFromView();
            };
            visibleInput.addEventListener('change', applyVisible);
            visibleRow.appendChild(visibleLabel);
            visibleRow.appendChild(visibleInput);
            section.appendChild(visibleRow);

            const preview = document.createElement('div');
            preview.className = 'viz-ndim-preview';
            preview.textContent = buildTensorViewPreview(spec);
            section.appendChild(preview);

            if (spec.hiddenGroups.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'viz-ndim-hint';
                empty.textContent = 'all dimensions are visible.';
                section.appendChild(empty);
                root.appendChild(section);
                return;
            }
            spec.hiddenGroups.forEach((group) => {
                const row = document.createElement('div');
                row.className = 'viz-ndim-row';
                const label = document.createElement('label');
                label.className = 'viz-ndim-label';
                label.textContent = `${group.token}:`;
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = '0';
                slider.max = String(Math.max(0, group.size - 1));
                slider.value = String(group.value);
                slider.className = 'viz-ndim-slider';
                const value = document.createElement('input');
                value.type = 'number';
                value.min = '0';
                value.max = String(Math.max(0, group.size - 1));
                value.value = String(group.value);
                value.className = 'viz-ndim-index';
                const applyAxisValue = (raw: string): void => {
                    const currentSpec = state.tensorViews.get(name) || spec;
                    const currentGroup = currentSpec.hiddenGroups.find((entry) => entry.token === group.token);
                    if (!currentGroup) return;
                    const max = Math.max(0, currentGroup.size - 1);
                    const parsed = Number(raw);
                    const nextValue = Math.min(max, Math.max(0, Number.isFinite(parsed) ? Math.round(parsed) : 0));
                    const hiddenIndices = currentSpec.hiddenIndices.slice();
                    const expanded = unflattenAxesIndex(nextValue, currentGroup.axes, shape);
                    currentGroup.axes.forEach((axis, axisIdx) => {
                        hiddenIndices[axis] = expanded[axisIdx] ?? 0;
                    });
                    const nextSpec = buildTensorViewSpec(shape, currentSpec.visibleText, hiddenIndices);
                    state.tensorViews.set(name, nextSpec);
                    slider.value = String(nextValue);
                    value.value = String(nextValue);
                    slider.max = String(Math.max(0, currentGroup.size - 1));
                    value.max = String(Math.max(0, currentGroup.size - 1));
                    preview.textContent = buildTensorViewPreview(nextSpec);
                    rebuildAllTensorsFromView();
                };
                slider.addEventListener('input', () => applyAxisValue(slider.value));
                value.addEventListener('input', () => applyAxisValue(value.value));
                value.addEventListener('change', () => applyAxisValue(value.value));
                row.appendChild(label);
                row.appendChild(slider);
                row.appendChild(value);
                section.appendChild(row);
            });
            root.appendChild(section);
        });
    };
    applyTensorViewState(viewState?.tensorViews);
    renderTensorViewControls();
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
        items.push(createProgramSubsetLegendItem(
            getBaseCountColor(),
            programSubsets.subsets,
            state.programSubsetHues,
        ));
        createLegends(items);
        refreshDescriptorDimensionLines(vizCache);
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
        refreshDescriptorDimensionLines(vizCache);
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
                    refreshDescriptorDimensionLines(vizCache);
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
                    refreshDescriptorDimensionLines(vizCache);
                    requestRender();
                    return state.allProgramsOn;
                }
                return ensureProgramData();
            } : null,
            toggleEditTensorView: (): boolean => {
                state.editTensorViewOn = !state.editTensorViewOn;
                renderTensorViewControls();
                requestRender();
                return state.editTensorViewOn;
            },
        });
    }
    syncOpControlState();

    const fetchers = opUuid ? Array.from(tensors.entries()).map(([name, group]) => {
        return fetchTensorPayload(API_BASE, opUuid, group.userData.endpoint || 'getLoadTensor').then(p => {
            if (p) {
                state.payloads.set(name, p);
                refreshDescriptorDimensionLines(vizCache);
                renderShapeLegend();
                if (!supportsAllPrograms || !state.allProgramsOn) {
                    updateTensorHighlights(group, p.highlights, vizCache.highlightColor, group.userData.mesh.userData.color_base);
                }
            }
        });
    }) : [];

    renderShapeLegend();

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
        refreshDescriptorDimensionLines(vizCache);
        requestRender();
    });

    applyBackgroundTheme('#000000');
    requestRender();
    return vizCache.cleanup;
}
