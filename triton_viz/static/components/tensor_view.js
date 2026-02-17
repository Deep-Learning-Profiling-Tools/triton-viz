import { createCadDimension, createShapeLegend } from '../utils/dimension_utils.js';
import { clamp01, getHue, hslToRgb } from '../utils/colormap.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';
import { setupScene, setupGeometries, createTensor, setupCamera, fitCameraToBounds, setupEventListeners, cameraControls, CUBE_SIZE, GAP, COLOR_HOVER, updateTensorHighlights, canUseWebgl, renderWebglWarning, } from '../utils/three_utils.js';
import { createHistogramOverlay } from './histogram.js';
import { getApiBase, postJson } from '../core/api.js';
import { getState } from '../core/state.js';
import { createDisposer } from '../utils/dispose.js';
const VIZ_CACHE = new Map();
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
const DESCRIPTOR_AXIS_COLORS = {
    x: '#f59e0b',
    y: '#d946ef',
    z: '#22d3ee',
};
// --- Top-Level Helpers ---
function sampleValueFromCache(cache, coords) {
    if (!cache || !cache.values)
        return 0;
    let cursor = cache.values;
    for (let axis = 0; axis < coords.length; axis += 1) {
        if (!Array.isArray(cursor))
            return Number(cursor ?? 0);
        const idx = coords[axis] ?? 0;
        cursor = cursor[idx];
    }
    return Number(cursor ?? 0);
}
function getMeshFullCoords(mesh, index) {
    return (mesh.userData.coords_full?.[index]
        || mesh.userData.coords_display?.[index]
        || mesh.userData.coords?.[index]
        || [0, 0, 0]);
}
function getMeshDisplayCoords(mesh, index) {
    return mesh.userData.coords_display?.[index] || getMeshFullCoords(mesh, index);
}
function getAxisLabel(axis) {
    const letters = 'abcdefghijklmnopqrstuvwxyz';
    return letters[axis] || `d${axis}`;
}
function getAxisLabels(rank) {
    return Array.from({ length: rank }, (_, axis) => getAxisLabel(axis));
}
function defaultVisibleAxes(rank) {
    if (rank <= 3)
        return Array.from({ length: rank }, (_, axis) => axis);
    return Array.from({ length: 3 }, (_, offset) => rank - 3 + offset);
}
function normalizeViewShape(shapeRaw) {
    if (!Array.isArray(shapeRaw) || shapeRaw.length === 0)
        return [1];
    return shapeRaw.map((dim) => Math.max(1, Number(dim) || 1));
}
function buildTensorViewSpec(shapeRaw, visibleText = '', hiddenIndices = []) {
    const rank = shapeRaw.length;
    const axisLabels = getAxisLabels(rank);
    const axisFromLabel = new Map();
    axisLabels.forEach((label, axis) => axisFromLabel.set(label, axis));
    const parsedVisible = [];
    const usedVisible = new Set();
    const chars = (visibleText || '').toLowerCase().split('');
    chars.forEach((ch) => {
        const axis = axisFromLabel.get(ch);
        if (axis === undefined || usedVisible.has(axis))
            return;
        usedVisible.add(axis);
        parsedVisible.push(axis);
    });
    const visibleAxes = parsedVisible.length > 0 ? parsedVisible : defaultVisibleAxes(rank);
    const visibleSet = new Set(visibleAxes);
    const hiddenAxes = Array.from({ length: rank }, (_, axis) => axis).filter((axis) => !visibleSet.has(axis));
    const nextHiddenIndices = Array.from({ length: rank }, (_, axis) => {
        const dim = Math.max(1, shapeRaw[axis] ?? 1);
        const prev = Number(hiddenIndices[axis] ?? 0);
        return Math.min(dim - 1, Math.max(0, Number.isFinite(prev) ? prev : 0));
    });
    const displayShape = visibleAxes.map((axis) => Math.max(1, shapeRaw[axis] ?? 1));
    return {
        axisLabels,
        visibleAxes,
        hiddenAxes,
        hiddenIndices: nextHiddenIndices,
        visibleText: visibleAxes.map((axis) => axisLabels[axis]).join(''),
        displayShape,
    };
}
function mapDisplayToFullCoords(displayCoord, spec) {
    const full = spec.hiddenIndices.slice();
    spec.visibleAxes.forEach((axis, displayAxis) => {
        full[axis] = Number(displayCoord[displayAxis] ?? 0);
    });
    return full;
}
function arraysEqual(a = [], b = []) {
    if (a.length !== b.length)
        return false;
    for (let i = 0; i < a.length; i += 1) {
        if (a[i] !== b[i])
            return false;
    }
    return true;
}
function applyValueColormap(mesh, cache, paint) {
    if (!mesh || !cache)
        return;
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
    if (mesh.instanceColor)
        mesh.instanceColor.needsUpdate = true;
}
function applyCoordColormap(mesh, paint) {
    if (!mesh || !paint)
        return;
    const coordsList = mesh.userData.coords;
    if (!coordsList)
        return;
    const c = new THREE.Color();
    for (let i = 0; i < mesh.count; i += 1) {
        const coords = coordsList[i];
        if (!coords)
            continue;
        paint(c, coords, i);
        mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor)
        mesh.instanceColor.needsUpdate = true;
}
function createSideMenu(container) {
    const menu = document.createElement('div');
    Object.assign(menu.style, {
        position: 'absolute', right: '16px', bottom: '16px', width: '300px', padding: '12px',
        background: 'rgba(0,0,0,0.65)', color: '#fff', borderRadius: '8px',
        fontFamily: 'var(--font-sans)', fontSize: '12px', zIndex: 2500,
    });
    container.appendChild(menu);
    return menu;
}
function updateSideMenu(el, name, coords, val, shape, extraHtml = '') {
    const shapeStr = Array.isArray(shape) ? `[${shape.join(', ')}]` : '(unknown)';
    const indexExpr = (Array.isArray(coords) ? coords : []).map((coord) => `[${coord}]`).join('');
    const valueLine = indexExpr ? `tensor${indexExpr} = ${val}` : `tensor = ${val}`;
    el.innerHTML = `<h3>${name} Tensor</h3><p>${valueLine}</p><p>Shape: ${shapeStr}</p>${extraHtml}`;
}
async function fetchTensorPayload(apiBase, uuid, endpoint = 'getLoadTensor') {
    try {
        const data = await postJson(`/api/${endpoint}`, { uuid }, { base: apiBase });
        if (!data)
            return null;
        return {
            scaleMin: data.min ?? 0, scaleMax: data.max ?? 0,
            values: data.values, shape: data.shape, dims: data.dims,
            highlights: data.highlights,
        };
    }
    catch (e) {
        return null;
    }
}
async function fetchProgramCounts(apiBase, op) {
    if (!op || !op.overall_key || op.time_idx === undefined)
        return null;
    // fetch a sparse list of coords -> program count for this op
    try {
        return await postJson('/api/getLoadStoreAllPrograms', {
            type: op.type,
            overall_key: op.overall_key,
            time_idx: op.time_idx,
            op_index: op.op_index,
        }, { base: apiBase });
    }
    catch (e) {
        return null;
    }
}
function applyColorToMesh(mesh, cache, label) {
    if (!mesh || !cache)
        return;
    const hue = getHue(label);
    applyValueColormap(mesh, cache, (color, _coords, _val, t) => {
        const [r, g, b] = hslToRgb(hue, 0.9, t);
        color.setRGB(r, g, b);
    });
}
function applyDimmedColormap(mesh, cache, label, isHighlighted) {
    if (!mesh || !cache)
        return;
    const hue = getHue(label);
    applyValueColormap(mesh, cache, (color, coords, _val, t) => {
        if (isHighlighted(coords)) {
            const [r, g, b] = hslToRgb(hue, 0.9, t);
            color.setRGB(r, g, b);
        }
        else {
            color.setRGB(t, t, t);
        }
    });
}
function applyMonochromeColormap(mesh, cache) {
    if (!mesh || !cache)
        return;
    applyValueColormap(mesh, cache, (color, _coords, _val, t) => {
        color.setRGB(t, t, t);
    });
}
function normalizeProgramCounts(payload) {
    // normalize the sparse payload into a map for fast lookups
    const map = new Map();
    let maxCount = 0;
    (payload?.counts || []).forEach((entry) => {
        if (!entry || entry.length < 4)
            return;
        const [x, y, z, count] = entry;
        const safeCount = Number(count) || 0;
        map.set(`${x},${y},${z}`, safeCount);
        if (safeCount > maxCount)
            maxCount = safeCount;
    });
    maxCount = Math.max(maxCount, Number(payload?.max_count) || 0);
    return { map, maxCount };
}
function normalizeProgramSubsets(payload) {
    // normalize subset payload into lookup maps
    const subsetMap = new Map();
    const subsets = (payload?.subsets || {});
    const subsetCount = Number(payload?.subset_count) || Object.keys(subsets).length;
    const countMap = new Map();
    let maxCount = Number(payload?.max_count) || 0;
    (payload?.coords || []).forEach((entry) => {
        if (!entry || entry.length < 4)
            return;
        const [x, y, z, key] = entry;
        subsetMap.set(`${x},${y},${z}`, String(key));
    });
    (payload?.counts || []).forEach((entry) => {
        if (!entry || entry.length < 4)
            return;
        const [x, y, z, count] = entry;
        const safeCount = Number(count) || 0;
        countMap.set(`${x},${y},${z}`, safeCount);
        if (safeCount > maxCount)
            maxCount = safeCount;
    });
    return { subsetMap, subsets, subsetCount, countMap, maxCount };
}
function buildSubsetHues(subsets) {
    const keys = Object.keys(subsets || {}).sort();
    const hues = new Map();
    const used = [];
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
        const tooClose = (candidate) => {
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
function applyProgramCountColors(mesh, counts, baseColor, palette) {
    if (!mesh || !counts)
        return;
    const base = baseColor instanceof THREE.Color ? baseColor : new THREE.Color(baseColor);
    const countMap = counts.map;
    const colors = palette || PROGRAM_COUNT_PALETTE;
    applyCoordColormap(mesh, (color, coords) => {
        const count = coords ? countMap.get(`${coords[0]},${coords[1]},${coords[2]}`) || 0 : 0;
        if (count <= 0) {
            color.copy(base);
        }
        else {
            const idx = Math.min(count - 1, colors.length - 1);
            color.copy(colors[idx]);
        }
    });
}
function applyProgramCountHeatmap(mesh, cache, counts, palette, baseColor) {
    if (!mesh || !cache || !counts)
        return;
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
function applyProgramSubsetColors(mesh, subsetState, hues, baseColor) {
    if (!mesh || !subsetState || !hues)
        return;
    const base = baseColor instanceof THREE.Color ? baseColor : new THREE.Color(baseColor);
    const subsetMap = subsetState.subsetMap;
    applyCoordColormap(mesh, (color, coords) => {
        const key = coords ? subsetMap.get(`${coords[0]},${coords[1]},${coords[2]}`) : null;
        if (key && hues.has(key)) {
            color.setHSL(hues.get(key), 0.6, 0.55);
        }
        else {
            color.copy(base);
        }
    });
}
function applyProgramSubsetHeatmap(mesh, cache, subsetState, hues, baseColor) {
    if (!mesh || !cache || !subsetState || !hues)
        return;
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
function parseDescriptorHighlight(highlights) {
    if (!highlights || highlights.type !== 'descriptor')
        return null;
    const rank = Math.max(highlights.start?.length || 0, highlights.shape?.length || 0, highlights.stride?.length || 0);
    if (rank <= 0)
        return null;
    const start = Array.from({ length: rank }, (_, axis) => Number(highlights.start?.[axis] ?? 0));
    const shape = Array.from({ length: rank }, (_, axis) => Number(highlights.shape?.[axis] ?? 0));
    const stride = Array.from({ length: rank }, (_, axis) => Math.max(1, Math.abs(Number(highlights.stride?.[axis] ?? 1))));
    if (shape.some((dim) => dim <= 0))
        return null;
    return {
        start,
        shape,
        stride,
    };
}
function descriptorContainsCoord(descriptor, coords) {
    const inAxis = (coord, axisStart, axisShape, axisStride) => {
        if (axisShape <= 0)
            return false;
        const delta = coord - axisStart;
        if (delta < 0 || delta % axisStride !== 0)
            return false;
        return (delta / axisStride) < axisShape;
    };
    for (let axis = 0; axis < descriptor.shape.length; axis += 1) {
        if (!inAxis(coords[axis] ?? 0, descriptor.start[axis] ?? 0, descriptor.shape[axis] ?? 0, descriptor.stride[axis] ?? 1)) {
            return false;
        }
    }
    return true;
}
function projectDescriptorForView(descriptor, spec) {
    for (let i = 0; i < spec.hiddenAxes.length; i += 1) {
        const axis = spec.hiddenAxes[i];
        if (axis === undefined)
            continue;
        const idx = spec.hiddenIndices[axis] ?? 0;
        const start = descriptor.start[axis] ?? 0;
        const shape = descriptor.shape[axis] ?? 0;
        const stride = descriptor.stride[axis] ?? 1;
        if (shape <= 0)
            return null;
        const delta = idx - start;
        if (delta < 0 || delta % stride !== 0)
            return null;
        if ((delta / stride) >= shape)
            return null;
    }
    return {
        start: spec.visibleAxes.map((axis) => descriptor.start[axis] ?? 0),
        shape: spec.visibleAxes.map((axis) => descriptor.shape[axis] ?? 0),
        stride: spec.visibleAxes.map((axis) => descriptor.stride[axis] ?? 1),
    };
}
function reorderDescriptorForTensor(values, tensorRank, fallback) {
    const out = [];
    for (let i = 0; i < tensorRank; i += 1) {
        out.push(values[i] ?? fallback);
    }
    return out;
}
function getHighlightPredicate(highlights) {
    if (!highlights)
        return null;
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
function applyColorizedMesh(ctx, group, name) {
    const mesh = group.userData.mesh;
    const p = ctx.state.payloads.get(name);
    if (!p)
        return;
    const label = name === 'Global' ? ctx.type : name;
    const predicate = getHighlightPredicate(p.highlights);
    if (predicate) {
        applyDimmedColormap(mesh, p, label, predicate);
    }
    else {
        applyMonochromeColormap(mesh, p);
    }
}
function restoreTensorColors(ctx) {
    const { state, tensors } = ctx;
    const supportsAllPrograms = ctx.supportsAllPrograms;
    tensors.forEach((group, name) => {
        const mesh = group.userData.mesh;
        const p = state.payloads.get(name);
        if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'subset' && state.programSubsets) {
            if (state.colorizeOn && p) {
                applyProgramSubsetHeatmap(mesh, p, state.programSubsets, state.programSubsetHues, mesh.userData.color_base);
            }
            else {
                applyProgramSubsetColors(mesh, state.programSubsets, state.programSubsetHues, mesh.userData.color_base);
            }
        }
        else if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'count' && state.programCounts) {
            if (state.colorizeOn && p) {
                applyProgramCountHeatmap(mesh, p, state.programCounts, PROGRAM_COUNT_PALETTE, mesh.userData.color_base);
            }
            else {
                applyProgramCountColors(mesh, state.programCounts, mesh.userData.color_base, PROGRAM_COUNT_PALETTE);
            }
        }
        else if (state.colorizeOn && p) {
            applyColorizedMesh(ctx, group, name);
        }
        else {
            updateTensorHighlights(group, p?.highlights, ctx.highlightColor, mesh.userData.color_base);
        }
    });
}
function applyDotHoverHighlight(ctx, row, col) {
    const { tensors, state } = ctx;
    const aGroup = tensors.get('A');
    const bGroup = tensors.get('B');
    if (!aGroup || !bGroup)
        return;
    const aCache = state.payloads.get('A');
    const bCache = state.payloads.get('B');
    if (!aCache || !bCache)
        return;
    applyDimmedColormap(aGroup.userData.mesh, aCache, 'A', (coords) => coords[0] === row);
    applyDimmedColormap(bGroup.userData.mesh, bCache, 'B', (coords) => coords[1] === col);
}
function applyDotHoverOutline(ctx, row, col) {
    const { tensors } = ctx;
    const aGroup = tensors.get('A');
    const bGroup = tensors.get('B');
    if (!aGroup || !bGroup)
        return;
    updateTensorHighlights(aGroup, null, ctx.highlightColor, aGroup.userData.mesh.userData.color_base, (x, y) => y === row);
    updateTensorHighlights(bGroup, null, ctx.highlightColor, bGroup.userData.mesh.userData.color_base, (x) => x === col);
}
function captureHistogramState(histogramUI) {
    const overlay = histogramUI?.overlay;
    if (!overlay) {
        return { histogramVisible: false, histogramSource: null, histogramBins: null };
    }
    const select = overlay.querySelector('#histogram-source');
    const bins = overlay.querySelector('#histogram-bins');
    return {
        histogramVisible: overlay.style.display === 'block',
        histogramSource: select ? select.value : null,
        histogramBins: bins ? Number(bins.value) : null,
    };
}
function applyHistogramState(histogramUI, state) {
    const overlay = histogramUI?.overlay;
    if (!overlay || !state)
        return;
    const select = overlay.querySelector('#histogram-source');
    const bins = overlay.querySelector('#histogram-bins');
    if (select && state.histogramSource) {
        select.value = state.histogramSource;
    }
    if (bins && Number.isFinite(state.histogramBins)) {
        bins.value = String(state.histogramBins);
    }
    if (state.histogramVisible) {
        histogramUI.show?.();
    }
    else {
        histogramUI.hide?.();
    }
}
function createLegendItem(label, min, max) {
    const item = document.createElement('div');
    Object.assign(item.style, { display: 'grid', gap: '4px', fontFamily: 'monospace', fontSize: '12px' });
    const title = document.createElement('div');
    title.textContent = `${label} Value`;
    title.style.opacity = '0.9';
    title.style.fontWeight = 'bold';
    const canvas = document.createElement('canvas');
    canvas.width = 220;
    canvas.height = 10;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        item.appendChild(title);
        item.appendChild(canvas);
        return item;
    }
    for (let x = 0; x < canvas.width; x++) {
        const t = clamp01(x / (canvas.width - 1));
        const v = Math.round(t * 255);
        ctx.fillStyle = `rgb(${v},${v},${v})`;
        ctx.fillRect(x, 0, 1, canvas.height);
    }
    const labels = document.createElement('div');
    labels.style.display = 'flex';
    labels.style.justifyContent = 'space-between';
    labels.style.marginTop = '2px';
    labels.innerHTML = `<span>${min.toFixed(3)}</span><span>${max.toFixed(3)}</span>`;
    item.appendChild(title);
    item.appendChild(canvas);
    item.appendChild(labels);
    return item;
}
function createProgramCountLegendItem(baseColor, maxCount, palette = PROGRAM_COUNT_PALETTE) {
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
    const addRow = (label, color) => {
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
function createProgramSubsetLegendItem(baseColor, subsets, hues) {
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
    const addRow = (label, color) => {
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
            ? pids.map(([x, y, z]) => `(${x},${y},${z})`).join(' ')
            : '(empty)';
        const color = new THREE.Color();
        if (hues && hues.has(key)) {
            color.setHSL(hues.get(key), 0.6, 0.55);
        }
        else {
            color.copy(base);
        }
        addRow(label, color);
    });
    item.appendChild(rows);
    return item;
}
function addDimensionLines(scene, tensorGroup, dimColors = []) {
    const mesh = tensorGroup?.userData?.mesh;
    if (!mesh)
        return [];
    const shape = mesh.userData.shape;
    const shapeRaw = mesh.userData.shape_raw || [];
    if (shapeRaw.length > 3)
        return [];
    const bbox = new THREE.Box3().setFromObject(tensorGroup);
    const offsetBase = (CUBE_SIZE + GAP) * 1.5;
    const axisDefaults = { x: '#f87171', y: '#4ade80', z: '#60a5fa' };
    const getColor = (axis) => {
        if (shapeRaw.length === 1 && axis === 'x')
            return dimColors[0] || axisDefaults.x;
        if (shapeRaw.length === 2 && axis === 'y')
            return dimColors[0] || axisDefaults.y;
        if (shapeRaw.length === 2 && axis === 'x')
            return dimColors[1] || axisDefaults.x;
        if (shapeRaw.length >= 3 && axis === 'z')
            return dimColors[0] || axisDefaults.z;
        if (shapeRaw.length >= 3 && axis === 'y')
            return dimColors[1] || axisDefaults.y;
        if (shapeRaw.length >= 3 && axis === 'x')
            return dimColors[2] || axisDefaults.x;
        return axisDefaults[axis];
    };
    const groups = [];
    if (shapeRaw.length >= 2) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${shape.width}`, 'x', getColor('x'), { offset: offsetBase }));
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), `${shape.height}`, 'y', getColor('y'), { offset: offsetBase }));
    }
    else if (shapeRaw.length === 1) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${shape.width}`, 'x', getColor('x'), { offset: offsetBase }));
    }
    if (shapeRaw.length >= 3) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z), new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), `${shape.depth}`, 'z', getColor('z'), { offset: offsetBase }));
    }
    return groups;
}
function getDescriptorSelectionBounds(mesh, descriptor) {
    const tensorShape = mesh.userData.shape;
    if (!descriptor || descriptor.shape.length < 1 || descriptor.shape.length > 3) {
        const fallback = new THREE.Vector3(0, 0, 0);
        return { min: fallback.clone(), max: fallback };
    }
    const spacing = CUBE_SIZE + GAP;
    const centerX = (tensorShape.width - 1) * spacing / 2;
    const centerY = -((tensorShape.height - 1) * spacing / 2);
    const centerZ = -((tensorShape.depth - 1) * spacing / 2);
    const axisFromDisplay = (axis) => {
        if (descriptor.shape.length === 1)
            return 0;
        if (descriptor.shape.length === 2)
            return axis === 'x' ? 1 : 0;
        if (axis === 'x')
            return 2;
        if (axis === 'y')
            return 1;
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
    const toX = (x) => x * spacing - centerX;
    const toY = (y) => -y * spacing - centerY;
    const toZ = (z) => -z * spacing - centerZ;
    const half = CUBE_SIZE / 2;
    const min = new THREE.Vector3(Math.min(toX(sx), toX(ex)) - half, Math.min(toY(sy), toY(ey)) - half, Math.min(toZ(sz), toZ(ez)) - half);
    const max = new THREE.Vector3(Math.max(toX(sx), toX(ex)) + half, Math.max(toY(sy), toY(ey)) + half, Math.max(toZ(sz), toZ(ez)) + half);
    return { min, max };
}
function addDescriptorDimensionLines(scene, tensorGroup, highlights, spec = null) {
    const mesh = tensorGroup?.userData?.mesh;
    if (!mesh)
        return [];
    const descriptorRaw = parseDescriptorHighlight(highlights);
    const descriptor = descriptorRaw && spec ? projectDescriptorForView(descriptorRaw, spec) : descriptorRaw;
    if (!descriptor)
        return [];
    const shapeRaw = mesh.userData.shape_raw || [];
    if (shapeRaw.length > 3 || descriptor.shape.length > 3)
        return [];
    const bbox = getDescriptorSelectionBounds(mesh, descriptor);
    const offsetBase = (CUBE_SIZE + GAP) * 0.85;
    const axisFromDisplay = (axis) => {
        if (descriptor.shape.length === 1)
            return 0;
        if (descriptor.shape.length === 2)
            return axis === 'x' ? 1 : 0;
        if (axis === 'x')
            return 2;
        if (axis === 'y')
            return 1;
        return 0;
    };
    const groups = [];
    if (shapeRaw.length >= 2) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${descriptor.shape[axisFromDisplay('x')] ?? 1}`, 'x', DESCRIPTOR_AXIS_COLORS.x, { offset: offsetBase, opacity: 0.95 }));
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), `${descriptor.shape[axisFromDisplay('y')] ?? 1}`, 'y', DESCRIPTOR_AXIS_COLORS.y, { offset: offsetBase, opacity: 0.95 }));
    }
    else if (shapeRaw.length === 1) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${descriptor.shape[axisFromDisplay('x')] ?? 1}`, 'x', DESCRIPTOR_AXIS_COLORS.x, { offset: offsetBase, opacity: 0.95 }));
    }
    if (shapeRaw.length >= 3) {
        groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z), new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), `${descriptor.shape[axisFromDisplay('z')] ?? 1}`, 'z', DESCRIPTOR_AXIS_COLORS.z, { offset: offsetBase, opacity: 0.95 }));
    }
    return groups;
}
function clearDescriptorDimensionLines(ctx) {
    ctx.descriptorDimLineGroups.forEach((group) => ctx.scene.remove(group));
    ctx.descriptorDimLineGroups = [];
}
function refreshDescriptorDimensionLines(ctx) {
    clearDescriptorDimensionLines(ctx);
    if (!ctx.showDimLines || ctx.state.allProgramsOn)
        return;
    ctx.tensors.forEach((group, name) => {
        const p = ctx.state.payloads.get(name);
        if (!p)
            return;
        const spec = ctx.state.tensorViews.get(name) || null;
        ctx.descriptorDimLineGroups.push(...addDescriptorDimensionLines(ctx.scene, group, p.highlights, spec));
    });
}
// --- Interaction Handlers ---
function onMouseMove(event, ctx) {
    const { renderer, camera, tensors, state, sideMenu, requestRender, raycaster, mouse, API_BASE, op } = ctx;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const meshes = Array.from(tensors.values()).map((t) => t.userData.mesh);
    const hits = raycaster.intersectObjects(meshes);
    if (hits.length > 0) {
        const hit = hits[0];
        const mesh = hit.object;
        const instanceId = hit.instanceId ?? 0;
        const tensorName = mesh.userData.tensorName || '';
        const key = `${tensorName}_${instanceId}`;
        if (key !== state.lastHoverKey) {
            state.lastHoverKey = key;
        }
        const coords3 = (mesh.userData.coords?.[instanceId] ?? [0, 0, 0]);
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
            const key = `${coords3[0]},${coords3[1]},${coords3[2]}`;
            if (state.programSubsets) {
                const subsetKey = state.programSubsets.subsetMap.get(key);
                const subset = subsetKey ? state.programSubsets.subsets?.[subsetKey] || [] : [];
                const label = subset.length
                    ? subset.map(([x, y, z]) => `(${x},${y},${z})`).join(' ')
                    : 'none';
                extraHtml = `<p>Programs: ${label}</p>`;
            }
            else if (state.programCounts) {
                const count = state.programCounts.map.get(key) || 0;
                extraHtml = `<p>Programs: ${count}</p>`;
            }
            else if (state.programDataLoading) {
                extraHtml = '<p>Programs: loading...</p>';
            }
        }
        updateSideMenu(sideMenu, tensorName, coordsDisplay, val, currentShape || null, extraHtml);
        if (ctx.type === 'Dot' && tensorName === 'C') {
            const row = Number(coordsDisplay[0] ?? 0);
            const col = Number(coordsDisplay[1] ?? 0);
            const hoverKey = `${row},${col}`;
            if (state.dotHoverKey !== hoverKey) {
                state.dotHoverKey = hoverKey;
                if (state.colorizeOn) {
                    applyDotHoverHighlight(ctx, row, col);
                }
                else {
                    applyDotHoverOutline(ctx, row, col);
                }
            }
        }
        else if (state.dotHoverKey) {
            state.dotHoverKey = null;
            restoreTensorColors(ctx);
        }
        requestRender();
    }
    else {
        if (state.lastHoverKey !== null) {
            state.lastHoverKey = null;
            if (state.activeHoverOutline)
                state.activeHoverOutline.visible = false;
            sideMenu.innerHTML = '';
            if (state.dotHoverKey) {
                state.dotHoverKey = null;
                restoreTensorColors(ctx);
            }
            requestRender();
        }
    }
}
function onMouseUp(ctx) { ctx.state.isDragging = false; if (ctx.stage)
    ctx.stage.style.cursor = ''; }
// --- Main Exports ---
export function createTensorVisualization(containerElement, op, options = {}) {
    const { type = 'Load', colors = {}, tensorConfigs = [], dimColors = {}, showDimLines = true, viewState = null, layoutBounds = null, fitToTensors = true, cameraPadding = 1.15 } = options;
    const API_BASE = getApiBase();
    const initialToggles = getState().toggles;
    const configs = tensorConfigs.length > 0 ? tensorConfigs : [
        { name: 'Global', shape: op.global_shape || [], color: colors.GLOBAL || '#333', position: [0, 0, 0], endpoint: 'getLoadTensor' }
    ];
    const supportsAllPrograms = (type === 'Load' || type === 'Store') && configs.every((cfg) => (cfg.shape || []).length <= 3);
    const configByNameMap = new Map(configs.map((cfg) => [cfg.name, cfg]));
    let cache = VIZ_CACHE.get(containerElement);
    const shapeKey = JSON.stringify({ shapes: configs.map(c => c.shape), layoutBounds });
    const isSameContext = cache && cache.type === type && cache.shapeKey === shapeKey;
    if (!isSameContext) {
        if (cache && cache.cleanup)
            cache.cleanup();
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
        let scene;
        let camera;
        let renderer;
        try {
            ({ scene, camera, renderer } = setupScene(stage, 0x000000));
        }
        catch (err) {
            // webgl can still fail even after a feature test.
            return renderWebglWarning(containerElement);
        }
        const disposer = createDisposer();
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();
        const initialTensorViews = new Map();
        configs.forEach((cfg) => {
            const snapshot = viewState?.tensorViews?.[cfg.name];
            initialTensorViews.set(cfg.name, buildTensorViewSpec(normalizeViewShape(cfg.shape || []), snapshot?.visible || '', snapshot?.hiddenIndices || []));
        });
        const tensors = new Map();
        configs.forEach(cfg => {
            const spec = initialTensorViews.get(cfg.name) || buildTensorViewSpec(normalizeViewShape(cfg.shape || []));
            const group = createTensor(spec.displayShape, null, cfg.color, cfg.name, cubeGeometry, edgesGeometry, lineMaterial, { mapDisplayCoordToFull: (coord) => mapDisplayToFullCoords(coord, spec) });
            group.position.set(...(cfg.position || [0, 0, 0]));
            if (cfg.endpoint) {
                group.userData.endpoint = cfg.endpoint;
            }
            else if (group.userData.endpoint) {
                delete group.userData.endpoint;
            }
            scene.add(group);
            tensors.set(cfg.name, group);
        });
        if (layoutBounds) {
            const depth = layoutBounds.depth ?? CUBE_SIZE;
            const layoutBox = new THREE.Mesh(new THREE.BoxGeometry(layoutBounds.width, layoutBounds.height, depth), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 }));
            const center = layoutBounds.center ?? [0, 0, 0];
            layoutBox.position.set(center[0], center[1], center[2] ?? 0);
            scene.add(layoutBox);
        }
        const hoverOutline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05)), new THREE.LineBasicMaterial({ color: COLOR_HOVER }));
        hoverOutline.visible = false;
        scene.add(hoverOutline);
        const dimLineGroups = [];
        if (showDimLines) {
            tensors.forEach((group, name) => {
                dimLineGroups.push(...addDimensionLines(scene, group, dimColors[name]));
            });
        }
        let cameraCenter = new THREE.Vector3(0, 0, 0);
        let fitRadius = 0;
        const bounds = new THREE.Box3();
        let hasBounds = false;
        let tensorCenter = null;
        if (fitToTensors) {
            tensors.forEach((group) => bounds.union(new THREE.Box3().setFromObject(group)));
            hasBounds = true;
            if (!bounds.isEmpty())
                tensorCenter = bounds.getCenter(new THREE.Vector3());
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
        }
        else {
            const { center } = setupCamera(scene, camera);
            cameraCenter = center;
        }
        const orbitControls = new OrbitControls(camera, renderer.domElement);
        orbitControls.enableDamping = false;
        orbitControls.target.copy(cameraCenter);
        orbitControls.update();
        disposer.add(() => orbitControls.dispose());
        const syncClipPlanes = () => {
            if (!fitRadius)
                return;
            const dist = camera.position.distanceTo(orbitControls.target);
            camera.near = Math.max(0.05, dist - fitRadius * 2.5);
            camera.far = Math.max(camera.far, dist + fitRadius * 2.5);
            camera.updateProjectionMatrix();
        };
        const state = {
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
            tensorViews: new Map(initialTensorViews),
        };
        const highlightColor = (colors.HIGHLIGHT instanceof THREE.Color) ? colors.HIGHLIGHT : new THREE.Color(colors.HIGHLIGHT || 0x00b3ff);
        const ctx = {
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
            descriptorDimLineGroups: [],
            showDimLines,
            supportsAllPrograms,
            highlightColor,
            tensorViewControls: null,
            requestRender: () => { },
            applyBackgroundTheme: () => { },
            destroyLegends: () => { },
            createLegends: () => { },
        };
        ctx.requestRender = () => {
            if (state.rafId !== null) {
                state.renderPending = true;
                return;
            }
            state.rafId = requestAnimationFrame(() => { state.rafId = null; orbitControls.update(); syncClipPlanes(); renderer.render(scene, camera); if (state.renderPending) {
                state.renderPending = false;
                ctx.requestRender();
            } });
        };
        let lastWidth = 0;
        let lastHeight = 0;
        const resizeRenderer = () => {
            const width = Math.max(1, Math.floor(stage.clientWidth));
            const height = Math.max(1, Math.floor(stage.clientHeight));
            if (width === lastWidth && height === lastHeight)
                return;
            lastWidth = width;
            lastHeight = height;
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
        ctx.applyBackgroundTheme = (hex) => {
            const isLight = (hex || '').toLowerCase() === '#ffffff';
            const baseGlobalLight = new THREE.Color('#fefce8');
            tensors.forEach((group, name) => {
                const mesh = group.userData.mesh;
                const baseColor = isLight ? baseGlobalLight : mesh.userData.color_base;
                if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'subset' && state.programSubsets) {
                    if (state.colorizeOn) {
                        const payload = state.payloads.get(name);
                        if (payload)
                            applyProgramSubsetHeatmap(mesh, payload, state.programSubsets, state.programSubsetHues, baseColor);
                    }
                    else {
                        applyProgramSubsetColors(mesh, state.programSubsets, state.programSubsetHues, baseColor);
                    }
                }
                else if (supportsAllPrograms && state.allProgramsOn && state.allProgramsMode === 'count' && state.programCounts) {
                    if (state.colorizeOn) {
                        const payload = state.payloads.get(name);
                        if (payload)
                            applyProgramCountHeatmap(mesh, payload, state.programCounts, PROGRAM_COUNT_PALETTE, baseColor);
                    }
                    else {
                        applyProgramCountColors(mesh, state.programCounts, baseColor, PROGRAM_COUNT_PALETTE);
                    }
                }
                else {
                    updateTensorHighlights(group, state.payloads.get(name)?.highlights, highlightColor, baseColor);
                }
                group.children.forEach(c => { if (c?.userData?.edges)
                    c.userData.edges.visible = !isLight; });
            });
            if (!isLight && lineMaterial) {
                lineMaterial.color.set('#ffffff');
                lineMaterial.opacity = 0.28;
            }
            ctx.requestRender();
        };
        ctx.destroyLegends = () => { if (ctx.legendContainer?.remove)
            ctx.legendContainer.remove(); ctx.legendContainer = null; };
        ctx.createLegends = (items) => {
            ctx.destroyLegends();
            if (!items.length)
                return;
            const shapeLegend = containerElement.querySelector('.viz-shape-legend');
            const wrapper = document.createElement('div');
            Object.assign(wrapper.style, { display: 'grid', gap: '12px', marginTop: '10px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.15)' });
            items.forEach(i => wrapper.appendChild(i));
            if (shapeLegend) {
                shapeLegend.appendChild(wrapper);
            }
            else {
                Object.assign(wrapper.style, { position: 'absolute', left: '24px', bottom: '20px', background: 'rgba(0,0,0,0.65)', color: '#fff', padding: '12px', borderRadius: '8px', zIndex: '2000', pointerEvents: 'auto', border: '1px solid rgba(255,255,255,0.1)' });
                containerElement.appendChild(wrapper);
            }
            ctx.legendContainer = wrapper;
        };
        const serializeTensorViews = () => {
            const out = {};
            state.tensorViews.forEach((spec, name) => {
                out[name] = {
                    visible: spec.visibleText,
                    hiddenIndices: spec.hiddenIndices.slice(),
                };
            });
            return out;
        };
        const applyViewState = (nextState) => {
            if (!nextState)
                return;
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
        const getViewState = () => {
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
        const cleanupListeners = setupEventListeners(stage, camera, renderer, (e) => onMouseMove(e, ctx), cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ')), ctx.requestRender);
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
            if (state.rafId)
                cancelAnimationFrame(state.rafId);
            ctx.disposer.dispose();
            ctx.destroyLegends();
            ctx.dimLineGroups.forEach((group) => scene.remove(group));
            ctx.dimLineGroups = [];
            clearDescriptorDimensionLines(ctx);
            if (stage.parentElement)
                stage.parentElement.removeChild(stage);
            if (sideMenu.parentElement)
                sideMenu.parentElement.removeChild(sideMenu);
            if (ctx.tensorViewControls?.parentElement)
                ctx.tensorViewControls.parentElement.removeChild(ctx.tensorViewControls);
            histogramUI.destroy?.();
            if (histogramUI.overlay?.parentElement)
                histogramUI.overlay.parentElement.removeChild(histogramUI.overlay);
            if (containerElement.__vizGetState) {
                delete containerElement.__vizGetState;
            }
            VIZ_CACHE.delete(containerElement);
        };
        cache = ctx;
        VIZ_CACHE.set(containerElement, cache);
    }
    else if (cache) {
        if (cache.stage.parentElement !== containerElement) {
            containerElement.innerHTML = '';
            containerElement.appendChild(cache.stage);
            containerElement.appendChild(cache.sideMenu);
            if (cache.histogramUI.overlay)
                containerElement.appendChild(cache.histogramUI.overlay);
            if (cache.legendContainer)
                containerElement.appendChild(cache.legendContainer);
        }
    }
    if (!cache)
        return;
    const vizCache = cache;
    const { state, tensors, sideMenu, requestRender, applyBackgroundTheme, createLegends, destroyLegends, configByName, cubeGeometry, edgesGeometry, scene, stage, } = vizCache;
    const syncOpControlState = () => {
        if (!window.setOpControlState)
            return;
        window.setOpControlState({
            colorize: state.colorizeOn,
            histogram: vizCache.histogramUI.overlay.style.display === 'block',
            allPrograms: state.allProgramsOn,
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
    const renderShapeLegend = () => {
        createShapeLegend(containerElement, Array.from(tensors.entries()).map(([name, group]) => {
            const entry = {
                name: name === 'Global' ? type : `Matrix ${name}`,
                color: '#' + group.userData.mesh.userData.color_base.getHexString(),
            };
            const shape = group.userData.mesh.userData.shape_raw;
            if (shape)
                entry.shape = shape;
            const dimColor = dimColors?.[name];
            if (dimColor)
                entry.dimColors = dimColor;
            const spec = state.tensorViews.get(name) || null;
            const descriptorRaw = parseDescriptorHighlight(state.payloads.get(name)?.highlights);
            const descriptor = descriptorRaw && spec ? projectDescriptorForView(descriptorRaw, spec) : descriptorRaw;
            if (descriptor && shape && shape.length > 0) {
                const selectionColor = vizCache.highlightColor instanceof THREE.Color
                    ? `#${vizCache.highlightColor.getHexString()}`
                    : String(vizCache.highlightColor || '#00b3ff');
                const selectionDimColors = reorderDescriptorForTensor([DESCRIPTOR_AXIS_COLORS.x, DESCRIPTOR_AXIS_COLORS.y, DESCRIPTOR_AXIS_COLORS.z], descriptor.shape.length, selectionColor);
                entry.descriptor = {
                    shape: descriptor.shape.slice(),
                    color: selectionColor,
                    dimColors: selectionDimColors,
                };
            }
            return entry;
        }));
    };
    const rebuildDimensionLines = () => {
        vizCache.dimLineGroups.forEach((group) => scene.remove(group));
        vizCache.dimLineGroups = [];
        if (!vizCache.showDimLines)
            return;
        tensors.forEach((group, name) => {
            vizCache.dimLineGroups.push(...addDimensionLines(scene, group, dimColors[name]));
        });
    };
    const rebuildTensorFromView = (name) => {
        const cfg = configByName.get(name);
        const oldGroup = tensors.get(name);
        if (!cfg || !oldGroup)
            return;
        const shape = normalizeViewShape(cfg.shape || []);
        const spec = state.tensorViews.get(name) || buildTensorViewSpec(shape);
        const nextGroup = createTensor(spec.displayShape, null, oldGroup.userData.mesh.userData.color_base || cfg.color, name, cubeGeometry, edgesGeometry, vizCache.lineMaterial, { mapDisplayCoordToFull: (coord) => mapDisplayToFullCoords(coord, spec) });
        nextGroup.position.copy(oldGroup.position);
        const endpoint = oldGroup.userData.endpoint || cfg.endpoint;
        if (endpoint)
            nextGroup.userData.endpoint = endpoint;
        else if (nextGroup.userData.endpoint)
            delete nextGroup.userData.endpoint;
        scene.remove(oldGroup);
        const oldMaterial = oldGroup.userData.mesh?.material;
        if (oldMaterial && typeof oldMaterial.dispose === 'function')
            oldMaterial.dispose();
        scene.add(nextGroup);
        tensors.set(name, nextGroup);
    };
    const rebuildAllTensorsFromView = () => {
        Array.from(tensors.keys()).forEach((name) => rebuildTensorFromView(name));
        rebuildDimensionLines();
        renderShapeLegend();
        restoreTensorColors(vizCache);
        refreshDescriptorDimensionLines(vizCache);
        requestRender();
    };
    const ensureTensorViewsInitialized = () => {
        configByName.forEach((cfg, name) => {
            if (state.tensorViews.has(name))
                return;
            state.tensorViews.set(name, buildTensorViewSpec(normalizeViewShape(cfg.shape || [])));
        });
    };
    const applyTensorViewState = (snapshots) => {
        ensureTensorViewsInitialized();
        if (!snapshots)
            return;
        let changed = false;
        configByName.forEach((cfg, name) => {
            const snapshot = snapshots[name];
            if (!snapshot)
                return;
            const shape = normalizeViewShape(cfg.shape || []);
            const nextSpec = buildTensorViewSpec(shape, snapshot.visible, snapshot.hiddenIndices || []);
            const prevSpec = state.tensorViews.get(name);
            if (!prevSpec
                || prevSpec.visibleText !== nextSpec.visibleText
                || !arraysEqual(prevSpec.hiddenIndices, nextSpec.hiddenIndices)) {
                state.tensorViews.set(name, nextSpec);
                changed = true;
            }
        });
        if (changed)
            rebuildAllTensorsFromView();
    };
    const renderTensorViewControls = () => {
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
        const root = vizCache.tensorViewControls;
        root.innerHTML = '';
        Array.from(configByName.keys()).sort().forEach((name) => {
            const cfg = configByName.get(name);
            if (!cfg)
                return;
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
            const applyVisible = () => {
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
            if (spec.hiddenAxes.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'viz-ndim-hint';
                empty.textContent = 'all dimensions are visible.';
                section.appendChild(empty);
                root.appendChild(section);
                return;
            }
            spec.hiddenAxes.forEach((axis) => {
                const row = document.createElement('div');
                row.className = 'viz-ndim-row';
                const label = document.createElement('label');
                label.className = 'viz-ndim-label';
                label.textContent = `${spec.axisLabels[axis]}:`;
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = '0';
                slider.max = String(Math.max(0, (shape[axis] ?? 1) - 1));
                slider.value = String(spec.hiddenIndices[axis] ?? 0);
                slider.className = 'viz-ndim-slider';
                const value = document.createElement('input');
                value.type = 'number';
                value.min = '0';
                value.max = String(Math.max(0, (shape[axis] ?? 1) - 1));
                value.value = String(spec.hiddenIndices[axis] ?? 0);
                value.className = 'viz-ndim-index';
                const applyAxisValue = (raw) => {
                    const currentSpec = state.tensorViews.get(name) || spec;
                    const max = Math.max(0, (shape[axis] ?? 1) - 1);
                    const parsed = Number(raw);
                    const nextValue = Math.min(max, Math.max(0, Number.isFinite(parsed) ? Math.round(parsed) : 0));
                    const hiddenIndices = currentSpec.hiddenIndices.slice();
                    hiddenIndices[axis] = nextValue;
                    state.tensorViews.set(name, buildTensorViewSpec(shape, currentSpec.visibleText, hiddenIndices));
                    slider.value = String(nextValue);
                    value.value = String(nextValue);
                    rebuildAllTensorsFromView();
                };
                slider.addEventListener('input', () => applyAxisValue(slider.value));
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
    const getValueLegendItems = () => {
        const items = [];
        if (!state.colorizeOn)
            return items;
        tensors.forEach((group, name) => {
            const p = state.payloads.get(name);
            if (p) {
                items.push(createLegendItem(name === 'Global' ? type : name, p.scaleMin, p.scaleMax));
            }
        });
        return items;
    };
    const renderProgramCounts = () => {
        // paint counts across all cubes and attach the legend
        const programCounts = state.programCounts;
        if (!programCounts)
            return;
        destroyLegends();
        tensors.forEach((group, name) => {
            const p = state.payloads.get(name);
            if (state.colorizeOn && p) {
                applyProgramCountHeatmap(group.userData.mesh, p, programCounts, PROGRAM_COUNT_PALETTE, group.userData.mesh.userData.color_base);
            }
            else {
                applyProgramCountColors(group.userData.mesh, programCounts, group.userData.mesh.userData.color_base, PROGRAM_COUNT_PALETTE);
            }
        });
        const items = getValueLegendItems();
        items.push(createProgramCountLegendItem(getBaseCountColor(), programCounts.maxCount, PROGRAM_COUNT_PALETTE));
        createLegends(items);
    };
    const renderProgramSubsets = () => {
        // paint subset colors across all cubes and attach the legend
        const programSubsets = state.programSubsets;
        if (!programSubsets)
            return;
        destroyLegends();
        tensors.forEach((group, name) => {
            const p = state.payloads.get(name);
            if (state.colorizeOn && p) {
                applyProgramSubsetHeatmap(group.userData.mesh, p, programSubsets, state.programSubsetHues, group.userData.mesh.userData.color_base);
            }
            else {
                applyProgramSubsetColors(group.userData.mesh, programSubsets, state.programSubsetHues, group.userData.mesh.userData.color_base);
            }
        });
        const items = getValueLegendItems();
        items.push(createProgramSubsetLegendItem(getBaseCountColor(), programSubsets.subsets, state.programSubsetHues));
        createLegends(items);
        refreshDescriptorDimensionLines(vizCache);
    };
    const applyProgramPayload = (payload) => {
        const subsets = normalizeProgramSubsets(payload);
        if (subsets.subsetCount > PROGRAM_SUBSET_LIMIT) {
            state.allProgramsMode = 'count';
            state.programCounts = normalizeProgramCounts(payload);
            state.programSubsets = null;
            state.programSubsetHues = null;
        }
        else {
            state.allProgramsMode = 'subset';
            state.programSubsets = subsets;
            state.programSubsetHues = buildSubsetHues(subsets.subsets);
            state.programCounts = null;
        }
    };
    const ensureProgramData = async () => {
        // fetch subset data for the current op if needed
        if (!supportsAllPrograms || !state.allProgramsOn)
            return false;
        if (!state.programSubsets && !state.programCounts) {
            const payload = await fetchProgramCounts(API_BASE, op);
            if (!payload) {
                state.allProgramsOn = false;
                if (window.setOpControlState)
                    window.setOpControlState({ allPrograms: false });
                return false;
            }
            applyProgramPayload(payload);
        }
        if (state.allProgramsMode === 'subset' && state.programSubsets) {
            renderProgramSubsets();
        }
        else if (state.programCounts) {
            renderProgramCounts();
        }
        refreshDescriptorDimensionLines(vizCache);
        requestRender();
        return true;
    };
    const ensureProgramDataForHover = () => {
        if (!supportsAllPrograms)
            return;
        if (state.programSubsets || state.programCounts || state.programDataLoading)
            return;
        state.programDataLoading = true;
        fetchProgramCounts(API_BASE, op).then((payload) => {
            if (payload) {
                applyProgramPayload(payload);
                if (state.allProgramsOn) {
                    if (state.allProgramsMode === 'subset' && state.programSubsets) {
                        renderProgramSubsets();
                    }
                    else if (state.programCounts) {
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
                    }
                    else if (state.programCounts) {
                        renderProgramCounts();
                    }
                }
                else if (!state.colorizeOn) {
                    destroyLegends();
                    tensors.forEach((group, name) => updateTensorHighlights(group, state.payloads.get(name)?.highlights, vizCache.highlightColor, group.userData.mesh.userData.color_base));
                }
                else {
                    const items = [];
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
            toggleHistogram: () => {
                const isVisible = vizCache.histogramUI.overlay.style.display === 'block';
                if (isVisible) {
                    vizCache.histogramUI.hide?.();
                }
                else {
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
        }
        else if (state.colorizeOn) {
            const items = [];
            tensors.forEach((group, name) => {
                const p = state.payloads.get(name);
                if (p) {
                    items.push(createLegendItem(name === 'Global' ? type : name, p.scaleMin, p.scaleMax));
                    applyColorizedMesh(vizCache, group, name);
                }
            });
            createLegends(items);
        }
        else {
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
                    }
                    else {
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
