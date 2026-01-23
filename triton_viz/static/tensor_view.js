import { createCadDimension, createShapeLegend } from './dimension_utils.js';
import { clamp01, getHue, hslToRgb } from './colormap.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';
import {
    setupScene,
    setupGeometries,
    createTensor,
    calculateTensorSize,
    setupCamera,
    setupEventListeners,
    cameraControls,
    addLabels,
    COLOR_EDGE,
    CUBE_SIZE,
    GAP,
    COLOR_HOVER,
    updateTensorHighlights,
} from './load_utils.js';
import { createHistogramOverlay } from './histogram.js';
import { enableDrag } from './ui_helpers.js';

const VIZ_CACHE = new Map();

// --- Top-Level Helpers ---

function coordsFromIndex(index, shape) {
    const w = Math.max(1, shape.width), h = Math.max(1, shape.height);
    const z = Math.floor(index / (w * h)), rem = index % (w * h);
    const y = Math.floor(rem / w), x = rem % w;
    return [x, y, z];
}

function sampleValueFromCache(cache, coords) {
    if (!cache || !cache.values) return 0;
    const [x, y, z] = coords;
    const dims = cache.dims || (Array.isArray(cache.values[0]) ? (Array.isArray(cache.values[0][0]) ? 3 : 2) : 1);
    if (dims >= 3) return cache.values?.[y]?.[x]?.[z] ?? 0;
    if (dims === 2) return cache.values?.[y]?.[x] ?? 0;
    return cache.values?.[x] ?? 0;
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
    el.innerHTML = `<h3>${name} Tensor</h3><p>Coords: [${coords.join(', ')}]</p><p>Value: ${val}</p><p>Shape: ${shapeStr}</p>${extraHtml}`;
}

async function fetchTensorPayload(apiBase, uuid, endpoint = 'getLoadTensor') {
    try {
        const res = await fetch(`${apiBase}/api/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uuid })
        });
        const data = await res.json();
        if (!data || data.error) return null;
        return {
            scaleMin: data.min ?? 0, scaleMax: data.max ?? 0,
            values: data.values, shape: data.shape, dims: data.dims,
            highlights: data.highlights,
        };
    } catch (e) { return null; }
}

function applyColorToMesh(mesh, cache, label) {
    if (!mesh || !cache) return;
    const min = cache.scaleMin, max = cache.scaleMax, denom = max - min || 1;
    const hue = getHue(label), count = mesh.count, shape = mesh.userData.shape;
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
        const coords = coordsFromIndex(i, shape);
        const val = sampleValueFromCache(cache, coords);
        const t = clamp01((val - min) / denom);
        const [r, g, b] = hslToRgb(hue, 0.9, t);
        c.setRGB(r, g, b);
        mesh.setColorAt(i, c);
    }
    mesh.instanceColor.needsUpdate = true;
}

function applyDimmedColormap(mesh, cache, label, isHighlighted) {
    if (!mesh || !cache) return;
    const min = cache.scaleMin, max = cache.scaleMax, denom = max - min || 1;
    const hue = getHue(label), count = mesh.count, shape = mesh.userData.shape;
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
        const coords = coordsFromIndex(i, shape);
        const val = sampleValueFromCache(cache, coords);
        const t = clamp01((val - min) / denom);
        if (isHighlighted(coords)) {
            const [r, g, b] = hslToRgb(hue, 0.9, t);
            c.setRGB(r, g, b);
        } else {
            c.setRGB(t, t, t);
        }
        mesh.setColorAt(i, c);
    }
    mesh.instanceColor.needsUpdate = true;
}

function applyMonochromeColormap(mesh, cache) {
    if (!mesh || !cache) return;
    const min = cache.scaleMin, max = cache.scaleMax, denom = max - min || 1;
    const count = mesh.count, shape = mesh.userData.shape;
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
        const coords = coordsFromIndex(i, shape);
        const val = sampleValueFromCache(cache, coords);
        const t = clamp01((val - min) / denom);
        c.setRGB(t, t, t);
        mesh.setColorAt(i, c);
    }
    mesh.instanceColor.needsUpdate = true;
}

function getHighlightPredicate(highlights) {
    if (!highlights) return null;
    if (highlights.type === 'descriptor') {
        const { start, shape } = highlights;
        const [sx, sy, sz] = start || [0, 0, 0];
        const [dx, dy, dz] = shape || [0, 0, 0];
        if ((dx || 0) <= 0 || (dy || 0) <= 0 || (dz || 0) <= 0) return null;
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

function applyColorizedMesh(ctx, group, name) {
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

function restoreTensorColors(ctx) {
    const { state, tensors, type } = ctx;
    tensors.forEach((group, name) => {
        const mesh = group.userData.mesh;
        const p = state.payloads.get(name);
        if (state.colorizeOn && p) {
            applyColorizedMesh(ctx, group, name);
        } else {
            updateTensorHighlights(group, p?.highlights, ctx.highlightColor, mesh.userData.color_base);
        }
    });
}

function applyDotHoverHighlight(ctx, row, col) {
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

function applyDotHoverOutline(ctx, row, col) {
    const { tensors } = ctx;
    const aGroup = tensors.get('A');
    const bGroup = tensors.get('B');
    if (!aGroup || !bGroup) return;
    updateTensorHighlights(aGroup, null, ctx.highlightColor, aGroup.userData.mesh.userData.color_base, (x, y) => y === row);
    updateTensorHighlights(bGroup, null, ctx.highlightColor, bGroup.userData.mesh.userData.color_base, (x) => x === col);
}

function captureHistogramState(histogramUI) {
    const overlay = histogramUI?.overlay;
    if (!overlay) return {};
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
    if (!overlay || !state) return;
    const select = overlay.querySelector('#histogram-source');
    const bins = overlay.querySelector('#histogram-bins');
    if (select && state.histogramSource) {
        select.value = state.histogramSource;
    }
    if (bins && Number.isFinite(state.histogramBins)) {
        bins.value = state.histogramBins;
    }
    if (state.histogramVisible) {
        histogramUI.show?.();
    } else {
        histogramUI.hide?.();
    }
}

function createLegendItem(label, min, max) {
    const item = document.createElement('div');
    Object.assign(item.style, { display: 'grid', gap: '4px', fontFamily: 'monospace', fontSize: '12px' });
    const title = document.createElement('div');
    title.textContent = `${label} Value`;
    title.style.opacity = '0.9'; title.style.fontWeight = 'bold';
    const canvas = document.createElement('canvas');
    canvas.width = 220; canvas.height = 10;
    const ctx = canvas.getContext('2d');
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

function addDimensionLines(scene, tensorGroup, dimColors = []) {
    const mesh = tensorGroup?.userData?.mesh;
    if (!mesh) return [];
    const shape = mesh.userData.shape;
    const shapeRaw = mesh.userData.shape_raw || [];
    const bbox = new THREE.Box3().setFromObject(tensorGroup);
    const offsetBase = (CUBE_SIZE + GAP) * 1.5;
    const axisDefaults = { x: '#f87171', y: '#4ade80', z: '#60a5fa' };
    const getColor = (axis) => {
        if (shapeRaw.length === 1 && axis === 'x') return dimColors[0] || axisDefaults.x;
        if (shapeRaw.length === 2 && axis === 'y') return dimColors[0] || axisDefaults.y;
        if (shapeRaw.length === 2 && axis === 'x') return dimColors[1] || axisDefaults.x;
        if (shapeRaw.length >= 3 && axis === 'z') return dimColors[0] || axisDefaults.z;
        if (shapeRaw.length >= 3 && axis === 'y') return dimColors[1] || axisDefaults.y;
        if (shapeRaw.length >= 3 && axis === 'x') return dimColors[2] || axisDefaults.x;
        return axisDefaults[axis];
    };
    const groups = [];
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

function onMouseMove(event, ctx) {
    const { renderer, camera, tensors, state, sideMenu, requestRender, raycaster, mouse, API_BASE, op } = ctx;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const meshes = Array.from(tensors.values()).map(t => t.userData.mesh);
    const hits = raycaster.intersectObjects(meshes);

    if (hits.length > 0) {
        const hit = hits[0];
        const mesh = hit.object;
        const instanceId = hit.instanceId;
        const tensorName = mesh.userData.tensorName;
        const key = `${tensorName}_${instanceId}`;

        if (key !== state.lastHoverKey) {
            state.lastHoverKey = key;
            const coords = mesh.userData.coords[instanceId];
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

            updateSideMenu(sideMenu, tensorName, coords, val, currentShape);
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
        }
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

function onMouseUp(ctx) { ctx.state.isDragging = false; if (ctx.stage) ctx.stage.style.cursor = ''; }

// --- Main Exports ---

export function createTensorVisualization(containerElement, op, options = {}) {
    const { type = 'Load', colors = {}, tensorConfigs = [], dimColors = {}, showDimLines = true, viewState = null } = options;
    const API_BASE = window.__TRITON_VIZ_API__ || '';
    const configs = tensorConfigs.length > 0 ? tensorConfigs : [
        { name: 'Global', shape: op.global_shape, color: colors.GLOBAL || '#333', position: [0,0,0], endpoint: 'getLoadTensor' }
    ];

    let cache = VIZ_CACHE.get(containerElement);
    const shapeKey = JSON.stringify(configs.map(c => c.shape));
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
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();
        const tensors = new Map();
        configs.forEach(cfg => {
            const group = createTensor(cfg.shape, null, cfg.color, cfg.name, cubeGeometry, edgesGeometry, lineMaterial);
            group.position.set(...(cfg.position || [0,0,0]));
            group.userData.endpoint = cfg.endpoint;
            scene.add(group);
            tensors.set(cfg.name, group);
        });
        const hoverOutline = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05)), new THREE.LineBasicMaterial({ color: COLOR_HOVER }));
        hoverOutline.visible = false;
        scene.add(hoverOutline);
        const { center } = setupCamera(scene, camera);
        const orbitControls = new OrbitControls(camera, renderer.domElement);
        orbitControls.enableDamping = false;
        orbitControls.target.copy(center);
        orbitControls.update();
        const state = { colorizeOn: false, payloads: new Map(), rafId: null, renderPending: false, lastHoverKey: null, activeHoverOutline: hoverOutline, dotHoverKey: null };
        const highlightColor = colors.HIGHLIGHT || new THREE.Color(0.0, 0.7, 1.0);
        const ctx = { type, shapeKey, containerElement, sideMenu, histogramUI, stage, API_BASE, op, scene, camera, renderer, tensors, orbitControls, lineMaterial, state, raycaster: new THREE.Raycaster(), mouse: new THREE.Vector2(), legendContainer: null, dimLineGroups: [], highlightColor };
        ctx.requestRender = () => {
            if (state.rafId !== null) { state.renderPending = true; return; }
            state.rafId = requestAnimationFrame(() => { state.rafId = null; orbitControls.update(); renderer.render(scene, camera); if (state.renderPending) { state.renderPending = false; ctx.requestRender(); } });
        };
        orbitControls.addEventListener('change', ctx.requestRender);
        ctx.applyBackgroundTheme = (hex) => {
            const isLight = (hex || '').toLowerCase() === '#ffffff';
            const baseGlobalLight = new THREE.Color('#fefce8');
            tensors.forEach((group, name) => {
                const mesh = group.userData.mesh;
                const baseColor = isLight ? baseGlobalLight : mesh.userData.color_base;
                updateTensorHighlights(group, state.payloads.get(name)?.highlights, highlightColor, baseColor);
                group.children.forEach(c => { if (c?.userData?.edges) c.userData.edges.visible = !isLight; });
            });
            if (!isLight && lineMaterial) { lineMaterial.color.set('#ffffff'); lineMaterial.opacity = 0.28; }
            ctx.requestRender();
        };
        ctx.destroyLegends = () => { if (ctx.legendContainer?.remove) ctx.legendContainer.remove(); ctx.legendContainer = null; };
        ctx.createLegends = (items) => {
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
        const applyViewState = (nextState) => {
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
            applyHistogramState(histogramUI, nextState);
            if (window.setOpControlState) {
                window.setOpControlState({
                    colorize: state.colorizeOn,
                    histogram: !!nextState.histogramVisible,
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
                ...captureHistogramState(histogramUI),
            };
        };
        setupEventListeners(stage, camera, renderer, (e) => onMouseMove(e, ctx), cameraControls(camera, new THREE.Euler(0,0,0,'YXZ')), ctx.requestRender);
        if (showDimLines) {
            tensors.forEach((group, name) => {
                ctx.dimLineGroups.push(...addDimensionLines(scene, group, dimColors[name]));
            });
        }
        applyViewState(viewState);
        containerElement.__vizGetState = getViewState;
        ctx.cleanup = () => {
            if (state.rafId) cancelAnimationFrame(state.rafId);
            ctx.destroyLegends();
            ctx.dimLineGroups.forEach((group) => scene.remove(group));
            ctx.dimLineGroups = [];
            if (stage.parentElement) stage.parentElement.removeChild(stage);
            if (sideMenu.parentElement) sideMenu.parentElement.removeChild(sideMenu);
            if (histogramUI.overlay?.parentElement) histogramUI.overlay.parentElement.removeChild(histogramUI.overlay);
            if (containerElement.__vizGetState) {
                containerElement.__vizGetState = null;
            }
            VIZ_CACHE.delete(containerElement);
        };
        cache = ctx;
        VIZ_CACHE.set(containerElement, cache);
    } else {
        if (cache.stage.parentElement !== containerElement) {
            containerElement.innerHTML = '';
            containerElement.appendChild(cache.stage);
            containerElement.appendChild(cache.sideMenu);
            if (cache.histogramUI.overlay) containerElement.appendChild(cache.histogramUI.overlay);
            if (cache.legendContainer) containerElement.appendChild(cache.legendContainer);
        }
    }

    const { state, tensors, sideMenu, requestRender, applyBackgroundTheme, createLegends, destroyLegends } = cache;
    state.payloads.clear();
    sideMenu.innerHTML = '';
    window.current_op_uuid = op.uuid;
    if (window.setOpControlHandlers) {
        window.setOpControlHandlers({
            toggleColorize: async () => {
                state.colorizeOn = !state.colorizeOn;
                if (!state.colorizeOn) {
                    destroyLegends();
                    tensors.forEach((group, name) => updateTensorHighlights(group, state.payloads.get(name)?.highlights, cache.highlightColor, group.userData.mesh.userData.color_base));
                } else {
                    const items = [];
                    tensors.forEach((group, name) => {
                        const p = state.payloads.get(name);
                        if (p) {
                            items.push(createLegendItem(name === 'Global' ? type : name, p.scaleMin, p.scaleMax));
                            applyColorizedMesh(cache, group, name);
                        }
                    });
                    createLegends(items);
                }
                requestRender();
                return state.colorizeOn;
            },
            toggleShowCode: () => window.__tritonVizCodeToggle?.(),
            toggleHistogram: () => { if (cache.histogramUI.overlay.style.display === 'block') cache.histogramUI.hide?.(); else cache.histogramUI.show?.(); },
        });
    }

    const fetchers = Array.from(tensors.entries()).map(([name, group]) => {
        return fetchTensorPayload(API_BASE, op.uuid, group.userData.endpoint).then(p => {
            if (p) {
                state.payloads.set(name, p);
                updateTensorHighlights(group, p.highlights, cache.highlightColor, group.userData.mesh.userData.color_base);
            }
        });
    });

    Promise.all(fetchers).then(() => {
        if (state.colorizeOn) {
            const items = [];
            tensors.forEach((group, name) => {
                const p = state.payloads.get(name);
                if (p) {
                    items.push(createLegendItem(name === 'Global' ? type : name, p.scaleMin, p.scaleMax));
                    applyColorizedMesh(cache, group, name);
                }
            });
            createLegends(items);
        } else {
            tensors.forEach((group, name) => {
                const p = state.payloads.get(name);
                if (p) {
                    updateTensorHighlights(group, p.highlights, cache.highlightColor, group.userData.mesh.userData.color_base);
                }
            });
            if (state.dotHoverKey && type === 'Dot') {
                const [row, col] = state.dotHoverKey.split(',').map((val) => Number(val));
                if (Number.isFinite(row) && Number.isFinite(col)) {
                    if (state.colorizeOn) {
                        applyDotHoverHighlight(cache, row, col);
                    } else {
                        applyDotHoverOutline(cache, row, col);
                    }
                }
            }
        }
        requestRender();
    });

    createShapeLegend(containerElement, Array.from(tensors.entries()).map(([name, group]) => ({
        name: name === 'Global' ? type : `Matrix ${name}`,
        shape: group.userData.mesh.userData.shape_raw,
        color: '#' + group.userData.mesh.userData.color_base.getHexString(),
        dimColors: dimColors?.[name]
    })));

    applyBackgroundTheme('#000000');
    requestRender();
    return cache.cleanup;
}

export function createOverallVisualization(containerElement, op, options = {}) {
    const { type = 'Load' } = options;
    const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);
    const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);
    const tiles = op.overall_tiles || [];
    const globalShape = op.overall_shape || op.global_shape || [];
    containerElement.innerHTML = '';
    const sceneRoot = document.createElement('div');
    sceneRoot.className = 'viz-stage';
    containerElement.appendChild(sceneRoot);
    const legend = document.createElement('div');
    Object.assign(legend.style, { position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.65)', color: '#fff', padding: '8px 10px', borderRadius: '6px', maxHeight: '200px', overflow: 'auto', fontSize: '12px', zIndex: 10 });
    legend.innerHTML = '<strong>Program Blocks</strong><br/>';
    sceneRoot.appendChild(legend);
    let currentBackground = COLOR_BACKGROUND;
    const controlBar = document.createElement('div');
    Object.assign(controlBar.style, { position: 'absolute', top: '10px', left: '10px', zIndex: 12, display: 'flex', gap: '8px', alignItems: 'center', background: 'rgba(0,0,0,0.55)', color: '#fff', padding: '6px 8px', borderRadius: '6px', fontSize: '12px' });
    const bgLabel = document.createElement('span');
    bgLabel.textContent = 'Background';
    controlBar.appendChild(bgLabel);
    const backgroundSelect = document.createElement('select');
    [['Dark', '#000000'], ['Paper', '#ffffff'], ['Beige', '#f7f0e3'], ['Slate', '#0f172a']].forEach(([label, value]) => {
        const opt = document.createElement('option'); opt.value = value; opt.textContent = label; backgroundSelect.appendChild(opt);
    });
    backgroundSelect.value = '#000000';
    controlBar.appendChild(backgroundSelect);
    sceneRoot.appendChild(controlBar);
    const { scene, camera, renderer } = setupScene(sceneRoot, currentBackground);
    const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();
    const globalTensor = createTensor(globalShape, [], COLOR_GLOBAL, 'Global', cubeGeometry, edgesGeometry, lineMaterial);
    scene.add(globalTensor);
    createShapeLegend(containerElement, [{ name: 'Global', shape: globalShape, color: '#' + COLOR_GLOBAL.getHexString() }]);
    let labelSprites = addLabels(scene, globalTensor, null, currentBackground);
    const refreshLabels = () => { (labelSprites || []).forEach((sprite) => scene.remove(sprite)); labelSprites = addLabels(scene, globalTensor, null, currentBackground) || []; };
    const buildCubeMap = (tensor) => {
        const map = new Map();
        tensor.children.forEach((cube) => { if (cube.userData && typeof cube.userData.tensor0 === 'number') { const key = `${cube.userData.tensor0},${cube.userData.tensor1},${cube.userData.tensor2}`; map.set(key, cube); } });
        return map;
    };
    const paintCoords = (map, coords = [], color) => { coords.forEach(([x, y, z = 0]) => { const key = `${Math.round(x)},${Math.round(y)},${Math.round(z)}`; const cube = map.get(key); if (cube) cube.material.color.copy(color); }); };
    const createLegendRow = (idx, color) => {
        const row = document.createElement('div');
        row.style.display = 'flex'; row.style.alignItems = 'center'; row.style.gap = '6px';
        const swatch = document.createElement('span');
        Object.assign(swatch.style, { display: 'inline-block', width: '14px', height: '14px', background: `#${color.getHexString()}` });
        row.appendChild(swatch);
        const label = document.createElement('span');
        label.textContent = `Program Block ${idx + 1}`;
        row.appendChild(label);
        return row;
    };
    const globalMap = buildCubeMap(globalTensor);
    const renderTilesForBackground = (bgHex) => {
        const isLight = (bgHex || '').toLowerCase() === '#ffffff' || (bgHex || '').toLowerCase() === '#f7f0e3';
        const baseGlobal = isLight ? new THREE.Color('#fefce8') : COLOR_GLOBAL;
        globalTensor.children.forEach((cube) => { if (cube && cube.material && cube.material.color) cube.material.color.copy(baseGlobal); });
        legend.innerHTML = '<strong>Program Blocks</strong><br/>';
        tiles.forEach((tile, idx) => {
            const color = isLight ? new THREE.Color(['#ff8a3c', '#ffd54a', '#a3e635', '#34d399', '#5fd4ff', '#4f8bff', '#6366f1', '#f9739b'][idx % 8]) : new THREE.Color().setHSL((idx * 0.17) % 1, 0.65, 0.55);
            legend.appendChild(createLegendRow(idx, color));
            paintCoords(globalMap, tile.global_coords, color);
        });
    };
    renderTilesForBackground(backgroundSelect.value || '#000000');
    const { center } = setupCamera(scene, camera);
    const orbitControls = new OrbitControls(camera, renderer.domElement);
    orbitControls.enableDamping = false; orbitControls.target.copy(center); orbitControls.update();
    renderer.setClearColor(currentBackground, 1);
    sceneRoot.appendChild(renderer.domElement);
    backgroundSelect.addEventListener('change', (event) => { const value = event.target.value; currentBackground = new THREE.Color(value); scene.background = currentBackground; renderer.setClearColor(currentBackground, 1); renderTilesForBackground(value); refreshLabels(); requestRender(); });
    let rafId = null; let renderPending = false;
    const requestRender = () => { if (rafId !== null) { renderPending = true; return; } rafId = requestAnimationFrame(renderFrame); };
    const renderFrame = () => { orbitControls.update(); renderer.render(scene, camera); if (renderPending) { renderPending = false; rafId = requestAnimationFrame(renderFrame); } else rafId = null; };
    requestRender();
    return () => { if (rafId) cancelAnimationFrame(rafId); renderer.dispose(); containerElement.innerHTML = ''; };
}
