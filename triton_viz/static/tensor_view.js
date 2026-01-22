import { createShapeLegend } from './dimension_utils.js';
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
    COLOR_HOVER,
    updateTensorHighlights,
} from './load_utils.js';
import { createHistogramOverlay } from './histogram.js';
import { enableDrag } from './ui_helpers.js';

const COLORMAPS = {
    Load: [[0.1, 0.8, 0.9], [0.1, 0.2, 1.0]],   // Cyan to Blue
    Store: [[1.0, 0.95, 0.2], [1.0, 0.3, 0.0]], // Yellow to Orange
    A: [[0.1, 0.8, 0.9], [0.1, 0.2, 1.0]],
    B: [[1.0, 0.95, 0.2], [1.0, 0.3, 0.0]],
    C: [[0.4, 1.0, 0.4], [0.8, 0.2, 0.9]],
};

function clamp01(value) { return Math.min(1, Math.max(0, value)); }
function lerp(a, b, t) { return a + (b - a) * t; }

function getColormap(label) {
    return COLORMAPS[label] || COLORMAPS.Load;
}

export function createTensorVisualization(containerElement, op, options = {}) {
    const {
        type = 'Load',
        colors = {},
    } = options;

    const API_BASE = window.__TRITON_VIZ_API__ || '';
    fetch(`${API_BASE}/api/setop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uuid: op.uuid }),
    })
    .then(response => response.json())
    .catch((error) => console.error('Error:', error));

    try { window.current_op_uuid = op.uuid; } catch(e){}

    containerElement.innerHTML = '';
    containerElement.style.position = 'relative';
    const stage = document.createElement('div');
    stage.className = 'viz-stage';
    containerElement.appendChild(stage);

    const sideMenu = createSideMenu(containerElement);

    const histogramUI = createHistogramOverlay(containerElement, {
        title: `${type} Value Distribution`,
        apiBase: API_BASE,
        sources: [
            { value: 'GLOBAL', label: 'Global Tensor' }
        ],
        buildRequestBody: (source, bins) => ({
            uuid: op.uuid,
            source,
            bins
        }),
    });
    let histogramVisible = false;

    let colorizeOn = false;
    let tensorCache = null;
    let hoveredCube = null;
    let lastHoverKey = null;
    let rafId = null;
    let renderPending = false;
    let legendContainer = null;
    let labelSprites = [];
    let allProgramsOn = false;
    let allProgramTiles = null;
    let overlapMeshes = { global: null };
    let codePanel = null;

    let mouseDx = Number(localStorage.getItem('viz_mouse_dx') || 0);
    let mouseDy = Number(localStorage.getItem('viz_mouse_dy') || 0);
    let dragModeOn = false;
    let isDragging = false;
    let dragTarget = null;
    const dragPlane = new THREE.Plane();
    const planeIntersect = new THREE.Vector3();
    const worldPosHelper = new THREE.Vector3();
    const dragOffset = new THREE.Vector3();

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const tmpColor = new THREE.Color();

    // --- Helper Functions Definitions ---

    function requestRender() {
        if (rafId !== null) { renderPending = true; return; }
        rafId = requestAnimationFrame(renderFrame);
    }

    function renderFrame() {
        const needsMore = orbitControls.update();
        renderer.render(scene, camera);
        if (needsMore || renderPending) {
            renderPending = false;
            rafId = requestAnimationFrame(renderFrame);
            return;
        }
        rafId = null;
    }

    async function fetchTensorPayload() {
        try {
            const res = await fetch(`${API_BASE}/api/getLoadTensor`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uuid: op.uuid })
            });
            const data = await res.json();
            if (!data || data.error) return null;
            return {
                scaleMin: data.min ?? 0,
                scaleMax: data.max ?? 0,
                global: { dims: data.dims, values: data.values, shape: data.shape },
                highlights: data.highlights,
            };
        } catch (e) { return null; }
    }

    async function fetchAllProgramTiles() {
        if (!op.overall_key) return null;
        const endpoint = type === 'Store' ? 'store_overall' : 'load_overall';
        const res = await fetch(`${API_BASE}/api/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: op.overall_key, time_idx: op.time_idx })
        });
        const data = await res.json();
        if (!res.ok || data.error) return null;
        return data.tiles || [];
    }

    function destroyLegends() {
        if (legendContainer && legendContainer.remove) legendContainer.remove();
        legendContainer = null;
    }

    function createLegendItem(label, min, max) {
        const map = getColormap(label);
        const item = document.createElement('div');
        item.style.display = 'grid';
        item.style.gap = '4px';
        const title = document.createElement('div');
        title.textContent = `${label} Value`;
        title.style.opacity = '0.9';
        const canvas = document.createElement('canvas');
        canvas.width = 220; canvas.height = 10;
        const ctx = canvas.getContext('2d');
        for (let x = 0; x < canvas.width; x++) {
            const t = clamp01(x / (canvas.width - 1));
            const r = lerp(map[0][0], map[1][0], t);
            const g = lerp(map[0][1], map[1][1], t);
            const b = lerp(map[0][2], map[1][2], t);
            ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
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

    function createLegends(items) {
        destroyLegends();
        if (!items.length) return;
        const wrapper = document.createElement('div');
        Object.assign(wrapper.style, {
            position: 'absolute', left: '10px', bottom: '10px', background: 'rgba(0,0,0,0.6)',
            color: '#fff', padding: '8px', borderRadius: '6px', zIndex: '2000',
            display: 'grid', gap: '8px', pointerEvents: 'auto',
        });
        items.forEach((item) => wrapper.appendChild(item));
        containerElement.appendChild(wrapper);
        legendContainer = wrapper;
    }

    function applyColorToMesh(mesh, cache) {
        if (!mesh || !cache) return;
        const min = cache.scaleMin ?? 0;
        const max = cache.scaleMax ?? 0;
        const denom = max - min || 1;
        const shape = mesh.userData.shape || {};
        const map = getColormap(type);
        const count = mesh.count;
        const values = cache.global.values;

        for (let idx = 0; idx < count; idx++) {
            const coords = coordsFromIndex(idx, shape);
            const val = sampleValueFromCache(cache.global, coords);
            const t = clamp01((val - min) / denom);
            const r = lerp(map[0][0], map[1][0], t);
            const g = lerp(map[0][1], map[1][1], t);
            const b = lerp(map[0][2], map[1][2], t);
            tmpColor.setRGB(r, g, b);
            mesh.setColorAt(idx, tmpColor);
        }
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    async function toggleColorize() {
        if (allProgramsOn) {
            allProgramsOn = false;
            clearOverlapMeshes();
            if (window.setOpControlState) window.setOpControlState({ allPrograms: false });
        }
        colorizeOn = !colorizeOn;
        if (!colorizeOn) {
            destroyLegends();
            resetGlobalColors();
            requestRender();
            return colorizeOn;
        }
        if (!tensorCache) tensorCache = await fetchTensorPayload();
        if (!tensorCache) {
            colorizeOn = false;
            requestRender();
            return colorizeOn;
        }
        const item = createLegendItem(type, tensorCache.scaleMin, tensorCache.scaleMax);
        createLegends([item]);
        applyColorToMesh(globalMesh, tensorCache);
        requestRender();
        return colorizeOn;
    }

    async function toggleShowCode() {
        if (window.__tritonVizCodeToggle) return window.__tritonVizCodeToggle();
        return false;
    }

    function toggleHistogram() {
        histogramVisible = !histogramVisible;
        if (histogramVisible) {
            if (histogramUI.show) histogramUI.show();
            else if (histogramUI.overlay) histogramUI.overlay.style.display = 'block';
            return histogramVisible;
        }
        if (histogramUI.hide) histogramUI.hide();
        else if (histogramUI.overlay) histogramUI.overlay.style.display = 'none';
        return histogramVisible;
    }

    async function toggleAllPrograms() {
        allProgramsOn = !allProgramsOn;
        if (allProgramsOn) {
            if (colorizeOn) {
                colorizeOn = false;
                destroyLegends();
                if (window.setOpControlState) window.setOpControlState({ colorize: false });
            }
            if (!allProgramTiles) allProgramTiles = await fetchAllProgramTiles();
            if (!allProgramTiles || !allProgramTiles.length) {
                allProgramsOn = false;
                requestRender();
                return false;
            }
            applyAllProgramsTiles(allProgramTiles);
            requestRender();
            return true;
        }
        resetGlobalColors();
        clearOverlapMeshes();
        requestRender();
        return false;
    }

    function _updateMouseNDC(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        const dpr = (window.devicePixelRatio || 1);
        const px = (event.clientX - rect.left + mouseDx) * dpr;
        const py = (event.clientY - rect.top  + mouseDy) * dpr;
        const w = rect.width * dpr, h = rect.height * dpr;
        mouse.x = (px / w) * 2 - 1;
        mouse.y = -(py / h) * 2 + 1;
    }

    function _raycastAll() {
        raycaster.setFromCamera(mouse, camera);
        const targets = [];
        if (globalMesh) targets.push(globalMesh);
        return raycaster.intersectObjects(targets, false);
    }

    function _toTopLevelCube(obj) {
        let node = obj;
        while (node && !(node.userData && node.userData.tensorName)) node = node.parent;
        return node;
    }

    function resolveHitCoords(hit, mesh) {
        if (!hit || !mesh || typeof hit.instanceId !== 'number' || hit.instanceId < 0) return null;
        const coords = mesh.userData.coords;
        if (Array.isArray(coords) && coords[hit.instanceId]) return coords[hit.instanceId];
        const shape = mesh.userData.shape || {};
        const width = Math.max(1, shape.width || 1);
        const height = Math.max(1, shape.height || 1);
        const depth = Math.max(1, shape.depth || 1);
        const layerSize = Math.max(1, width * height);
        const z = Math.min(depth - 1, Math.floor(hit.instanceId / layerSize));
        const remainder = hit.instanceId - z * layerSize;
        const y = Math.min(height - 1, Math.floor(remainder / Math.max(1, width)));
        const x = Math.min(width - 1, remainder % Math.max(1, width));
        return [x, y, z];
    }

    async function onMouseMove(event) {
        _updateMouseNDC(event);
        raycaster.setFromCamera(mouse, camera);

        if (isDragging && dragTarget) {
            if (raycaster.ray.intersectPlane(dragPlane, planeIntersect)) {
                const newWorld = planeIntersect.add(dragOffset);
                dragTarget.parent.worldToLocal(newWorld);
                dragTarget.position.copy(newWorld);
            }
        }

        const intersects = _raycastAll();
        let currentKey = null;
        if (intersects.length > 0) {
            const hit = intersects[0];
            currentKey = `${hit.object.uuid}_${hit.instanceId}`;
        }

        if (currentKey === lastHoverKey) return;
        lastHoverKey = currentKey;

        hideHoverOutlines();
        hoveredCube = null;

        if (intersects.length > 0) {
            const hit = intersects[0];
            hoveredCube = _toTopLevelCube(hit.object);
            if (hoveredCube) {
                showHoverOutlineForMesh(hoveredCube, hit.instanceId);
                const { tensorName } = hoveredCube.userData;
                const coords = resolveHitCoords(hit, hoveredCube);
                if (!coords) {
                    updateSideMenu(null);
                    requestRender();
                    return;
                }
                const [tensor0, tensor1, tensor2] = coords;

                let cachedVal;
                if (tensorCache) {
                    const sub = tensorName === 'Global' ? tensorCache.global : null;
                    if (sub) cachedVal = sampleValueFromCache(sub, [tensor0, tensor1, tensor2]);
                } else {
                     if (!window._isFetchingTensor) {
                         window._isFetchingTensor = true;
                         fetchTensorPayload().then(data => {
                             window._isFetchingTensor = false;
                             if (data) tensorCache = data;
                         });
                     }
                }

                if (cachedVal !== undefined) updateSideMenu(tensorName, tensor0, tensor1, tensor2, cachedVal);
                else updateSideMenu(tensorName, tensor0, tensor1, tensor2, 'Loading data...');
            }
        } else {
            updateSideMenu(null);
        }
        requestRender();
    }

    function onMouseDown(event) {
        if (!dragModeOn) return;
        _updateMouseNDC(event);
        const hits = _raycastAll();
        if (hits.length === 0) return;
        const cube = _toTopLevelCube(hits[0].object);
        if (!cube) return;
        const normal = new THREE.Vector3();
        camera.getWorldDirection(normal);
        cube.getWorldPosition(worldPosHelper);
        dragPlane.setFromNormalAndCoplanarPoint(normal, worldPosHelper);
        raycaster.setFromCamera(mouse, camera);
        if (!raycaster.ray.intersectPlane(dragPlane, planeIntersect)) return;
        dragOffset.copy(worldPosHelper).sub(planeIntersect);
        isDragging = true;
        dragTarget = cube;
        stage.style.cursor = 'grabbing';
        requestRender();
    }

    function onMouseUp() {
        if (!dragModeOn) return;
        isDragging = false;
        dragTarget = null;
        stage.style.cursor = '';
        requestRender();
    }

    function coordsFromIndex(index, shape) {
        const width = Math.max(1, shape?.width || 1);
        const height = Math.max(1, shape?.height || 1);
        const depth = Math.max(1, shape?.depth || 1);
        const layerSize = Math.max(1, width * height);
        const z = Math.min(depth - 1, Math.floor(index / layerSize));
        const remainder = index - z * layerSize;
        const y = Math.min(height - 1, Math.floor(remainder / width));
        const x = Math.min(width - 1, remainder % width);
        return [x, y, z];
    }

    function sampleValueFromCache(cache, coords) {
        if (!cache || !cache.values) return 0;
        const dims = cache.dims || 0;
        const [x, y, z] = coords;
        const values = cache.values;
        if (dims >= 3) return values?.[y]?.[x]?.[z] ?? 0;
        if (dims === 2) return values?.[y]?.[x] ?? 0;
        if (dims === 1) return values?.[x] ?? 0;
        return 0;
    }

    function resetGlobalColors() {
        if (!globalMesh) return;
        updateTensorHighlights(globalTensor, coordsData, COLOR_HIGHLIGHT, currentBaseGlobal);
    }

    function resetBaseColors() {
        for (let idx = 0; idx < globalMesh.count; idx += 1) globalMesh.setColorAt(idx, currentBaseGlobal);
        if (globalMesh.instanceColor) globalMesh.instanceColor.needsUpdate = true;
    }

    function getRainbowColor(idx, total) {
        const denom = total > 0 ? total : 1;
        const hue = (idx / denom) % 1;
        return new THREE.Color().setHSL(hue, 0.7, 0.55);
    }

    function coordToInstanceId(mesh, coord) {
        if (!mesh || !coord) return null;
        const coords = mesh.userData.coords;
        if (coords) {
            if (!mesh.userData.coordIndex) {
                const map = new Map();
                coords.forEach((entry, idx) => map.set(entry.join(','), idx));
                mesh.userData.coordIndex = map;
            }
            return mesh.userData.coordIndex.get(coord.join(','));
        }
        const shape = mesh.userData.shape;
        if (!shape) return null;
        const perm = mesh.userData.coordPerm;
        const base = [coord[0], coord[1], coord[2] || 0];
        const mapped = perm && perm.length === 3
            ? [base[perm.indexOf(0)], base[perm.indexOf(1)], base[perm.indexOf(2)]]
            : base;
        const width = shape.width;
        const height = shape.height;
        const depth = shape.depth;
        const [x, y, z] = mapped;
        if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) return null;
        return z * width * height + y * width + x;
    }

    function clearOverlapMeshes() {
        const meshes = [overlapMeshes.global];
        meshes.forEach((mesh) => {
            if (!mesh) return;
            if (mesh.parent) mesh.parent.remove(mesh);
            if (mesh.material && mesh.material.dispose) mesh.material.dispose();
        });
        overlapMeshes = { global: null };
    }

    function buildSelectionMap(mesh, tiles, coordKey) {
        const selection = new Map();
        const total = tiles.length;
        tiles.forEach((tile, idx) => {
            const coords = tile[coordKey] || [];
            if (!coords.length) return;
            const color = getRainbowColor(idx, total);
            coords.forEach((coord) => {
                const instanceId = coordToInstanceId(mesh, coord);
                if (instanceId === null || instanceId === undefined) return;
                const entry = selection.get(instanceId);
                if (entry) entry.colors.push(color);
                else selection.set(instanceId, { colors: [color] });
            });
        });
        return selection;
    }

    function applySelectionMap(mesh, selection) {
        const overlaps = [];
        selection.forEach((entry, instanceId) => {
            const colors = entry.colors || [];
            if (!colors.length) return;
            mesh.setColorAt(instanceId, colors[0]);
            if (colors.length > 1) {
                const secondary = new THREE.Color(0, 0, 0);
                colors.slice(1).forEach((color) => secondary.add(color));
                secondary.multiplyScalar(1 / (colors.length - 1));
                overlaps.push({ instanceId, color: secondary });
            }
        });
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        return overlaps;
    }

    function buildOverlapMesh(baseMesh, entries) {
        if (!entries.length) return null;
        const overlayMaterial = new THREE.MeshBasicMaterial({
            vertexColors: true, transparent: true, opacity: 0.9,
        });
        const overlayMesh = new THREE.InstancedMesh(baseMesh.geometry, overlayMaterial, entries.length);
        overlayMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        overlayMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(entries.length * 3), 3);
        const matrix = new THREE.Matrix4();
        const position = new THREE.Vector3();
        const rotation = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        const shrink = 0.7;
        entries.forEach((entry, idx) => {
            baseMesh.getMatrixAt(entry.instanceId, matrix);
            matrix.decompose(position, rotation, scale);
            scale.multiplyScalar(shrink);
            matrix.compose(position, rotation, scale);
            overlayMesh.setMatrixAt(idx, matrix);
            overlayMesh.setColorAt(idx, entry.color);
        });
        overlayMesh.instanceMatrix.needsUpdate = true;
        if (overlayMesh.instanceColor) overlayMesh.instanceColor.needsUpdate = true;
        overlayMesh.renderOrder = 1;
        return overlayMesh;
    }

    function applyAllProgramsTiles(tiles) {
        if (!tiles || !tiles.length) return;
        resetBaseColors();
        clearOverlapMeshes();
        const globalSelection = buildSelectionMap(globalMesh, tiles, 'global_coords');
        const globalOverlaps = applySelectionMap(globalMesh, globalSelection);
        overlapMeshes.global = buildOverlapMesh(globalMesh, globalOverlaps);
        if (overlapMeshes.global) globalTensor.add(overlapMeshes.global);
    }

    function updateSideMenu(tensorName, x, y, z, value) {
        if (!tensorName) { sideMenu.innerHTML = ''; return; }
        let dims = op.global_shape;
        sideMenu.innerHTML = `
            <h3 style="margin-top: 0;">${tensorName} Tensor</h3>
            <p>X: ${x + 1}</p>
            <p>Y: ${y + 1}</p>
            <p>Z: ${z + 1}</p>
            <p>Dimensions: ${dims.join(' x ')}</p>
            <p>Value: ${value !== undefined ? value : 'Loading...'}</p>
        `;
    }

    function createSideMenu(container) {
        const menu = document.createElement('div');
        Object.assign(menu.style, {
            position: 'absolute', right: '16px', bottom: '16px', width: '220px', padding: '10px',
            background: 'rgba(0,0,0,0.65)', color: '#fff', borderRadius: '8px',
            fontFamily: 'var(--font-sans)', fontSize: '12px', zIndex: 2500,
        });
        container.appendChild(menu);
        return menu;
    }

    function isLightBackgroundHex(hex) {
        const h = (hex || '').toLowerCase();
        return h === '#ffffff' || h === '#f7f0e3' || h === '#f5f7fb';
    }

    function applyBackgroundTheme(hex) {
        const isLight = isLightBackgroundHex(hex);
        currentBaseGlobal.copy(isLight ? baseGlobalLight : baseGlobalDark);
        resetGlobalColors();
        globalTensor.children.forEach((cube) => {
            if (cube?.userData?.edges) cube.userData.edges.visible = !isLight;
        });
        if (!isLight && lineMaterial) {
            lineMaterial.color.set('#ffffff');
            lineMaterial.opacity = 0.28;
        }
        requestRender();
    }

    // --- End Helper Functions ---

    if (window.setOpControlHandlers) {
        window.setOpControlHandlers({
            toggleColorize, toggleShowCode, toggleHistogram, toggleAllPrograms,
        });
    }
    if (window.setOpControlState) {
        window.setOpControlState({
            colorize: colorizeOn, showCode: false, histogram: false, allPrograms: false,
        });
    }

    const COLOR_GLOBAL = colors.GLOBAL || new THREE.Color(0.2, 0.2, 0.2);
    const COLOR_HIGHLIGHT = colors.HIGHLIGHT || new THREE.Color(0.0, 0.7, 1.0);
    const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);

    let currentBackground = COLOR_BACKGROUND;
    let sceneBundle = setupScene(stage, currentBackground);
    let { scene, camera, renderer } = sceneBundle;
    const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

    const globalTensor = createTensor(op.global_shape, null, COLOR_GLOBAL, 'Global', cubeGeometry, edgesGeometry, lineMaterial);
    const globalMesh = globalTensor.userData.mesh;
    scene.add(globalTensor);

    createShapeLegend(containerElement, [
        { name: 'Global', shape: op.global_shape, color: '#' + COLOR_GLOBAL.getHexString() }
    ]);

    const hoverGeometry = new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05);
    const hoverEdgesGeometry = new THREE.EdgesGeometry(hoverGeometry);
    const hoverMaterial = new THREE.LineBasicMaterial({ color: COLOR_HOVER });
    const globalHoverOutline = new THREE.LineSegments(hoverEdgesGeometry, hoverMaterial);
    globalHoverOutline.visible = false;
    globalTensor.add(globalHoverOutline);
    let activeHoverOutline = null;

    function hideHoverOutlines() {
        globalHoverOutline.visible = false;
        activeHoverOutline = null;
    }

    function showHoverOutlineForMesh(mesh, instanceId) {
        if (!mesh || typeof instanceId !== 'number' || instanceId < 0) return false;
        const outline = mesh === globalMesh ? globalHoverOutline : null;
        if (!outline) return false;
        mesh.getMatrixAt(instanceId, hoverMatrix);
        hoverMatrix.decompose(hoverPosition, hoverQuaternion, hoverScale);
        outline.position.copy(hoverPosition);
        outline.updateMatrixWorld();
        outline.visible = true;
        if (activeHoverOutline && activeHoverOutline !== outline) activeHoverOutline.visible = false;
        activeHoverOutline = outline;
        return true;
    }

    let coordsData = null;
    fetchTensorPayload().then(data => {
        if (data) {
            tensorCache = data;
            coordsData = data.highlights;
            updateTensorHighlights(globalTensor, coordsData, COLOR_HIGHLIGHT, currentBaseGlobal);
            requestRender();
        }
    });

    const baseGlobalDark = COLOR_GLOBAL.clone();
    const baseGlobalLight = new THREE.Color('#fefce8');
    let currentBaseGlobal = baseGlobalDark.clone();

    labelSprites = addLabels(scene, globalTensor, null, currentBackground);

    const refreshTextOverlays = () => {
        (labelSprites || []).forEach((s) => scene.remove(s));
        labelSprites = addLabels(scene, globalTensor, null, currentBackground) || [];
        createShapeLegend(containerElement, [
            { name: 'Global', shape: op.global_shape, color: '#' + COLOR_GLOBAL.getHexString() }
        ]);
    };

    try {
        const badge = document.createElement('div');
        badge.className = 'viz-floating-badge';
        const ms = (op.mem_src||'').toUpperCase();
        const md = (op.mem_dst||'').toUpperCase();
        const by = Number(op.bytes||0);
        if (ms && md) {
            badge.innerHTML = `<strong>Memory Flow</strong><br/>${ms} → ${md}${by?`<br/>${by} B`:''}`;
            stage.appendChild(badge);
        }
    } catch(e){}

    const { center } = setupCamera(scene, camera);
    const orbitControls = new OrbitControls(camera, renderer.domElement);
    orbitControls.enableDamping = false;
    orbitControls.dampingFactor = 0;
    orbitControls.target.copy(center);
    orbitControls.update();

    const viewStateStore = window.__tritonVizCameraState || (window.__tritonVizCameraState = {});
    const viewStateKey = `${type.toLowerCase()}:${op.overall_key || op.uuid}`;
    const saveCameraState = () => {
        viewStateStore[viewStateKey] = {
            position: camera.position.toArray(),
            target: orbitControls.target.toArray(),
        };
    };
    const restoreCameraState = () => {
        const state = viewStateStore[viewStateKey];
        if (!state) return;
        if (state.position) camera.position.set(...state.position);
        if (state.target) orbitControls.target.set(...state.target);
        orbitControls.update();
    };
    restoreCameraState();
    saveCameraState();
    orbitControls.addEventListener('change', () => {
        saveCameraState();
        requestRender();
    });

    const onKeyDown = cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ'));
    setupEventListeners(stage, camera, renderer, onMouseMove, onKeyDown, requestRender, saveCameraState, refreshTextOverlays);

    stage.addEventListener('mousedown', onMouseDown);
    stage.addEventListener('mouseup', onMouseUp);
    stage.addEventListener('mouseleave', onMouseUp);

    const defaultBackground = '#000000';
    currentBackground = new THREE.Color(defaultBackground);
    if (scene && scene.background) scene.background = currentBackground;
    applyBackgroundTheme(defaultBackground);
    refreshTextOverlays();
    requestRender();

    const hoverMatrix = new THREE.Matrix4();
    const hoverPosition = new THREE.Vector3();
    const hoverQuaternion = new THREE.Quaternion();
    const hoverScale = new THREE.Vector3();

    return () => {
        if (window.setOpControlHandlers) window.setOpControlHandlers(null);
        if (window.setOpControlState) window.setOpControlState({
            colorize: false, showCode: false, histogram: false, allPrograms: false,
        });
        if (window.__tritonVizCodeHide && !window.__tritonVizPreserveCodePanel) window.__tritonVizCodeHide();
        if (histogramUI.hide) histogramUI.hide();
        destroyLegends();
        saveCameraState();
        clearOverlapMeshes();
        if (rafId !== null) cancelAnimationFrame(rafId);
        if (renderer && renderer.dispose) renderer.dispose();
        if (renderer && renderer.domElement && renderer.domElement.parentElement) renderer.domElement.parentElement.removeChild(renderer.domElement);
        if (containerElement) containerElement.innerHTML = '';
    };
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
    Object.assign(legend.style, {
        position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.65)', color: '#fff',
        padding: '8px 10px', borderRadius: '6px', maxHeight: '200px', overflow: 'auto', fontSize: '12px', zIndex: 10,
    });
    legend.innerHTML = '<strong>Program Blocks</strong><br/>';
    sceneRoot.appendChild(legend);

    let currentBackground = COLOR_BACKGROUND;
    const controlBar = document.createElement('div');
    Object.assign(controlBar.style, {
        position: 'absolute', top: '10px', left: '10px', zIndex: 12, display: 'flex', gap: '8px', alignItems: 'center',
        background: 'rgba(0,0,0,0.55)', color: '#fff', padding: '6px 8px', borderRadius: '6px', fontSize: '12px',
    });
    const bgLabel = document.createElement('span');
    bgLabel.textContent = 'Background';
    controlBar.appendChild(bgLabel);
    const backgroundSelect = document.createElement('select');
    [['Dark', '#000000'], ['Paper', '#ffffff'], ['Beige', '#f7f0e3'], ['Slate', '#0f172a']].forEach(([label, value]) => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = label;
        backgroundSelect.appendChild(opt);
    });
    backgroundSelect.value = '#000000';
    controlBar.appendChild(backgroundSelect);
    sceneRoot.appendChild(controlBar);

    const { scene, camera, renderer } = setupScene(sceneRoot, currentBackground);
    const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

    const globalTensor = createTensor(globalShape, [], COLOR_GLOBAL, 'Global', cubeGeometry, edgesGeometry, lineMaterial);
    scene.add(globalTensor);
    createShapeLegend(containerElement, [
        { name: 'Global', shape: globalShape, color: '#' + COLOR_GLOBAL.getHexString() }
    ]);
    let labelSprites = addLabels(scene, globalTensor, null, currentBackground);
    const refreshLabels = () => {
        (labelSprites || []).forEach((sprite) => scene.remove(sprite));
        labelSprites = addLabels(scene, globalTensor, null, currentBackground) || [];
    };

    const globalMap = buildCubeMap(globalTensor);

    function isLightBackgroundHex(hex) {
        const h = (hex || '').toLowerCase();
        return h === '#ffffff' || h === '#f7f0e3';
    }

    function getTileColorForBackground(bgHex, idx) {
        if (isLightBackgroundHex(bgHex)) {
            const paperPalette = ['#ff8a3c', '#ffd54a', '#a3e635', '#34d399', '#5fd4ff', '#4f8bff', '#6366f1', '#f9739b'];
            return new THREE.Color(paperPalette[idx % paperPalette.length]);
        }
        return new THREE.Color().setHSL((idx * 0.17) % 1, 0.65, 0.55);
    }

    function recolorBackgroundCubes(bgHex) {
        const isLight = isLightBackgroundHex(bgHex);
        const baseGlobal = isLight ? new THREE.Color('#fefce8') : COLOR_GLOBAL;
        globalTensor.children.forEach((cube) => {
            if (cube && cube.material && cube.material.color) cube.material.color.copy(baseGlobal);
        });
        const switchMaterialForGroup = (group) => {
            group.traverse((obj) => {
                if (!obj || !obj.isMesh || !obj.material || !obj.material.color) return;
                if (isLight) {
                    if (!obj.userData.phongMaterial) obj.userData.phongMaterial = obj.material;
                    if (!(obj.material instanceof THREE.MeshBasicMaterial)) {
                        const color = obj.material.color.clone();
                        obj.material = new THREE.MeshBasicMaterial({ color, toneMapped: false });
                    }
                } else if (obj.userData.phongMaterial) obj.material = obj.userData.phongMaterial;
            });
        };
        switchMaterialForGroup(globalTensor);
        if (lineMaterial) {
            lineMaterial.visible = true;
            if (isLight) {
                lineMaterial.color.set('#111827');
                lineMaterial.opacity = 0.16;
            } else {
                lineMaterial.color.set('#ffffff');
                lineMaterial.opacity = 0.28;
            }
        }
    }

    function renderTilesForBackground(bgHex) {
        recolorBackgroundCubes(bgHex);
        legend.innerHTML = '<strong>Program Blocks</strong><br/>';
        tiles.forEach((tile, idx) => {
            const color = getTileColorForBackground(bgHex, idx);
            legend.appendChild(createLegendRow(idx, color));
            paintCoords(globalMap, tile.global_coords, color);
        });
    }

    renderTilesForBackground(backgroundSelect.value || '#000000');

    const { center } = setupCamera(scene, camera);
    const orbitControls = new OrbitControls(camera, renderer.domElement);
    orbitControls.enableDamping = false;
    orbitControls.dampingFactor = 0;
    orbitControls.target.copy(center);
    orbitControls.update();

    renderer.setClearColor(currentBackground, 1);
    sceneRoot.appendChild(renderer.domElement);

    backgroundSelect.addEventListener('change', (event) => {
        const value = event.target.value;
        currentBackground = new THREE.Color(value);
        scene.background = currentBackground;
        renderer.setClearColor(currentBackground, 1);
        renderTilesForBackground(value);
        refreshLabels();
        requestRender();
    });

    let rafId = null;
    let renderPending = false;
    function requestRender() {
        if (rafId !== null) { renderPending = true; return; }
        rafId = requestAnimationFrame(renderFrame);
    }
    function renderFrame() {
        const needsMore = orbitControls.update();
        renderer.render(scene, camera);
        if (needsMore || renderPending) {
            renderPending = false;
            rafId = requestAnimationFrame(renderFrame);
            return;
        }
        rafId = null;
    }
    requestRender();

    return () => {
        if (rafId) cancelAnimationFrame(rafId);
        renderer.dispose();
        if (renderer.domElement && renderer.domElement.parentElement) renderer.domElement.parentElement.removeChild(renderer.domElement);
        if (legend && legend.parentElement) legend.parentElement.removeChild(legend);
        containerElement.innerHTML = '';
    };

    function buildCubeMap(tensor) {
        const map = new Map();
        tensor.children.forEach((cube) => {
            if (cube.userData && typeof cube.userData.tensor0 === 'number') {
                const key = `${cube.userData.tensor0},${cube.userData.tensor1},${cube.userData.tensor2}`;
                map.set(key, cube);
            }
        });
        return map;
    }
    function paintCoords(map, coords = [], color) {
        coords.forEach(([x, y, z = 0]) => {
            const key = `${Math.round(x)},${Math.round(y)},${Math.round(z)}`;
            const cube = map.get(key);
            if (cube) cube.material.color.copy(color);
        });
    }
    function createLegendRow(idx, color) {
        const row = document.createElement('div');
        row.style.display = 'flex'; row.style.alignItems = 'center'; row.style.gap = '6px';
        const swatch = document.createElement('span');
        Object.assign(swatch.style, { display: 'inline-block', width: '14px', height: '14px', background: `#${color.getHexString()}` });
        row.appendChild(swatch);
        const label = document.createElement('span');
        label.textContent = `Program Block ${idx + 1}`;
        row.appendChild(label);
        return row;
    }
}
