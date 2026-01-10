import * as THREE from 'https://esm.sh/three@0.155.0/build/three.module.js';
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
} from './load_utils.js';
import { createHistogramOverlay } from './histogram.js';
import { enableDrag } from './ui_helpers.js';

export function createLoadVisualization(containerElement, op) {

        console.log(op.uuid);
        const API_BASE = window.__TRITON_VIZ_API__ || '';
        fetch(`${API_BASE}/api/setop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ uuid: op.uuid }),
        })
        .then(response => response.json())
        .then(data => console.log('Set current op:', data))
        .catch((error) => console.error('Error:', error));

        // expose current op uuid globally for generic code panel in gridblock
        try { window.current_op_uuid = op.uuid; } catch(e){}

        containerElement.innerHTML = '';
        containerElement.style.position = 'relative';
        const stage = document.createElement('div');
        stage.className = 'viz-stage';
        containerElement.appendChild(stage);

        let currentStep = 0;
        let frame = 0;
        let isPaused = false;

        const sideMenu = createSideMenu(containerElement);
        const histogramUI = createHistogramOverlay(containerElement, {
            title: 'Load Value Distribution',
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

        // expose for debugging
        try {
            window.last_op_global_shape = op.global_shape;
            window.last_global_coords = op.global_coords;
            window.last_slice_shape = op.slice_shape;
            window.last_slice_coords = op.slice_coords;
        } catch (e) {}

        let colorizeOn = false;
        let tensorCache = null; // {scaleMin, scaleMax, global:{dims,values}, slice:{dims,values}}
        let hoveredCube = null;
        let legendEl = null;
        let scheme = 'mono';
        let monoBaseHex = '#3b82f6';
let labelSprites = [];
const TEXT_LIGHT = '#ffffff';
const TEXT_DARK = '#111111';
        let allProgramsOn = false;
        let allProgramTiles = null;

function getTextColor(bgColor) {
    try {
        const c = (bgColor instanceof THREE.Color) ? bgColor : new THREE.Color(bgColor);
        const lum = c.getLuminance();
        return lum > 0.6 ? TEXT_DARK : TEXT_LIGHT;
    } catch (e) {
        return TEXT_LIGHT;
    }
}

        const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);    // Dark Gray (base for dark themes)
        const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);     // Cyan (highlighted global coords)
        const COLOR_LEFT_SLICE = new THREE.Color(1.0, 0.0, 1.0); // Magenta (slice base in dark themes)
        const COLOR_LOADED = new THREE.Color(1.0, 0.8, 0.0);    // Gold (final color for both slices)
        const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);  // Black

        let currentBackground = COLOR_BACKGROUND;
        let sceneBundle = setupScene(stage, currentBackground);
        let { scene, camera, renderer } = sceneBundle;
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

        const globalTensor = createTensor(op.global_shape, op.global_coords, COLOR_GLOBAL, 'Global', cubeGeometry, edgesGeometry, lineMaterial);
        const sliceTensor = createTensor(op.slice_shape, op.slice_coords, COLOR_LEFT_SLICE, 'Slice', cubeGeometry, edgesGeometry, lineMaterial);
        const globalMesh = globalTensor.userData.mesh;
        const sliceMesh = sliceTensor.userData.mesh;

        // Position slice tensor
        const globalSize = calculateTensorSize(op.global_shape);
        sliceTensor.position.set(globalSize.x + 5, 0, 0); // Adjusted tensor spacing

        scene.add(globalTensor);
        scene.add(sliceTensor);

        const hoverGeometry = new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05);
        const hoverEdgesGeometry = new THREE.EdgesGeometry(hoverGeometry);
        const hoverMaterial = new THREE.LineBasicMaterial({ color: COLOR_HOVER });
        const globalHoverOutline = new THREE.LineSegments(hoverEdgesGeometry, hoverMaterial);
        const sliceHoverOutline = new THREE.LineSegments(hoverEdgesGeometry, hoverMaterial);
        globalHoverOutline.visible = false;
        sliceHoverOutline.visible = false;
        globalTensor.add(globalHoverOutline);
        sliceTensor.add(sliceHoverOutline);

        // Precompute highlighted coords in Global tensor for quick reset
        const highlightedGlobalSet = new Set(
            op.global_coords.map(([x, y, z]) => `${x},${y},${z}`)
        );

        // Theme-aware base colors for Global / Slice;会随着背景切换而更新
        const baseGlobalDark = COLOR_GLOBAL.clone();
        const baseSliceDark = COLOR_LEFT_SLICE.clone();
        // 浅色主题下用更明亮、带一点色温的 pastel 色，避免灰蒙蒙
        const baseGlobalLight = new THREE.Color('#fefce8'); // very light warm yellow
        const baseSliceLight = new THREE.Color('#dbeafe');  // light blue
        let currentBaseGlobal = baseGlobalDark.clone();
        let currentBaseSlice = baseSliceDark.clone();

        labelSprites = addLabels(scene, globalTensor, sliceTensor, currentBackground);

        const refreshTextOverlays = () => {
            const list = Array.isArray(labelSprites) ? labelSprites : [];
            list.forEach((s) => scene.remove(s));
            labelSprites = addLabels(scene, globalTensor, sliceTensor, currentBackground) || [];
        };

        // Overlay memory flow badges if available (NKI only)
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
        orbitControls.enableDamping = true;
        orbitControls.dampingFactor = 0.05;
        orbitControls.target.copy(center);
        orbitControls.update();

        const totalFrames = op.global_coords.length * 2 + 30;

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        // persistent calibration offsets
        let mouseDx = Number(localStorage.getItem('viz_mouse_dx') || 0);
        let mouseDy = Number(localStorage.getItem('viz_mouse_dy') || 0);
        // Drag state
        let dragModeOn = false;
        let isDragging = false;
        let dragTarget = null; // THREE.Mesh (cube)
        const dragPlane = new THREE.Plane();
        const planeIntersect = new THREE.Vector3();
        const worldPosHelper = new THREE.Vector3();
        const dragOffset = new THREE.Vector3();

        function isLightBackgroundHex(hex) {
            const h = (hex || '').toLowerCase();
            return h === '#ffffff' || h === '#f7f0e3' || h === '#f5f7fb';
        }

        function applyBackgroundTheme(hex) {
            const isLight = isLightBackgroundHex(hex);

            // 1) Global/Slice 底色：暗色下用原色，浅色下用更柔和的浅色
            currentBaseGlobal.copy(isLight ? baseGlobalLight : baseGlobalDark);
            currentBaseSlice.copy(isLight ? baseSliceLight : baseSliceDark);
            resetGlobalColors();
            resetSliceColors();

            // 2) 边线：Dark / Slate 显示白线，浅色背景则隐藏
            globalTensor.children.forEach((cube) => {
                if (cube?.userData?.edges) {
                    cube.userData.edges.visible = !isLight;
                }
            });
            sliceTensor.children.forEach((cube) => {
                if (cube?.userData?.edges) {
                    cube.userData.edges.visible = !isLight;
                }
            });
            if (!isLight) {
                if (lineMaterial) {
                    lineMaterial.color.set('#ffffff');
                    lineMaterial.opacity = 0.28;
                }
            }
        }

        const defaultBackground = '#000000';
        currentBackground = new THREE.Color(defaultBackground);
        if (scene && scene.background) {
            scene.background = currentBackground;
        }
        applyBackgroundTheme(defaultBackground);
        refreshTextOverlays();

        const onKeyDown = cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ'));
        setupEventListeners(stage, camera, renderer, onMouseMove, onKeyDown);

        // Additional pointer events for dragging
        stage.addEventListener('mousedown', onMouseDown);
        stage.addEventListener('mouseup', onMouseUp);
        stage.addEventListener('mouseleave', onMouseUp);
        animate();

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
            const allTensorChildren = [
                ...globalTensor.children,
                ...sliceTensor.children
            ];
            return raycaster.intersectObjects(allTensorChildren, true);
        }

        function _toTopLevelCube(obj) {
            let node = obj;
            while (node && !(node.userData && node.userData.tensorName)) {
                node = node.parent;
            }
            return node;
        }

        async function onMouseMove(event) {
            _updateMouseNDC(event);

            raycaster.setFromCamera(mouse, camera);

            // Dragging: update target position on plane
            if (isDragging && dragTarget) {
                if (raycaster.ray.intersectPlane(dragPlane, planeIntersect)) {
                    const newWorld = planeIntersect.add(dragOffset);
                    // Convert world -> parent local
                    dragTarget.parent.worldToLocal(newWorld);
                    dragTarget.position.copy(newWorld);
                }
            }

            const intersects = _raycastAll();

            if (hoveredCube) {
                hoveredCube.getObjectByName('hoverOutline').visible = false;
                hoveredCube = null;
            }

            if (intersects.length > 0) {
                    hoveredCube = _toTopLevelCube(intersects[0].object);

                if (hoveredCube) {
                    const hoverOutline = hoveredCube.getObjectByName('hoverOutline');
                    if (hoverOutline) {
                        hoverOutline.visible = true;
                    }

                    const { tensorName, tensor0, tensor1, tensor2 } = hoveredCube.userData;
                    updateSideMenu(tensorName, tensor0, tensor1, tensor2, undefined);

                    const res = await getElementValue(tensorName, tensor0, tensor1, tensor2);

                    updateSideMenu(tensorName, tensor0, tensor1, tensor2, res.value);

                    console.log(`Value: ${res.value}`);
                }
            } else {
                updateSideMenu(null);
            }
        }

        // --------- Colormap (Viridis-like) and Legend ---------
        function lerp(a, b, t) { return a + (b - a) * t; }
        function lerpColor(c1, c2, t) {
            const r = Math.round(lerp(c1[0], c2[0], t)) / 255;
            const g = Math.round(lerp(c1[1], c2[1], t)) / 255;
            const b = Math.round(lerp(c1[2], c2[2], t)) / 255;
            return new THREE.Color(r, g, b);
        }
        // Viridis palette control points (sRGB)
        const VIRIDIS = [
            [0.00, [68, 1, 84]],
            [0.25, [59, 82, 139]],
            [0.50, [33, 145, 140]],
            [0.75, [94, 201, 98]],
            [1.00, [253, 231, 37]]
        ];
        function viridisColor(t) {
            if (t <= 0) return new THREE.Color().setRGB(...VIRIDIS[0][1].map(v=>v/255));
            if (t >= 1) return new THREE.Color().setRGB(...VIRIDIS[VIRIDIS.length-1][1].map(v=>v/255));
            for (let i = 0; i < VIRIDIS.length - 1; i++) {
                const [p, c] = VIRIDIS[i];
                const [pn, cn] = VIRIDIS[i+1];
                if (t >= p && t <= pn) {
                    const tt = (t - p) / (pn - p);
                    return lerpColor(c, cn, tt);
                }
            }
            return new THREE.Color(1,1,1);
        }

        function monoColor(t, hex) {
            // map to HSL with same hue/sat, varying lightness from 0.9 -> 0.25
            const base = new THREE.Color(hex);
            const hsl = {h:0,s:0,l:0};
            base.getHSL(hsl);
            const l = lerp(0.9, 0.25, t);
            const s = Math.min(1.0, Math.max(0.4, hsl.s)); // ensure enough saturation
            return new THREE.Color().setHSL(hsl.h, s, l);
        }

        function destroyLegend() {
            if (legendEl && legendEl.remove) legendEl.remove();
            legendEl = null;
        }

        function createLegend(min, max) {
            destroyLegend();
            const wrapper = document.createElement('div');
            wrapper.className = 'viz-floating-badge value-legend';
            wrapper.style.left = '24px';
            wrapper.style.top = '120px';
            wrapper.style.right = 'auto';
            wrapper.style.width = '260px';

            const header = document.createElement('div');
            header.style.display = 'flex';
            header.style.alignItems = 'center';
            header.style.justifyContent = 'space-between';
            header.style.marginBottom = '8px';

            const title = document.createElement('span');
            title.textContent = scheme === 'mono' ? 'Value (Mono)' : 'Value (Viridis)';
            title.style.fontWeight = '600';
            header.appendChild(title);

            const legendHandle = document.createElement('button');
            legendHandle.type = 'button';
            legendHandle.className = 'viz-drag-handle drag-handle';
            legendHandle.innerHTML = '<span aria-hidden="true">⠿</span>';
            legendHandle.style.marginLeft = '12px';
            header.appendChild(legendHandle);

            wrapper.appendChild(header);

            const canvas = document.createElement('canvas');
            canvas.width = 220; canvas.height = 12;
            const ctx2 = canvas.getContext('2d');
            for (let x = 0; x < canvas.width; x++) {
                const t = x / (canvas.width - 1);
                const c = (scheme === 'mono') ? monoColor(t, monoBaseHex) : viridisColor(t);
                ctx2.fillStyle = `rgb(${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)})`;
                ctx2.fillRect(x, 0, 1, canvas.height);
            }
            wrapper.appendChild(canvas);

            const labels = document.createElement('div');
            labels.style.display = 'flex';
            labels.style.justifyContent = 'space-between';
            labels.style.marginTop = '4px';
            labels.innerHTML = `<span>${min.toFixed(3)}</span><span>${max.toFixed(3)}</span>`;
            wrapper.appendChild(labels);

            stage.appendChild(wrapper);
            legendEl = wrapper;
            enableDrag(wrapper, { handle: legendHandle, bounds: stage, initialLeft: 32, initialTop: 140 });
        }

        function destroyCodePanel() {
            if (codePanel && codePanel.remove) codePanel.remove();
            codePanel = null;
        }

        async function createCodePanel(frameIdx = 0, context = 8) {
            destroyCodePanel();
            const wrapper = document.createElement('div');
            wrapper.className = 'show-code-panel';
            wrapper.style.top = '140px';
            wrapper.style.right = '32px';
            const header = document.createElement('div');
            header.className = 'panel-header drag-handle';
            header.style.marginBottom = '8px';
            header.innerHTML = '<span>Operation Code & Context</span><span class="drag-grip" aria-hidden="true">⠿</span>';
            const closeBtn = document.createElement('button');
            closeBtn.className = 'viz-button ghost';
            closeBtn.textContent = 'Close';
            closeBtn.style.marginLeft = 'auto';
            closeBtn.addEventListener('pointerdown', (e) => e.stopPropagation());
            closeBtn.addEventListener('click', () => {
                destroyCodePanel();
                if (window.setOpControlState) {
                    window.setOpControlState({ showCode: false });
                }
            });
            header.appendChild(closeBtn);
            wrapper.appendChild(header);

            try {
                const res = await fetch(`${API_BASE}/api/op_code`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ uuid: op.uuid, frame_idx: frameIdx, context })
                });
                const data = await res.json();
                const meta = document.createElement('div');
                meta.style.marginBottom = '6px';
                meta.style.fontSize = '12px';
                meta.textContent = `${data.filename || ''}:${data.lineno || ''}`;
                wrapper.appendChild(meta);
                const pre = document.createElement('pre');
                pre.style.margin = '0';
                const lines = (data.lines || []).map(l => {
                    const mark = (data.highlight === l.no) ? '▶ ' : '  ';
                    return `${mark}${String(l.no).padStart(6,' ')} | ${l.text||''}`;
                }).join('\n');
                pre.textContent = lines || '(no code available)';
                wrapper.appendChild(pre);
            } catch (e) {
                const err = document.createElement('div');
                err.textContent = 'Failed to load code context.';
                wrapper.appendChild(err);
            }

            containerElement.appendChild(wrapper);
            enableDrag(wrapper, { handle: header, bounds: window });
            codePanel = wrapper;
        }

        function sampleValue(cache, userData, fallback) {
            if (!cache) return fallback;
            const { dims, values } = cache;
            try {
                if (dims >= 3) {
                    return values[userData.tensor1][userData.tensor0][userData.tensor2];
                } else if (dims === 2) {
                    return values[userData.tensor1][userData.tensor0];
                } else if (dims === 1) {
                    return values[userData.tensor0];
                }
            } catch (e) {
                /* ignore */
            }
            return fallback;
        }

        function applyColorToTensor(targetTensor, cache) {
            if (!cache || !tensorCache) return;
            const min = tensorCache.scaleMin;
            const max = tensorCache.scaleMax;
            const denom = max - min || 1;
            targetTensor.children.forEach((cube) => {
                const u = cube.userData;
                if (!u) return;
                const val = sampleValue(cache, u, min);
                const t = Math.max(0, Math.min(1, (val - min) / denom));
                const color = scheme === 'mono' ? monoColor(t, monoBaseHex) : viridisColor(t);
                cube.material.color.copy(color);
            });
        }

        function applyColorMapIfNeeded() {
            if (!colorizeOn || !tensorCache) return;
            applyColorToTensor(globalTensor, tensorCache.global);
            applyColorToTensor(sliceTensor, tensorCache.slice);
        }

        function resetGlobalColors() {
            // Restore base colors according to current theme:
            // - Global cubes use currentBaseGlobal
            // - highlighted coords (in op.global_coords) use COLOR_SLICE
            globalTensor.children.forEach((cube) => {
                const u = cube.userData;
                if (!u) return;
                const key = `${u.tensor0},${u.tensor1},${u.tensor2}`;
                const baseColor = highlightedGlobalSet.has(key) ? COLOR_SLICE : currentBaseGlobal;
                cube.material.color.copy(baseColor);
            });
        }

        function resetSliceColors() {
            sliceTensor.children.forEach((cube) => {
                cube.material.color.copy(currentBaseSlice);
            });
        }

        function resetBaseColors() {
            for (let idx = 0; idx < globalMesh.count; idx += 1) {
                globalMesh.setColorAt(idx, currentBaseGlobal);
            }
            for (let idx = 0; idx < sliceMesh.count; idx += 1) {
                sliceMesh.setColorAt(idx, currentBaseSlice);
            }
            if (globalMesh.instanceColor) globalMesh.instanceColor.needsUpdate = true;
            if (sliceMesh.instanceColor) sliceMesh.instanceColor.needsUpdate = true;
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
            if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) {
                return null;
            }
            return z * width * height + y * width + x;
        }

        function paintCoords(mesh, coords, color) {
            if (!mesh || !coords) return;
            coords.forEach((coord) => {
                const idx = coordToInstanceId(mesh, coord);
                if (idx === null || idx === undefined) return;
                mesh.setColorAt(idx, color);
            });
        }

        async function fetchAllProgramTiles() {
            if (!op.overall_key) return null;
            const res = await fetch(`${API_BASE}/api/load_overall`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: op.overall_key })
            });
            const data = await res.json();
            if (!res.ok || data.error) {
                console.warn('load_overall error:', data && data.error);
                return null;
            }
            return data.tiles || [];
        }

        function applyAllProgramsTiles(tiles) {
            if (!tiles || !tiles.length) return;
            resetBaseColors();
            const total = tiles.length;
            tiles.forEach((tile, idx) => {
                const color = getRainbowColor(idx, total);
                paintCoords(globalMesh, tile.global_coords, color);
                paintCoords(sliceMesh, tile.slice_coords, color);
            });
            if (globalMesh.instanceColor) globalMesh.instanceColor.needsUpdate = true;
            if (sliceMesh.instanceColor) sliceMesh.instanceColor.needsUpdate = true;
        }

        function onMouseDown(event) {
            if (!dragModeOn) return;
            _updateMouseNDC(event);
            const hits = _raycastAll();
            if (hits.length === 0) return;
            const cube = _toTopLevelCube(hits[0].object);
            if (!cube) return;
            // Prepare drag plane using camera forward as normal, passing through cube
            const normal = new THREE.Vector3();
            camera.getWorldDirection(normal);
            cube.getWorldPosition(worldPosHelper);
            dragPlane.setFromNormalAndCoplanarPoint(normal, worldPosHelper);
            // Compute offset between intersection and cube world position
            raycaster.setFromCamera(mouse, camera);
            if (!raycaster.ray.intersectPlane(dragPlane, planeIntersect)) return;
            dragOffset.copy(worldPosHelper).sub(planeIntersect);
            isDragging = true;
            dragTarget = cube;
            stage.style.cursor = 'grabbing';
        }

        function onMouseUp() {
            if (!dragModeOn) return;
            isDragging = false;
            dragTarget = null;
            stage.style.cursor = '';
        }

        function animate() {
            requestAnimationFrame(animate);
            // If colormap is OFF, ensure colors are reset every frame before animations
            if (!colorizeOn && !allProgramsOn) {
                resetGlobalColors();
                resetSliceColors();
            }
            orbitControls.update();

            // Run highlight animation regardless of Color by Value state
            if (!isPaused && frame < totalFrames) {
                frame++;
            }

            applyColorMapIfNeeded();
            renderer.render(scene, camera);
        }


        async function getElementValue(tensorName, x, y, z) {
            let uuid = op.uuid;
            const response = await fetch(`${API_BASE}/api/getLoadValue`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ uuid, tensorName, x, y, z }),
            });
            return await response.json();
        }

        async function fetchTensorPayload() {
            try {
                const res = await fetch(`${API_BASE}/api/getLoadTensor`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ uuid: op.uuid })
                });
                const data = await res.json();
                if (!data || data.error) {
                    console.warn('getLoadTensor error:', data && data.error);
                    return null;
                }
                const sliceMin = data.slice?.min;
                const sliceMax = data.slice?.max;
                const scaleMin = Math.min(
                    data.min ?? 0,
                    sliceMin !== undefined ? sliceMin : data.min ?? 0
                );
                const scaleMax = Math.max(
                    data.max ?? 0,
                    sliceMax !== undefined ? sliceMax : data.max ?? 0
                );
                return {
                    scaleMin,
                    scaleMax,
                    global: {
                        dims: data.dims,
                        values: data.values,
                        shape: data.shape,
                    },
                    slice: data.slice
                        ? {
                            dims: data.slice.dims,
                            values: data.slice.values,
                            shape: data.slice.shape,
                        }
                        : null,
                };
            } catch (e) {
                console.error('getLoadTensor failed', e);
                return null;
            }
        }

        async function toggleColorize() {
            if (allProgramsOn) {
                allProgramsOn = false;
                if (window.setOpControlState) {
                    window.setOpControlState({ allPrograms: false });
                }
            }
            colorizeOn = !colorizeOn;
            if (!colorizeOn) {
                resetGlobalColors();
                resetSliceColors();
                destroyLegend();
                return colorizeOn;
            }
            if (!tensorCache) {
                tensorCache = await fetchTensorPayload();
            }
            if (!tensorCache) {
                colorizeOn = false;
                return colorizeOn;
            }
            createLegend(tensorCache.scaleMin, tensorCache.scaleMax);
            applyColorMapIfNeeded();
            return colorizeOn;
        }

        async function toggleShowCode() {
            if (window.__tritonVizCodeToggle) {
                return window.__tritonVizCodeToggle();
            }
            return false;
        }

        function toggleHistogram() {
            histogramVisible = !histogramVisible;
            if (histogramVisible) {
                if (histogramUI.show) {
                    histogramUI.show();
                } else if (histogramUI.overlay) {
                    histogramUI.overlay.style.display = 'block';
                }
                return histogramVisible;
            }
            if (histogramUI.hide) {
                histogramUI.hide();
            } else if (histogramUI.overlay) {
                histogramUI.overlay.style.display = 'none';
            }
            return histogramVisible;
        }

        async function toggleAllPrograms() {
            allProgramsOn = !allProgramsOn;
            if (allProgramsOn) {
                if (colorizeOn) {
                    colorizeOn = false;
                    destroyLegend();
                    if (window.setOpControlState) {
                        window.setOpControlState({ colorize: false });
                    }
                }
                if (!allProgramTiles) {
                    allProgramTiles = await fetchAllProgramTiles();
                }
                if (!allProgramTiles || !allProgramTiles.length) {
                    allProgramsOn = false;
                    return false;
                }
                applyAllProgramsTiles(allProgramTiles);
                return true;
            }
            resetGlobalColors();
            resetSliceColors();
            return false;
        }

        if (window.setOpControlHandlers) {
            window.setOpControlHandlers({
                toggleColorize,
                toggleShowCode,
                toggleHistogram,
                toggleAllPrograms: op.overall_key ? toggleAllPrograms : null,
            });
        }
        if (window.setOpControlState) {
            window.setOpControlState({
                colorize: colorizeOn,
                showCode: false,
                histogram: false,
                allPrograms: false,
            });
        }

        function updateSideMenu(tensorName, x, y, z, value) {
            if (!tensorName) {
                sideMenu.innerHTML = '';
                return;
            }

            let dims = tensorName === 'Global' ? op.global_shape : op.slice_shape;
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
                position: 'absolute',
                top: '16px',
                right: '16px',
                width: '220px',
                padding: '10px',
                background: 'rgba(0,0,0,0.65)',
                color: '#fff',
                borderRadius: '8px',
                fontFamily: 'var(--font-sans)',
                fontSize: '12px',
                zIndex: 2500,
            });
            container.appendChild(menu);
            return menu;
        }

        return () => {
            if (window.setOpControlHandlers) {
                window.setOpControlHandlers(null);
            }
            if (window.setOpControlState) {
                window.setOpControlState({
                    colorize: false,
                    showCode: false,
                    histogram: false,
                    allPrograms: false,
                });
            }
            if (window.__tritonVizCodeHide && !window.__tritonVizPreserveCodePanel) {
                window.__tritonVizCodeHide();
            }
            if (histogramUI.hide) {
                histogramUI.hide();
            }
            destroyLegend();
            if (renderer && renderer.dispose) {
                renderer.dispose();
            }
            if (renderer && renderer.domElement && renderer.domElement.parentElement) {
                renderer.domElement.parentElement.removeChild(renderer.domElement);
            }
            if (containerElement) {
                containerElement.innerHTML = '';
            }
        };

}

export function createLoadOverallVisualization(containerElement, op) {
        const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);
        const COLOR_LEFT_SLICE = new THREE.Color(1.0, 0.0, 1.0);
        const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);
        const tiles = op.overall_tiles || [];
        const globalShape = op.overall_shape || op.global_shape || [];
        const sliceShape = op.overall_slice_shape || op.slice_shape || [];

        containerElement.innerHTML = '';
        const sceneRoot = document.createElement('div');
        sceneRoot.className = 'viz-stage';
        containerElement.appendChild(sceneRoot);

        const legend = document.createElement('div');
        Object.assign(legend.style, {
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(0,0,0,0.65)',
            color: '#fff',
            padding: '8px 10px',
            borderRadius: '6px',
            maxHeight: '200px',
            overflow: 'auto',
            fontSize: '12px',
            zIndex: 10,
        });
        legend.innerHTML = '<strong>Program Blocks</strong><br/>';
        sceneRoot.appendChild(legend);

        // 背景选择
        let currentBackground = COLOR_BACKGROUND;
        const controlBar = document.createElement('div');
        Object.assign(controlBar.style, {
            position: 'absolute',
            top: '10px',
            left: '10px',
            zIndex: 12,
            display: 'flex',
            gap: '8px',
            alignItems: 'center',
            background: 'rgba(0,0,0,0.55)',
            color: '#fff',
            padding: '6px 8px',
            borderRadius: '6px',
            fontSize: '12px',
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
        const sliceTensor = createTensor(sliceShape, [], COLOR_LEFT_SLICE, 'Slice', cubeGeometry, edgesGeometry, lineMaterial);
        const globalSize = calculateTensorSize(globalShape);
        sliceTensor.position.set(globalSize.x + 5, 0, 0);
        scene.add(globalTensor);
        scene.add(sliceTensor);

        const globalMap = buildCubeMap(globalTensor);
        const sliceMap = buildCubeMap(sliceTensor);

        const PAPER_BG_HEX = '#ffffff';
        const BEIGE_BG_HEX = '#f7f0e3';

        function isLightBackgroundHex(hex) {
            const h = (hex || '').toLowerCase();
            return h === PAPER_BG_HEX || h === BEIGE_BG_HEX;
        }

        function getTileColorForBackground(bgHex, idx) {
            const hex = (bgHex || '').toLowerCase();
            if (isLightBackgroundHex(hex)) {
                // Paper/Beige：更多 block 时也能区分开的亮色调色板（8 种色相）
                const paperPalette = [
                    '#ff8a3c', // orange
                    '#ffd54a', // yellow
                    '#a3e635', // lime
                    '#34d399', // green
                    '#5fd4ff', // cyan
                    '#4f8bff', // blue
                    '#6366f1', // indigo
                    '#f9739b', // rose
                ];
                const chosen = paperPalette[idx % paperPalette.length];
                return new THREE.Color(chosen);
            }
            // 暗色背景下沿用原来的 HSL 方案（保持 Dark 外观不变）
            return new THREE.Color().setHSL((idx * 0.17) % 1, 0.65, 0.55);
        }

        function recolorBackgroundCubes(bgHex) {
            const hex = (bgHex || '').toLowerCase();
            const isLight = isLightBackgroundHex(hex);

            // Paper/Beige：用更明亮的 pastel 底色；Dark/Slate：使用原始配色
            const baseGlobal = isLight ? new THREE.Color('#fefce8') : COLOR_GLOBAL;
            const baseSlice  = isLight ? new THREE.Color('#dbeafe') : COLOR_LEFT_SLICE;

            // 先把所有 cube 恢复为"底色"，避免旧 tile 颜色残留
            globalTensor.children.forEach((cube) => {
                if (cube && cube.material && cube.material.color) {
                    cube.material.color.copy(baseGlobal);
                }
            });
            sliceTensor.children.forEach((cube) => {
                if (cube && cube.material && cube.material.color) {
                    cube.material.color.copy(baseSlice);
                }
            });

            // 根据背景切换 MeshPhong <-> MeshBasic：深色保持 3D，高光；浅色用平涂颜色
            const switchMaterialForGroup = (group) => {
                group.traverse((obj) => {
                    if (!obj || !obj.isMesh) return;
                    if (!obj.material || !obj.material.color) return;
                    if (isLight) {
                        if (!obj.userData.phongMaterial) {
                            obj.userData.phongMaterial = obj.material;
                        }
                        if (!(obj.material instanceof THREE.MeshBasicMaterial)) {
                            const color = obj.material.color.clone();
                            obj.material = new THREE.MeshBasicMaterial({
                                color,
                                toneMapped: false,
                            });
                        }
                    } else if (obj.userData.phongMaterial) {
                        obj.material = obj.userData.phongMaterial;
                    }
                });
            };
            switchMaterialForGroup(globalTensor);
            switchMaterialForGroup(sliceTensor);

            // 所有背景都加一层细衬线：暗底用半透明白线，浅底用半透明深灰线
            if (lineMaterial) {
                lineMaterial.visible = true;
                if (isLight) {
                    lineMaterial.color.set('#111827'); // 深灰
                    lineMaterial.opacity = 0.16;
                } else {
                    lineMaterial.color.set('#ffffff');
                    lineMaterial.opacity = 0.28;
                }
            }
        }

        function renderTilesForBackground(bgHex) {
            recolorBackgroundCubes(bgHex);

            // 重置图例，并按当前背景重新上色所有块
            legend.innerHTML = '<strong>Program Blocks</strong><br/>';

            tiles.forEach((tile, idx) => {
                const color = getTileColorForBackground(bgHex, idx);
                legend.appendChild(createLegendRow(idx, color));
                paintCoords(globalMap, tile.global_coords, color);
                paintCoords(sliceMap, tile.slice_coords, color);
            });
        }

        // 初始渲染（根据当前背景选择器的值）
        renderTilesForBackground(backgroundSelect.value || '#000000');

        const { center } = setupCamera(scene, camera);
        const orbitControls = new OrbitControls(camera, renderer.domElement);
        orbitControls.enableDamping = true;
        orbitControls.dampingFactor = 0.05;
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
        });

        let rafId = null;
        const animate = () => {
            orbitControls.update();
            renderer.render(scene, camera);
            rafId = requestAnimationFrame(animate);
        };
        animate();

        return () => {
            if (rafId) cancelAnimationFrame(rafId);
            renderer.dispose();
            if (renderer.domElement && renderer.domElement.parentElement) {
                renderer.domElement.parentElement.removeChild(renderer.domElement);
            }
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
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '6px';
            const swatch = document.createElement('span');
            Object.assign(swatch.style, {
                display: 'inline-block',
                width: '14px',
                height: '14px',
                background: `#${color.getHexString()}`,
            });
            row.appendChild(swatch);
            const label = document.createElement('span');
            label.textContent = `Program Block ${idx + 1}`;
            row.appendChild(label);
            return row;
        }
}
