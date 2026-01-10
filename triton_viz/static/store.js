import * as THREE from 'https://esm.sh/three@0.155.0/build/three.module.js';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';
import {
    setupScene,
    setupGeometries,
    createTensor,
    calculateTensorSize,
    updateCubeColor,
    setupCamera,
    setupEventListeners,
    cameraControls,
    addLabels,
    CUBE_SIZE
} from './load_utils.js';
import { createHistogramOverlay } from './histogram.js';
import { createFlipDemo } from './flip_demo.js';
import { createFlip3D } from './flip_3d.js';

function createHeatmapOverlay(apiBase, uuid, onDataLoaded) {
        const button = document.createElement('button');
        button.textContent = 'Value Heatmap';

        const overlay = document.createElement('div');
        Object.assign(overlay.style, {
            position: 'fixed',
            top: '120px',
            left: '20px',
            width: '520px',
            padding: '12px',
            background: 'rgba(0,0,0,0.9)',
            color: '#fff',
            borderRadius: '8px',
            border: '1px solid #444',
            zIndex: 3200,
            display: 'none'
        });

        const header = document.createElement('div');
        header.textContent = 'Store Value Heatmap';
        header.style.fontWeight = 'bold';
        header.style.marginBottom = '8px';
        overlay.appendChild(header);

        const info = document.createElement('div');
        info.style.fontSize = '12px';
        info.style.marginBottom = '6px';
        overlay.appendChild(info);

        const canvas = document.createElement('canvas');
        canvas.width = 460;
        canvas.height = 340;
        canvas.style.background = '#111';
        canvas.style.border = '1px solid #555';
        overlay.appendChild(canvas);

        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Close';
        closeBtn.style.marginTop = '8px';
        closeBtn.addEventListener('click', () => (overlay.style.display = 'none'));
        overlay.appendChild(closeBtn);

        button.addEventListener('click', () => {
            overlay.style.display = 'block';
            updateHeatmap();
        });

        async function updateHeatmap() {
            info.textContent = 'Loading...';
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            try {
                const res = await fetch(`${apiBase}/api/getLoadTensor`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ uuid })
                });
                const data = await res.json();
                if (data.error) {
                    info.textContent = data.error;
                    if (typeof onDataLoaded === 'function') onDataLoaded(null);
                    return;
                }
                drawHeatmap(ctx, data);
                info.textContent = `Shape: ${data.shape.join(' x ')} | Min: ${data.min.toFixed(4)} | Max: ${data.max.toFixed(4)}`;
                if (typeof onDataLoaded === 'function') onDataLoaded(data);
            } catch (err) {
                info.textContent = `Heatmap error: ${err}`;
                if (typeof onDataLoaded === 'function') onDataLoaded(null);
            }
        }

        function drawHeatmap(ctx, tensorData) {
            let matrix = tensorData.values;
            if (!Array.isArray(matrix) || matrix.length === 0) {
                ctx.fillStyle = '#ccc';
                ctx.fillText('No data', 20, 30);
                return;
            }
            if (tensorData.dims >= 3) {
                matrix = matrix[0];
            }
            if (!Array.isArray(matrix[0])) {
                matrix = [matrix];
            }
            const rows = matrix.length;
            const cols = matrix[0].length;
            const margin = 20;
            const width = canvas.width - margin * 2;
            const height = canvas.height - margin * 2;
            const cellW = width / cols;
            const cellH = height / rows;
            const min = tensorData.min;
            const max = tensorData.max;
            const denom = max - min || 1;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const val = matrix[r][c];
                    const t = (val - min) / denom;
                    const color = heatColor(t);
                    ctx.fillStyle = color;
                    ctx.fillRect(margin + c * cellW, margin + r * cellH, cellW, cellH);
                }
            }
        }

        function heatColor(t) {
            const clamped = Math.max(0, Math.min(1, t));
            const r = Math.floor(255 * clamped);
            const g = Math.floor(255 * (1 - Math.abs(clamped - 0.5) * 2));
            const b = Math.floor(255 * (1 - clamped));
            return `rgb(${r},${g},${b})`;
        }

        document.body.appendChild(overlay);
        return { button, overlay };
}

export function createStoreVisualization(containerElement, op) {

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
        try { window.current_op_uuid = op.uuid; } catch (e) {}

        let currentStep = 0;
        let frame = 0;
        let isPaused = false;

        containerElement.innerHTML = '';
        containerElement.style.position = 'relative';
        const stage = document.createElement('div');
        stage.className = 'viz-stage';
        containerElement.appendChild(stage);

        const sideMenu = createSideMenu(containerElement);
        const histogramUI = createHistogramOverlay(containerElement, {
            title: 'Store Value Distribution',
            apiBase: API_BASE,
            sources: [
                { value: 'GLOBAL', label: 'Global Tensor' }
            ],
            buildRequestBody: (source, bins) => ({
                uuid: op.uuid,
                source,
                bins,
            }),
        });
        let dragModeOn = false;
        let hoveredCube = null;
        let flipCleanup = null;
        let colorizeOn = false;
        let tensorCache = null; // {scaleMin, scaleMax, global:{dims,values}, slice:{dims,values}}
        let legendEl = null;

        const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);    // Dark Gray (base for dark themes)
        const COLOR_SLICE = new THREE.Color(1.0, 0.55, 0.0);    // Orange (highlight for store target)
        const COLOR_LEFT_SLICE = new THREE.Color(1.0, 0.0, 1.0); // Magenta (starting color for left slice)
        const COLOR_LOADED = new THREE.Color(1.0, 0.8, 0.0);    // Gold (final color for both slices)
        const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);  // Black
        const COLOR_HIGHLIGHT = new THREE.Color(1.0, 0.55, 0.0);
        const COLOR_FILLED = new THREE.Color(0.0, 0.8, 1.0);
        const COLOR_COOL = new THREE.Color(0.2, 0.4, 1.0);
        const COLOR_HOT = new THREE.Color(1.0, 0.3, 0.1);
        const TEMP_COLOR = new THREE.Color();
        const highlightedGlobalSet = new Set((op.global_coords || []).map(([x, y, z]) => `${x},${y},${z}`));
        const HOVER_COLOR = new THREE.Color(1.0, 0.55, 0.0);

        const baseGlobalDark = COLOR_GLOBAL.clone();
        const baseSliceDark = COLOR_LEFT_SLICE.clone();
        // 浅色主题下的更清爽 pastel 底色，减少"灰雾感"
        const baseGlobalLight = new THREE.Color('#fefce8'); // very light warm yellow
        const baseSliceLight = new THREE.Color('#dbeafe');  // light blue
        let currentBaseGlobal = baseGlobalDark.clone();
        let currentBaseSlice = baseSliceDark.clone();

        let currentBackground = COLOR_BACKGROUND;

        const { scene, camera, renderer } = setupScene(stage, currentBackground);
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

        const globalTensor = createTensor(op.global_shape, op.global_coords, COLOR_GLOBAL, 'Global', cubeGeometry, edgesGeometry, lineMaterial);
        const sliceTensor = createTensor(op.slice_shape, op.slice_coords, COLOR_LEFT_SLICE, 'Slice', cubeGeometry, edgesGeometry, lineMaterial);

        // Position slice tensor
        const globalSize = calculateTensorSize(op.global_shape);
        sliceTensor.position.set(globalSize.x + 5, 0, 0); // Adjusted tensor spacing

        scene.add(globalTensor);
        scene.add(sliceTensor);

        const hoverGeometry = new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05);
        const hoverEdgesGeometry = new THREE.EdgesGeometry(hoverGeometry);
        const hoverMaterial = new THREE.LineBasicMaterial({ color: HOVER_COLOR });
        const globalHoverOutline = new THREE.LineSegments(hoverEdgesGeometry, hoverMaterial);
        const sliceHoverOutline = new THREE.LineSegments(hoverEdgesGeometry, hoverMaterial);
        globalHoverOutline.visible = false;
        sliceHoverOutline.visible = false;
        globalTensor.add(globalHoverOutline);
        sliceTensor.add(sliceHoverOutline);
        // Arrow helper (+ label) showing flow Slice -> Global
        const ARROW_COLOR = 0xff9800; // orange for store
        const arrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0,0,0), 1.0, ARROW_COLOR, 0.25, 0.12);
        arrow.visible = true;
        scene.add(arrow);

        function createTextSprite(text) {
            const c = document.createElement('canvas');
            const ctx2 = c.getContext('2d');
            const P = 4;
            ctx2.font = 'bold 18px Arial';
            const metrics = ctx2.measureText(text);
            const w = Math.ceil(metrics.width) + P*2;
            const h = 28 + P*2;
            c.width = w; c.height = h;
            const ctx3 = c.getContext('2d');
            ctx3.fillStyle = 'rgba(0,0,0,0.65)';
            ctx3.fillRect(0, 0, w, h);
            ctx3.fillStyle = '#ffffff';
            ctx3.font = 'bold 18px Arial';
            ctx3.textBaseline = 'middle';
            ctx3.fillText(text, P, h/2);
            const tex = new THREE.CanvasTexture(c);
            tex.needsUpdate = true;
            const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
            const spr = new THREE.Sprite(mat);
            const scaleX = Math.max(1, w / 64);
            const scaleY = Math.max(0.5, h / 64);
            spr.scale.set(scaleX, scaleY, 1);
            return spr;
        }

        const labelText = (()=>{
            const ms = (op.mem_src||'').toUpperCase();
            const md = (op.mem_dst||'').toUpperCase();
            if (ms && md) return `Store ${ms} → ${md}`;
            return 'Store Slice → Global';
        })();
        const arrowLabel = createTextSprite(labelText);
        arrowLabel.visible = true;
        const midX = (globalTensor.position.x + sliceTensor.position.x) / 2;
        arrowLabel.position.set(midX, 1.4, 0);
        arrowLabel.material.color = new THREE.Color(getTextColor(currentBackground));
        scene.add(arrowLabel);

        labelSprites = addLabels(scene, globalTensor, sliceTensor, currentBackground);

        const refreshTextOverlays = () => {
            if (arrowLabel) scene.remove(arrowLabel);
            const newLabel = createTextSprite(labelText);
            newLabel.visible = true;
            const midX2 = (globalTensor.position.x + sliceTensor.position.x) / 2;
            newLabel.position.set(midX2, 1.4, 0);
            newLabel.material.color = new THREE.Color(getTextColor(currentBackground));
            scene.add(newLabel);
            labelSprites.forEach(s => scene.remove(s));
            labelSprites = addLabels(scene, globalTensor, sliceTensor, currentBackground);
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
        // Drag state (optional)
        let isDragging = false;
        let dragTarget = null;
        const dragPlane = new THREE.Plane();
        const planeIntersect = new THREE.Vector3();
        const worldPosHelper = new THREE.Vector3();
        const dragOffset = new THREE.Vector3();

        const onKeyDown = cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ'));
        setupEventListeners(stage, camera, renderer, onMouseMove, onKeyDown);
        stage.addEventListener('mousedown', onMouseDown);
        stage.addEventListener('mouseup', onMouseUp);
        stage.addEventListener('mouseleave', onMouseUp);
        async function toggleColorize() {
            colorizeOn = !colorizeOn;
            if (colorizeOn) {
                if (!tensorCache) {
                    tensorCache = await fetchStoreTensor();
                }
                if (tensorCache) {
                    applyColorByValue();
                    createLegend(tensorCache.scaleMin, tensorCache.scaleMax);
                } else {
                    colorizeOn = false;
                }
            } else {
                resetColorByValue();
                destroyLegend();
            }
            return colorizeOn;
        }

        async function toggleShowCode() {
            if (window.__tritonVizCodeToggle) {
                return window.__tritonVizCodeToggle();
            }
            return false;
        }

        function showHistogram() {
            if (histogramUI.show) {
                histogramUI.show();
            } else if (histogramUI.overlay) {
                histogramUI.overlay.style.display = 'block';
            }
        }

        if (window.setOpControlHandlers) {
            window.setOpControlHandlers({
                toggleColorize,
                toggleShowCode,
                showHistogram,
            });
        }
        if (window.setOpControlState) {
            window.setOpControlState({ colorize: colorizeOn, showCode: false });
        }

        let flowArrowOn = true;
        function isLightBackgroundHex(hex) {
            const h = (hex || '').toLowerCase();
            return h === '#ffffff' || h === '#f7f0e3' || h === '#f5f7fb';
        }

        function applyBackgroundTheme(hex) {
            const isLight = isLightBackgroundHex(hex);

            currentBaseGlobal.copy(isLight ? baseGlobalLight : baseGlobalDark);
            currentBaseSlice.copy(isLight ? baseSliceLight : baseSliceDark);
            resetColorByValue();

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
        if (scene && scene.background) scene.background = currentBackground;
        applyBackgroundTheme(defaultBackground);
        refreshTextOverlays();
        // Removed Flip demo button from Store view; Flip visualization is available under Flip op.
        animate();

        function _updateMouseNDC(event) {
            mouse.x = (event.clientX / stage.clientWidth) * 2 - 1;
            mouse.y = -(event.clientY / stage.clientHeight) * 2 + 1;
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
            if (isDragging && dragTarget) {
                if (raycaster.ray.intersectPlane(dragPlane, planeIntersect)) {
                    const newWorld = planeIntersect.add(dragOffset);
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
        }

        function onMouseUp() {
            if (!dragModeOn) return;
            isDragging = false;
            dragTarget = null;
            stage.style.cursor = '';
        }

        function animate() {
            requestAnimationFrame(animate);
            orbitControls.update();

            if (!isPaused && frame < totalFrames) {
                const index = Math.floor(frame / 2);
                const factor = (frame % 2) / 1.0;

                if (index < op.global_coords.length) {
                    const globalCoord = op.global_coords[index];
                    const sliceCoord = op.slice_coords[index];

                    updateCubeColor(globalTensor, globalCoord, COLOR_GLOBAL, COLOR_SLICE, factor);
                    updateCubeColor(sliceTensor, sliceCoord, COLOR_LEFT_SLICE, COLOR_LOADED, factor);

                    highlightCurrentOperation(globalTensor, globalCoord, sliceTensor, sliceCoord);
                    updateInfoPanel(globalCoord, sliceCoord, index);

                    // Update arrow position/direction and label at midpoint (Slice -> Global)
                    if (flowArrowOn) {
                        const gCube = globalTensor.children.find(c => c.userData && c.userData.tensor0 === globalCoord[0] && c.userData.tensor1 === globalCoord[1] && c.userData.tensor2 === globalCoord[2]);
                        const sCube = sliceTensor.children.find(c => c.userData && c.userData.tensor0 === sliceCoord[0] && c.userData.tensor1 === sliceCoord[1] && c.userData.tensor2 === sliceCoord[2]);
                        if (gCube && sCube) {
                            const src = new THREE.Vector3();
                            const dst = new THREE.Vector3();
                            sCube.getWorldPosition(src); // from slice
                            gCube.getWorldPosition(dst); // to global
                            const dir = new THREE.Vector3().subVectors(dst, src);
                            const len = dir.length();
                            if (len > 1e-6) {
                                arrow.visible = true;
                                arrow.setDirection(dir.normalize());
                                const safeLen = Math.max(0.1, len - 0.3);
                                arrow.setLength(safeLen, 0.25, 0.12);
                                arrow.position.copy(src);
                                const mid = new THREE.Vector3().addVectors(src, dst).multiplyScalar(0.5);
                                arrowLabel.position.copy(mid);
                                arrowLabel.visible = true;
                            } else {
                                arrow.visible = false;
                                arrowLabel.visible = false;
                            }
                        } else {
                            arrow.visible = false;
                            arrowLabel.visible = false;
                        }
                    }
                }

                frame++;
            }

            if (colorizeOn && tensorCache) {
                applyColorByValue();
            } else if (!colorizeOn) {
                // ensure cached colors stay reset when exiting mode
            }
            renderer.render(scene, camera);
        }


        function highlightCurrentOperation(globalTensor, globalCoord, sliceTensor, sliceCoord) {
            globalTensor.children.forEach(cube => cube.material.emissive.setHex(0x000000));
            sliceTensor.children.forEach(cube => cube.material.emissive.setHex(0x000000));

            const globalCube = globalTensor.children.find(c =>
                c.userData && c.userData.tensor0 === globalCoord[0] && c.userData.tensor1 === globalCoord[1] && c.userData.tensor2 === globalCoord[2]
            );
            const sliceCube = sliceTensor.children.find(c =>
                c.userData && c.userData.tensor0 === sliceCoord[0] && c.userData.tensor1 === sliceCoord[1] && c.userData.tensor2 === sliceCoord[2]
            );

            if (globalCube) globalCube.material.emissive.setHex(0x444444);
            if (sliceCube) sliceCube.material.emissive.setHex(0x444444);
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

        async function fetchStoreTensor() {
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
                return normalizeTensorPayload(data);
            } catch (e) {
                console.warn('getLoadTensor failed', e);
                return null;
            }
        }

        function normalizeTensorPayload(data) {
            if (!data || data.error) return null;
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
        }

        function resetColorByValue() {
            globalTensor.children.forEach((cube) => {
                const u = cube.userData;
                if (!u) return;
                const key = `${u.tensor0},${u.tensor1},${u.tensor2}`;
                const base = highlightedGlobalSet.has(key) ? COLOR_SLICE : currentBaseGlobal;
                cube.material.color.copy(base);
            });
            sliceTensor.children.forEach((cube) => {
                cube.material.color.copy(currentBaseSlice);
            });
        }

        function sampleStoreValue(cache, userData, fallback) {
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

        function applyColorsToTensor(targetTensor, cache) {
            if (!cache || !tensorCache) return;
            const min = tensorCache.scaleMin;
            const max = tensorCache.scaleMax;
            const denom = max - min || 1;
            targetTensor.children.forEach((cube) => {
                const u = cube.userData;
                if (!u) return;
                const val = sampleStoreValue(cache, u, min);
                const t = Math.max(0, Math.min(1, (val - min) / denom));
                cube.material.color.copy(TEMP_COLOR.copy(COLOR_COOL).lerp(COLOR_HOT, t));
            });
        }

        function applyColorByValue() {
            if (!colorizeOn || !tensorCache) return;
            applyColorsToTensor(globalTensor, tensorCache.global);
            applyColorsToTensor(sliceTensor, tensorCache.slice);
        }

        function destroyLegend() {
            if (legendEl && legendEl.remove) legendEl.remove();
            legendEl = null;
        }

        function createLegend(min, max) {
            destroyLegend();
            const wrapper = document.createElement('div');
            wrapper.style.position = 'absolute';
            wrapper.style.top = '32px';
            wrapper.style.left = '32px';
            wrapper.style.background = 'rgba(0,0,0,0.7)';
            wrapper.style.color = '#fff';
            wrapper.style.padding = '8px 10px';
            wrapper.style.borderRadius = '8px';
            wrapper.style.fontSize = '12px';
            wrapper.style.fontFamily = 'var(--font-sans)';
            wrapper.style.zIndex = '2500';

            const header = document.createElement('div');
            header.style.fontWeight = '600';
            header.style.marginBottom = '6px';
            header.textContent = 'Value Scale';
            wrapper.appendChild(header);

            const canvas = document.createElement('canvas');
            canvas.width = 220;
            canvas.height = 12;
            const ctx2 = canvas.getContext('2d');
            for (let x = 0; x < canvas.width; x += 1) {
                const t = x / (canvas.width - 1);
                const color = TEMP_COLOR.copy(COLOR_COOL).lerp(COLOR_HOT, t);
                ctx2.fillStyle = `#${color.getHexString()}`;
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
                <p>Value: ${value !== undefined ? value : 'Storeing...'}</p>
            `;
        }

        function updateInfoPanel(globalCoord, sliceCoord, index) {
            sideMenu.innerHTML = `
                <h3>Current Operation</h3>
                <p>Global Coords: (${globalCoord.join(', ')})</p>
                <p>Slice Coords: (${sliceCoord.join(', ')})</p>
                <p>Progress: ${index + 1}/${op.global_coords.length}</p>
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
                window.setOpControlState({ colorize: false, showCode: false });
            }
            if (window.__tritonVizCodeHide) {
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

export function createStoreOverallVisualization(containerElement, op) {
        const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);
        const COLOR_SLICE = new THREE.Color(1.0, 0.0, 1.0);
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
        const sliceTensor = createTensor(sliceShape, [], COLOR_SLICE, 'Slice', cubeGeometry, edgesGeometry, lineMaterial);
        const globalSize = calculateTensorSize(globalShape);
        sliceTensor.position.set(globalSize.x + 5, 0, 0);
        scene.add(globalTensor);
        scene.add(sliceTensor);

        // Overall 视图使用 MeshBasic 平涂材质，让颜色与图例更一致
        const applyFlatMaterial = (group) => {
            group.traverse((obj) => {
                if (obj && obj.isMesh && obj.material && obj.material.color) {
                    const color = obj.material.color.clone();
                    obj.material = new THREE.MeshBasicMaterial({
                        color,
                        toneMapped: false,
                    });
                }
            });
        };
        applyFlatMaterial(globalTensor);
        applyFlatMaterial(sliceTensor);

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
                // Paper/Beige：与 Load overall 一致的 8 色高亮调色板
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
            // 暗色背景沿用原来的 HSL 分布
            return new THREE.Color().setHSL((idx * 0.17) % 1, 0.65, 0.55);
        }

        function recolorBackgroundCubes(bgHex) {
            const hex = (bgHex || '').toLowerCase();
            const isLight = isLightBackgroundHex(hex);

            const baseGlobal = isLight ? new THREE.Color('#fefce8') : COLOR_GLOBAL;
            const baseSlice  = isLight ? new THREE.Color('#dbeafe') : COLOR_SLICE;

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

            // 根据背景切换 MeshPhong <-> MeshBasic，让深色背景保持原始 3D 效果
            const switchMaterialForGroup = (group) => {
                group.traverse((obj) => {
                    if (!obj || !obj.isMesh) return;
                    if (!obj.material || !obj.material.color) return;
                    if (isLight) {
                        // 深色材质 -> 记住原来的，再换成 MeshBasic 平涂
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
                paintCoords(sliceMap, tile.slice_coords, color);
            });
        }

        const { center } = setupCamera(scene, camera);
        const orbitControls = new OrbitControls(camera, renderer.domElement);
        orbitControls.enableDamping = true;
        orbitControls.dampingFactor = 0.05;
        orbitControls.target.copy(center);
        orbitControls.update();

        sceneRoot.appendChild(renderer.domElement);

        // 初始化一次
        renderTilesForBackground(backgroundSelect.value || '#000000');

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
