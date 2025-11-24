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
    cameraControls
} from './load_utils.js';

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

        let currentStep = 0;
        let frame = 0;
        let isPaused = false;

        const sideMenu = createSideMenu(containerElement);
        // Color map UI
        const controlBar = document.createElement('div');
        controlBar.style.position = 'fixed';
        controlBar.style.top = '10px';
        controlBar.style.left = '10px';
        controlBar.style.display = 'flex';
        controlBar.style.gap = '8px';
        controlBar.style.zIndex = '2000';
        controlBar.style.pointerEvents = 'auto';
        const colorizeToggle = document.createElement('button');
        colorizeToggle.textContent = 'Color by Value: OFF';
        controlBar.appendChild(colorizeToggle);
        // Color scheme selector + color picker
        const schemeSelect = document.createElement('select');
        schemeSelect.style.padding = '4px 6px';
        schemeSelect.style.borderRadius = '4px';
        schemeSelect.style.border = '1px solid #555';
        schemeSelect.style.background = '#2a2a2a';
        schemeSelect.style.color = '#fff';
        schemeSelect.innerHTML = '<option value="mono">Mono</option><option value="viridis">Viridis</option>';
        controlBar.appendChild(schemeSelect);
        const colorPicker = document.createElement('input');
        colorPicker.type = 'color';
        colorPicker.value = '#3b82f6'; // default blue
        colorPicker.style.width = '36px';
        colorPicker.style.height = '28px';
        colorPicker.style.border = 'none';
        colorPicker.style.outline = 'none';
        colorPicker.title = 'Choose base color for Mono';
        controlBar.appendChild(colorPicker);
        const dragToggle = document.createElement('button');
        dragToggle.textContent = 'Drag Cubes: OFF';
        controlBar.appendChild(dragToggle);
        const codeToggle = document.createElement('button');
        codeToggle.textContent = 'Show Code: OFF';
        controlBar.appendChild(codeToggle);
        // Mouse pick calibration (dx/dy in pixels)
        const calibWrap = document.createElement('div');
        calibWrap.style.display = 'flex';
        calibWrap.style.alignItems = 'center';
        calibWrap.style.gap = '4px';
        const calibLabel = document.createElement('span');
        calibLabel.textContent = 'Calib:';
        calibLabel.style.opacity = '0.8';
        const dxMinus = document.createElement('button'); dxMinus.textContent = '−X';
        const dxPlus  = document.createElement('button'); dxPlus.textContent  = '+X';
        const dyMinus = document.createElement('button'); dyMinus.textContent = '−Y';
        const dyPlus  = document.createElement('button'); dyPlus.textContent  = '+Y';
        const dxdyInfo = document.createElement('span'); dxdyInfo.style.minWidth = '70px'; dxdyInfo.style.textAlign = 'center';
        const dxdyReset = document.createElement('button'); dxdyReset.textContent = 'Reset';
        calibWrap.appendChild(calibLabel);
        calibWrap.appendChild(dxMinus); calibWrap.appendChild(dxPlus);
        calibWrap.appendChild(dyMinus); calibWrap.appendChild(dyPlus);
        calibWrap.appendChild(dxdyInfo); calibWrap.appendChild(dxdyReset);
        controlBar.appendChild(calibWrap);
        containerElement.appendChild(controlBar);

        // expose for debugging
        try {
            window.last_op_global_shape = op.global_shape;
            window.last_global_coords = op.global_coords;
            window.last_slice_shape = op.slice_shape;
            window.last_slice_coords = op.slice_coords;
        } catch (e) {}

        let colorizeOn = false;
        let tensorCache = null; // {min, max, shape, dims, values}
        let hoveredCube = null;
        let legendEl = null;
        let scheme = 'mono';
        let monoBaseHex = '#3b82f6';
        let codePanel = null;

        const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);    // Dark Gray
        const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);     // Cyan (starting color for global slice)
        const COLOR_LEFT_SLICE = new THREE.Color(1.0, 0.0, 1.0); // Magenta (starting color for left slice)
        const COLOR_LOADED = new THREE.Color(1.0, 0.8, 0.0);    // Gold (final color for both slices)
        const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);  // Black

        const { scene, camera, renderer } = setupScene(containerElement, COLOR_BACKGROUND);
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

        const globalTensor = createTensor(op.global_shape, op.global_coords, COLOR_GLOBAL, 'Global', cubeGeometry, edgesGeometry, lineMaterial);
        const sliceTensor = createTensor(op.slice_shape, op.slice_coords, COLOR_LEFT_SLICE, 'Slice', cubeGeometry, edgesGeometry, lineMaterial);

        // Position slice tensor
        const globalSize = calculateTensorSize(op.global_shape);
        sliceTensor.position.set(globalSize.x + 5, 0, 0); // Adjusted tensor spacing

        scene.add(globalTensor);
        scene.add(sliceTensor);

        // Precompute highlighted coords in Global tensor for quick reset
        const highlightedGlobalSet = new Set(
            op.global_coords.map(([x, y, z]) => `${x},${y},${z}`)
        );

        addLabels(scene, globalTensor, sliceTensor);

        // Overlay memory flow badges if available (NKI only)
        try {
            const badge = document.createElement('div');
            badge.style.position = 'fixed';
            badge.style.right = '10px';
            badge.style.top = '60px';
            badge.style.zIndex = '2500';
            badge.style.background = 'rgba(0,0,0,0.65)';
            badge.style.color = '#fff';
            badge.style.padding = '6px 8px';
            badge.style.borderRadius = '6px';
            badge.style.font = '12px Arial';
            const ms = (op.mem_src||'').toUpperCase();
            const md = (op.mem_dst||'').toUpperCase();
            const by = Number(op.bytes||0);
            if (ms && md) {
                badge.innerHTML = `<b>Memory Flow</b><br/>${ms} → ${md}${by?`<br/>${by} B`:''}`;
                containerElement.appendChild(badge);
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
        function updateDxDyLabel(){ dxdyInfo.textContent = `dx=${mouseDx}, dy=${mouseDy}`; }
        updateDxDyLabel();
        // Drag state
        let dragModeOn = false;
        let isDragging = false;
        let dragTarget = null; // THREE.Mesh (cube)
        const dragPlane = new THREE.Plane();
        const planeIntersect = new THREE.Vector3();
        const worldPosHelper = new THREE.Vector3();
        const dragOffset = new THREE.Vector3();

        const onKeyDown = cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ'));
        setupEventListeners(containerElement, camera, renderer, onMouseMove, onKeyDown);

        // Additional pointer events for dragging
        containerElement.addEventListener('mousedown', onMouseDown);
        containerElement.addEventListener('mouseup', onMouseUp);
        containerElement.addEventListener('mouseleave', onMouseUp);
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
            wrapper.style.position = 'fixed';
            wrapper.style.left = '10px';
            wrapper.style.top = '50px';
            wrapper.style.padding = '6px 8px';
            wrapper.style.background = 'rgba(0,0,0,0.6)';
            wrapper.style.color = '#fff';
            wrapper.style.font = '12px Arial, sans-serif';
            wrapper.style.borderRadius = '6px';
            wrapper.style.zIndex = '2000';

            const title = document.createElement('div');
            title.textContent = scheme === 'mono' ? 'Value (Mono)' : 'Value (Viridis)';
            title.style.marginBottom = '4px';
            title.style.opacity = '0.9';
            wrapper.appendChild(title);

            const canvas = document.createElement('canvas');
            canvas.width = 220; canvas.height = 10;
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
            labels.style.marginTop = '2px';
            labels.innerHTML = `<span>${min.toFixed(3)}</span><span>${max.toFixed(3)}</span>`;
            wrapper.appendChild(labels);

            containerElement.appendChild(wrapper);
            legendEl = wrapper;
        }

        function destroyCodePanel() {
            if (codePanel && codePanel.remove) codePanel.remove();
            codePanel = null;
        }

        async function createCodePanel(frameIdx = 0, context = 8) {
            destroyCodePanel();
            const wrapper = document.createElement('div');
            wrapper.style.position = 'fixed';
            wrapper.style.right = '10px';
            wrapper.style.top = '50px';
            wrapper.style.width = '520px';
            wrapper.style.maxHeight = '60vh';
            wrapper.style.overflow = 'auto';
            wrapper.style.padding = '8px 10px';
            wrapper.style.background = 'rgba(0,0,0,0.65)';
            wrapper.style.color = '#fff';
            wrapper.style.font = '12px Menlo, Consolas, monospace';
            wrapper.style.borderRadius = '6px';
            wrapper.style.zIndex = '2000';

            const header = document.createElement('div');
            header.textContent = 'Operation Code & Context';
            header.style.marginBottom = '6px';
            header.style.opacity = '0.9';
            wrapper.appendChild(header);

            try {
                const res = await fetch(`${API_BASE}/api/op_code`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ uuid: op.uuid, frame_idx: frameIdx, context })
                });
                const data = await res.json();
                const meta = document.createElement('div');
                meta.style.marginBottom = '4px';
                meta.textContent = `${data.filename || ''}:${data.lineno || ''}`;
                wrapper.appendChild(meta);
                const pre = document.createElement('pre');
                pre.style.margin = '0';
                pre.style.whiteSpace = 'pre';
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
            codePanel = wrapper;
        }

        function applyColorMapIfNeeded() {
            if (!colorizeOn || !tensorCache) return;
            const { min, max, dims, values } = tensorCache;
            const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
            const norm = (v) => (max === min ? 0.5 : (v - min) / (max - min));
            const toColor = (t) => {
                const u = clamp(norm(t), 0, 1);
                return scheme === 'mono' ? monoColor(u, monoBaseHex) : viridisColor(u);
            };
            globalTensor.children.forEach((cube) => {
                const u = cube.userData;
                if (!u) return;
                let v = 0.0;
                try {
                    if (dims === 3) v = values[u.tensor0][u.tensor1][u.tensor2];
                    else if (dims === 2) v = values[u.tensor0][u.tensor1];
                    else if (dims === 1) v = values[u.tensor0];
                } catch (e) { /* ignore bad index */ }
                cube.material.color.copy(toColor(v));
            });
        }

        function resetGlobalColors() {
            // Restore original colors: Global cubes default to COLOR_GLOBAL,
            // highlighted coords (in op.global_coords) are COLOR_SLICE
            globalTensor.children.forEach((cube) => {
                const u = cube.userData;
                if (!u) return;
                const key = `${u.tensor0},${u.tensor1},${u.tensor2}`;
                const baseColor = highlightedGlobalSet.has(key) ? COLOR_SLICE : COLOR_GLOBAL;
                cube.material.color.copy(baseColor);
            });
        }

        function resetSliceColors() {
            sliceTensor.children.forEach((cube) => {
                cube.material.color.copy(COLOR_LEFT_SLICE);
            });
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
            containerElement.style.cursor = 'grabbing';
        }

        function onMouseUp() {
            if (!dragModeOn) return;
            isDragging = false;
            dragTarget = null;
            containerElement.style.cursor = '';
        }

        function animate() {
            requestAnimationFrame(animate);
            // If colormap is OFF, ensure colors are reset every frame before animations
            if (!colorizeOn) {
                resetGlobalColors();
                resetSliceColors();
            }
            orbitControls.update();

            // Run highlight animation regardless of Color by Value state
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
                }

                frame++;
            }

            applyColorMapIfNeeded();
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

        async function fetchGlobalTensor() {
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
                return data;
            } catch (e) {
                console.error('getLoadTensor failed', e);
                return null;
            }
        }

        colorizeToggle.addEventListener('click', async () => {
            colorizeOn = !colorizeOn;
            colorizeToggle.textContent = `Color by Value: ${colorizeOn ? 'ON' : 'OFF'}`;
            if (colorizeOn && !tensorCache) {
                tensorCache = await fetchGlobalTensor();
            }
            if (!colorizeOn) {
                // When turning OFF, immediately restore original colors
                resetGlobalColors();
                // Also reset slice side to its base color to avoid mixing styles
                sliceTensor.children.forEach((cube) => {
                    cube.material.color.copy(COLOR_LEFT_SLICE);
                });
                destroyLegend();
            } else if (tensorCache) {
                createLegend(tensorCache.min, tensorCache.max);
            }
        });

        schemeSelect.addEventListener('change', () => {
            scheme = schemeSelect.value;
            if (colorizeOn && tensorCache) {
                applyColorMapIfNeeded();
                createLegend(tensorCache.min, tensorCache.max);
            }
            // color picker visible only for mono
            colorPicker.style.display = scheme === 'mono' ? 'block' : 'none';
        });

        colorPicker.addEventListener('input', (e) => {
            monoBaseHex = e.target.value || '#3b82f6';
            if (scheme === 'mono' && colorizeOn && tensorCache) {
                applyColorMapIfNeeded();
                createLegend(tensorCache.min, tensorCache.max);
            }
        });

        // initialize picker visibility
        colorPicker.style.display = 'block';

        dragToggle.addEventListener('click', () => {
            dragModeOn = !dragModeOn;
            dragToggle.textContent = `Drag Cubes: ${dragModeOn ? 'ON' : 'OFF'}`;
            orbitControls.enabled = !dragModeOn;
        });

        codeToggle.addEventListener('click', async () => {
            const on = codeToggle.textContent.endsWith('OFF');
            codeToggle.textContent = `Show Code: ${on ? 'ON' : 'OFF'}`;
            if (on) {
                await createCodePanel(0, 8);
            } else {
                destroyCodePanel();
            }
        });

        // calibration handlers
        dxMinus.addEventListener('click', ()=>{ mouseDx -= 1; localStorage.setItem('viz_mouse_dx', String(mouseDx)); updateDxDyLabel(); });
        dxPlus.addEventListener('click',  ()=>{ mouseDx += 1; localStorage.setItem('viz_mouse_dx', String(mouseDx)); updateDxDyLabel(); });
        dyMinus.addEventListener('click', ()=>{ mouseDy -= 1; localStorage.setItem('viz_mouse_dy', String(mouseDy)); updateDxDyLabel(); });
        dyPlus.addEventListener('click',  ()=>{ mouseDy += 1; localStorage.setItem('viz_mouse_dy', String(mouseDy)); updateDxDyLabel(); });
        dxdyReset.addEventListener('click', ()=>{ mouseDx = 0; mouseDy = 0; localStorage.setItem('viz_mouse_dx','0'); localStorage.setItem('viz_mouse_dy','0'); updateDxDyLabel(); });

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
            menu.style.position = 'absolute';
            menu.style.top = '10px';
            menu.style.right = '10px';
            menu.style.width = '200px';
            menu.style.padding = '10px';
            menu.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
            menu.style.color = 'white';
            menu.style.fontFamily = 'Arial, sans-serif';
            menu.style.fontSize = '14px';
            menu.style.borderRadius = '5px';
            container.appendChild(menu);
            return menu;
        }

        function addLabels(scene, globalTensor, sliceTensor) {
            addLabel(scene, "Global Tensor", globalTensor.position);
            addLabel(scene, "Slice Tensor", sliceTensor.position);
        }

        function addLabel(scene, text, position) {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            context.font = 'Bold 24px Arial';
            context.fillStyle = 'white';
            context.fillText(text, 0, 24);

            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(material);
            sprite.position.set(position.x, position.y + 2, position.z);
            sprite.scale.set(4, 2, 1);
            scene.add(sprite);
        }

}
