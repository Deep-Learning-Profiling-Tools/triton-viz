import * as THREE from 'https://esm.sh/three@0.155.0/build/three.module.js';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';
import {
    setupScene,
    setupGeometries,
    createTensor,
    setupCamera,
    setupEventListeners,
    cameraControls,
    CUBE_SIZE,
    COLOR_HOVER
} from './load_utils.js';

export function createLoadVisualization(containerElement, op, viewState = null) {

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

        const sideMenu = createSideMenu(containerElement);
        // Color map UI
        const controlBar = document.createElement('div');
        controlBar.style.position = 'absolute';
        controlBar.style.top = '10px';
        controlBar.style.left = '10px';
        controlBar.style.display = 'flex';
        controlBar.style.gap = '8px';
        controlBar.style.zIndex = '2000';
        controlBar.style.pointerEvents = 'auto';
        const colorizeToggle = document.createElement('button');
        colorizeToggle.textContent = 'Color by Value: OFF';
        controlBar.appendChild(colorizeToggle);
        containerElement.appendChild(controlBar);
        const sliceNote = document.createElement('div');
        sliceNote.innerHTML = '<span style="color:#00b3ff">■</span> blue = data selected by the slice';
        Object.assign(sliceNote.style, {
            position: 'absolute',
            left: '10px',
            bottom: '10px',
            padding: '6px 8px',
            background: 'rgba(0,0,0,0.6)',
            color: '#fff',
            font: '12px Arial, sans-serif',
            borderRadius: '6px',
            zIndex: '2000',
            pointerEvents: 'none'
        });
        containerElement.appendChild(sliceNote);

        // expose for debugging
        try {
            window.last_op_global_shape = op.global_shape;
            window.last_global_coords = op.global_coords;
        } catch (e) {}

        let colorizeOn = false;
        let tensorCache = null; // {min, max, shape, dims, values}
        let hoveredHit = null;
        let lastHoverKey = null;
        let hoverToken = 0;
        let hoverRaf = null;
        let lastMouseEvent = null;
        let legendEl = null;
        let rafId = null;
        let renderPending = false;

        const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);    // Dark Gray
        const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);     // Cyan (starting color for global slice)
        const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);  // Black

        const { scene, camera, renderer } = setupScene(containerElement, COLOR_BACKGROUND);
        const { cubeGeometry } = setupGeometries();

        const globalTensor = createTensor(op.global_shape, op.global_coords, COLOR_GLOBAL, 'Global', cubeGeometry);
        const globalMesh = globalTensor.userData.mesh;
        scene.add(globalTensor);

        // Precompute highlighted coords in Global tensor for quick reset
        const highlightedGlobalIndices = globalTensor.userData.highlightedIndices;

        const allTensorChildren = [globalMesh];
        const hoverGeometry = new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05);
        const hoverEdgesGeometry = new THREE.EdgesGeometry(hoverGeometry);
        const hoverMaterial = new THREE.LineBasicMaterial({ color: COLOR_HOVER });
        const globalHoverOutline = new THREE.LineSegments(hoverEdgesGeometry, hoverMaterial);
        globalHoverOutline.visible = false;
        globalTensor.add(globalHoverOutline);

        const { center } = setupCamera(scene, camera);
        const orbitControls = new OrbitControls(camera, renderer.domElement);
        orbitControls.enableDamping = true;
        orbitControls.dampingFactor = 0.05;
        orbitControls.target.copy(center);
        orbitControls.update();
        const applyCameraState = () => {
            if (!viewState || !viewState.camera) return;
            const { position, target } = viewState.camera;
            if (position) camera.position.set(position[0], position[1], position[2]);
            if (target) orbitControls.target.set(target[0], target[1], target[2]);
            orbitControls.update();
        };
        const saveCameraState = () => {
            if (!viewState) return;
            viewState.camera = {
                position: camera.position.toArray(),
                target: orbitControls.target.toArray()
            };
        };
        applyCameraState();
        saveCameraState();
        orbitControls.addEventListener('change', () => {
            saveCameraState();
            requestRender();
        });

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const mouseDx = 0;
        const mouseDy = 0;
        const onKeyDown = cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ'));
        setupEventListeners(containerElement, camera, renderer, onMouseMove, onKeyDown, requestRender, saveCameraState);
        requestRender();

        function _updateMouseNDC(event) {
            const rect = renderer.domElement.getBoundingClientRect();
            const dpr = renderer.getPixelRatio();
            const px = (event.clientX - rect.left + mouseDx) * dpr;
            const py = (event.clientY - rect.top  + mouseDy) * dpr;
            const w = rect.width * dpr, h = rect.height * dpr;
            mouse.x = (px / w) * 2 - 1;
            mouse.y = -(py / h) * 2 + 1;
        }

        const tmpMatrix = new THREE.Matrix4();
        const tmpPosition = new THREE.Vector3();
        const tmpQuat = new THREE.Quaternion();
        const tmpScale = new THREE.Vector3();

        function _raycastAll() {
            raycaster.setFromCamera(mouse, camera);
            return raycaster.intersectObjects(allTensorChildren, true);
        }

        function instanceCoord(mesh, instanceId) {
            if (mesh.userData.coords) return mesh.userData.coords[instanceId];
            const { width, height } = mesh.userData.shape;
            const z = Math.floor(instanceId / (width * height));
            const rem = instanceId - z * width * height;
            const y = Math.floor(rem / width);
            const x = rem - y * width;
            return [x, y, z];
        }

        function instanceLocalPosition(mesh, instanceId, target) {
            mesh.getMatrixAt(instanceId, tmpMatrix);
            tmpMatrix.decompose(target, tmpQuat, tmpScale);
            return target;
        }

        function setOutlinePosition(outline, mesh, instanceId) {
            instanceLocalPosition(mesh, instanceId, tmpPosition);
            outline.position.copy(tmpPosition);
            outline.visible = true;
        }

        function baseGlobalColorByIndex(idx) {
            return highlightedGlobalIndices.has(idx) ? COLOR_SLICE : COLOR_GLOBAL;
        }

        function onMouseMove(event) {
            lastMouseEvent = event;
            if (hoverRaf) return;
            hoverRaf = requestAnimationFrame(() => {
                hoverRaf = null;
                handleMouseMove(lastMouseEvent);
            });
        }

        function handleMouseMove(event) {
            _updateMouseNDC(event);
            raycaster.setFromCamera(mouse, camera);

            let needsRender = false;

            const intersects = _raycastAll();
            const hit = intersects.length > 0 ? intersects[0] : null;
            const nextHover = hit && hit.instanceId !== undefined ? { mesh: hit.object, instanceId: hit.instanceId } : null;
            const prevOutline = hoveredHit ? globalHoverOutline : null;
            const nextOutline = nextHover ? globalHoverOutline : null;
            if (prevOutline && (!nextHover || hoveredHit.mesh !== nextHover.mesh || hoveredHit.instanceId !== nextHover.instanceId)) {
                if (prevOutline.visible) {
                    prevOutline.visible = false;
                    needsRender = true;
                }
            }
            hoveredHit = nextHover;

            if (hoveredHit) {
                if (nextOutline && !nextOutline.visible) {
                    setOutlinePosition(nextOutline, hoveredHit.mesh, hoveredHit.instanceId);
                    needsRender = true;
                }

                const tensorName = hoveredHit.mesh.userData.tensorName;
                const [tensor0, tensor1, tensor2] = instanceCoord(hoveredHit.mesh, hoveredHit.instanceId);
                const hoverKey = `${tensorName}:${tensor0},${tensor1},${tensor2}`;
                if (hoverKey !== lastHoverKey) {
                    lastHoverKey = hoverKey;
                    updateSideMenu(tensorName, tensor0, tensor1, tensor2, undefined);
                    const token = ++hoverToken;
                    getElementValue(tensorName, tensor0, tensor1, tensor2).then((res) => {
                        if (token !== hoverToken) return;
                        updateSideMenu(tensorName, tensor0, tensor1, tensor2, res.value);
                    });
                }
            } else {
                if (lastHoverKey !== null) updateSideMenu(null);
                lastHoverKey = null;
            }

            if (needsRender) requestRender();
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

        function destroyLegend() {
            if (legendEl && legendEl.remove) legendEl.remove();
            legendEl = null;
        }

        function createLegend(min, max) {
            destroyLegend();
            const wrapper = document.createElement('div');
            wrapper.style.position = 'absolute';
            wrapper.style.left = '10px';
            wrapper.style.top = '50px';
            wrapper.style.padding = '6px 8px';
            wrapper.style.background = 'rgba(0,0,0,0.6)';
            wrapper.style.color = '#fff';
            wrapper.style.font = '12px Arial, sans-serif';
            wrapper.style.borderRadius = '6px';
            wrapper.style.zIndex = '2000';

            const title = document.createElement('div');
            title.textContent = 'Value (Viridis)';
            title.style.marginBottom = '4px';
            title.style.opacity = '0.9';
            wrapper.appendChild(title);

            const canvas = document.createElement('canvas');
            canvas.width = 220; canvas.height = 10;
            const ctx2 = canvas.getContext('2d');
            for (let x = 0; x < canvas.width; x++) {
                const t = x / (canvas.width - 1);
                const c = viridisColor(t);
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

        function applyColorMap() {
            if (!colorizeOn || !tensorCache) return;
            const { min, max, dims, values } = tensorCache;
            const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
            const norm = (v) => (max === min ? 0.5 : (v - min) / (max - min));
            const toColor = (t) => viridisColor(clamp(norm(t), 0, 1));
            for (let idx = 0; idx < globalMesh.count; idx++) {
                const [x, y, z] = instanceCoord(globalMesh, idx);
                let v = 0.0;
                try {
                    if (dims === 3) v = values[x][y][z];
                    else if (dims === 2) v = values[x][y];
                    else if (dims === 1) v = values[x];
                } catch (e) { /* ignore bad index */ }
                globalMesh.setColorAt(idx, toColor(v));
            }
            if (globalMesh.instanceColor) globalMesh.instanceColor.needsUpdate = true;
        }

        function resetGlobalColors() {
            // restore original colors: global cubes default to COLOR_GLOBAL,
            // highlighted coords (in op.global_coords) are COLOR_SLICE
            for (let idx = 0; idx < globalMesh.count; idx++) {
                globalMesh.setColorAt(idx, baseGlobalColorByIndex(idx));
            }
            if (globalMesh.instanceColor) globalMesh.instanceColor.needsUpdate = true;
        }

        function requestRender() {
            if (rafId !== null) {
                renderPending = true;
                return;
            }
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
            if (colorizeOn) {
                if (!tensorCache) {
                    tensorCache = await fetchGlobalTensor();
                }
                if (tensorCache) {
                    applyColorMap();
                    createLegend(tensorCache.min, tensorCache.max);
                }
                requestRender();
                return;
            }
            resetGlobalColors();
            destroyLegend();
            requestRender();
        });

        function updateSideMenu(tensorName, x, y, z, value) {
            if (!tensorName) {
                sideMenu.innerHTML = '';
                return;
            }

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

}
