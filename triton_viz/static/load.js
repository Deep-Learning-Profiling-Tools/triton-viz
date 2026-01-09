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

        const controlBar = document.createElement('div');
        controlBar.className = 'viz-floating-bar';
        controlBar.style.flexWrap = 'wrap';

        const dragHandle = document.createElement('button');
        dragHandle.type = 'button';
        dragHandle.className = 'viz-drag-handle drag-handle';
        dragHandle.setAttribute('aria-label', 'Drag controls');
        dragHandle.innerHTML = '<span aria-hidden="true">⠿</span> Drag';
        controlBar.appendChild(dragHandle);

        const makeGhostButton = (label) => {
            const btn = document.createElement('button');
            btn.className = 'viz-button ghost';
            btn.textContent = label;
            return btn;
        };

        const colorizeToggle = makeGhostButton('Color by Value: OFF');
        controlBar.appendChild(colorizeToggle);

        const dragToggle = makeGhostButton('Drag Cubes: OFF');
        controlBar.appendChild(dragToggle);

        const codeToggle = makeGhostButton('Show Code: OFF');
        controlBar.appendChild(codeToggle);
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
        histogramUI.button.className = 'viz-button ghost';
        controlBar.appendChild(histogramUI.button);
        // Per-op summary (bytes) toggle
        const summaryBtn = makeGhostButton('Summary: OFF');
        controlBar.appendChild(summaryBtn);
        // Flow arrow toggle
        const flowToggle = makeGhostButton('Flow Arrow: ON');
        controlBar.appendChild(flowToggle);
        // Background selector + color scheme controls
        const backgroundSelect = document.createElement('select');
        backgroundSelect.className = 'viz-select';
        backgroundSelect.title = 'Canvas background';
        backgroundSelect.innerHTML = `
            <option value="#000000">Night</option>
            <option value="#0b1120">Midnight</option>
            <option value="#111827">Deep Blue</option>
            <option value="#ffffff">Paper</option>
            <option value="#f7f0e3">Beige</option>
            <option value="#f5f7fb">Soft Gray</option>
        `;
        controlBar.appendChild(backgroundSelect);

        const schemeSelect = document.createElement('select');
        schemeSelect.className = 'viz-select';
        schemeSelect.innerHTML = '<option value="mono">Mono</option><option value="viridis">Viridis</option>';
        controlBar.appendChild(schemeSelect);

        const colorPicker = document.createElement('input');
        colorPicker.type = 'color';
        colorPicker.value = '#3b82f6';
        colorPicker.title = 'Choose base color for Mono';
        controlBar.appendChild(colorPicker);
        // Mouse pick calibration (dx/dy in pixels)
        const calibWrap = document.createElement('div');
        calibWrap.className = 'viz-inline-controls';
        const calibLabel = document.createElement('span');
        calibLabel.textContent = 'Calib';
        const dxMinus = makeGhostButton('−X');
        const dxPlus  = makeGhostButton('+X');
        const dyMinus = makeGhostButton('−Y');
        const dyPlus  = makeGhostButton('+Y');
        const dxdyInfo = document.createElement('span');
        dxdyInfo.className = 'value-pill';
        const dxdyReset = makeGhostButton('Reset');
        calibWrap.appendChild(calibLabel);
        calibWrap.appendChild(dxMinus); calibWrap.appendChild(dxPlus);
        calibWrap.appendChild(dyMinus); calibWrap.appendChild(dyPlus);
        calibWrap.appendChild(dxdyInfo); calibWrap.appendChild(dxdyReset);
        controlBar.appendChild(calibWrap);
        containerElement.appendChild(controlBar);
        enableDrag(controlBar, { handle: dragHandle, bounds: window, initialLeft: 32, initialTop: 32 });

        // Load view summary panel (per-op bytes)
        let summaryPanel = null;
        const destroySummaryPanel = () => {
            if (summaryPanel && summaryPanel.remove) summaryPanel.remove();
            summaryPanel = null;
        };

        function openSummaryPanel() {
            destroySummaryPanel();
            const panel = document.createElement('div');
            panel.className = 'info-card';
            panel.style.position = 'fixed';
            panel.style.left = '32px';
            panel.style.maxWidth = '260px';
            panel.style.zIndex = '2200';

            const header = document.createElement('div');
            header.className = 'panel-header drag-handle';
            header.style.marginBottom = '6px';
            header.innerHTML = '<span>Load Summary</span><span class="drag-grip" aria-hidden="true">⠿</span>';
            const closeBtn = document.createElement('button');
            closeBtn.className = 'viz-button ghost';
            closeBtn.textContent = 'Close';
            closeBtn.style.marginLeft = 'auto';
            closeBtn.addEventListener('pointerdown', (e) => e.stopPropagation());
            closeBtn.addEventListener('click', () => {
                destroySummaryPanel();
                summaryBtn.textContent = 'Summary: OFF';
            });
            header.appendChild(closeBtn);
            panel.appendChild(header);

            const body = document.createElement('div');
            body.style.fontSize = '11px';
            body.innerHTML = `
                <div style="font-weight:600;margin-bottom:4px;">Current op</div>
                <div>Type: Load</div>
                <div>Bytes: ${Number(op.bytes || 0)}</div>
            `;

            panel.appendChild(body);
            document.body.appendChild(panel);
            enableDrag(panel, { handle: header, bounds: window, initialLeft: 32, initialTop: window.innerHeight - 220 });
            summaryPanel = panel;
        }

        summaryBtn.addEventListener('click', () => {
            const turnOn = summaryBtn.textContent.endsWith('OFF');
            summaryBtn.textContent = `Summary: ${turnOn ? 'ON' : 'OFF'}`;
            if (turnOn) {
                openSummaryPanel();
            } else {
                destroySummaryPanel();
            }
        });

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
        let codePanel = null;
let labelSprites = [];
const TEXT_LIGHT = '#ffffff';
const TEXT_DARK = '#111111';

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

        // Position slice tensor
        const globalSize = calculateTensorSize(op.global_shape);
        sliceTensor.position.set(globalSize.x + 5, 0, 0); // Adjusted tensor spacing

        scene.add(globalTensor);
        scene.add(sliceTensor);

        // Arrow helper (+ label) showing flow Global -> Slice
        const ARROW_COLOR = 0xffcc00; // gold
        const arrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0,0,0), 1.0, ARROW_COLOR, 0.25, 0.12);
        arrow.visible = true;
        scene.add(arrow);

        function createTextSprite(text, backgroundColor) {
            const c = document.createElement('canvas');
            const ctx2 = c.getContext('2d');
            const P = 4; // padding
            ctx2.font = 'bold 18px Arial';
            const metrics = ctx2.measureText(text);
            const w = Math.ceil(metrics.width) + P*2;
            const h = 28 + P*2;
            c.width = w; c.height = h;
            // redraw with proper size
            const ctx3 = c.getContext('2d');
            const luminance = (backgroundColor && backgroundColor.getLuminance) ? backgroundColor.getLuminance() : 0;
            const textColor = luminance > 0.6 ? '#111111' : '#ffffff';
            const bgColor = luminance > 0.6 ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.65)';
            ctx3.fillStyle = bgColor;
            ctx3.fillRect(0, 0, w, h);
            ctx3.fillStyle = textColor;
            ctx3.font = 'bold 18px Arial';
            ctx3.textBaseline = 'middle';
            ctx3.fillText(text, P, h/2);
            const tex = new THREE.CanvasTexture(c);
            tex.needsUpdate = true;
            const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
            const spr = new THREE.Sprite(mat);
            // scale proportional to text size
            const scaleX = Math.max(1, w / 64);
            const scaleY = Math.max(0.5, h / 64);
            spr.scale.set(scaleX, scaleY, 1);
            return spr;
        }

        const labelText = (()=>{
            const ms = (op.mem_src||'').toUpperCase();
            const md = (op.mem_dst||'').toUpperCase();
            if (ms && md) return `Load ${ms} → ${md}`;
            return 'Load Global → Slice';
        })();
        let arrowLabel = createTextSprite(labelText, currentBackground);
        arrowLabel.visible = true;
        // place label above the midpoint between tensors
        const midX = (globalTensor.position.x + sliceTensor.position.x) / 2;
        const textColor = getTextColor(currentBackground);
        arrowLabel.position.set(midX, 1.4, 0);
        arrowLabel.material.color = new THREE.Color(textColor);
        scene.add(arrowLabel);

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
            if (arrowLabel) scene.remove(arrowLabel);
            arrowLabel = createTextSprite(labelText, currentBackground);
            arrowLabel.visible = true;
            const midX = (globalTensor.position.x + sliceTensor.position.x) / 2;
            arrowLabel.position.set(midX, 1.4, 0);
            arrowLabel.material.color = new THREE.Color(getTextColor(currentBackground));
            scene.add(arrowLabel);
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

        backgroundSelect.addEventListener('change', (event) => {
            const value = event.target.value;
            currentBackground = new THREE.Color(value);
            if (scene && scene.background) {
                scene.background = currentBackground;
            }
            applyBackgroundTheme(value);
            refreshTextOverlays();
        });

        // 初始化一次，和默认背景同步
        applyBackgroundTheme(backgroundSelect.value || '#000000');

        const onKeyDown = cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ'));
        setupEventListeners(stage, camera, renderer, onMouseMove, onKeyDown);

        // Additional pointer events for dragging
        stage.addEventListener('mousedown', onMouseDown);
        stage.addEventListener('mouseup', onMouseUp);
        stage.addEventListener('mouseleave', onMouseUp);
        let flowArrowOn = true;
        flowToggle.addEventListener('click', () => {
            flowArrowOn = !flowArrowOn;
            flowToggle.textContent = `Flow Arrow: ${flowArrowOn ? 'ON' : 'OFF'}`;
            arrow.visible = flowArrowOn;
            arrowLabel.visible = flowArrowOn;
        });
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
            wrapper.className = 'viz-floating-badge';
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
                codeToggle.textContent = 'Show Code: OFF';
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
                    return values[userData.tensor0][userData.tensor1][userData.tensor2];
                } else if (dims === 2) {
                    return values[userData.tensor0][userData.tensor1];
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

                    // Update arrow position/direction and label at midpoint
                    if (flowArrowOn) {
                        const gCube = globalTensor.children.find(c => c.userData && c.userData.tensor0 === globalCoord[0] && c.userData.tensor1 === globalCoord[1] && c.userData.tensor2 === globalCoord[2]);
                        const sCube = sliceTensor.children.find(c => c.userData && c.userData.tensor0 === sliceCoord[0] && c.userData.tensor1 === sliceCoord[1] && c.userData.tensor2 === sliceCoord[2]);
                        if (gCube && sCube) {
                            const src = new THREE.Vector3();
                            const dst = new THREE.Vector3();
                            gCube.getWorldPosition(src);
                            sCube.getWorldPosition(dst);
                            const dir = new THREE.Vector3().subVectors(dst, src);
                            const len = dir.length();
                            if (len > 1e-6) {
                                arrow.visible = true;
                                arrow.setDirection(dir.normalize());
                                // arrow length a bit shorter than full to avoid piercing cubes
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

        colorizeToggle.addEventListener('click', async () => {
            colorizeOn = !colorizeOn;
            colorizeToggle.textContent = `Color by Value: ${colorizeOn ? 'ON' : 'OFF'}`;
            if (!colorizeOn) {
                resetGlobalColors();
                resetSliceColors();
                destroyLegend();
                return;
            }
            if (!tensorCache) {
                tensorCache = await fetchTensorPayload();
            }
            if (!tensorCache) {
                colorizeOn = false;
                colorizeToggle.textContent = 'Color by Value: OFF';
                return;
            }
            createLegend(tensorCache.scaleMin, tensorCache.scaleMax);
            applyColorMapIfNeeded();
        });

        schemeSelect.addEventListener('change', () => {
            scheme = schemeSelect.value;
            if (colorizeOn && tensorCache) {
                applyColorMapIfNeeded();
                createLegend(tensorCache.scaleMin, tensorCache.scaleMax);
            }
            // color picker visible only for mono
            colorPicker.style.display = scheme === 'mono' ? 'block' : 'none';
        });

        colorPicker.addEventListener('input', (e) => {
            monoBaseHex = e.target.value || '#3b82f6';
            if (scheme === 'mono' && colorizeOn && tensorCache) {
                applyColorMapIfNeeded();
                createLegend(tensorCache.scaleMin, tensorCache.scaleMax);
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
            menu.className = 'viz-side-panel';
            menu.style.width = '240px';
            container.appendChild(menu);
            return menu;
        }

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
