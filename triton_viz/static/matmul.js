import * as THREE from 'https://esm.sh/three@0.155.0/build/three.module.js';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';

export function createMatMulVisualization(containerElement, op, viewState = null) {
    const API_BASE = window.__TRITON_VIZ_API__ || '';
    const { input_shape, other_shape, output_shape } = op;
    console.log(op.uuid)
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
    let currentStep = 0; // kept for compatibility with getValue; not used for animation


    const sideMenu = document.createElement('div');
    // Use fixed position and high z-index to ensure it's above WebGL canvas
    sideMenu.style.position = 'absolute';
    sideMenu.style.top = '10px';
    sideMenu.style.right = '10px';
    sideMenu.style.width = '200px';
    sideMenu.style.padding = '10px';
    sideMenu.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    sideMenu.style.color = 'white';
    sideMenu.style.fontFamily = 'Arial, sans-serif';
    sideMenu.style.fontSize = '14px';
    sideMenu.style.borderRadius = '5px';
    sideMenu.style.zIndex = '3000';
    sideMenu.style.pointerEvents = 'auto';
    containerElement.appendChild(sideMenu);

    let hoveredHit = null;
    let lastHoverKey = null;
    let hoverToken = 0;
    let hoverRaf = null;
    let lastMouseEvent = null;
    let highlightA = [];
    let highlightB = [];
    let highlightC = [];
    let rafId = null;
    let renderPending = false;




    async function getElementValue( matrixName, row, col) {
        let uuid = op.uuid;
        const response = await fetch(`${API_BASE}/api/getValue`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ uuid, matrixName, row, col, currentStep }),
        });
        return await response.json();
    }


    function updateSideMenu(matrixOrName, x, y, vectors, value) {
        if (!matrixOrName) {
            sideMenu.innerHTML = '';
            return;
        }

        let matrixName;
        let dims;
        // Accept both string name ('A'|'B'|'C') and matrix group object
        if (typeof matrixOrName === 'string') {
            const name = matrixOrName.toUpperCase();
            if (name === 'A') { matrixName = 'A'; dims = input_shape; }
            else if (name === 'B') { matrixName = 'B'; dims = other_shape; }
            else if (name === 'C') { matrixName = 'C'; dims = output_shape; }
            else { sideMenu.innerHTML = ''; return; }
        } else {
            if (matrixOrName === matrixA) { matrixName = 'A'; dims = input_shape; }
            else if (matrixOrName === matrixB) { matrixName = 'B'; dims = other_shape; }
            else if (matrixOrName === matrixC) { matrixName = 'C'; dims = output_shape; }
            else { sideMenu.innerHTML = ''; return; }
        }
        console.log(matrixName, "x:", (x + 1), "y:", (y + 1));
        let extra = '';
        if (matrixName === 'C' && vectors && !vectors.error) {
            const aRow = vectors.a_row || [];
            const bCol = vectors.b_col || [];
            const k = vectors.k || 0;
            extra = `
                <hr/>
                <div><b>From A row</b>: [${aRow.slice(0,8).join(', ')}${aRow.length>8?' …':''}]</div>
                <div><b>From B col</b>: [${bCol.slice(0,8).join(', ')}${bCol.length>8?' …':''}]</div>
                <div><b>k</b>: ${k}</div>
            `;
        }

        sideMenu.innerHTML = `
            <h3 style="margin-top: 0;">Matrix ${matrixName}</h3>
            <p>Row: ${y + 1}</p>
            <p>Column: ${x + 1}</p>
            <p>Dimensions: ${dims[0]} x ${dims[1]}</p>
            <p>Value: ${value !== undefined ? value : 'Loading...'}</p>
            ${extra}
        `;
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const mouseDx = 0;
    const mouseDy = 0;


    function _updateMouseNDC(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        const dpr = renderer.getPixelRatio();
        const px = (event.clientX - rect.left + mouseDx) * dpr;
        const py = (event.clientY - rect.top  + mouseDy) * dpr;
        const w = rect.width * dpr, h = rect.height * dpr;
        mouse.x = (px / w) * 2 - 1;
        mouse.y = -(py / h) * 2 + 1;
    }

    function instanceCoord(mesh, instanceId) {
        const cols = mesh.userData.cols;
        const row = Math.floor(instanceId / cols);
        const col = instanceId - row * cols;
        return [row, col];
    }

    const tmpMatrix = new THREE.Matrix4();
    const tmpPosition = new THREE.Vector3();
    const tmpQuat = new THREE.Quaternion();
    const tmpScale = new THREE.Vector3();

    function instanceLocalPosition(mesh, instanceId, target) {
        mesh.getMatrixAt(instanceId, tmpMatrix);
        tmpMatrix.decompose(target, tmpQuat, tmpScale);
        return target;
    }

    function setOutlinePosition(mesh, instanceId) {
        instanceLocalPosition(mesh, instanceId, tmpPosition);
        mesh.localToWorld(tmpPosition);
        hoverOutline.position.copy(tmpPosition);
        hoverOutline.visible = true;
    }

    function clearHighlight(list, mesh, baseColor) {
        if (list.length === 0) return;
        list.forEach((idx) => mesh.setColorAt(idx, baseColor));
        list.length = 0;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    function applyHighlights(row, col) {
        clearHighlight(highlightA, matrixA, COLOR_A);
        clearHighlight(highlightB, matrixB, COLOR_B);
        if (!colorOn) clearHighlight(highlightC, matrixC, COLOR_C);

        const aCols = matrixA.userData.cols;
        const bCols = matrixB.userData.cols;
        const cCols = matrixC.userData.cols;

        for (let j = 0; j < aCols; j++) {
            const idx = row * aCols + j;
            highlightA.push(idx);
            matrixA.setColorAt(idx, COLOR_HIGHLIGHT);
        }
        for (let i = 0; i < matrixB.userData.rows; i++) {
            const idx = i * bCols + col;
            highlightB.push(idx);
            matrixB.setColorAt(idx, COLOR_HIGHLIGHT);
        }
        if (!colorOn) {
            const cIdx = row * cCols + col;
            highlightC.push(cIdx);
            matrixC.setColorAt(cIdx, COLOR_FILLED);
        }

        if (matrixA.instanceColor) matrixA.instanceColor.needsUpdate = true;
        if (matrixB.instanceColor) matrixB.instanceColor.needsUpdate = true;
        if (matrixC.instanceColor) matrixC.instanceColor.needsUpdate = true;
    }

    function clearHighlights(includeC) {
        clearHighlight(highlightA, matrixA, COLOR_A);
        clearHighlight(highlightB, matrixB, COLOR_B);
        if (includeC) clearHighlight(highlightC, matrixC, COLOR_C);
    }

    function onMouseMove(event) {
        lastMouseEvent = event;
        if (hoverRaf) return;
        hoverRaf = requestAnimationFrame(() => {
            hoverRaf = null;
            handleMouseMove(lastMouseEvent);
        });
    }

    async function handleMouseMove(event) {
        _updateMouseNDC(event);
        raycaster.setFromCamera(mouse, camera);

        let needsRender = false;
        const intersects = raycaster.intersectObjects([matrixA, matrixB, matrixC], true);
        const hit = intersects.length > 0 ? intersects[0] : null;
        const nextHover = hit && hit.instanceId !== undefined ? { mesh: hit.object, instanceId: hit.instanceId } : null;

        if (hoveredHit && (!nextHover || hoveredHit.mesh !== nextHover.mesh || hoveredHit.instanceId !== nextHover.instanceId)) {
            if (hoverOutline.visible) {
                hoverOutline.visible = false;
                needsRender = true;
            }
        }
        hoveredHit = nextHover;

        if (hoveredHit) {
            setOutlinePosition(hoveredHit.mesh, hoveredHit.instanceId);
            needsRender = true;

            const matrixName = hoveredHit.mesh.userData.matrixName;
            const [row, col] = instanceCoord(hoveredHit.mesh, hoveredHit.instanceId);
            const hoverKey = `${matrixName}:${row},${col}`;
            if (hoverKey !== lastHoverKey) {
                lastHoverKey = hoverKey;
                updateSideMenu(matrixName, col, row, null, undefined);
                const token = ++hoverToken;
                const res = await getElementValue(matrixName, row, col);
                if (token !== hoverToken) return;

                let vectors = null;
                let valueForPanel = res && res.value !== undefined ? res.value : undefined;
                if (matrixName === 'C') {
                    try {
                        const resp = await fetch(`${API_BASE}/api/getMatmulVectors`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ uuid: op.uuid, row, col })
                        });
                        vectors = await resp.json();
                        if (vectors && !vectors.error && Array.isArray(vectors.a_row) && Array.isArray(vectors.b_col)) {
                            const aRow = vectors.a_row;
                            const bCol = vectors.b_col;
                            const len = Math.min(aRow.length, bCol.length);
                            let sum = 0.0;
                            for (let i = 0; i < len; i++) sum += aRow[i] * bCol[i];
                            valueForPanel = sum;
                        }
                    } catch (e) { vectors = { error: String(e) }; }

                    applyHighlights(row, col);
                    needsRender = true;
                } else {
                    clearHighlights(!colorOn);
                    needsRender = true;
                }
                updateSideMenu(matrixName, col, row, vectors, valueForPanel);
            }
        } else {
            lastHoverKey = null;
            updateSideMenu(null);
            clearHighlights(!colorOn);
            needsRender = true;
        }

        if (needsRender) requestRender();
    }





    const CUBE_SIZE = 0.2;
    const GAP = 0.05;

    const COLOR_A = new THREE.Color(0.21, 0.46, 1.0);
    const COLOR_B = new THREE.Color(1.0, 1.0, 0.31);
    const COLOR_C = new THREE.Color(0.59, 1.0, 0.37);
    const COLOR_HIGHLIGHT = new THREE.Color(0.0, 0.0, 1.0);
    const COLOR_FILLED = new THREE.Color(0.0, 0.0, 1.0);
    const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);
    const COLOR_HOVER = new THREE.Color(1.0, 1.0, 0.0);

    const scene = new THREE.Scene();
    scene.background = COLOR_BACKGROUND;
    const camera = new THREE.PerspectiveCamera(45, containerElement.clientWidth / containerElement.clientHeight, 0.1, 1000);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    renderer.setPixelRatio(dpr);
    renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    containerElement.appendChild(renderer.domElement);
    // Ensure canvas is under overlays
    renderer.domElement.style.position = 'relative';
    renderer.domElement.style.zIndex = '1';
    try { containerElement.style.pointerEvents = 'auto'; } catch(e){}

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    const cubeGeometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    const normals = cubeGeometry.attributes.normal;
    const colorArray = new Float32Array(cubeGeometry.attributes.position.count * 3);
    const lightDir = new THREE.Vector3(0.35, 0.6, 0.7).normalize();
    for (let i = 0; i < normals.count; i++) {
        const nx = normals.getX(i);
        const ny = normals.getY(i);
        const nz = normals.getZ(i);
        const ndotl = Math.max(0, nx * lightDir.x + ny * lightDir.y + nz * lightDir.z);
        const shade = 0.6 + 0.4 * ndotl;
        const base = i * 3;
        colorArray[base] = shade;
        colorArray[base + 1] = shade;
        colorArray[base + 2] = shade;
    }
    cubeGeometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    const spacing = CUBE_SIZE + GAP;

    function matrixSize(dimensions) {
        const rows = dimensions[0];
        const cols = dimensions[1];
        const width = (cols - 1) * spacing + CUBE_SIZE;
        const height = (rows - 1) * spacing + CUBE_SIZE;
        return {
            rows,
            cols,
            width,
            height
        };
    }

    function createMatrixMesh(dimensions, position, color, matrixName) {
        const { rows, cols } = matrixSize(dimensions);
        const count = rows * cols;
        const material = new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true });
        const mesh = new THREE.InstancedMesh(cubeGeometry, material, count);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
        const matrix = new THREE.Matrix4();
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const idx = i * cols + j;
                matrix.setPosition(position.x + j * spacing, position.y - i * spacing, position.z);
                mesh.setMatrixAt(idx, matrix);
                mesh.setColorAt(idx, color);
            }
        }
        mesh.userData.matrixName = matrixName;
        mesh.userData.rows = rows;
        mesh.userData.cols = cols;
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        mesh.computeBoundingBox();
        mesh.computeBoundingSphere();
        return mesh;
    }

    const gap = GAP;
    const sizeA = matrixSize(input_shape);
    const sizeB = matrixSize(other_shape);
    const sizeC = matrixSize(output_shape);
    const posC = new THREE.Vector3(0, 0, 0);
    const posA = new THREE.Vector3(-(sizeA.width + gap), 0, 0);
    const posB = new THREE.Vector3(0, sizeB.height + gap, 0);

    const matrixA = createMatrixMesh(input_shape, posA, COLOR_A, 'A');
    const matrixB = createMatrixMesh(other_shape, posB, COLOR_B, 'B');
    const matrixC = createMatrixMesh(output_shape, posC, COLOR_C, 'C');

    scene.add(matrixA);
    scene.add(matrixB);
    scene.add(matrixC);
    const hoverGeometry = new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05);
    const hoverEdgesGeometry = new THREE.EdgesGeometry(hoverGeometry);
    const hoverOutline = new THREE.LineSegments(hoverEdgesGeometry, new THREE.LineBasicMaterial({ color: COLOR_HOVER }));
    hoverOutline.visible = false;
    scene.add(hoverOutline);

    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    const box = new THREE.Box3().setFromObject(scene);
    box.getCenter(center);
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5;

    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.lookAt(center);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.copy(center);
    controls.update();
    const applyCameraState = () => {
        if (!viewState || !viewState.camera) return;
        const { position, target } = viewState.camera;
        if (position) camera.position.set(position[0], position[1], position[2]);
        if (target) controls.target.set(target[0], target[1], target[2]);
        controls.update();
    };
    const saveCameraState = () => {
        if (!viewState) return;
        viewState.camera = {
            position: camera.position.toArray(),
            target: controls.target.toArray()
        };
    };
    applyCameraState();
    saveCameraState();
    controls.addEventListener('change', () => {
        saveCameraState();
        requestRender();
    });

    function resetColors() {
        clearHighlights(true);
        for (let idx = 0; idx < matrixA.count; idx++) {
            matrixA.setColorAt(idx, COLOR_A);
        }
        for (let idx = 0; idx < matrixB.count; idx++) {
            matrixB.setColorAt(idx, COLOR_B);
        }
        for (let idx = 0; idx < matrixC.count; idx++) {
            matrixC.setColorAt(idx, COLOR_C);
        }
        if (matrixA.instanceColor) matrixA.instanceColor.needsUpdate = true;
        if (matrixB.instanceColor) matrixB.instanceColor.needsUpdate = true;
        if (matrixC.instanceColor) matrixC.instanceColor.needsUpdate = true;
    }

    function requestRender() {
        if (rafId !== null) {
            renderPending = true;
            return;
        }
        rafId = requestAnimationFrame(renderFrame);
    }

    function renderFrame() {
        const needsMore = controls.update();
        renderer.render(scene, camera);
        if (needsMore || renderPending) {
            renderPending = false;
            rafId = requestAnimationFrame(renderFrame);
            return;
        }
        rafId = null;
    }

    function onResize() {
        camera.aspect = containerElement.clientWidth / containerElement.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
        requestRender();
    }

    const controlPanel = document.createElement('div');
    controlPanel.style.position = 'absolute';
    controlPanel.style.bottom = '10px';
    controlPanel.style.left = '10px';
    controlPanel.style.display = 'flex';
    controlPanel.style.gap = '10px';
    controlPanel.style.zIndex = '3000';
    controlPanel.style.pointerEvents = 'auto';

    // Removed animation controls; keep panel for color-by-value toggle only

    // Color by Value (for C matrix) toggle + legend (mono blue)
    const colorToggle = document.createElement('button');
    colorToggle.textContent = 'Color by Value: OFF';
    controlPanel.appendChild(colorToggle);
    let colorOn = false;
    let legendEl = null;
    function destroyLegend(){ if(legendEl && legendEl.remove) legendEl.remove(); legendEl=null; }
    function createLegend(min,max){
        destroyLegend();
        const w = document.createElement('div');
        Object.assign(w.style,{position:'absolute', left:'10px', bottom:'60px', background:'rgba(0,0,0,0.6)', color:'#fff', padding:'6px 8px', borderRadius:'6px', zIndex:'2000', pointerEvents:'auto'});
        const c = document.createElement('canvas'); c.width=220; c.height=10; const ctx=c.getContext('2d');
        for(let x=0;x<c.width;x++){ const t=x/(c.width-1); const r=t,g=0.2,b=1-t; ctx.fillStyle=`rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`; ctx.fillRect(x,0,1,c.height);}
        const lab=document.createElement('div'); lab.style.display='flex'; lab.style.justifyContent='space-between'; lab.style.marginTop='2px'; lab.innerHTML=`<span>${min.toFixed(3)}</span><span>${max.toFixed(3)}</span>`;
        const ttl=document.createElement('div'); ttl.textContent='Value (C)'; ttl.style.marginBottom='4px'; ttl.style.opacity='0.9';
        w.appendChild(ttl); w.appendChild(c); w.appendChild(lab); containerElement.appendChild(w); legendEl=w;
    }
    async function fetchCValues(){
        try{ const res=await fetch(`${API_BASE}/api/getMatmulC`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uuid: op.uuid})});
            return await res.json(); }catch(e){ return {error:String(e)} }
    }
    function applyColorCWithData(data){ if(!colorOn||!data||data.error) return; try{
            const M=(data.shape||[])[0]||0, N=(data.shape||[])[1]||0; const vals2d=data.values||[];
            const vals=[]; const rows=matrixC.userData.rows; const cols=matrixC.userData.cols;
            for(let r=0;r<rows;r++){ for(let c=0;c<cols;c++){ vals.push((r<M&&c<N)? vals2d[r][c]:0.0); } }
            const mn=Math.min(...vals), mx=Math.max(...vals);
            for(let idx=0;idx<vals.length;idx++){
                const v=vals[idx]; const u=(mx===mn)?0.5:(v-mn)/(mx-mn); const r=u,g=0.2,b=1-u;
                matrixC.setColorAt(idx, new THREE.Color(r,g,b));
            }
            if (matrixC.instanceColor) matrixC.instanceColor.needsUpdate = true;
            createLegend(mn,mx);
        }catch(e){}
    }
    colorToggle.addEventListener('click', async ()=>{ colorOn=!colorOn; colorToggle.textContent=`Color by Value: ${colorOn?'ON':'OFF'}`; if(!colorOn){ destroyLegend(); resetColors(); } else { clearHighlights(true); const data=await fetchCValues(); applyColorCWithData(data); } requestRender(); });

    containerElement.appendChild(controlPanel);

    const PAN_SPEED = 0.1;
    const TILT_SPEED = 0.02;
    const ZOOM_SPEED = 0.5;
    let cameraRotation = new THREE.Euler(0, 0, 0, 'YXZ');

    function onKeyDown(event) {
        switch (event.key.toLowerCase()) {
            case 'w':
                camera.position.y += PAN_SPEED;
                break;
            case 's':
                camera.position.y -= PAN_SPEED;
                break;
            case 'a':
                camera.position.x -= PAN_SPEED;
                break;
            case 'd':
                camera.position.x += PAN_SPEED;
                break;
            case 'arrowup':
                cameraRotation.x -= TILT_SPEED;
                break;
            case 'arrowdown':
                cameraRotation.x += TILT_SPEED;
                break;
            case 'arrowleft':
                cameraRotation.y -= TILT_SPEED;
                break;
            case 'arrowright':
                cameraRotation.y += TILT_SPEED;
                break;
            case 'o':
                camera.position.z += ZOOM_SPEED;
                break;
            case 'p':
                camera.position.z -= ZOOM_SPEED;
                break;
        }
        camera.setRotationFromEuler(cameraRotation);
        camera.updateProjectionMatrix();
        saveCameraState();
        requestRender();
    }

    window.addEventListener('resize', onResize);
    window.addEventListener('keydown', onKeyDown);
    containerElement.addEventListener('mousemove', onMouseMove);
    try { window.current_op_uuid = op.uuid; } catch (e) {}

    // Mouse wheel zoom for matmul view
    const WHEEL_ZOOM_SPEED = 0.5;
    containerElement.addEventListener('wheel', (event) => {
        event.preventDefault();
        const direction = event.deltaY > 0 ? 1 : -1;
        camera.position.z += direction * WHEEL_ZOOM_SPEED;
        camera.updateProjectionMatrix();
        saveCameraState();
        requestRender();
    }, { passive: false });
    requestRender();


    function cleanup() {
        window.removeEventListener('resize', onResize);
        window.removeEventListener('keydown', onKeyDown);
        containerElement.removeEventListener('mousemove', onMouseMove);
        containerElement.innerHTML = '';
        renderer.dispose();
        scene.clear();
    }

    return cleanup;
}
